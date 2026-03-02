"""
Telegram command and message handlers.
All handlers are async. Only responds to the authorized user.
"""

import html
import io
import logging
import re
import tempfile
import time
from pathlib import Path

from PIL import Image as _PILImage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from agent import asset_gen, asset_library, auto_state, brain, chat, compositor, compositor_config, conversation_context, engine, feedback, generation_history, guidelines, image_gen, intent_router, onboarding, publisher, schedule_queue, scheduler, state
from agent import compositor_config as _cc
from config import settings

logger = logging.getLogger(__name__)

# Rate limiting — minimum seconds between generation requests per user
_RATE_LIMIT_SECONDS = 10
_last_request_time: dict[int, float] = {}

# Bulk upload batch tracking — maps user_id to pending asyncio task
import asyncio as _aio
_bulk_upload_tasks: dict[int, _aio.Task] = {}


def _authorized(user_id: int) -> bool:
    """Check if a Telegram user is the authorized operator."""
    return user_id == settings.TELEGRAM_ALLOWED_USER_ID


# Patterns that indicate the user wants to generate a template from their reference image
_TEMPLATE_FROM_REF_PATTERNS = [
    "make a template",
    "make template",
    "create a template",
    "create template",
    "generate a template",
    "generate template",
    "use this layout",
    "use this as a template",
    "use this as template",
    "turn this into a template",
    "template from this",
    "template this",
    "copy this layout",
    "replicate this layout",
    "recreate this layout",
    "use this format",
    "copy this format",
]


def _is_template_from_ref_intent(caption: str) -> bool:
    """Check if a photo caption expresses intent to generate a template from reference."""
    lower = caption.lower().strip()
    return any(p in lower for p in _TEMPLATE_FROM_REF_PATTERNS)


# Patterns indicating the user wants to use their uploaded photo directly (no AI generation)
_DIRECT_PHOTO_PATTERNS = [
    "use this", "post this", "announce this", "use this photo",
    "use this image", "publish this", "tweet this", "share this",
    "put this in", "use my photo", "use my image",
]


# Keywords that suggest the user is describing template region positions
_REGION_POSITION_KEYWORDS = [
    "top", "bottom", "left", "right", "centered", "center",
    "full canvas", "full width", "entire background", "background",
    "text goes", "text across", "image zone", "image area",
    "title", "subtitle", "headline",
]


def _is_template_region_update(message: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if a message describes template region positions after a recent upload.

    Returns True if:
    - A template was uploaded within the last 2 messages (user_data has last_uploaded_template_id)
    - The message contains position keywords + percentage or layout terms
    """
    user_data = context.user_data if context else {}
    template_id = user_data.get("last_uploaded_template_id")
    if not template_id:
        return False

    lower = message.lower()
    keyword_hits = sum(1 for kw in _REGION_POSITION_KEYWORDS if kw in lower)
    has_percentage = "%" in lower
    has_region_type = any(w in lower for w in ("text", "image", "logo"))

    # Need at least 2 keyword hits AND (a percentage or a region type word)
    return keyword_hits >= 2 and (has_percentage or has_region_type)


def _is_direct_photo_intent(caption: str) -> bool:
    """Check if a photo caption means 'use this photo directly, don't regenerate it'."""
    lower = caption.lower().strip()
    return any(p in lower for p in _DIRECT_PHOTO_PATTERNS)


def _rate_limited(user_id: int) -> bool:
    """Check if user is sending requests too fast. Returns True if blocked."""
    now = time.time()
    last = _last_request_time.get(user_id, 0)
    if now - last < _RATE_LIMIT_SECONDS:
        return True
    _last_request_time[user_id] = now
    return False


# ---------------------------------------------------------------------------
# Shared core logic — used by slash commands, NL router, and inline buttons
# ---------------------------------------------------------------------------

async def _do_approve(update: Update, context: ContextTypes.DEFAULT_TYPE, option_num: int = 1, source: str = "command") -> None:
    """Core approve logic shared by /approve, NL router, and inline buttons."""
    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("Nothing to approve. Send me a content request first.")
        return

    # If multiple image options exist, select the chosen one
    image_urls = pending.get("image_urls", [])
    if image_urls and 1 <= option_num <= len(image_urls):
        pending["image_url"] = image_urls[option_num - 1]
        logger.info("Approve (%s): selected option %d of %d", source, option_num, len(image_urls))
    elif image_urls and option_num > len(image_urls):
        await update.message.reply_text(
            f"Only {len(image_urls)} options available. Use /approve 1-{len(image_urls)}."
        )
        return

    await update.message.chat.send_action("typing")

    # Log feedback
    count = await feedback.async_log_feedback(
        request=pending.get("original_request", ""),
        draft=pending,
        accepted=True,
        resources_used=pending.get("resources_used", []),
    )

    # Update generation history status
    try:
        ts = pending.get("timestamp", 0)
        if ts:
            await generation_history.async_update_generation_status(ts, "approved")
    except Exception as e:
        logger.debug("Generation history update failed: %s", e)

    # Save approved composed image to brand/references/ for style consistency
    composed_path, composed_ct = state.get_last_composed()
    if composed_path and Path(composed_path).exists():
        try:
            refs_dir = Path(settings.BRAND_FOLDER) / "references"
            refs_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            save_name = f"approved_{composed_ct}_{ts}.png"
            save_path = refs_dir / save_name
            import shutil
            shutil.copy2(composed_path, save_path)
            logger.info("Saved approved reference: %s", save_path)

            # Cap at 5 per content_type — delete oldest
            existing = sorted(refs_dir.glob(f"approved_{composed_ct}_*.png"))
            if len(existing) > 5:
                for old in existing[:-5]:
                    old.unlink(missing_ok=True)
                    logger.info("Pruned old reference: %s", old.name)
        except Exception as e:
            logger.warning("Failed to save approved reference: %s", e)

        # Save into active style profile if one is set for this content_type
        try:
            active_profile = state.get_active_profile(composed_ct)
            if active_profile:
                count_p = state.add_profile_image(active_profile, composed_path)
                logger.info(
                    "Saved approved image to profile %s (%d images)",
                    active_profile, count_p,
                )
        except Exception as e:
            logger.warning("Failed to save to style profile: %s", e)

        # Add to asset library
        try:
            asset_library.add(
                composed_path, "approved", composed_ct or "general",
                prompt=pending.get("image_prompt", ""),
                tags=["approved"],
            )
        except Exception as e:
            logger.debug("Asset library add failed: %s", e)

    # Save approved mascot outputs to grow character reference library
    _mascot_kw = re.compile(r"mascot|character", re.IGNORECASE)
    _is_mascot_draft = (
        _mascot_kw.search(pending.get("original_request", ""))
        or _mascot_kw.search(pending.get("image_prompt", ""))
    )
    if _is_mascot_draft and pending.get("image_url"):
        try:
            from agent._client import get_httpx as _get_httpx
            _r = await _get_httpx().get(pending["image_url"])
            _r.raise_for_status()
            ts = int(time.time())
            save_path = Path(settings.BRAND_FOLDER) / "assets" / f"mascot_approved_{ts}.png"
            _PILImage.open(io.BytesIO(_r.content)).convert("RGB").save(str(save_path), "PNG")
            logger.info("Saved approved mascot output: %s", save_path)
        except Exception as e:
            logger.warning("Failed to save mascot output: %s", e)

    # Add to LoRA training set
    if pending.get("image_url"):
        try:
            from agent import lora_pipeline
            img_url = pending["image_url"]
            img_prompt = pending.get("image_prompt", "")
            ct = pending.get("content_type", composed_ct or "announcement")
            lora_count, threshold_hit = await lora_pipeline.add_training_image_from_url(
                img_url, img_prompt, ct,
            )
            logger.info("LoRA training image added (%d total)", lora_count)
            if threshold_hit:
                await update.message.reply_text(
                    f"Training set reached {lora_count} images! "
                    f"Use /train_lora to start LoRA training.",
                )
        except Exception as e:
            logger.warning("Failed to add LoRA training image: %s", e)

    # Post to X — prefer composed image (template/compositor) over raw URL
    tweet_url = None
    try:
        await update.message.chat.send_action("typing")
        publish_image = pending.get("image_url")
        if composed_path and Path(composed_path).exists():
            publish_image = composed_path
        tweet_url = await publisher.post_to_x(
            pending.get("caption", ""),
            pending.get("hashtags", []),
            publish_image,
        )
    except Exception as e:
        logger.error("Failed to post to X: %s", e)
        await update.message.reply_text(
            f"Approved, but X posting failed. Check logs for details.\n"
            f"Feedback logged ({count} total entries).",
            parse_mode="HTML",
        )
        state.clear_pending()
        state.clear_draft_history()
        if composed_path and Path(composed_path).exists():
            Path(composed_path).unlink(missing_ok=True)
            state.clear_last_composed()
        return

    # If this draft came from the auto-post scheduler, record it
    auto_slot = pending.get("auto_slot")
    if auto_slot:
        auto_state.record_post(
            slot_name=auto_slot,
            caption=pending.get("caption", ""),
            tweet_url=tweet_url,
            event_ids=pending.get("auto_event_ids"),
        )
        logger.info("Auto-post slot '%s' recorded via approve (%s)", auto_slot, source)

    state.clear_pending()
    state.clear_draft_history()
    slot_note = f"  (auto-slot: {_esc(auto_slot)})" if auto_slot else ""
    await update.message.reply_text(
        f"Posted to X!{slot_note}\n"
        f"{_esc(tweet_url)}\n\n"
        f"Feedback logged ({count} total entries).",
        parse_mode="HTML",
    )
    logger.info("Draft approved (%s) and posted: %s (feedback #%d)", source, tweet_url, count)

    # Track context — draft approved, nothing pending
    try:
        user_id = update.effective_user.id if update.effective_user else 0
        if user_id:
            conversation_context.update_context(
                user_id,
                last_bot_action="sent_content",
                pending_draft_exists=False,
                last_command="/approve",
            )
    except Exception as e:
        logger.debug("Context tracking failed in _do_approve: %s", e)

    # Clean up temp composed file (after publish so it's still available for X upload)
    if composed_path and Path(composed_path).exists():
        try:
            Path(composed_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("Composed cleanup failed for %s: %s", composed_path, e)
        state.clear_last_composed()

    # Auto-summarize preferences at threshold
    if count % settings.FEEDBACK_SUMMARIZE_EVERY == 0:
        try:
            await update.message.reply_text("Auto-learning preferences from feedback history...")
            summary = await feedback.summarize_preferences()
            await update.message.reply_text(
                f"Learned preferences updated ({len(summary)} chars).",
            )
        except Exception as e:
            logger.error("Auto-summarize failed: %s", e)


async def _do_reject(update: Update, context: ContextTypes.DEFAULT_TYPE, feedback_text: str = "", source: str = "command") -> None:
    """Core reject logic shared by /reject, NL router, and inline buttons."""
    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("Nothing to reject. Send me a content request first.")
        return

    if not feedback_text:
        await update.message.reply_text(
            "Please include feedback: /reject <i>make it more urgent and add a CTA</i>",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")

    # Log the rejection
    count = await feedback.async_log_feedback(
        request=pending.get("original_request", ""),
        draft=pending,
        accepted=False,
        feedback_text=feedback_text,
        resources_used=pending.get("resources_used", []),
    )
    logger.info("Draft rejected (%s, feedback #%d): %s", source, count, feedback_text[:100])

    # Update generation history status
    try:
        ts = pending.get("timestamp", 0)
        if ts:
            await generation_history.async_update_generation_status(ts, "rejected")
    except Exception as e:
        logger.debug("Generation history update failed: %s", e)

    # Auto-summarize at threshold
    if count % settings.FEEDBACK_SUMMARIZE_EVERY == 0:
        try:
            await feedback.summarize_preferences()
            logger.info("Auto-summarized preferences after %d entries", count)
        except Exception as e:
            logger.error("Auto-summarize failed: %s", e)

    # Clear the old pending before running revision
    state.clear_pending()

    # Track context — draft rejected, revision incoming
    try:
        user_id = update.effective_user.id if update.effective_user else 0
        if user_id:
            conversation_context.update_context(
                user_id,
                last_bot_action="idle",
                pending_draft_exists=False,
                last_command="/reject",
            )
    except Exception as e:
        logger.debug("Context tracking failed in _do_reject: %s", e)

    # Branch: agent mode re-runs with revision context, pipeline mode uses revise_draft
    if settings.AGENT_MODE == "agent":
        await _handle_agent_revision(update, pending, feedback_text)
    else:
        await _handle_pipeline_revision(update, pending, feedback_text)


def _esc(text: str) -> str:
    """HTML-escape text for Telegram messages."""
    return html.escape(str(text))


_STEP_ICONS = {
    "Analyze": "\U0001F50D",          # magnifying glass
    "Plan": "\U0001F4DD",             # memo
    "Verify": "\u2705",               # check mark
    "Plan & Verify": "\U0001F4DD\u2705",
    "Generate": "\u2728",             # sparkles
}

_TOOL_ICONS = {
    "read_brand_guidelines": "\U0001F4DA",   # books
    "read_references": "\U0001F4C2",         # folder
    "check_figma_design": "\U0001F3A8",      # palette
    "generate_image": "\U0001F5BC",          # framed picture
    "img2img": "\U0001F5BC",                 # framed picture
    "read_feedback_history": "\U0001F4AC",   # speech bubble
    "log_resource_usage": "\U0001F4CB",      # clipboard
    "execute_openclaw_script": "\u26D3",     # chain
}


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show available commands."""
    if not _authorized(update.effective_user.id):
        return

    mode = settings.AGENT_MODE
    msg = (
        f"<b>BrandMover Local</b> (mode: {_esc(mode)})\n\n"
        "Send any message to generate a branded post draft.\n"
        "I'll think through it in multiple steps and show you my reasoning.\n\n"
        "<b>Commands:</b>\n"
        "/approve [N] — Approve the pending draft (option N if multiple)\n"
        "/reject <i>reason</i> — Revise the draft with feedback\n"
        "/edit <i>feedback</i> — Surgical edit on the last generated image\n"
        "/status — Show pending draft details\n"
        "/refs — Show loaded reference materials\n"
        "/feedback — Show approval/rejection stats\n"
        "/learn — Trigger preference learning from feedback history\n"
        "/style — Manage visual style profiles\n"
        "/brand — Show active brand config\n"
        "/setup — Bootstrap guidelines from a PDF upload\n"
        "/cancel — Clear pending draft\n"
        "/schedule <i>time prompt</i> — Schedule a post for a specific time\n"
        "/scheduled — List upcoming scheduled posts\n"
        "/unschedule <i>id</i> — Cancel a scheduled post\n"
        "/autostatus — Auto-posting scheduler status\n"
        "/autopause — Pause/resume auto-posting\n"
        "/autoforce <i>slot</i> — Force a specific auto-post slot\n"
        "/generate <i>type description</i> — Generate a standalone asset\n"
        "/logo — View/set brand logo\n"
        "/ingest — Extract brand info from an image\n"
        "/brand_check — Check an image against brand guidelines\n"
        "/train_lora — Trigger LoRA training from approved images\n"
        "/lora_status — Show LoRA training status and versions\n"
        "/lora_versions — List all trained LoRA versions\n"
        "/lora_switch <i>N</i> — Switch active LoRA to version N\n"
        "/lora_rollback — Roll back to previous LoRA version\n"
        "/history — Show generation history and stats\n"
        "/analytics — Show approval rates by content type and model\n"
        "/apply — Apply extracted brand info to guidelines\n"
        "/template — Toggle image composition on/off\n"
        "/template_upload — Upload a custom visual template\n"
        "/template_from_reference — Generate a template from a reference image\n"
        "/template_import — Import a template from Figma\n"
        "/font_upload — Upload a custom TTF/OTF font\n"
        "/onboard — Start conversational brand onboarding\n"
        "/onboard_cancel — Cancel onboarding\n"
        "/library — List or search the asset library\n"
        "/strategy — View current brand strategy and config\n"
        "/preview [topic] — Generate a sample post (no rate limit)\n"
        "/regen_guidelines — Regenerate guidelines from asset inventory\n"
        "/reset_brand — Wipe brand config and start fresh\n"
        "/upload — Add images to your brand asset library\n"
        "/done — Finish asset upload session\n"
        "/help — Show this message"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def refs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /refs — show loaded reference materials."""
    if not _authorized(update.effective_user.id):
        return

    summary = guidelines.get_reference_summary()
    await update.message.reply_text(
        f"<b>Reference Vault</b>\n\n<pre>{_esc(summary)}</pre>",
        parse_mode="HTML",
    )


async def brand_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /brand — show active brand config from guidelines.md."""
    if not _authorized(update.effective_user.id):
        return

    from agent import compositor_config
    summary = compositor_config.get_brand_summary()

    lines = [f"<b>{_esc(summary['brand_name'] or 'Brand Config')}</b>"]
    if summary["tagline"]:
        lines.append(f"<i>{_esc(summary['tagline'])}</i>")
    if summary["website"]:
        lines.append(f"Web: {_esc(summary['website'])}")
    if summary["x_handle"]:
        lines.append(f"X: {_esc(summary['x_handle'])}")

    if summary["colors"]:
        lines.append("\n<b>Colors</b>")
        for role, c in summary["colors"].items():
            lines.append(f"  {_esc(role)} — {_esc(c['name'])} <code>{_esc(c['hex'])}</code>")

    if summary["fonts"]:
        lines.append("\n<b>Fonts</b>")
        for use, f in summary["fonts"].items():
            lines.append(f"  {_esc(use)} — {_esc(f['family'])} {_esc(f['weight'])}")

    if summary["style_keywords"]:
        lines.append(f"\n<b>Style:</b> {_esc(', '.join(summary['style_keywords']))}")

    if summary["parsed_at"]:
        import datetime
        ts = datetime.datetime.fromtimestamp(summary["parsed_at"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"\n<i>Parsed {ts} from {_esc(summary['source_path'])}</i>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show pending draft info."""
    if not _authorized(update.effective_user.id):
        return

    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("No pending draft. Send me a content request to get started.")
        return

    age = int(time.time() - pending.get("timestamp", 0))
    minutes = age // 60
    revision = state.get_draft_revision_count()
    rev_tag = f" (revision {revision})" if revision > 1 else ""
    msg = (
        f"<b>Pending Draft{_esc(rev_tag)}</b>\n\n"
        f"<b>Request:</b> {_esc(pending['original_request'])}\n\n"
        f"<b>Caption:</b>\n{_esc(pending['caption'])}\n\n"
        f"<b>Image:</b> {'Yes' if pending.get('image_url') else 'No'}\n"
        f"<b>Waiting:</b> {minutes} min\n\n"
        f"Reply /approve to post or /reject <i>feedback</i> to revise."
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel — clear pending draft."""
    if not _authorized(update.effective_user.id):
        return

    if not state.has_pending():
        await update.message.reply_text("Nothing to cancel — no pending draft.")
        return

    state.clear_pending()
    state.clear_draft_history()
    await update.message.reply_text("Draft cancelled. Send a new request whenever you're ready.")


async def approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /approve [N] — approve the pending draft and log feedback."""
    if not _authorized(update.effective_user.id):
        return

    # Parse optional option number from "/approve N"
    text = (update.message.text or "").strip()
    parts = text.split()
    option_num = 1
    if len(parts) >= 2:
        try:
            option_num = int(parts[1])
        except ValueError:
            pass

    await _do_approve(update, context, option_num=option_num, source="command")


async def edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /edit <feedback> — surgical img2img edit on the last generated image."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    feedback_text = text.partition("/edit")[2].strip()
    if not feedback_text:
        await update.message.reply_text(
            "Usage: /edit <i>make the background darker</i>\n\n"
            "This applies a light img2img edit to the last generated image.",
            parse_mode="HTML",
        )
        return

    last_url, content_type = state.get_last_generated()
    if not last_url:
        await update.message.reply_text("No image to edit — generate one first with a brand_3d request.")
        return

    await update.message.chat.send_action("upload_photo")
    await update.message.reply_text(f"\U0001F58C Editing: {_esc(feedback_text)}", parse_mode="HTML")

    try:
        # Download the last generated image to a temp file
        import httpx
        ts = int(time.time())
        tmp_path = str(Path(tempfile.gettempdir()) / f"edit_ref_{ts}.jpg")
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(last_url)
            resp.raise_for_status()
            _PILImage.open(io.BytesIO(resp.content)).convert("RGB").save(tmp_path, "JPEG", quality=95)

        # Build edit prompt with brand constraints
        edit_prompt = (
            f"Edit this image: {feedback_text}. "
            f"Keep everything else identical. Maintain the brand's visual style."
        )

        # Low strength for surgical edits
        url = await image_gen.generate_img2img(edit_prompt, tmp_path, strength=0.2)

        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("Temp cleanup failed for %s: %s", tmp_path, e)

        if not url:
            await update.message.reply_text("Edit failed — image generation returned no result. Try again.")
            return

        # Save as new last generated
        state.save_last_generated(url, content_type or "brand_3d")

        # Get existing pending draft for compositing context
        pending = state.get_pending()
        ct = content_type or "brand_3d"

        # Build a draft dict the compositor can use (needs title + subtitle)
        if pending:
            draft = dict(pending)
            ct = content_type or pending.get("content_type", ct)
        else:
            draft = {"caption": "Edited image", "alt_text": "Edited brand image"}

        # Ensure the compositor has title/subtitle — synthesize from caption if missing
        if not draft.get("title") and not draft.get("subtitle"):
            caption_text = draft.get("caption", "")
            if caption_text:
                # First sentence or first 60 chars → title, rest → subtitle
                sentences = caption_text.split(". ", 1)
                draft["title"] = sentences[0].rstrip(".")
                draft["subtitle"] = sentences[1] if len(sentences) > 1 else ""

        photo, composed = await _maybe_compose(draft, url, ct)

        # Save composed for archiving
        if composed and isinstance(composed, io.BytesIO):
            try:
                tmp_composed = str(Path(tempfile.gettempdir()) / f"brandmover_edit_composed_{ts}.png")
                with open(tmp_composed, "wb") as f:
                    f.write(composed.getvalue())
                composed.seek(0)
                photo = composed  # reset after reading for save
                state.set_last_composed(tmp_composed, ct)
            except Exception as e:
                logger.debug("Failed to save edit composed image: %s", e)

        # Update pending with the new image URL
        if pending:
            state.save_pending(
                caption=pending.get("caption", ""),
                hashtags=pending.get("hashtags", []),
                image_url=url,
                alt_text=pending.get("alt_text", ""),
                image_prompt=pending.get("image_prompt", ""),
                original_request=pending.get("original_request", ""),
            )

        try:
            await update.message.reply_photo(
                photo=photo,
                caption=(
                    f"<b>Edited</b>: {_esc(feedback_text)}\n\n"
                    f"/approve to post\n"
                    f"/edit <i>more changes</i>\n"
                    f"/reject <i>feedback</i> to start over"
                ),
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Failed to send edited image: %s", e)
            await update.message.reply_text(
                f"Edit complete but couldn't send image: {_esc(str(e))}",
                parse_mode="HTML",
            )

    except Exception as e:
        logger.error("Edit command failed: %s", e)
        await update.message.reply_text(
            f"Edit failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


async def reject_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reject [reason] — revise draft with feedback."""
    if not _authorized(update.effective_user.id):
        return

    # Extract feedback after "/reject"
    text = update.message.text or ""
    feedback_text = text.partition("/reject")[2].strip()

    await _do_reject(update, context, feedback_text=feedback_text, source="command")


async def _handle_pipeline_revision(update: Update, pending: dict, feedback_text: str) -> None:
    """Revise a draft using the pipeline mode."""
    try:
        brand_context = guidelines.get_brand_context()
        draft = await brain.revise_draft(
            original_draft=pending,
            feedback=feedback_text,
            brand_context=brand_context,
        )

        # Generate new image if prompt changed
        image_url = pending.get("image_url")
        if draft.get("image_prompt") != pending.get("image_prompt"):
            await update.message.chat.send_action("upload_photo")
            content_type = draft.get("content_type", "announcement")
            from agent import template_memory as _tm
            template_aspect = _tm.get_aspect_ratio_for_content_type(content_type)
            image_url = await image_gen.generate_image(draft["image_prompt"], content_type=content_type, aspect_ratio=template_aspect)

        # Save revised draft (carry forward auto-post metadata through revisions)
        state.save_pending(
            caption=draft["caption"],
            hashtags=draft.get("hashtags", []),
            image_url=image_url,
            alt_text=draft["alt_text"],
            image_prompt=draft["image_prompt"],
            original_request=pending["original_request"],
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
        )

        await _send_draft(update, draft, image_url)

    except Exception as e:
        logger.error("Revision failed: %s", e)
        # Restore the old pending so user can retry
        state.save_pending(
            caption=pending.get("caption", ""),
            hashtags=pending.get("hashtags", []),
            image_url=pending.get("image_url"),
            alt_text=pending.get("alt_text", ""),
            image_prompt=pending.get("image_prompt", ""),
            original_request=pending.get("original_request", ""),
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
        )
        await update.message.reply_text(
            f"Revision failed: {_esc(str(e))}\n\nOriginal draft still pending. Try again or /cancel.",
            parse_mode="HTML",
        )


async def _handle_agent_revision(update: Update, pending: dict, feedback_text: str) -> None:
    """Revise a draft using agent mode — re-runs the agent with revision context."""
    revision_context = (
        f"PREVIOUS DRAFT (REJECTED):\n"
        f"Caption: {pending.get('caption', '')}\n"
        f"Image prompt: {pending.get('image_prompt', '')}\n\n"
        f"USER FEEDBACK: {feedback_text}\n\n"
        f"Please revise the draft based on this feedback. Address the specific concerns raised."
    )

    async def on_tool_call(tool_name: str, description: str):
        icon = _TOOL_ICONS.get(tool_name, "\u26A1")
        await update.message.reply_text(
            f"{icon} {_esc(description)}",
            parse_mode="HTML",
        )
        await update.message.chat.send_action("typing")

    try:
        result = await engine.run_agent(
            request=pending.get("original_request", ""),
            on_tool_call=on_tool_call,
            revision_context=revision_context,
        )

        if not result.draft:
            await update.message.reply_text(
                f"Agent couldn't produce a valid revision.\n\n<pre>{_esc(result.final_text[:500])}</pre>",
                parse_mode="HTML",
            )
            return

        image_url = result.image_url or pending.get("image_url")

        # Carry forward auto-post slot metadata through revisions
        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft.get("hashtags", []),
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=pending["original_request"],
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
        )

        await _send_draft(update, result.draft, image_url, resources=result.resources)
        return

    except Exception as e:
        logger.error("Agent revision failed: %s", e)
        # Restore the old pending so user can retry
        state.save_pending(
            caption=pending.get("caption", ""),
            hashtags=pending.get("hashtags", []),
            image_url=pending.get("image_url"),
            alt_text=pending.get("alt_text", ""),
            image_prompt=pending.get("image_prompt", ""),
            original_request=pending.get("original_request", ""),
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
        )
        await update.message.reply_text(
            f"Agent revision failed: {_esc(str(e))}\n\nOriginal draft still pending. Try again or /cancel.",
            parse_mode="HTML",
        )


async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /feedback — show feedback stats."""
    if not _authorized(update.effective_user.id):
        return

    stats = feedback.get_feedback_stats()
    await update.message.reply_text(
        f"<b>Feedback Stats</b>\n\n<pre>{_esc(stats)}</pre>",
        parse_mode="HTML",
    )


async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /learn — trigger preference learning from feedback history."""
    if not _authorized(update.effective_user.id):
        return

    await update.message.chat.send_action("typing")

    try:
        summary = await feedback.summarize_preferences()
        await update.message.reply_text(
            f"<b>Preferences Updated</b>\n\n{_esc(summary[:2000])}",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Learn command failed: %s", e)
        await update.message.reply_text(
            f"Failed to summarize preferences: {_esc(str(e))}",
            parse_mode="HTML",
        )


async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /style — manage style profiles."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    args = text.partition("/style")[2].strip().split()

    # /style — list all profiles
    if not args:
        profiles = state.list_profiles()
        if not profiles:
            await update.message.reply_text(
                "<b>Style Profiles</b>\n\n"
                "No profiles yet.\n\n"
                "<code>/style create &lt;name&gt; &lt;description&gt;</code> — create one\n"
                "<code>/style &lt;name&gt; &lt;content_type&gt;</code> — set active\n"
                "Upload a photo with the profile name as caption to add references.",
                parse_mode="HTML",
            )
            return

        lines = ["<b>Style Profiles</b>\n"]
        for p in profiles:
            active = ", ".join(p["active_for"]) if p["active_for"] else "none"
            lines.append(
                f"<b>{_esc(p['name'])}</b> — {_esc(p['description'])}\n"
                f"  images: {p['image_count']} | strength: {p['strength']} | active for: {active}"
            )
        await update.message.reply_text("\n\n".join(lines), parse_mode="HTML")
        return

    # /style create <name> <description...>
    if args[0] == "create":
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: <code>/style create &lt;name&gt; &lt;description&gt;</code>",
                parse_mode="HTML",
            )
            return
        name = args[1]
        description = " ".join(args[2:]) if len(args) > 2 else ""
        try:
            state.add_style_profile(name, description=description)
            await update.message.reply_text(
                f"Created style profile <b>{_esc(name)}</b>\n"
                f"Upload reference photos with caption <code>{_esc(name)}</code> to add images.",
                parse_mode="HTML",
            )
        except ValueError as e:
            await update.message.reply_text(f"Error: {_esc(str(e))}", parse_mode="HTML")
        return

    # First arg is a profile name
    profile_name = args[0]
    profiles = state.get_style_profiles()
    if profile_name not in profiles:
        await update.message.reply_text(
            f"Profile <b>{_esc(profile_name)}</b> not found. "
            f"Use <code>/style create {_esc(profile_name)} description</code> to create it.",
            parse_mode="HTML",
        )
        return

    # /style <name> info
    if len(args) >= 2 and args[1] == "info":
        p_data = profiles[profile_name]
        refs = state.get_profile_refs(profile_name)
        data = state._read_styles()
        active_for = [ct for ct, p in data["active"].items() if p == profile_name]
        active_str = ", ".join(active_for) if active_for else "none"
        await update.message.reply_text(
            f"<b>{_esc(profile_name)}</b>\n\n"
            f"<b>Description:</b> {_esc(p_data.get('description', ''))}\n"
            f"<b>Strength:</b> {p_data.get('strength', 0.3)}\n"
            f"<b>Prompt prefix:</b> {_esc(p_data.get('prompt_prefix', '') or '(none)')}\n"
            f"<b>Images:</b> {len(refs)}\n"
            f"<b>Active for:</b> {active_str}",
            parse_mode="HTML",
        )
        return

    # /style <name> remove
    if len(args) >= 2 and args[1] == "remove":
        state.remove_active_profile(profile_name)
        await update.message.reply_text(
            f"Removed <b>{_esc(profile_name)}</b> from all active mappings (images kept).",
            parse_mode="HTML",
        )
        return

    # /style <name> <content_type> — set active
    if len(args) >= 2:
        content_type = args[1]
        try:
            state.set_active_profile(content_type, profile_name)
            await update.message.reply_text(
                f"Set <b>{_esc(profile_name)}</b> as active style for <b>{_esc(content_type)}</b>",
                parse_mode="HTML",
            )
        except ValueError as e:
            await update.message.reply_text(f"Error: {_esc(str(e))}", parse_mode="HTML")
        return

    # Shouldn't reach here, but show info as fallback
    await update.message.reply_text(
        f"Usage:\n"
        f"<code>/style</code> — list profiles\n"
        f"<code>/style create &lt;name&gt; &lt;desc&gt;</code> — create\n"
        f"<code>/style &lt;name&gt; &lt;content_type&gt;</code> — set active\n"
        f"<code>/style &lt;name&gt; info</code> — details\n"
        f"<code>/style &lt;name&gt; remove</code> — deactivate",
        parse_mode="HTML",
    )


async def setup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /setup — start PDF brand bootstrap."""
    if not _authorized(update.effective_user.id):
        return

    await update.message.reply_text(
        "<b>Brand Setup from PDF</b>\n\n"
        "Send me a PDF of your brand guidelines and I'll extract them into "
        "a structured <code>guidelines.md</code> that the bot can use.\n\n"
        "Just upload the PDF as a document in this chat.",
        parse_mode="HTML",
    )
    # Set a flag so the document handler knows we're in setup mode
    context.user_data["awaiting_setup_pdf"] = True


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads — PDF brand bootstrap when in setup mode."""
    if not _authorized(update.effective_user.id):
        return

    # Onboarding intercept — accept documents as brand assets during upload phase
    ob_session = onboarding.get_session(update.effective_user.id)
    if ob_session and ob_session.state == onboarding.OnboardingState.UPLOADS.value:
        document = update.message.document
        if document:
            import tempfile
            tg_file = await document.get_file()
            tmp_path = str(Path(tempfile.gettempdir()) / f"onboard_doc_{int(time.time())}_{document.file_name or 'doc'}")
            await tg_file.download_to_drive(tmp_path)
            ob_session.uploaded_assets.append({"path": tmp_path, "type": "document"})
            onboarding.save_session(ob_session)
            count = len(ob_session.uploaded_assets)
            await update.message.reply_text(
                f"Document received ({count} assets total). Send more, or /onboard_skip when done.",
            )
        return

    document = update.message.document
    if not document:
        return

    file_name = document.file_name or ""

    # Font upload handling
    is_font = file_name.lower().endswith((".ttf", ".otf"))
    if is_font or context.user_data.get("awaiting_font_upload"):
        context.user_data.pop("awaiting_font_upload", None)
        if not file_name.lower().endswith((".ttf", ".otf")):
            await update.message.reply_text("Please send a .ttf or .otf font file.")
            return
        try:
            # Sanitize filename to prevent path traversal
            import os as _os
            safe_name = _os.path.basename(file_name)
            if not safe_name.lower().endswith((".ttf", ".otf")):
                await update.message.reply_text("Invalid font filename.")
                return
            # Reject oversized files (10 MB limit)
            _fsize = getattr(document, "file_size", None)
            if isinstance(_fsize, int) and _fsize > 10 * 1024 * 1024:
                await update.message.reply_text("Font file too large (max 10 MB).")
                return
            fonts_dir = Path(settings.BRAND_FOLDER) / "assets" / "fonts"
            fonts_dir.mkdir(parents=True, exist_ok=True)
            tg_file = await document.get_file()
            save_path = fonts_dir / safe_name
            await tg_file.download_to_drive(str(save_path))
            # Clear font caches
            try:
                from agent.font_manager import clear_cache as _fm_clear
                _fm_clear()
            except ImportError:
                pass
            try:
                from agent.compositor import clear_font_cache
                clear_font_cache()
            except ImportError:
                pass
            await update.message.reply_text(
                f"Font <b>{_esc(file_name)}</b> saved to brand fonts.\n"
                "It's now available for templates and compositions.",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("Font upload failed: %s", e)
            await update.message.reply_text("Font upload failed. Check logs for details.")
        return

    is_pdf = file_name.lower().endswith(".pdf")

    if not is_pdf:
        await update.message.reply_text(
            "I can accept PDFs, image files, and font files (.ttf/.otf). "
            "Send a PDF or image to add to your brand."
        )
        return

    # Smart PDF handling — save to references, auto-extract if no guidelines exist
    if not context.user_data.get("awaiting_setup_pdf"):
        await update.message.chat.send_action("typing")
        try:
            # Sanitize filename to prevent path traversal
            import os as _os
            safe_name = _os.path.basename(file_name)
            if not safe_name:
                await update.message.reply_text("Invalid filename.")
                return
            # Reject oversized files (50 MB limit)
            _fsize = getattr(document, "file_size", None)
            if isinstance(_fsize, int) and _fsize > 50 * 1024 * 1024:
                await update.message.reply_text("File too large (max 50 MB).")
                return
            tg_file = await document.get_file()
            refs_dir = Path(settings.BRAND_FOLDER) / "references"
            refs_dir.mkdir(parents=True, exist_ok=True)
            ref_path = refs_dir / safe_name
            await tg_file.download_to_drive(str(ref_path))

            guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
            if not guidelines_path.exists():
                # No guidelines yet — auto-extract from PDF
                await update.message.reply_text(
                    f"saved <code>{_esc(file_name)}</code> — extracting brand guidelines...",
                    parse_mode="HTML",
                )
                guidelines_md = await guidelines.extract_brand_from_pdf(str(ref_path))
                if guidelines_md:
                    guidelines_path.write_text(guidelines_md, encoding="utf-8")
                    compositor_config.invalidate_cache()
                    compositor.clear_font_cache()
                    guidelines.invalidate_brand_context()
                    preview = guidelines_md[:1500]
                    if len(guidelines_md) > 1500:
                        preview += "\n\n[... truncated ...]"
                    await update.message.reply_text(
                        f"<b>Guidelines Generated</b> ({len(guidelines_md)} chars)\n\n"
                        f"<pre>{_esc(preview)}</pre>\n\n"
                        "you're all set! send me a content request to try it out.",
                        parse_mode="HTML",
                    )
                else:
                    await update.message.reply_text(
                        "saved PDF to references but couldn't extract guidelines. try /setup for manual setup."
                    )
            else:
                await update.message.reply_text(
                    f"saved <code>{_esc(file_name)}</code> to brand references.\n\n"
                    "you already have brand guidelines. use /setup to rebuild from this PDF.",
                    parse_mode="HTML",
                )
        except Exception as e:
            logger.error("PDF handling failed: %s", e)
            await update.message.reply_text(
                f"failed to process PDF: {_esc(str(e))}",
                parse_mode="HTML",
            )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Extracting text from PDF and generating guidelines...")

    try:
        # Download the PDF
        tg_file = await document.get_file()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            await tg_file.download_to_drive(tmp_path)

        # Extract and generate guidelines
        guidelines_md = await guidelines.extract_brand_from_pdf(tmp_path)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        if not guidelines_md:
            await update.message.reply_text("Failed to extract guidelines from the PDF. Try a different file.")
            return

        # Save to brand folder
        brand_path = Path(settings.BRAND_FOLDER)
        guidelines_path = brand_path / "guidelines.md"

        # Backup existing guidelines
        if guidelines_path.exists():
            backup_path = brand_path / "guidelines.md.bak"
            import shutil
            shutil.copy2(guidelines_path, backup_path)
            await update.message.reply_text(
                f"Backed up existing guidelines to <code>guidelines.md.bak</code>",
                parse_mode="HTML",
            )

        guidelines_path.write_text(guidelines_md, encoding="utf-8")
        compositor_config.invalidate_cache()
        compositor.clear_font_cache()
        guidelines.invalidate_brand_context()

        # Also save PDF to references
        refs_dir = brand_path / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        ref_path = refs_dir / file_name
        await tg_file.download_to_drive(str(ref_path))

        context.user_data["awaiting_setup_pdf"] = False

        # Show preview
        preview = guidelines_md[:1500]
        if len(guidelines_md) > 1500:
            preview += "\n\n[... truncated ...]"

        await update.message.reply_text(
            f"<b>Guidelines Generated</b> ({len(guidelines_md)} chars)\n\n"
            f"<pre>{_esc(preview)}</pre>\n\n"
            f"Saved to: <code>{guidelines_path}</code>\n"
            f"PDF saved to: <code>{ref_path}</code>\n\n"
            f"You're all set! Send me a content request to try it out.",
            parse_mode="HTML",
        )
        logger.info("Brand setup complete: %d chars from %s", len(guidelines_md), file_name)

    except Exception as e:
        logger.error("Setup PDF extraction failed: %s", e)
        context.user_data["awaiting_setup_pdf"] = False
        await update.message.reply_text(
            f"Setup failed: {_esc(str(e))}\n\nTry again with /setup.",
            parse_mode="HTML",
        )


def _merge_extracted(results: list[dict]) -> dict:
    """Merge multiple ingest extraction results into one deduplicated set."""
    colors: dict[str, dict] = {}
    fonts: dict[str, dict] = {}
    style_keywords: set[str] = set()
    logo_descriptions: list[str] = []

    for r in results:
        for c in r.get("colors", []):
            h = c.get("hex", "")
            if h and h not in colors:
                colors[h] = c
        for f in r.get("fonts", []):
            family = f.get("family", "")
            if family and family not in fonts:
                fonts[family] = f
        for kw in r.get("style_keywords", []):
            style_keywords.add(kw)
        desc = r.get("logo_description", "")
        if desc and desc not in logo_descriptions:
            logo_descriptions.append(desc)

    return {
        "colors": list(colors.values()),
        "fonts": list(fonts.values()),
        "style_keywords": sorted(style_keywords),
        "logo_description": " | ".join(logo_descriptions) if logo_descriptions else "",
    }


async def _delayed_bulk_process(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: int,
) -> None:
    """Wait 3 seconds for more photos, then process the batch."""
    import asyncio
    await asyncio.sleep(3)
    await _process_bulk_upload(context, user_id, chat_id)


async def _process_bulk_upload(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: int,
) -> None:
    """Process batched photo uploads — auto-ingest if multiple, prompt if single."""
    batch = context.user_data.pop("_bulk_uploads", [])
    if not batch:
        return

    if len(batch) == 1:
        # Single image — ask what to do
        await context.bot.send_message(
            chat_id,
            "got it. what should i do with this? reply with:\n"
            "reference / mascot / style <name> / background",
        )
        return

    # Multiple images — auto-ingest all
    count = len(batch)
    await context.bot.send_message(
        chat_id,
        f"received {count} images — analyzing with AI vision...",
    )

    from agent import ingest
    import json
    import shutil

    all_extracted = []
    for i, path in enumerate(batch, 1):
        try:
            await context.bot.send_chat_action(chat_id, "typing")
            extracted = await ingest.extract_brand_from_image(path)
            all_extracted.append(extracted)
        except Exception as e:
            logger.warning("Ingest failed for image %d: %s", i, e)

    # Save images to brand/references/
    refs_dir = Path(settings.BRAND_FOLDER) / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(batch, 1):
        try:
            dest = refs_dir / f"ref_{time.time_ns()}_{i}.jpg"
            shutil.copy2(path, str(dest))
        except Exception as e:
            logger.warning("Failed to save reference image %d: %s", i, e)

    if not all_extracted:
        await context.bot.send_message(
            chat_id,
            f"saved {count} images to brand/references/ but couldn't analyze them.\n"
            "try /ingest to analyze one at a time.",
        )
        return

    merged = _merge_extracted(all_extracted)
    extracted_text = json.dumps(merged, indent=2)
    if len(extracted_text) > 3000:
        extracted_text = extracted_text[:3000] + "\n..."

    await context.bot.send_message(
        chat_id,
        f"<b>analyzed {count} images — extracted brand elements:</b>\n"
        f"<pre>{_esc(extracted_text)}</pre>",
        parse_mode="HTML",
    )

    # Diff against guidelines
    try:
        report = await ingest.diff_against_guidelines(merged)
        await context.bot.send_message(
            chat_id,
            f"<b>Compliance report:</b>\n{_esc(report)}",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.warning("Diff against guidelines failed: %s", e)

    # Store for /apply
    context.user_data["last_ingest_extracted"] = merged

    await context.bot.send_message(
        chat_id,
        f"saved {count} images to brand/references/\n"
        "reply /apply to update your guidelines with the extracted info.",
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo uploads and image documents as reference images."""
    if not _authorized(update.effective_user.id):
        return
    if not update.message:
        return

    # Determine source: photo[] or image document
    tg_file = None
    if update.message.photo:
        tg_file = await update.message.photo[-1].get_file()
    elif update.message.document:
        mime = update.message.document.mime_type or ""
        if not mime.startswith("image/"):
            return
        tg_file = await update.message.document.get_file()
    else:
        return

    tmp_path = str(Path(tempfile.gettempdir()) / f"brandmover_upload_{time.time_ns()}.jpg")

    try:
        await tg_file.download_to_drive(tmp_path)
    except Exception as e:
        logger.error("Failed to download uploaded image: %s", e)
        await update.message.reply_text(
            "couldn't download that image, try sending it as a photo instead of a file"
        )
        return

    # Convert to JPEG
    try:
        img = _PILImage.open(tmp_path).convert("RGB")
        img.save(tmp_path, "JPEG", quality=95)
    except Exception as e:
        logger.warning("Image conversion failed (using as-is): %s", e)

    # --- Onboarding upload intercept ---
    ob_session = onboarding.get_session(update.effective_user.id)
    if ob_session and ob_session.state == onboarding.OnboardingState.UPLOADS.value:
        # Check if this looks like a template
        from agent import template_memory as _tm
        try:
            is_tpl = await _tm.detect_if_template(tmp_path)
        except Exception:
            is_tpl = False
        if is_tpl:
            try:
                template = await _tm.register_template(tmp_path, name=f"Onboarding Template")
                regions_str = ", ".join(f"{r.type}" for r in template.regions)
                await update.message.reply_text(
                    f"That looks like a template! Registered as <code>{_esc(template.id)}</code> "
                    f"({template.aspect_ratio}, regions: {_esc(regions_str) or 'none'}).\n"
                    f"Send more assets, or /onboard_skip when done.",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("Onboarding template registration failed: %s", e)
                # Fall through to normal asset handling
                ob_session.uploaded_assets.append({"path": tmp_path, "type": "image"})
                onboarding.save_session(ob_session)
                count = len(ob_session.uploaded_assets)
                await update.message.reply_text(
                    f"Asset {count} received. Send more, or /onboard_skip when done.",
                )
            return

        ob_session.uploaded_assets.append({"path": tmp_path, "type": "image"})
        onboarding.save_session(ob_session)
        count = len(ob_session.uploaded_assets)
        await update.message.reply_text(
            f"Asset {count} received. Send more, or /onboard_skip when done.",
        )
        return

    # --- Template upload intercept ---
    user_data = context.user_data if context else {}
    if user_data.get("awaiting_template"):
        await _handle_template_upload(update, context, tmp_path)
        return

    # --- Template from reference intercept ---
    if user_data.get("awaiting_template_from_ref"):
        state.clear_reference_image()
        await _handle_template_from_reference(update, context, tmp_path)
        return

    # --- /upload asset library intercept ---
    if user_data.get("awaiting_asset_upload"):
        try:
            from agent import asset_audit
            import asyncio
            # Tag via filename heuristic (fast, no API call)
            ct = asset_library._guess_content_type(Path(tmp_path))
            entry = asset_library.add(tmp_path, "uploaded", ct, tags=[ct, "uploaded"])
            upload_count = user_data.get("_asset_upload_count", 0) + 1
            user_data["_asset_upload_count"] = upload_count
            await update.message.reply_text(
                f"Added to library: <code>{entry.id}</code> ({ct})\n"
                f"{upload_count} asset(s) uploaded this session. Send more or /done when finished.",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Asset upload failed: %s", e)
            await update.message.reply_text(f"Failed to add asset: {_esc(str(e))}", parse_mode="HTML")
        return

    # --- Priority flag checks (logo > ingest > brand_check) ---
    if user_data.get("awaiting_logo_upload"):
        user_data["awaiting_logo_upload"] = False
        logo_dir = Path(settings.BRAND_FOLDER) / "assets"
        logo_dir.mkdir(parents=True, exist_ok=True)
        logo_dest = logo_dir / "logo.png"
        try:
            _PILImage.open(tmp_path).convert("RGBA").save(str(logo_dest), "PNG")
            await update.message.reply_text(
                f"Brand logo saved to <code>{_esc(str(logo_dest))}</code>",
                parse_mode="HTML",
            )
            logger.info("Brand logo updated: %s", logo_dest)
        except Exception as e:
            logger.error("Failed to save logo: %s", e)
            await update.message.reply_text(f"Failed to save logo: {_esc(str(e))}", parse_mode="HTML")
        return

    if user_data.get("awaiting_ingest_image"):
        user_data["awaiting_ingest_image"] = False
        await update.message.chat.send_action("typing")
        await update.message.reply_text("Analyzing image for brand elements...")
        try:
            from agent import ingest
            extracted = await ingest.extract_brand_from_image(tmp_path)
            report = await ingest.diff_against_guidelines(extracted)
            # Send extracted info
            import json
            extracted_text = json.dumps(extracted, indent=2)
            if len(extracted_text) > 3000:
                extracted_text = extracted_text[:3000] + "\n..."
            await update.message.reply_text(
                f"<b>Extracted brand elements:</b>\n<pre>{_esc(extracted_text)}</pre>",
                parse_mode="HTML",
            )
            await update.message.reply_text(
                f"<b>Compliance report:</b>\n{_esc(report)}",
                parse_mode="HTML",
            )
            # Store extracted data for /apply
            context.user_data["last_ingest_extracted"] = extracted
            await update.message.reply_text(
                "Reply /apply to update guidelines with the extracted info.",
            )
        except Exception as e:
            logger.error("Brand ingestion failed: %s", e)
            await update.message.reply_text(f"Ingestion failed: {_esc(str(e))}", parse_mode="HTML")
        return

    if user_data.get("awaiting_brand_check"):
        user_data["awaiting_brand_check"] = False
        await update.message.chat.send_action("typing")
        await update.message.reply_text("Checking image against brand guidelines...")
        try:
            from agent import brand_check
            report = await brand_check.check_brand_compliance(tmp_path)
            formatted = brand_check.format_compliance_report(report)
            await update.message.reply_text(formatted, parse_mode="HTML")
        except Exception as e:
            logger.error("Brand check failed: %s", e)
            await update.message.reply_text(f"Brand check failed: {_esc(str(e))}", parse_mode="HTML")
        return

    # --- Normal flow: set as reference image ---
    state.set_reference_image(tmp_path)
    logger.info("Reference image saved to state: %s", tmp_path)

    caption = (update.message.caption or "").strip()

    # Caption-based /template_upload — image sent with "/template_upload [name]" as caption
    if caption.lower().startswith("/template_upload"):
        template_name = caption[len("/template_upload"):].strip()
        if context:
            context.user_data["template_name"] = template_name
        state.clear_reference_image()  # Undo the reference save above
        await _handle_template_upload(update, context, tmp_path)
        return

    # Caption-based /template_from_reference — image sent with "/template_from_reference [name]"
    if caption.lower().startswith("/template_from_reference"):
        template_name = caption[len("/template_from_reference"):].strip()
        if context:
            context.user_data["template_from_ref_name"] = template_name
        state.clear_reference_image()
        await _handle_template_from_reference(update, context, tmp_path)
        return

    # Caption-based /brand_check — image sent with "/brand_check" as caption
    if caption.lower().startswith("/brand_check"):
        await update.message.chat.send_action("typing")
        await update.message.reply_text("Checking image against brand guidelines...")
        try:
            from agent import brand_check
            report = await brand_check.check_brand_compliance(tmp_path)
            formatted = brand_check.format_compliance_report(report)
            await update.message.reply_text(formatted, parse_mode="HTML")
        except Exception as e:
            logger.error("Brand check failed: %s", e)
            await update.message.reply_text(f"Brand check failed: {_esc(str(e))}", parse_mode="HTML")
        return

    # Natural language template-from-reference intent detection
    if caption and _is_template_from_ref_intent(caption):
        state.clear_reference_image()
        if context:
            context.user_data["template_from_ref_name"] = ""
        await _handle_template_from_reference(update, context, tmp_path)
        return

    # --- Direct photo mode: user wants to use their photo as-is in a template ---
    if caption and _is_direct_photo_intent(caption):
        from agent import template_memory as _tm
        cfg = _cc.get_config()
        memory = _tm.TemplateMemory()
        template = memory.get_template_for_content_type("announcement")
        if template and cfg.compositor_enabled:
            state.clear_reference_image()
            if context:
                context.user_data["direct_photo_path"] = tmp_path
            await update.message.reply_text("got it, composing with your photo directly...")

            if _rate_limited(update.effective_user.id):
                await update.message.reply_text(
                    f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
                )
                return

            if state.has_pending():
                await update.message.reply_text(
                    "You have a pending draft. /approve, /reject, or /cancel it first.",
                    parse_mode="HTML",
                )
                return

            # Strip the direct-photo keywords from caption to get a content hint
            content_hint = caption
            for p in _DIRECT_PHOTO_PATTERNS:
                content_hint = content_hint.lower().replace(p, "").strip()
            if not content_hint:
                content_hint = "create an announcement"
            request = f"{content_hint}\n\n[DIRECT PHOTO: {tmp_path}]\n[generate text only, do NOT call generate_image]"

            if settings.AGENT_MODE == "agent":
                await _handle_agent_mode(update, request)
            else:
                await _handle_pipeline_mode(update, request)
            return

    # Check if caption matches a style profile name (e.g. "3d_card" or "style 3d_card")
    if caption:
        style_name = caption
        if caption.lower().startswith("style "):
            style_name = caption[6:].strip()

        profiles = state.get_style_profiles()
        if style_name in profiles:
            count = state.add_profile_image(style_name, tmp_path)
            await update.message.reply_text(
                f"added to <b>{_esc(style_name)}</b> profile ({count} images total)",
                parse_mode="HTML",
            )
            return

    if caption:
        await update.message.reply_text("got it, generating with your image as reference...")

        if _rate_limited(update.effective_user.id):
            await update.message.reply_text(
                f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
            )
            return

        if state.has_pending():
            await update.message.reply_text(
                "You have a pending draft. /approve, /reject, or /cancel it first.",
                parse_mode="HTML",
            )
            return

        if settings.AGENT_MODE == "agent":
            await _handle_agent_mode(update, caption)
        else:
            await _handle_pipeline_mode(update, caption)
    else:
        # Batch uploads: collect images for 3 seconds, then auto-ingest if bulk
        import asyncio as _aio

        batch = context.user_data.setdefault("_bulk_uploads", [])
        batch.append(tmp_path)
        chat_id = update.message.chat_id
        user_id = update.effective_user.id

        # Cancel existing batch timer and reschedule
        existing = _bulk_upload_tasks.get(user_id)
        if existing and not existing.done():
            existing.cancel()

        _bulk_upload_tasks[user_id] = _aio.create_task(
            _delayed_bulk_process(context, user_id, chat_id)
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any plain text message — routes through intent router before generation."""
    if not _authorized(update.effective_user.id):
        return

    if not update.message or not update.message.text:
        return

    request = update.message.text.strip()
    if not request:
        return

    # Template-from-reference adjustment intercept
    tplref = (context.user_data or {}).get("tplref_pending") if context else None
    if isinstance(tplref, dict) and "design" in tplref:
        await _handle_tplref_adjustment(update, context, request)
        return

    # Template region update intercept — user describing region positions after upload
    if _is_template_region_update(request, context):
        await _handle_template_region_update(update, context, request)
        return

    # Onboarding intercept — handle messages during onboarding flow (highest priority)
    session = onboarding.get_session(update.effective_user.id)
    if session and session.state not in (
        onboarding.OnboardingState.IDLE.value,
        onboarding.OnboardingState.COMPLETE.value,
    ):
        # DISCOVERY state uses Claude-driven async conversation
        if session.state == onboarding.OnboardingState.DISCOVERY.value:
            await update.message.chat.send_action("typing")
            session, response = await onboarding.advance_async(session, request)
        else:
            session, response = onboarding.advance(session, request)
        onboarding.save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")

        # Handle async state transitions
        if session.state == onboarding.OnboardingState.AUDITING.value:
            await _run_onboarding_audit(update, session)
        elif session.state == onboarding.OnboardingState.STRATEGY.value:
            await _run_onboarding_strategy(update, session)
        elif session.state == onboarding.OnboardingState.COMPLETE.value:
            summary = await onboarding.finalize_onboarding(session)
            await update.message.reply_text(summary, parse_mode="HTML")
        return

    # Intent routing — classify message and dispatch if confident
    if settings.INTENT_ROUTER_ENABLED:
        try:
            handled = await _route_intent(update, context, request)
            if handled:
                return
        except Exception as e:
            logger.warning("Intent router error, falling through to generation: %s", e)

    # Fallback: generation path (rate limited, pending draft blocked)
    if _rate_limited(update.effective_user.id):
        await update.message.reply_text(
            f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
        )
        return

    if state.has_pending():
        await update.message.reply_text(
            "You have a pending draft. /approve, /reject, or /cancel it first.",
            parse_mode="HTML",
        )
        return

    if settings.AGENT_MODE == "agent":
        await _handle_agent_mode(update, request)
    else:
        await _handle_pipeline_mode(update, request)


async def _route_intent(update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
    """Classify intent and dispatch. Returns True if handled, False to fall through."""
    user_id = update.effective_user.id
    ctx = conversation_context.get_context(user_id)
    # Sync pending draft state from actual state file
    ctx.pending_draft_exists = state.has_pending()

    result = await intent_router.classify_intent(message, ctx)
    intent = result.intent
    confidence = result.confidence

    logger.info(
        "Intent: %s (%.2f) via %s for: %s",
        intent, confidence, result.routed_via, message[:60],
    )

    # Track classified intent
    try:
        recent = list(ctx.recent_intents) + [intent]
        conversation_context.update_context(user_id, recent_intents=recent)
    except Exception as e:
        logger.debug("Failed to track intent: %s", e)

    # High-confidence actions
    if intent == "approve" and confidence >= 0.8:
        await _do_approve(update, context, source="router")
        return True

    if intent == "reject" and confidence >= 0.8:
        if result.parameters.get("needs_feedback_prompt"):
            # Bare reject word (e.g. "no") — prompt for specific feedback
            fb = ""
        else:
            fb = result.parameters.get("feedback", message)
        await _do_reject(update, context, feedback_text=fb, source="router")
        return True

    if intent == "edit_request" and confidence >= 0.5:
        fb = result.parameters.get("feedback", message)
        if state.has_pending():
            await _do_reject(update, context, feedback_text=fb, source="router")
        else:
            return False  # Fall through to generation
        return True

    if intent == "reroll" and confidence >= 0.8:
        pending = state.get_pending()
        if pending:
            original = pending.get("original_request", "")
            state.clear_pending()
            state.clear_draft_history()
            await update.message.reply_text("Regenerating...")
            if original:
                if settings.AGENT_MODE == "agent":
                    await _handle_agent_mode(update, original)
                else:
                    await _handle_pipeline_mode(update, original)
            return True
        return False

    if intent == "modify_last" and confidence >= 0.5:
        fb = result.parameters.get("feedback", message)
        modified = await chat.handle_modify_last(fb, ctx)
        if modified:
            # Save modified draft and re-send
            state.save_pending(
                caption=modified.get("caption", ""),
                hashtags=modified.get("hashtags", []),
                image_url=modified.get("image_url"),
                alt_text=modified.get("alt_text", ""),
                image_prompt=modified.get("image_prompt", ""),
                original_request=modified.get("original_request", ""),
            )
            await _send_draft(update, modified, modified.get("image_url"))
            return True
        return False

    # Info commands
    if intent == "show_status" and confidence >= 0.8:
        await status_command(update, context)
        return True

    if intent == "show_help" and confidence >= 0.8:
        await help_command(update, context)
        return True

    if intent == "show_analytics" and confidence >= 0.8:
        await analytics_command(update, context)
        return True

    if intent == "show_history" and confidence >= 0.8:
        await history_command(update, context)
        return True

    if intent == "brand_check" and confidence >= 0.8:
        await brand_check_command(update, context)
        return True

    if intent == "upload_assets":
        await update.message.reply_text(
            "go ahead — send your images or PDFs and I'll analyze them automatically.\n\n"
            "bulk uploads are auto-ingested with AI vision. "
            "single images can be tagged as reference / mascot / style / background.",
        )
        return True

    if intent == "schedule_post" and confidence >= 0.5:
        time_expr = result.parameters.get("time", "")
        topic = result.parameters.get("topic", "")

        # If Haiku extracted time and topic, try to schedule directly
        if time_expr and topic:
            combined = f"{time_expr} {topic}"
            prompt, ts, recurrence, display = schedule_queue.parse_schedule_command(combined)
            if prompt and ts:
                item = schedule_queue.add_scheduled(prompt, ts, recurrence or "once")
                await update.message.reply_text(
                    f"<b>Post scheduled</b>\n\n"
                    f"<b>Time:</b> {_esc(display)}\n"
                    f"<b>Prompt:</b> {_esc(prompt[:200])}\n"
                    f"<b>ID:</b> <code>{item['id']}</code>\n\n"
                    f"I'll generate a draft at the scheduled time and send it for your approval.\n"
                    f"Use /unschedule <code>{item['id']}</code> to cancel.",
                    parse_mode="HTML",
                )
                return True

        # If no time/topic extracted or parse failed, show bare keyword help
        # or list existing schedule
        items = schedule_queue.list_scheduled()
        if items:
            await scheduled_command(update, context)
        else:
            await update.message.reply_text(
                "<b>Schedule a post</b>\n\n"
                "Tell me when and what to post:\n"
                "  <i>\"schedule a post about our launch at 3pm\"</i>\n"
                "  <i>\"post about community updates tomorrow 9am\"</i>\n\n"
                "Or use: /schedule <i>time</i> <i>prompt</i>",
                parse_mode="HTML",
            )
        return True

    # Conversational
    if intent == "greeting":
        user = update.effective_user
        name = user.first_name if user else ""
        reply = await chat.handle_greeting(name)
        await update.message.reply_text(reply)
        conversation_context.update_context(user_id, last_bot_action="sent_content")
        return True

    if intent == "casual_chat" and confidence >= 0.5:
        reply = await chat.handle_casual_chat(message, ctx)
        await update.message.reply_text(reply)
        conversation_context.update_context(user_id, last_bot_action="sent_content")
        return True

    # generate_content or unknown / low confidence → fall through to generation
    return False


async def _handle_agent_mode(update: Update, request: str) -> None:
    """Run the agent tool-use loop for a content request."""
    await update.message.chat.send_action("typing")

    # Extract direct photo path if embedded in the request
    direct_photo_path = None
    import re as _re_mod
    _dp_match = _re_mod.search(r"\[DIRECT PHOTO: (.+?)\]", request)
    if _dp_match:
        direct_photo_path = _dp_match.group(1)

    # If a reference image is stored, inject it into the request for the agent
    ref_path = state.get_reference_image()
    if ref_path and Path(ref_path).exists():
        request = f"{request}\n\n[REFERENCE IMAGE: {ref_path}]"

    async def on_tool_call(tool_name: str, description: str):
        icon = _TOOL_ICONS.get(tool_name, "\u26A1")
        await update.message.reply_text(
            f"{icon} {_esc(description)}",
            parse_mode="HTML",
        )
        await update.message.chat.send_action("typing")

    try:
        result = await engine.run_agent(
            request=request,
            on_tool_call=on_tool_call,
        )

        if not result.draft:
            # Agent gave a conversational response (strategy, advice, etc.)
            # rather than a structured social media draft — show it cleanly
            text = result.final_text or "I processed your request but didn't generate a draft."
            # Truncate for Telegram's 4096 char limit, leaving room for footer
            max_len = 3900
            if len(text) > max_len:
                text = text[:max_len] + "..."
            footer = (
                f"\n\n<i>({result.turns_used} turns, {result.total_time:.1f}s"
                f"{', tools: ' + ', '.join(result.tool_calls_made) if result.tool_calls_made else ''})</i>"
            )
            await update.message.reply_text(
                f"{text}{footer}",
                parse_mode="HTML",
            )
            return

        image_url = result.image_url
        image_urls = result.image_urls

        # Direct photo mode: use the user's photo for template composition
        if direct_photo_path and Path(direct_photo_path).exists():
            image_url = direct_photo_path
            logger.info("Direct photo mode: using %s as image source", direct_photo_path)

        # Save pending state
        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft.get("hashtags", []),
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=request,
            image_urls=image_urls if len(image_urls) > 1 else None,
            content_type=result.draft.get("content_type"),
        )

        # Clean up reference image temp file if one was used
        ref_cleanup = state.get_reference_image()
        if ref_cleanup:
            try:
                Path(ref_cleanup).unlink(missing_ok=True)
            except Exception as e:
                logger.debug("Ref cleanup failed for %s: %s", ref_cleanup, e)
            state.clear_reference_image()

        await _send_draft(update, result.draft, image_url, resources=result.resources, image_urls=image_urls)

    except Exception as e:
        logger.error("Agent error: %s", e)
        await update.message.reply_text(
            f"Something went wrong: {_esc(str(e))}\n\nPlease try again.",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# Onboarding commands
# ---------------------------------------------------------------------------


async def onboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /onboard — start conversational onboarding."""
    if not _authorized(update.effective_user.id):
        return

    user_id = update.effective_user.id
    session = onboarding.get_session(user_id)

    # Resume existing session
    if session and session.state not in (
        onboarding.OnboardingState.IDLE.value,
        onboarding.OnboardingState.COMPLETE.value,
    ):
        await update.message.reply_text(
            f"You have an onboarding session in progress "
            f"(state: {_esc(session.state)}, brand: {_esc(session.brand_name)}).\n\n"
            f"Continue where you left off, or /onboard_cancel to start fresh.",
            parse_mode="HTML",
        )
        return

    # Start new session
    session = onboarding.OnboardingSession(user_id=user_id)
    session, response = onboarding.advance(session, None)
    onboarding.save_session(session)
    await update.message.reply_text(response, parse_mode="HTML")


async def onboard_cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /onboard_cancel — cancel onboarding session."""
    if not _authorized(update.effective_user.id):
        return

    onboarding.delete_session(update.effective_user.id)
    await update.message.reply_text("Onboarding cancelled. Use /onboard to start again.")


async def onboard_skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /onboard_skip — skip upload phase during onboarding."""
    if not _authorized(update.effective_user.id):
        return

    session = onboarding.get_session(update.effective_user.id)
    if not session or session.state != onboarding.OnboardingState.UPLOADS.value:
        await update.message.reply_text("Not in upload phase. Use /onboard to start onboarding.")
        return

    session, response = onboarding.advance(session, "/onboard_skip")
    onboarding.save_session(session)
    await update.message.reply_text(response, parse_mode="HTML")

    # If we transitioned to AUDITING, run the audit
    if session.state == onboarding.OnboardingState.AUDITING.value:
        await _run_onboarding_audit(update, session)


async def _run_onboarding_audit(update: Update, session: onboarding.OnboardingSession) -> None:
    """Run Claude Vision audit on uploaded assets and advance the session."""
    from agent import asset_audit

    await update.message.chat.send_action("typing")

    paths = [a["path"] for a in session.uploaded_assets if Path(a["path"]).exists()]
    if not paths:
        session.state = onboarding.OnboardingState.VISUAL_PREF.value
        onboarding.save_session(session)
        await update.message.reply_text(
            "No valid assets found. Let's pick a visual style instead.\n\n"
            "Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
            "<b>minimal</b> / <b>bold</b> / <b>elegant</b>",
            parse_mode="HTML",
        )
        return

    try:
        inventory = await asset_audit.audit_batch(paths)
        asset_audit.save_inventory(inventory)

        entries_creative = [
            {
                "first_impression": e.first_impression,
                "creative_dna": e.creative_dna,
                "overall_energy": e.overall_energy,
                "what_makes_it_special": e.what_makes_it_special,
                "never_do": e.never_do,
                "character_system": e.character_system,
            }
            for e in inventory.entries
            if e.first_impression
        ]
        audit_data = {
            "archetype": inventory.archetype,
            "consolidated_colors": inventory.consolidated_colors,
            "consolidated_style": inventory.consolidated_style,
            "missing_items": inventory.missing_items,
            "entry_count": len(inventory.entries),
            "collection_analysis": inventory.collection_analysis,
            "brand_insights": inventory.brand_insights,
            "entries_creative": entries_creative,
        }
        session, response = onboarding.finalize_audit(session, audit_data)
        onboarding.save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")
    except Exception as e:
        logger.error("Onboarding audit failed: %s", e)
        session.state = onboarding.OnboardingState.VISUAL_PREF.value
        onboarding.save_session(session)
        await update.message.reply_text(
            f"Asset analysis failed: {_esc(str(e))}\n\nLet's pick a visual style instead.\n"
            "Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
            "<b>minimal</b> / <b>bold</b> / <b>elegant</b>",
            parse_mode="HTML",
        )


async def _run_onboarding_strategy(update: Update, session: onboarding.OnboardingSession) -> None:
    """Run strategy recommendation and advance session to CONFIRM."""
    from agent import strategy as strategy_mod
    from agent.asset_audit import AssetInventory, load_inventory

    await update.message.chat.send_action("typing")

    try:
        inventory = load_inventory()
        rec = await strategy_mod.recommend_strategy(
            brand_name=session.brand_name,
            description=session.description,
            platforms=session.platforms,
            inventory=inventory,
            visual_preferences=session.visual_preferences,
        )

        strategy_data = {
            "archetype": rec.archetype,
            "compositor_enabled": rec.compositor_enabled,
            "badge_text": rec.badge_text,
            "default_mode": rec.default_mode,
            "recommended_content_types": rec.recommended_content_types,
            "visual_style_notes": rec.visual_style_notes,
            "reasoning": rec.reasoning,
        }
        session, response = onboarding.finalize_strategy(session, strategy_data)
        onboarding.save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")
    except Exception as e:
        logger.error("Onboarding strategy failed: %s", e)
        # Use defaults and continue to confirm
        from agent.strategy import _ARCHETYPE_DEFAULTS
        archetype = session.asset_audit.get("archetype", "starting_fresh")
        defaults = _ARCHETYPE_DEFAULTS.get(archetype, _ARCHETYPE_DEFAULTS["starting_fresh"])
        strategy_data = {"archetype": archetype, **defaults, "reasoning": f"Auto-configured (strategy generation failed: {e})"}
        session, response = onboarding.finalize_strategy(session, strategy_data)
        onboarding.save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")


async def _maybe_compose(draft: dict, image_url: str, content_type: str):
    """Compositor guard. Returns (photo_to_send, composed_bytes_or_None).

    Priority chain: template > compositor > raw.
    /template off disables both templates and compositor.
    """
    cfg = _cc.get_config()
    if not cfg.compositor_enabled:
        return image_url, None

    from agent import template_memory as _tm

    # Priority 1: Template
    try:
        memory = _tm.TemplateMemory()
        template = memory.get_template_for_content_type(content_type)
        if template:
            composed = await _tm.apply_template(template, image_url, draft)
            if composed:
                return composed, composed
    except Exception as e:
        logger.debug("Template composition failed, falling through: %s", e)

    # Priority 2: Compositor
    composed = await compositor.compose_branded_image(draft, image_url, content_type)
    return (composed if composed else image_url), composed


async def template_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template — toggle image composition on/off."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    arg = text.partition("/template")[2].strip().lower()

    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"

    if not arg:
        # Show current status
        cfg = _cc.get_config()
        status = "ON" if cfg.compositor_enabled else "OFF"
        badge = cfg.badge_text or "(none)"
        mode = cfg.default_mode

        # Show active templates
        from agent import template_memory as _tm
        memory = _tm.TemplateMemory()
        templates = memory.list_templates()
        tpl_lines = ""
        if templates:
            tpl_lines = "\n\n<b>Active Templates:</b>\n"
            for t in templates:
                types_str = ", ".join(t.content_types) if t.content_types else "all"
                source_tag = f" [{t.source}]" if t.source else ""
                has_spec = " (spec)" if t.spec_json else ""
                tpl_lines += (
                    f"- <code>{_esc(t.id)}</code> {_esc(t.name)} "
                    f"({t.aspect_ratio}, {types_str}){source_tag}{has_spec}\n"
                )

        await update.message.reply_text(
            f"<b>Compositor Status</b>\n\n"
            f"Enabled: <b>{status}</b>\n"
            f"Badge: <code>{_esc(badge)}</code>\n"
            f"Mode: <code>{_esc(mode)}</code>\n\n"
            f"<code>/template on</code> — enable\n"
            f"<code>/template off</code> — disable\n"
            f"<code>/template_upload</code> — upload a custom template\n"
            f"<code>/template_from_reference</code> — generate from reference image\n"
            f"<code>/template_import</code> — import from Figma\n"
            f"<code>/font_upload</code> — upload custom font"
            f"{tpl_lines}",
            parse_mode="HTML",
        )
        return

    if arg not in ("on", "off"):
        await update.message.reply_text(
            "Usage: /template on | /template off | /template",
        )
        return

    enabled_value = "true" if arg == "on" else "false"

    # Read current guidelines
    if not guidelines_path.exists():
        await update.message.reply_text("No guidelines.md found. Run /setup first.")
        return

    content = guidelines_path.read_text(encoding="utf-8")

    # Check if COMPOSITOR section already exists
    import re as _re
    section_match = _re.search(r"##\s*COMPOSITOR(.*?)(?=\n##|\Z)", content, _re.DOTALL)
    if section_match:
        # Update the Enabled row
        section = section_match.group(0)
        updated = _re.sub(
            r"(\|\s*Enabled\s*\|\s*)(true|false|yes|no|on|off)(\s*\|)",
            rf"\g<1>{enabled_value}\g<3>",
            section,
            flags=_re.IGNORECASE,
        )
        content = content.replace(section, updated)
    else:
        # Add COMPOSITOR section at the end
        content += f"\n\n## COMPOSITOR\n\n| Setting        | Value          |\n|----------------|----------------|\n| Enabled        | {enabled_value}           |\n"

    guidelines_path.write_text(content, encoding="utf-8")
    _cc.invalidate_cache()
    compositor.clear_font_cache()
    guidelines.invalidate_brand_context()

    status_str = "ON" if arg == "on" else "OFF"
    await update.message.reply_text(
        f"Compositor <b>{status_str}</b>",
        parse_mode="HTML",
    )


async def template_upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_upload [name] — start template upload flow."""
    if not _authorized(update.effective_user.id):
        return

    template_name = " ".join(context.args) if context.args else ""
    context.user_data["awaiting_template"] = True
    context.user_data["template_name"] = template_name
    name_note = f" (name: <b>{_esc(template_name)}</b>)" if template_name else ""
    await update.message.reply_text(
        f"Send me a template image (frame, mockup, bordered layout).{name_note}\n"
        "I'll analyze it and register it for future posts.",
        parse_mode="HTML",
    )


async def template_test_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_test [content_type] — render a test composition with placeholder content."""
    if not _authorized(update.effective_user.id):
        return

    from agent import template_memory as _tm

    content_type = " ".join(context.args).strip() if context.args else "announcement"
    memory = _tm.TemplateMemory()
    template = memory.get_template_for_content_type(content_type)
    if not template:
        await update.message.reply_text(
            "No templates registered. Use /template_upload to add one.",
        )
        return

    await update.message.chat.send_action("upload_photo")

    # Create placeholder image using brand primary color
    cfg = _cc.get_config()
    primary = cfg.colors.get("primary")
    if primary:
        primary_rgb = primary.rgb
    else:
        primary_rgb = (107, 159, 212)
    placeholder = _PILImage.new("RGBA", (template.width, template.height), primary_rgb + (255,))
    placeholder_path = str(Path(tempfile.gettempdir()) / f"test_placeholder_{int(time.time())}.png")
    placeholder.save(placeholder_path, "PNG")

    test_draft = {"title": "HEADLINE HERE", "subtitle": "Subtitle text goes here"}

    try:
        result = await _tm.apply_template(template, placeholder_path, test_draft)
        if result:
            regions_str = ", ".join(f"{r.type}({r.width}x{r.height})" for r in template.regions)
            await update.message.reply_photo(
                photo=result,
                caption=(
                    f"Template test: <b>{_esc(template.name)}</b> (<code>{_esc(template.id)}</code>)\n"
                    f"Aspect: {_esc(template.aspect_ratio)} | Regions: {_esc(regions_str)}\n"
                    f"Content type: {_esc(content_type)}"
                ),
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("Template composition failed — check that the template has an image region.")
    except Exception as e:
        logger.error("Template test failed: %s", e)
        await update.message.reply_text(f"Template test failed: {_esc(str(e))}", parse_mode="HTML")
    finally:
        try:
            Path(placeholder_path).unlink(missing_ok=True)
        except Exception:
            pass


async def template_from_reference_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_from_reference [name] — generate a branded template from a reference image."""
    if not _authorized(update.effective_user.id):
        return

    template_name = " ".join(context.args) if context.args else ""
    context.user_data["awaiting_template_from_ref"] = True
    context.user_data["template_from_ref_name"] = template_name
    name_note = f" (name: <b>{_esc(template_name)}</b>)" if template_name else ""
    await update.message.reply_text(
        f"Send me a reference image (screenshot, post, layout you like).{name_note}\n"
        "I'll analyze the layout and recreate it as a branded template using your colors, fonts, and logo.",
        parse_mode="HTML",
    )


async def _handle_template_from_reference(update: Update, context: ContextTypes.DEFAULT_TYPE, tmp_path: str) -> None:
    """Generate a branded template preview from a reference image (no registration yet)."""
    from agent import template_generator as _tg

    context.user_data["awaiting_template_from_ref"] = False
    await update.message.chat.send_action("typing")
    await update.message.reply_text("Analyzing layout and building template spec...")

    try:
        design, preview_img = await _tg.analyze_and_generate(tmp_path)

        # Store design and metadata in user_data for adjustments
        context.user_data["tplref_pending"] = {
            "design": _tg.design_to_dict(design),
            "name": (context.user_data or {}).get("template_from_ref_name", ""),
        }

        await _send_template_preview(update, context, design, preview_img)
    except Exception as e:
        logger.error("Template generation from reference failed: %s", e)
        context.user_data.pop("tplref_pending", None)
        await update.message.reply_text(
            f"Template generation failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


async def _send_template_preview(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    design,
    generated_img,
) -> None:
    """Send a template preview with inline save/adjust/discard buttons."""
    from agent import template_generator as _tg

    buf = io.BytesIO()
    generated_img.convert("RGB").save(buf, "PNG")
    buf.seek(0)

    regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in design.regions)
    aspect = _tg._compute_aspect_ratio(design.canvas_width, design.canvas_height)
    name = (context.user_data.get("tplref_pending") or {}).get("name", "")

    has_spec = hasattr(design, "spec") and design.spec is not None
    method_note = "spec-rendered" if has_spec else "analyzed"
    caption = (
        f"<b>Template Preview</b> ({method_note})\n\n"
        f"Size: {design.canvas_width}x{design.canvas_height} ({aspect})\n"
        f"Style: {_esc(design.visual_style or 'detected')}\n"
        f"Regions: {_esc(regions_str) or 'none detected'}\n"
    )
    if name:
        caption += f"Name: <b>{_esc(name)}</b>\n"
    caption += "\nTap <b>Save</b> to register, <b>Adjust</b> to modify, or reply with feedback."

    buttons = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Save Template", callback_data="tplref_save"),
            InlineKeyboardButton("Adjust", callback_data="tplref_adjust"),
            InlineKeyboardButton("Discard", callback_data="tplref_discard"),
        ]
    ])

    await update.message.reply_photo(
        photo=buf,
        caption=caption,
        parse_mode="HTML",
        reply_markup=buttons,
    )


async def _handle_tplref_adjustment(update: Update, context: ContextTypes.DEFAULT_TYPE, feedback: str) -> None:
    """Apply user feedback to the pending template design — adjusts prompt and regenerates."""
    from agent import template_generator as _tg

    pending = context.user_data.get("tplref_pending")
    if not pending:
        return

    # "looks good" / "save" / "done" → treat as save
    lower = feedback.lower().strip()
    if lower in ("looks good", "save", "save it", "done", "perfect", "yes", "keep it", "ok", "okay"):
        design = _tg.design_from_dict(pending["design"])
        name = pending.get("name") or None
        saved_path = await _tg.save_generated_image(design)  # renders locally if spec available
        template = _tg.register_design(design, saved_path, name)
        context.user_data.pop("tplref_pending", None)

        regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in template.regions)
        await update.message.reply_text(
            f"<b>Template Saved</b>\n\n"
            f"Name: <b>{_esc(template.name)}</b>\n"
            f"ID: <code>{_esc(template.id)}</code>\n"
            f"Size: {template.width}x{template.height} ({template.aspect_ratio})\n"
            f"Regions: {_esc(regions_str) or 'none detected'}\n\n"
            f"This template will be used for future posts.",
            parse_mode="HTML",
        )
        return

    # "discard" / "cancel" → discard
    if lower in ("discard", "cancel", "nevermind", "never mind", "nah", "no"):
        context.user_data.pop("tplref_pending", None)
        await update.message.reply_text("Template discarded.")
        return

    # Otherwise treat as adjustment feedback — modify spec and re-render locally
    await update.message.chat.send_action("typing")
    await update.message.reply_text("Adjusting template spec and re-rendering...")

    try:
        design = _tg.design_from_dict(pending["design"])
        adjusted = await _tg.adjust_spec(design, feedback)

        # Re-render preview locally (instant, no API cost for image gen)
        if adjusted.spec:
            from agent.template_renderer import render_preview
            preview_img = render_preview(adjusted.spec)
        elif adjusted.generated_image_url:
            preview_img = await _tg.download_image(adjusted.generated_image_url)
            if not preview_img:
                raise ValueError("Failed to download regenerated template image.")
        else:
            from PIL import Image as _PILImage
            preview_img = _PILImage.new("RGB", (adjusted.canvas_width, adjusted.canvas_height), (14, 15, 43))

        # Update stored design
        pending["design"] = _tg.design_to_dict(adjusted)
        context.user_data["tplref_pending"] = pending

        await _send_template_preview(update, context, adjusted, preview_img)
    except Exception as e:
        logger.error("Template adjustment failed: %s", e)
        await update.message.reply_text(
            f"Adjustment failed: {_esc(str(e))}\nYou can try again or tap Save/Discard.",
            parse_mode="HTML",
        )


async def tplref_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button presses for template-from-reference previews."""
    from agent import template_generator as _tg

    query = update.callback_query
    await query.answer()

    if not _authorized(query.from_user.id):
        return

    action = query.data.replace("tplref_", "")
    pending = (context.user_data or {}).get("tplref_pending")

    if not pending:
        await query.message.reply_text("No template preview pending.")
        return

    if action == "save":
        design = _tg.design_from_dict(pending["design"])
        name = pending.get("name") or None

        # Render and save the template frame (locally, no download needed)
        saved_path = await _tg.save_generated_image(design)
        template = _tg.register_design(design, saved_path, name)

        context.user_data.pop("tplref_pending", None)

        regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in template.regions)
        await query.edit_message_caption(
            caption=(
                f"<b>Template Saved</b>\n\n"
                f"Name: <b>{_esc(template.name)}</b>\n"
                f"ID: <code>{_esc(template.id)}</code>\n"
                f"Size: {template.width}x{template.height} ({template.aspect_ratio})\n"
                f"Regions: {_esc(regions_str) or 'none detected'}\n\n"
                f"This template will be used for future posts."
            ),
            parse_mode="HTML",
            reply_markup=None,
        )

    elif action == "adjust":
        await query.message.reply_text(
            "What should I adjust? Reply with your feedback, e.g.:\n"
            "<i>make the colors more vibrant</i>\n"
            "<i>use a darker background</i>\n"
            "<i>add more spacing between elements</i>",
            parse_mode="HTML",
        )

    elif action == "discard":
        context.user_data.pop("tplref_pending", None)
        await query.edit_message_caption(
            caption="<b>Template discarded.</b>",
            parse_mode="HTML",
            reply_markup=None,
        )


async def _handle_template_region_update(
    update: Update, context: ContextTypes.DEFAULT_TYPE, description: str,
) -> None:
    """Convert a natural language region description into pixel coordinates and update the template."""
    from agent import template_memory as _tm

    template_id = context.user_data.get("last_uploaded_template_id", "")
    if not template_id:
        await update.message.reply_text("No recently uploaded template to update.")
        return

    memory = _tm.TemplateMemory()
    templates = memory.list_templates()
    template = next((t for t in templates if t.id == template_id), None)
    if not template:
        await update.message.reply_text(
            f"Template <code>{_esc(template_id)}</code> not found in manifest.",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Parsing region positions...")

    try:
        new_regions = await _tm.parse_region_description(
            description, template.width, template.height,
        )
    except Exception as e:
        logger.error("Region parsing failed: %s", e)
        await update.message.reply_text(f"Region parsing failed: {_esc(str(e))}", parse_mode="HTML")
        return

    if not new_regions:
        await update.message.reply_text("Couldn't parse any regions from that description. Try being more specific.")
        return

    # Update the template
    updated = memory.update_template_regions(template_id, new_regions)
    if not updated:
        await update.message.reply_text("Failed to update template regions.")
        return

    # Clear the tracking flag so follow-up messages go to normal routing
    context.user_data.pop("last_uploaded_template_id", None)

    # Confirm with region coordinates
    region_lines = []
    for r in new_regions:
        region_lines.append(
            f"  {r.type}: ({r.x}, {r.y}, {r.width}, {r.height}) — {r.description}"
        )
    regions_display = "\n".join(region_lines)
    await update.message.reply_text(
        f"<b>Template updated</b> — <code>{_esc(template_id)}</code>\n\n"
        f"<pre>{_esc(regions_display)}</pre>",
        parse_mode="HTML",
    )

    # Auto-run template_test to show what it looks like
    await update.message.chat.send_action("upload_photo")

    # Find a content type for the template
    content_type = updated.content_types[0] if updated.content_types else "announcement"

    cfg = _cc.get_config()
    primary = cfg.colors.get("primary")
    primary_rgb = primary.rgb if primary else (107, 159, 212)
    placeholder = _PILImage.new("RGBA", (updated.width, updated.height), primary_rgb + (255,))
    placeholder_path = str(Path(tempfile.gettempdir()) / f"test_placeholder_{int(time.time())}.png")
    placeholder.save(placeholder_path, "PNG")

    test_draft = {"title": "HEADLINE HERE", "subtitle": "Subtitle text goes here"}

    try:
        result = await _tm.apply_template(updated, placeholder_path, test_draft)
        if result:
            regions_str = ", ".join(f"{r.type}({r.width}x{r.height})" for r in updated.regions)
            await update.message.reply_photo(
                photo=result,
                caption=(
                    f"Template test: <b>{_esc(updated.name)}</b>\n"
                    f"Regions: {_esc(regions_str)}"
                ),
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("Template test composition failed — check that the template has an image region.")
    except Exception as e:
        logger.warning("Auto template test failed: %s", e)
    finally:
        try:
            Path(placeholder_path).unlink(missing_ok=True)
        except Exception:
            pass


async def _handle_template_upload(update: Update, context: ContextTypes.DEFAULT_TYPE, tmp_path: str) -> None:
    """Process a template image upload — analyze and register."""
    from agent import template_memory as _tm

    context.user_data["awaiting_template"] = False
    await update.message.chat.send_action("typing")
    await update.message.reply_text("Analyzing template...")

    # Copy to templates dir
    templates_dir = Path(settings.BRAND_FOLDER) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    dest = templates_dir / f"template_{int(time.time())}.png"
    try:
        _PILImage.open(tmp_path).convert("RGBA").save(str(dest), "PNG")
    except Exception as e:
        await update.message.reply_text(f"Failed to process image: {_esc(str(e))}", parse_mode="HTML")
        return

    try:
        user_name = (context.user_data or {}).get("template_name", "")
        template = await _tm.register_template(
            str(dest),
            name=user_name or f"Template {int(time.time()) % 10000}",
        )

        # Track the uploaded template so follow-up messages can update its regions
        context.user_data["last_uploaded_template_id"] = template.id

        regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in template.regions)
        await update.message.reply_text(
            f"<b>Template Registered</b>\n\n"
            f"ID: <code>{_esc(template.id)}</code>\n"
            f"Size: {template.width}x{template.height} ({template.aspect_ratio})\n"
            f"Regions: {_esc(regions_str) or 'none detected'}\n"
            f"Notes: {_esc(template.analysis_notes or 'none')}\n\n"
            f"This template will be used for future posts.\n"
            f"You can describe region positions to adjust (e.g. \"top text across the top 15%, image fills the full canvas\").",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Template registration failed: %s", e)
        await update.message.reply_text(f"Template analysis failed: {_esc(str(e))}", parse_mode="HTML")


async def _handle_pipeline_mode(update: Update, request: str) -> None:
    """Run the existing multi-step pipeline for a content request."""
    await update.message.chat.send_action("typing")

    step_messages = {}

    async def on_step(step_num: int, total: int, step_name: str, summary: str):
        icon = _STEP_ICONS.get(step_name, "\u26A1")
        msg_text = f"{icon} <b>[{step_num}/{total}] {step_name}</b>\n{_esc(summary)}"
        sent = await update.message.reply_text(msg_text, parse_mode="HTML")
        step_messages[step_num] = sent
        await update.message.chat.send_action("typing")

    try:
        brand_context = guidelines.get_brand_context()
        logger.info("Brand context loaded: %d chars", len(brand_context))

        pipeline_result = await brain.pipeline_generate(
            request=request,
            brand_context=brand_context,
            on_step=on_step,
        )

        draft = pipeline_result.draft

        if pipeline_result.fell_back:
            await update.message.reply_text(
                "<i>Note: Pipeline had an issue, used direct generation instead.</i>",
                parse_mode="HTML",
            )

        logger.info("Draft generated: %s", draft.get("caption", "")[:80])

        # Generate image with smart model routing (mode-aware)
        image_url = None
        cfg = _cc.get_config()
        should_gen = draft.get("image_prompt") and cfg.default_mode != "text_only"
        if should_gen:
            await update.message.chat.send_action("upload_photo")
            content_type = draft.get("content_type", "announcement")
            from agent import template_memory as _tm
            template_aspect = _tm.get_aspect_ratio_for_content_type(content_type)
            image_url = await image_gen.generate_image(draft["image_prompt"], content_type=content_type, aspect_ratio=template_aspect)
            if not image_url:
                await update.message.reply_text(
                    "Image generation failed — sending text draft only.",
                )

        # Save pending state
        state.save_pending(
            caption=draft["caption"],
            hashtags=draft.get("hashtags", []),
            image_url=image_url,
            alt_text=draft["alt_text"],
            image_prompt=draft["image_prompt"],
            original_request=request,
        )

        await _send_draft(update, draft, image_url)

    except Exception as e:
        logger.error("Pipeline error: %s", e)
        await update.message.reply_text(
            f"Something went wrong: {_esc(str(e))}\n\nPlease try again.",
            parse_mode="HTML",
        )


async def _send_draft(
    update: Update,
    draft: dict,
    image_url: str | None,
    resources=None,
    image_urls: list[str] | None = None,
) -> None:
    """Send the generated draft to the user for review.

    When image_urls has >1 item, sends each as a numbered option with its own
    composed brand template so the CMO can compare side-by-side.
    """
    caption = draft["caption"]
    content_type = draft.get("content_type", "default")

    # Ensure the compositor has title/subtitle — synthesize from caption if missing
    if not draft.get("title") and not draft.get("subtitle") and caption:
        sentences = caption.split(". ", 1)
        draft["title"] = sentences[0].rstrip(".")
        draft["subtitle"] = sentences[1] if len(sentences) > 1 else ""

    # Build template section if title/subtitle present
    template_section = ""
    if draft.get("title") or draft.get("subtitle"):
        cfg = _cc.get_config()
        platform_line = (
            f"<b>Platform:</b> {_esc(draft.get('platform', cfg.badge_text or ''))}\n"
            if cfg.badge_text else ""
        )
        template_section = (
            f"\n<b>--- Image Template ---</b>\n"
            f"<b>Title:</b> {_esc(draft.get('title', ''))}\n"
            f"<b>Subtitle:</b> {_esc(draft.get('subtitle', ''))}\n"
            + platform_line
        )

    # --- Multi-option path (N>1 images) ---
    if image_urls and len(image_urls) > 1:
        # Compose all options in parallel for faster response
        import asyncio as _asyncio
        compose_tasks = [_maybe_compose(draft, url, content_type) for url in image_urls]
        compose_results = await _asyncio.gather(*compose_tasks, return_exceptions=True)

        for idx, result in enumerate(compose_results, 1):
            if isinstance(result, Exception):
                logger.warning("Failed to compose option %d: %s", idx, result)
                continue
            photo, composed = result

            # Save composed image for approve-time archiving (save last option)
            if composed and isinstance(composed, io.BytesIO):
                try:
                    tmp_composed = str(Path(tempfile.gettempdir()) / f"brandmover_composed_opt{idx}_{int(time.time())}.png")
                    with open(tmp_composed, "wb") as f:
                        f.write(composed.getvalue())
                    composed.seek(0)
                    # Store last composed for the first option by default
                    if idx == 1:
                        state.set_last_composed(tmp_composed, content_type)
                except Exception as e:
                    logger.warning("Failed to save composed option %d: %s", idx, e)

            opt_caption = f"<b>Option {idx} of {len(image_urls)}</b>"
            try:
                await update.message.reply_photo(
                    photo=photo,
                    caption=opt_caption,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("Failed to send option %d image: %s", idx, e)

        # Send summary text after all options
        approve_hints = " | ".join(f"/approve {i}" for i in range(1, len(image_urls) + 1))
        text_msg = (
            f"<b>Draft Ready — {len(image_urls)} options</b>\n\n"
            f"{_esc(caption)}\n"
            f"{template_section}\n"
            f"<i>Alt text:</i> {_esc(draft.get('alt_text', ''))}\n\n"
            f"{approve_hints}\n"
            f"/reject <i>feedback</i> to revise\n"
            f"/edit [feedback] to surgically fix specific elements"
        )
        if resources:
            text_msg += f"\n\n<i>Resources: {_esc(resources.to_summary())}</i>"
        await update.message.reply_text(text_msg, parse_mode="HTML")
        return

    # --- Single image path (original behavior) ---
    text_msg = (
        f"<b>Draft Ready</b>\n\n"
        f"{_esc(caption)}\n"
        f"{template_section}\n"
        f"<i>Alt text:</i> {_esc(draft.get('alt_text', ''))}\n\n"
        f"/approve to post to X\n"
        f"/reject <i>feedback</i> to revise\n"
        f"/edit [feedback] to surgically fix specific elements"
    )

    # Append resource summary if available
    if resources:
        text_msg += f"\n\n<i>Resources: {_esc(resources.to_summary())}</i>"

    # Inline keyboard for quick actions
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Approve", callback_data="draft_approve"),
            InlineKeyboardButton("Reject", callback_data="draft_reject"),
        ],
        [
            InlineKeyboardButton("Edit", callback_data="draft_edit"),
            InlineKeyboardButton("Reroll", callback_data="draft_reroll"),
        ],
    ])

    if image_url:
        photo, composed = await _maybe_compose(draft, image_url, content_type)

        # Save composed image to temp for approve-time archiving
        if composed and isinstance(composed, io.BytesIO):
            try:
                tmp_composed = str(Path(tempfile.gettempdir()) / f"brandmover_last_composed_{int(time.time())}.png")
                with open(tmp_composed, "wb") as f:
                    f.write(composed.getvalue())
                composed.seek(0)  # reset for Telegram send
                state.set_last_composed(tmp_composed, content_type)
            except Exception as e:
                logger.warning("Failed to save composed image for archiving: %s", e)

        try:
            await update.message.reply_photo(
                photo=photo,
                caption=text_msg[:1024],  # Telegram photo caption limit
                parse_mode="HTML",
                reply_markup=keyboard,
            )
            if len(text_msg) > 1024:
                await update.message.reply_text(text_msg, parse_mode="HTML")
        except Exception as e:
            logger.warning("Failed to send image via Telegram: %s — sending text only", e)
            await update.message.reply_text(text_msg, parse_mode="HTML", reply_markup=keyboard)
    else:
        await update.message.reply_text(text_msg, parse_mode="HTML", reply_markup=keyboard)

    # Track context — draft was sent
    try:
        user_id = update.effective_user.id if update.effective_user else 0
        if user_id:
            conversation_context.update_context(
                user_id,
                last_bot_action="sent_draft",
                pending_draft_exists=True,
                last_content_type=content_type,
            )
    except Exception as e:
        logger.debug("Context tracking failed in _send_draft: %s", e)


# ---------------------------------------------------------------------------
# Inline draft button callbacks
# ---------------------------------------------------------------------------


class _CallbackProxy:
    """Lightweight proxy so callback query responses go through query.message."""
    def __init__(self, update, query):
        self._update = update
        self.message = query.message
        self.effective_user = query.from_user
    def __getattr__(self, name):
        return getattr(self._update, name)


async def draft_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses on draft messages."""
    query = update.callback_query
    await query.answer()

    if not _authorized(query.from_user.id):
        return

    action = query.data.split("_", 1)[1]  # approve|reject|edit|reroll

    # Proxy so _do_approve/_do_reject reply via query.message
    proxy = _CallbackProxy(update, query)

    if action == "approve":
        await _do_approve(proxy, context, source="button")
    elif action == "reject":
        await query.message.reply_text(
            "What should I change? Reply with your feedback, e.g.:\n"
            "<i>make it more urgent and add a CTA</i>",
            parse_mode="HTML",
        )
    elif action == "edit":
        await query.message.reply_text(
            "What should I edit? Reply with your feedback, e.g.:\n"
            "<i>change the background to blue</i>",
            parse_mode="HTML",
        )
    elif action == "reroll":
        pending = state.get_pending()
        if pending:
            original = pending.get("original_request", "")
            state.clear_pending()
            state.clear_draft_history()
            await query.message.reply_text("Regenerating...")
            if original:
                if settings.AGENT_MODE == "agent":
                    await _handle_agent_mode(proxy, original)
                else:
                    await _handle_pipeline_mode(proxy, original)


# ---------------------------------------------------------------------------
# Auto-post draft delivery (called by scheduler, no Update object)
# ---------------------------------------------------------------------------


async def send_auto_draft(bot, draft: dict, image_url: str | None, slot_name: str) -> None:
    """Send an auto-generated draft to Telegram for review.

    Called by the in-process scheduler — uses the bot instance directly
    instead of replying to a user message.
    """
    chat_id = settings.TELEGRAM_ALLOWED_USER_ID
    caption = draft.get("caption", "")
    content_type = draft.get("content_type", "default")

    # Ensure title/subtitle for compositor
    if not draft.get("title") and not draft.get("subtitle") and caption:
        sentences = caption.split(". ", 1)
        draft["title"] = sentences[0].rstrip(".")
        draft["subtitle"] = sentences[1] if len(sentences) > 1 else ""

    template_section = ""
    if draft.get("title") or draft.get("subtitle"):
        template_section = (
            f"\n<b>--- Image Template ---</b>\n"
            f"<b>Title:</b> {_esc(draft.get('title', ''))}\n"
            f"<b>Subtitle:</b> {_esc(draft.get('subtitle', ''))}\n"
        )

    text_msg = (
        f"<b>Auto-Draft Ready</b>  [slot: <code>{_esc(slot_name)}</code>]\n\n"
        f"{_esc(caption)}\n"
        f"{template_section}\n"
        f"<i>Alt text:</i> {_esc(draft.get('alt_text', ''))}\n\n"
        f"/approve to post to X\n"
        f"/reject <i>feedback</i> to revise\n"
        f"/cancel to discard"
    )

    if image_url:
        photo, composed = await _maybe_compose(draft, image_url, content_type)

        # Save composed for archiving on approve
        if composed and isinstance(composed, io.BytesIO):
            try:
                tmp_composed = str(Path(tempfile.gettempdir()) / f"brandmover_auto_composed_{int(time.time())}.png")
                with open(tmp_composed, "wb") as f:
                    f.write(composed.getvalue())
                composed.seek(0)
                state.set_last_composed(tmp_composed, content_type)
            except Exception as e:
                logger.warning("Failed to save auto composed image: %s", e)

        try:
            await bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=text_msg[:1024],
                parse_mode="HTML",
            )
            if len(text_msg) > 1024:
                await bot.send_message(chat_id=chat_id, text=text_msg, parse_mode="HTML")
        except Exception as e:
            logger.warning("Failed to send auto-draft image: %s — sending text", e)
            await bot.send_message(chat_id=chat_id, text=text_msg, parse_mode="HTML")
    else:
        await bot.send_message(chat_id=chat_id, text=text_msg, parse_mode="HTML")

    logger.info("Auto-draft sent to Telegram for slot: %s", slot_name)


# ---------------------------------------------------------------------------
# Auto-post control commands
# ---------------------------------------------------------------------------


async def autostatus_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autostatus — show auto-posting scheduler status."""
    if not _authorized(update.effective_user.id):
        return

    status = auto_state.get_status_summary()
    schedule = scheduler.load_schedule()
    slots = schedule.get("slots", {})
    global_cfg = schedule.get("global", {})

    paused_str = "PAUSED" if status["paused"] else "ACTIVE"
    last_ts = status["last_post_timestamp"]
    last_str = time.strftime("%H:%M UTC", time.gmtime(last_ts)) if last_ts else "never"

    slot_lines = []
    for name, cfg in slots.items():
        enabled = cfg.get("enabled", True)
        posted = auto_state.is_slot_posted(name)
        icon = "\u2705" if posted else ("\u23F0" if enabled else "\u274C")
        slot_lines.append(f"  {icon} {name} ({cfg.get('hour_utc', '?')}:00 UTC)")

    recent = status.get("recent_captions", [])
    recent_str = "\n".join(f"  - {c}" for c in recent) if recent else "  (none)"

    # User-scheduled queue info
    scheduled_items = schedule_queue.list_scheduled()
    if scheduled_items:
        from datetime import datetime, timezone
        sched_lines = []
        for item in sorted(scheduled_items, key=lambda x: x.get("scheduled_utc", 0))[:5]:
            dt = datetime.fromtimestamp(item.get("scheduled_utc", 0), tz=timezone.utc)
            sched_lines.append(f"  \u23F0 {dt.strftime('%b %d %H:%M')} — {item.get('prompt', '')[:40]}")
        sched_section = f"\n<b>Scheduled ({len(scheduled_items)}):</b>\n" + "\n".join(sched_lines)
    else:
        sched_section = "\n<b>Scheduled:</b> none"

    msg = (
        f"<b>Auto-Post Status: {paused_str}</b>\n\n"
        f"<b>Enabled:</b> {settings.AUTO_POST_ENABLED}\n"
        f"<b>Dry run:</b> {settings.AUTO_POST_DRY_RUN}\n"
        f"<b>Posts today:</b> {status['posts_today']}/{global_cfg.get('max_posts_per_day', 6)}\n"
        f"<b>Last post:</b> {last_str}\n"
        f"<b>Min gap:</b> {global_cfg.get('min_gap_minutes', 120)} min\n\n"
        f"<b>Slots:</b>\n" + "\n".join(slot_lines) +
        sched_section + "\n\n"
        f"<b>Recent:</b>\n{recent_str}"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def autopause_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autopause — toggle auto-posting pause state."""
    if not _authorized(update.effective_user.id):
        return

    currently_paused = auto_state.is_paused()
    auto_state.set_paused(not currently_paused)

    if currently_paused:
        await update.message.reply_text("Auto-posting <b>resumed</b>.", parse_mode="HTML")
    else:
        await update.message.reply_text("Auto-posting <b>paused</b>.", parse_mode="HTML")


async def autoforce_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autoforce <slot> — force a specific slot to post now."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split()

    schedule = scheduler.load_schedule()
    slots = schedule.get("slots", {})

    if len(parts) < 2:
        slot_list = ", ".join(f"<code>{s}</code>" for s in slots.keys())
        await update.message.reply_text(
            f"Usage: /autoforce <i>slot_name</i>\n\n"
            f"Available: {slot_list}",
            parse_mode="HTML",
        )
        return

    slot_name = parts[1]
    if slot_name not in slots:
        slot_list = ", ".join(f"<code>{s}</code>" for s in slots.keys())
        await update.message.reply_text(
            f"Unknown slot: <code>{_esc(slot_name)}</code>\n\n"
            f"Available: {slot_list}",
            parse_mode="HTML",
        )
        return

    dry_run = "--dry-run" in text or settings.AUTO_POST_DRY_RUN

    # Block if there's already a pending draft
    if state.has_pending() and not dry_run:
        await update.message.reply_text(
            "A draft is already pending. /approve, /reject, or /cancel it first.",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text(
        f"Forcing slot <b>{_esc(slot_name)}</b>{'  (dry run)' if dry_run else ''}...\n"
        f"Generating draft for your review...",
        parse_mode="HTML",
    )

    from scripts.auto_post import process_slot

    global_config = schedule.get("global", {})
    slot_config = slots[slot_name]

    try:
        bot = context.bot
        success = await process_slot(
            slot_name, slot_config, global_config,
            dry_run=dry_run, bot=bot,
        )
        if success and not dry_run:
            # Draft was sent via send_auto_draft — no extra message needed
            pass
        elif success and dry_run:
            await update.message.reply_text(
                f"Dry run for <b>{_esc(slot_name)}</b> complete. Check logs for details.",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text(
                f"Slot <b>{_esc(slot_name)}</b> did not produce a draft. Check logs for details.",
                parse_mode="HTML",
            )
    except Exception as e:
        logger.error("autoforce failed for %s: %s", slot_name, e)
        await update.message.reply_text(
            f"Force draft failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /schedule, /scheduled, /unschedule — user-driven scheduling
# ---------------------------------------------------------------------------


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /schedule <time> <prompt> — schedule a post for a specific time.

    Examples:
        /schedule 3pm post about our new product launch
        /schedule tomorrow 9am morning engagement thread
        /schedule in 2 hours something cool about the community
        /schedule daily 3pm afternoon update
        /schedule weekly monday 9am week in review
    """
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    # Strip the /schedule command prefix
    args = text.split(None, 1)
    if len(args) < 2:
        await update.message.reply_text(
            "<b>Schedule a post</b>\n\n"
            "Usage: /schedule <i>time</i> <i>prompt</i>\n\n"
            "<b>Examples:</b>\n"
            "<code>/schedule 3pm post about our launch</code>\n"
            "<code>/schedule tomorrow 9am morning engagement</code>\n"
            "<code>/schedule in 2 hours community update</code>\n"
            "<code>/schedule daily 3pm afternoon post</code>\n"
            "<code>/schedule weekly monday 9am week in review</code>\n\n"
            "<b>Time formats:</b> 3pm, 9:30am, 15:00, tomorrow, monday, in 2 hours\n"
            "<b>Recurrence:</b> prefix with <code>daily</code> or <code>weekly</code>",
            parse_mode="HTML",
        )
        return

    prompt, ts, recurrence, display = schedule_queue.parse_schedule_command(args[1])

    if prompt is None:
        await update.message.reply_text(
            f"{_esc(display)}",
            parse_mode="HTML",
        )
        return

    item = schedule_queue.add_scheduled(prompt, ts, recurrence or "once")
    recurrence_tag = f" ({recurrence})" if recurrence and recurrence != "once" else ""

    await update.message.reply_text(
        f"<b>Post scheduled{_esc(recurrence_tag)}</b>\n\n"
        f"<b>Time:</b> {_esc(display)}\n"
        f"<b>Prompt:</b> {_esc(prompt[:200])}\n"
        f"<b>ID:</b> <code>{item['id']}</code>\n\n"
        f"I'll generate a draft at the scheduled time and send it here for your approval.\n"
        f"Use /unschedule <code>{item['id']}</code> to cancel.",
        parse_mode="HTML",
    )


async def scheduled_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /scheduled — list upcoming scheduled posts."""
    if not _authorized(update.effective_user.id):
        return

    items = schedule_queue.list_scheduled()

    if not items:
        await update.message.reply_text(
            "No scheduled posts.\n\n"
            "Use /schedule <i>time</i> <i>prompt</i> to schedule one.",
            parse_mode="HTML",
        )
        return

    from datetime import datetime, timezone

    lines = ["<b>Scheduled Posts</b>\n"]
    for item in sorted(items, key=lambda x: x.get("scheduled_utc", 0)):
        ts = item.get("scheduled_utc", 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_str = dt.strftime("%b %d %H:%M UTC")
        status = item.get("status", "pending")
        recurrence = item.get("recurrence", "once")
        rec_tag = f" [{recurrence}]" if recurrence != "once" else ""

        icon = "\u23F0" if status == "pending" else "\u2699\uFE0F"  # clock or gear
        lines.append(
            f"{icon} <code>{item['id']}</code> — {_esc(time_str)}{_esc(rec_tag)}\n"
            f"   {_esc(item.get('prompt', '')[:80])}"
        )

    lines.append(f"\n<i>{len(items)} scheduled post{'s' if len(items) != 1 else ''}</i>")
    lines.append("\nUse /unschedule <code>ID</code> to cancel.")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def unschedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /unschedule <id> — cancel a scheduled post."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split()

    if len(parts) < 2:
        items = schedule_queue.list_scheduled()
        if not items:
            await update.message.reply_text("No scheduled posts to cancel.")
            return

        item_list = ", ".join(f"<code>{i['id']}</code>" for i in items)
        await update.message.reply_text(
            f"Usage: /unschedule <i>id</i>\n\n"
            f"Active IDs: {item_list}\n\n"
            f"Use /scheduled to see details.",
            parse_mode="HTML",
        )
        return

    item_id = parts[1]
    if schedule_queue.cancel_scheduled(item_id):
        await update.message.reply_text(
            f"Cancelled scheduled post <code>{_esc(item_id)}</code>.",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            f"No active scheduled post with ID <code>{_esc(item_id)}</code>.\n"
            f"Use /scheduled to see current posts.",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /generate — standalone asset generation
# ---------------------------------------------------------------------------

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /generate <type> <description> — generate a standalone branded asset."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=2)  # "/generate", "type", "description..."

    asset_types = ", ".join(asset_gen.SUPPORTED_ASSET_TYPES)
    if len(parts) < 3:
        await update.message.reply_text(
            f"Usage: /generate <i>type</i> <i>description</i>\n\n"
            f"Types: {_esc(asset_types)}\n\n"
            f"Example: <code>/generate logo a shield with lightning bolt</code>",
            parse_mode="HTML",
        )
        return

    asset_type = parts[1].lower()
    description = parts[2]

    if _rate_limited(update.effective_user.id):
        await update.message.reply_text(
            f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
        )
        return

    await update.message.chat.send_action("upload_photo")
    await update.message.reply_text(
        f"Generating <b>{_esc(asset_type)}</b> options...",
        parse_mode="HTML",
    )

    try:
        result = await asset_gen.generate_asset(asset_type, description)
    except Exception as e:
        logger.error("generate_asset failed: %s", e)
        await update.message.reply_text(f"Generation failed: {_esc(str(e))}", parse_mode="HTML")
        return

    if result.get("error"):
        await update.message.reply_text(
            f"Error: {_esc(result['error'])}", parse_mode="HTML"
        )
        return

    urls = result.get("urls", [])
    if not urls:
        await update.message.reply_text("No images were generated. Check logs for details.")
        return

    # Log to generation history
    try:
        await generation_history.async_log_generation(
            asset_type=asset_type,
            content_type=result.get("content_type", ""),
            prompt=result.get("prompt", description),
            model_id=result.get("model_id", "auto"),
            image_urls=urls,
            original_request=f"/generate {asset_type} {description}",
        )
    except Exception as e:
        logger.warning("Failed to log generation history: %s", e)

    # Build a 2x2 grid image and send with inline keyboard
    grid_image = await _build_asset_grid(urls)

    # Save as pending for /approve N (and callback buttons)
    state.save_pending(
        caption=f"[{asset_type}] {description}",
        hashtags=[],
        image_url=urls[0] if urls else "",
        alt_text=description,
        image_prompt=result.get("prompt", description),
        original_request=f"/generate {asset_type} {description}",
        image_urls=urls,
        content_type=result.get("content_type"),
    )

    if grid_image:
        # Build inline keyboard: Approve 1-N + Reject All
        buttons = [
            InlineKeyboardButton(f"Approve {i}", callback_data=f"gen_approve:{i}")
            for i in range(1, len(urls) + 1)
        ]
        buttons.append(
            InlineKeyboardButton("Reject All", callback_data="gen_reject")
        )
        # Arrange: approve buttons on first row, reject on second
        keyboard = InlineKeyboardMarkup([
            buttons[:-1],
            [buttons[-1]],
        ])

        await update.message.reply_photo(
            photo=grid_image,
            caption=f"<b>{_esc(asset_type)}</b> — {len(urls)} options generated",
            parse_mode="HTML",
            reply_markup=keyboard,
        )
    else:
        # Fallback: send individual photos if grid fails
        for i, url in enumerate(urls, 1):
            try:
                await update.message.reply_photo(
                    photo=url,
                    caption=f"Option {i}/{len(urls)} — {_esc(asset_type)}",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("Failed to send option %d: %s", i, e)

        await update.message.reply_text(
            f"{len(urls)} option(s) generated. Use /approve N to select one.",
            parse_mode="HTML",
        )


async def _build_asset_grid(urls: list[str]) -> io.BytesIO | None:
    """Download images and compose a labeled 2x2 grid. Returns BytesIO or None."""
    import httpx as _httpx

    if not urls:
        return None

    images: list[_PILImage.Image] = []
    try:
        async with _httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            for url in urls[:4]:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    img = _PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning("Grid download failed for %s: %s", url[:60], e)
    except Exception as e:
        logger.warning("Grid build failed: %s", e)
        return None

    if not images:
        return None

    # Target cell size
    cell_w, cell_h = 640, 480
    pad = 8
    label_h = 32

    cols = 2 if len(images) > 1 else 1
    rows = (len(images) + cols - 1) // cols
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * (cell_h + label_h) + (rows + 1) * pad

    grid = _PILImage.new("RGB", (grid_w, grid_h), (20, 20, 40))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)

        try:
            font = ImageFont.load_default(size=20)
        except TypeError:
            font = ImageFont.load_default()

        for idx, img in enumerate(images):
            col = idx % cols
            row = idx // cols
            x = pad + col * (cell_w + pad)
            y = pad + row * (cell_h + label_h + pad)

            # Crop-fill to cell size
            sr = img.width / img.height
            tr = cell_w / cell_h
            if sr > tr:
                nw, nh = int(cell_h * sr), cell_h
            else:
                nw, nh = cell_w, int(cell_w / sr)
            resized = img.resize((nw, nh), _PILImage.LANCZOS)
            ox, oy = (nw - cell_w) // 2, (nh - cell_h) // 2
            cropped = resized.crop((ox, oy, ox + cell_w, oy + cell_h))

            # Draw label background
            draw.rectangle([x, y, x + cell_w, y + label_h], fill=(40, 40, 60))
            draw.text((x + 10, y + 6), f"Option {idx + 1}", fill=(255, 255, 255), font=font)

            grid.paste(cropped, (x, y + label_h))
    except Exception as e:
        logger.warning("Grid label drawing failed: %s", e)
        return None

    buf = io.BytesIO()
    grid.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf


async def generate_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks for /generate asset approval/rejection."""
    query = update.callback_query
    if not query or not query.data:
        return

    user_id = query.from_user.id if query.from_user else 0
    if not _authorized(user_id):
        await query.answer("Not authorized.")
        return

    data = query.data

    if data.startswith("gen_approve:"):
        # Parse option number
        try:
            option_num = int(data.split(":")[1])
        except (ValueError, IndexError):
            await query.answer("Invalid option.")
            return

        pending = state.get_pending()
        if not pending:
            await query.answer("Nothing pending to approve.")
            return

        image_urls = pending.get("image_urls", [])
        if not image_urls or option_num < 1 or option_num > len(image_urls):
            await query.answer(f"Invalid option {option_num}.")
            return

        # Select the chosen image
        pending["image_url"] = image_urls[option_num - 1]

        # Log feedback
        count = await feedback.async_log_feedback(
            request=pending.get("original_request", ""),
            draft=pending,
            accepted=True,
            resources_used=pending.get("resources_used", []),
        )

        # Update generation history
        try:
            ts = pending.get("timestamp", 0)
            if ts:
                await generation_history.async_update_generation_status(ts, "approved")
        except Exception as e:
            logger.debug("Generation history update failed: %s", e)

        # Add to LoRA training set
        if pending.get("image_url"):
            try:
                from agent import lora_pipeline
                lora_count, _ = await lora_pipeline.add_training_image_from_url(
                    pending["image_url"],
                    pending.get("image_prompt", ""),
                    pending.get("content_type", "brand_asset"),
                )
                logger.info("LoRA training image added (%d total)", lora_count)
            except Exception as e:
                logger.debug("LoRA training image add failed: %s", e)

        state.clear_pending()
        await query.answer(f"Option {option_num} approved!")
        await query.edit_message_caption(
            caption=f"Approved option {option_num}",
            parse_mode="HTML",
        )

    elif data == "gen_reject":
        pending = state.get_pending()
        if not pending:
            await query.answer("Nothing pending.")
            return

        # Log rejection
        await feedback.async_log_feedback(
            request=pending.get("original_request", ""),
            draft=pending,
            accepted=False,
            feedback_text="Rejected via button",
            resources_used=pending.get("resources_used", []),
        )

        try:
            ts = pending.get("timestamp", 0)
            if ts:
                await generation_history.async_update_generation_status(ts, "rejected")
        except Exception as e:
            logger.debug("Generation history update failed: %s", e)

        state.clear_pending()
        await query.answer("All options rejected.")
        await query.edit_message_caption(
            caption="Rejected. Use /generate again with feedback.",
            parse_mode="HTML",
        )

    else:
        await query.answer("Unknown action.")


# ---------------------------------------------------------------------------
# /logo — view/set brand logo
# ---------------------------------------------------------------------------

async def logo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /logo — show current logo or prepare for logo upload."""
    if not _authorized(update.effective_user.id):
        return

    logo_path = Path(settings.BRAND_FOLDER) / "assets" / "logo.png"

    if logo_path.exists():
        try:
            await update.message.reply_photo(
                photo=open(logo_path, "rb"),
                caption="Current brand logo. Send a new image to replace it.",
            )
        except Exception as e:
            logger.warning("Failed to send logo: %s", e)
            await update.message.reply_text("Logo file exists but couldn't be sent.")
    else:
        await update.message.reply_text("No logo set yet.")

    context.user_data["awaiting_logo_upload"] = True
    await update.message.reply_text(
        "Send me an image to set as the brand logo.",
    )


# ---------------------------------------------------------------------------
# /ingest — extract brand info from an image via Claude Vision
# ---------------------------------------------------------------------------

async def ingest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ingest — prepare to extract brand info from an uploaded image."""
    if not _authorized(update.effective_user.id):
        return

    context.user_data["awaiting_ingest_image"] = True
    await update.message.reply_text(
        "Send me a brand asset (logo, screenshot, marketing material) and I'll "
        "extract colors, fonts, and style keywords from it using AI vision.\n\n"
        "The extracted info will be compared against your current brand guidelines.",
    )


# ---------------------------------------------------------------------------
# /apply — apply extracted brand info to guidelines
# ---------------------------------------------------------------------------

async def apply_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /apply — merge last ingest extraction into guidelines.md."""
    if not _authorized(update.effective_user.id):
        return

    extracted = context.user_data.get("last_ingest_extracted")
    if not extracted:
        await update.message.reply_text(
            "No extracted data to apply. Use /ingest first and send a brand image.",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Merging extracted data into guidelines...")

    try:
        from agent import ingest
        import shutil

        guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"

        # Backup current guidelines
        if guidelines_path.exists():
            backup_path = guidelines_path.with_suffix(".md.bak")
            shutil.copy2(guidelines_path, backup_path)
            logger.info("Guidelines backed up to %s", backup_path)

        # Generate merged content
        new_content = await ingest.apply_extracted_to_guidelines(extracted)
        guidelines_path.write_text(new_content, encoding="utf-8")

        # Invalidate caches
        compositor_config.invalidate_cache()
        compositor.clear_font_cache()
        guidelines.invalidate_brand_context()

        # Clear stored extracted data
        context.user_data.pop("last_ingest_extracted", None)

        await update.message.reply_text(
            f"Guidelines updated ({len(new_content)} chars).\n"
            f"Backup saved to <code>guidelines.md.bak</code>\n"
            f"Config cache invalidated.",
            parse_mode="HTML",
        )
        logger.info("Guidelines updated from /apply (%d chars)", len(new_content))

    except Exception as e:
        logger.error("Apply command failed: %s", e)
        await update.message.reply_text(
            f"Failed to apply: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /brand_check — check an image against brand guidelines
# ---------------------------------------------------------------------------

async def brand_check_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /brand_check — check if an image matches brand guidelines.

    Supports three flows:
    1. Reply to an image message with /brand_check
    2. Send image with /brand_check as caption
    3. Send /brand_check alone, then upload an image
    """
    if not _authorized(update.effective_user.id):
        return

    # Flow 1: reply to an image
    reply = update.message.reply_to_message
    if reply and (reply.photo or (reply.document and (reply.document.mime_type or "").startswith("image/"))):
        tg_file = None
        if reply.photo:
            tg_file = await reply.photo[-1].get_file()
        elif reply.document:
            tg_file = await reply.document.get_file()

        if tg_file:
            await _run_brand_check(update, tg_file)
            return

    # Flow 2: image in same message (caption-based) — handled in handle_photo

    # Flow 3: no image attached, wait for next upload
    context.user_data["awaiting_brand_check"] = True
    await update.message.reply_text(
        "Send me an image and I'll check how well it matches your brand guidelines.\n\n"
        "I'll analyze colors, typography, visual style, brand elements, and layout.",
    )


async def _run_brand_check(update: Update, tg_file) -> None:
    """Download image, run brand compliance check, and send formatted report."""
    timestamp = int(time.time())
    tmp_path = str(Path(tempfile.gettempdir()) / f"brandmover_check_{timestamp}.jpg")

    try:
        await tg_file.download_to_drive(tmp_path)
    except Exception as e:
        logger.error("Failed to download image for brand check: %s", e)
        await update.message.reply_text(
            "couldn't download that image, try sending it as a photo instead of a file"
        )
        return

    # Convert to JPEG
    try:
        img = _PILImage.open(tmp_path).convert("RGB")
        img.save(tmp_path, "JPEG", quality=95)
    except Exception as e:
        logger.warning("Image conversion failed (using as-is): %s", e)

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Checking image against brand guidelines...")

    try:
        from agent import brand_check
        report = await brand_check.check_brand_compliance(tmp_path)
        formatted = brand_check.format_compliance_report(report)
        await update.message.reply_text(formatted, parse_mode="HTML")
    except Exception as e:
        logger.error("Brand check failed: %s", e)
        await update.message.reply_text(
            f"Brand check failed: {_esc(str(e))}", parse_mode="HTML"
        )


# ---------------------------------------------------------------------------
# /regen_guidelines — regenerate guidelines.md from asset inventory
# ---------------------------------------------------------------------------

async def regen_guidelines_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /regen_guidelines — regenerate guidelines.md from asset_inventory.json.

    If no asset_inventory.json exists, auto-scans brand/references/ and brand/assets/
    for images, runs audit_batch() to create the inventory, then regenerates.
    """
    if not _authorized(update.effective_user.id):
        return

    from agent import asset_audit, compositor_config
    from agent.strategy import StrategyRecommendation

    inv_path = Path(settings.BRAND_FOLDER) / "asset_inventory.json"

    # Auto-audit if no inventory exists but reference images are available
    if not inv_path.exists():
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
        scan_dirs = [
            Path(settings.BRAND_FOLDER) / "references",
            Path(settings.BRAND_FOLDER) / "assets",
        ]
        image_paths: list[str] = []
        for scan_dir in scan_dirs:
            if scan_dir.exists():
                for f in scan_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in image_exts:
                        image_paths.append(str(f))

        if not image_paths:
            await update.message.reply_text(
                "No asset inventory found and no images in brand/references/ or brand/assets/.\n"
                "Upload brand assets first via /upload or /onboard."
            )
            return

        await update.message.chat.send_action("typing")
        await update.message.reply_text(
            f"No asset inventory found. Auto-auditing {len(image_paths)} image(s) "
            f"from brand references..."
        )

        try:
            inventory = await asset_audit.audit_batch(image_paths)
            asset_audit.save_inventory(inventory)
        except Exception as e:
            logger.error("Auto-audit failed during regen_guidelines: %s", e)
            await update.message.reply_text(
                f"Auto-audit failed: {_esc(str(e))}", parse_mode="HTML"
            )
            return
    else:
        await update.message.chat.send_action("typing")
        inventory = asset_audit.load_inventory()
        if inventory is None:
            await update.message.reply_text(
                "Asset inventory file exists but couldn't be loaded. Try /upload to re-audit."
            )
            return

    await update.message.reply_text("Regenerating guidelines from your asset inventory...")

    try:
        # Build entries_creative from inventory entries
        entries_creative = []
        for entry in inventory.entries:
            ec: dict = {}
            if entry.first_impression:
                ec["first_impression"] = entry.first_impression
            if entry.creative_dna:
                ec["creative_dna"] = entry.creative_dna
            if entry.overall_energy:
                ec["overall_energy"] = entry.overall_energy
            if entry.what_makes_it_special:
                ec["what_makes_it_special"] = entry.what_makes_it_special
            if entry.never_do:
                ec["never_do"] = entry.never_do
            if entry.character_system:
                ec["character_system"] = entry.character_system
            if ec:
                entries_creative.append(ec)

        # Build a minimal session with inventory data
        session = onboarding.OnboardingSession(
            user_id=update.effective_user.id,
            brand_name=compositor_config.get_config().brand_name or "Brand",
            description=compositor_config.get_config().product_description or "",
            platforms=["x"],
            asset_audit={
                "archetype": inventory.archetype,
                "consolidated_colors": inventory.consolidated_colors,
                "consolidated_style": inventory.consolidated_style,
                "missing_items": inventory.missing_items,
                "entry_count": len(inventory.entries),
                "collection_analysis": inventory.collection_analysis,
                "brand_insights": inventory.brand_insights,
                "entries_creative": entries_creative,
            },
        )

        # Load existing strategy if available
        config_path = Path(settings.BRAND_FOLDER) / "config.json"
        rec_data = {}
        if config_path.exists():
            try:
                import json as _json
                rec_data = _json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        rec = StrategyRecommendation(
            archetype=rec_data.get("archetype", inventory.archetype),
            compositor_enabled=rec_data.get("compositor_enabled", False),
            badge_text=rec_data.get("badge_text"),
            default_mode=rec_data.get("default_mode", "image_optional"),
            recommended_content_types=rec_data.get("recommended_content_types", []),
            platforms=["x"],
        )

        # Load existing guidelines for merge (preserves voice/tone/positioning)
        guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
        existing_guidelines = ""
        if guidelines_path.exists():
            try:
                existing_guidelines = guidelines_path.read_text(encoding="utf-8")
            except OSError:
                pass

        guidelines_md = await onboarding.generate_guidelines_from_audit(
            session, rec, existing_guidelines=existing_guidelines,
        )
        guidelines_path.write_text(guidelines_md, encoding="utf-8")
        compositor_config.invalidate_cache()

        mode = "merged with" if existing_guidelines else "generated from"
        await update.message.reply_text(
            f"guidelines.md has been {mode} your asset inventory.\n"
            f"Voice/tone/positioning preserved, visuals updated from assets.\n\n"
            "Use /brand to review the updated config.",
        )
    except Exception as e:
        logger.error("Regen guidelines failed: %s", e)
        await update.message.reply_text(
            f"Failed to regenerate guidelines: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /train_lora — trigger LoRA training from approved images
# ---------------------------------------------------------------------------

async def train_lora_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /train_lora — trigger LoRA training on Replicate."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    stats = lora_pipeline.get_training_stats()
    total = stats["total_images"]
    threshold = stats["threshold"]

    if total < threshold:
        await update.message.reply_text(
            f"Not enough training images yet.\n\n"
            f"Images: <b>{total}</b> / {threshold} required\n"
            f"Keep approving drafts — each /approve adds to the training set.",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Starting LoRA training on Replicate...")

    try:
        result = await lora_pipeline.trigger_training(
            bot=context.bot,
            chat_id=update.effective_user.id,
        )
    except Exception as e:
        logger.error("train_lora failed: %s", e)
        await update.message.reply_text(
            f"Training failed: {_esc(str(e))}", parse_mode="HTML"
        )
        return

    if result.get("error"):
        await update.message.reply_text(
            f"Training error: {_esc(result['error'])}", parse_mode="HTML"
        )
        return

    await update.message.reply_text(
        f"LoRA training started!\n\n"
        f"Version: <b>{_esc(result.get('version', '?'))}</b>\n"
        f"Prediction ID: <code>{_esc(result.get('prediction_id', '?'))}</code>\n"
        f"Images: {result.get('image_count', '?')}\n"
        f"Trigger word: <code>{_esc(result.get('trigger_word', 'BRAND3D'))}</code>\n\n"
        f"Polling in background — I'll notify you when training completes "
        f"and auto-download the weights.\n"
        f"Use /lora_status to check progress.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /lora_status — show LoRA training status
# ---------------------------------------------------------------------------

async def lora_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_status — show LoRA training status and version history."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    stats = lora_pipeline.get_training_stats()
    total = stats["total_images"]
    threshold = stats["threshold"]
    versions = stats["versions"]
    lora_manifest = stats.get("lora_manifest", {})

    lora = lora_pipeline.get_active_lora()

    lines = ["<b>LoRA Training Status</b>\n"]

    if lora:
        lines.append(f"Active LoRA: <b>{_esc(lora.get('version', '?'))}</b>")
        lines.append(f"Trigger word: <code>{_esc(lora.get('trigger_word', 'BRAND3D'))}</code>")
        lines.append(f"Weights: <code>{_esc(lora.get('weights_path', 'N/A'))}</code>")
    else:
        lines.append("Active LoRA: <i>none</i>")

    lora_versions = lora_manifest.get("versions", [])
    lines.append(f"\nTotal versions: <b>{len(lora_versions)}</b>")
    lines.append(f"Training images: <b>{total}</b> / {threshold}")

    if lora_versions:
        lines.append(f"\n<b>Trained versions:</b>")
        lines.append(lora_pipeline.format_versions_list(lora_manifest))
    elif versions:
        lines.append("\n<b>Training history:</b>")
        for v in versions[-5:]:
            status_icon = {"completed": "\u2705", "training": "\u23F3", "failed": "\u274C"}.get(v.get("status", ""), "\u2753")
            lines.append(
                f"  {status_icon} {_esc(v.get('version', '?'))} — "
                f"{_esc(v.get('status', '?'))} "
                f"({v.get('image_count', '?')} images)"
            )

    lines.append("\nUse /lora_versions for details, /lora_switch N to change.")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /lora_versions — list all trained LoRA versions
# ---------------------------------------------------------------------------

async def lora_versions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_versions — list all trained LoRA versions."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    manifest = lora_pipeline.get_lora_manifest()
    versions = manifest.get("versions", [])

    if not versions:
        await update.message.reply_text(
            "No LoRA versions trained yet.\n"
            "Use /train_lora to start training.",
        )
        return

    formatted = lora_pipeline.format_versions_list(manifest)
    await update.message.reply_text(
        f"<b>LoRA Versions</b>\n\n{formatted}\n\n"
        f"Use /lora_switch <i>N</i> to switch, /lora_rollback to revert.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /lora_switch <version> — switch active LoRA version
# ---------------------------------------------------------------------------

async def lora_switch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_switch <N> — switch the active LoRA to version N."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    args = (update.message.text or "").split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        await update.message.reply_text(
            "Usage: /lora_switch <i>N</i>\n\n"
            "Example: <code>/lora_switch 2</code>",
            parse_mode="HTML",
        )
        return

    version_str = args[1].strip().lstrip("v")
    try:
        version_num = int(version_str)
    except ValueError:
        await update.message.reply_text(
            f"Invalid version number: {_esc(args[1].strip())}\n"
            f"Use /lora_versions to see available versions.",
            parse_mode="HTML",
        )
        return

    result = lora_pipeline.switch_active_version(version_num)

    if isinstance(result, str):
        # Error message
        await update.message.reply_text(
            f"Switch failed: {_esc(result)}",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            f"Switched active LoRA to <b>v{version_num}</b>\n\n"
            f"Trigger word: <code>{_esc(result.get('trigger_word', 'BRAND3D'))}</code>\n"
            f"Training images: {result.get('image_count', '?')}\n"
            f"Weights copied to brand3d.safetensors",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /lora_rollback — switch to previous LoRA version
# ---------------------------------------------------------------------------

async def lora_rollback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_rollback — roll back to previous LoRA version."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    result = lora_pipeline.rollback_version()

    if isinstance(result, str):
        await update.message.reply_text(
            f"Rollback failed: {_esc(result)}",
            parse_mode="HTML",
        )
    else:
        vn = result.get("version_number", "?")
        await update.message.reply_text(
            f"Rolled back to <b>v{vn}</b>\n\n"
            f"Trigger word: <code>{_esc(result.get('trigger_word', 'BRAND3D'))}</code>\n"
            f"Training images: {result.get('image_count', '?')}\n"
            f"Weights copied to brand3d.safetensors",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /history — generation history stats
# ---------------------------------------------------------------------------

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history — show generation stats and recent entries."""
    if not _authorized(update.effective_user.id):
        return

    stats = generation_history.get_generation_stats()
    recent = generation_history.get_recent_generations(5)

    lines = [f"<b>Generation History</b> ({stats['total']} total)\n"]

    if stats["by_status"]:
        status_parts = [f"{k}: {v}" for k, v in sorted(stats["by_status"].items())]
        lines.append(f"<b>By status:</b> {', '.join(status_parts)}")

    if stats["by_type"]:
        type_parts = [f"{k}: {v}" for k, v in sorted(stats["by_type"].items())]
        lines.append(f"<b>By type:</b> {', '.join(type_parts)}")

    if stats["by_model"]:
        model_parts = [f"{k}: {v}" for k, v in sorted(stats["by_model"].items())]
        lines.append(f"<b>By model:</b> {', '.join(model_parts)}")

    total_cost = stats.get("estimated_total_cost_usd", 0)
    if total_cost > 0:
        lines.append(f"<b>Est. total cost:</b> ${total_cost:.2f}")

    if recent:
        lines.append("\n<b>Recent:</b>")
        for e in recent:
            import datetime
            ts = datetime.datetime.fromtimestamp(e.get("timestamp", 0)).strftime("%m/%d %H:%M")
            status = e.get("status", "?")
            at = e.get("asset_type", e.get("content_type", "?"))
            req = e.get("original_request", "")[:50]
            lines.append(f"  [{status}] {ts} {at} — {_esc(req)}")

    if not stats["total"]:
        lines.append("No generations recorded yet.")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analytics — show approval rates by content type and model."""
    if not _authorized(update.effective_user.id):
        return

    data = generation_history.get_approval_analytics()
    lines = ["<b>Approval Rate Analytics</b>\n"]

    ct_data = data.get("by_content_type", {})
    if ct_data:
        lines.append("<b>By content type:</b>")
        for ct, stats in ct_data.items():
            total = stats["approved"] + stats["rejected"]
            lines.append(f"  {ct}: {stats['rate']:.0f}% ({stats['approved']}/{total})")
    else:
        lines.append("No reviewed drafts yet.")

    model_data = data.get("by_model", {})
    if model_data:
        lines.append("\n<b>By model:</b>")
        for model, stats in model_data.items():
            total = stats["approved"] + stats["rejected"]
            lines.append(f"  {model}: {stats['rate']:.0f}% ({stats['approved']}/{total})")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def library_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /library [query] — list or search the asset library."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    query = parts[1] if len(parts) > 1 else ""

    entries = asset_library.find(query=query, limit=10) if query else asset_library.list_all(limit=10)

    if not entries:
        await update.message.reply_text("Asset library is empty." if not query else f"No assets matching '{_esc(query)}'.")
        return

    lines = [f"<b>Asset Library</b> ({len(entries)} shown)\n"]
    for e in entries:
        used = f", used {e.used_count}x" if e.used_count else ""
        tags = f" [{', '.join(e.tags[:3])}]" if e.tags else ""
        prompt_short = (e.prompt[:40] + "...") if len(e.prompt) > 40 else e.prompt
        lines.append(f"<code>{e.id}</code> {e.source}/{e.content_type}{tags}{used}\n  {_esc(prompt_short)}")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def strategy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /strategy — show current brand strategy and config."""
    if not _authorized(update.effective_user.id):
        return

    config_path = Path(settings.BRAND_FOLDER) / "config.json"
    strategy_path = Path(settings.BRAND_FOLDER) / "strategy.md"

    if not config_path.exists() and not strategy_path.exists():
        await update.message.reply_text(
            "No strategy configured yet. Run /onboard to set up your brand."
        )
        return

    lines = ["<b>Brand Strategy</b>\n"]

    # Read config.json
    if config_path.exists():
        try:
            import json as _json
            cfg = _json.loads(config_path.read_text(encoding="utf-8"))
            pipeline = cfg.get("pipeline", {})
            lines.append(f"<b>Brand:</b> {_esc(cfg.get('brand_name', 'N/A'))}")
            lines.append(f"<b>Archetype:</b> {_esc(cfg.get('onboarding', {}).get('archetype', 'N/A'))}")
            lines.append(f"<b>Compositor:</b> {'ON' if pipeline.get('compositor_enabled') else 'OFF'}")
            badge = pipeline.get("badge_text")
            lines.append(f"<b>Badge:</b> {_esc(badge) if badge else '(none)'}")
            lines.append(f"<b>Mode:</b> {_esc(pipeline.get('default_mode', 'N/A'))}")
            platforms = cfg.get("platforms", [])
            if platforms:
                lines.append(f"<b>Platforms:</b> {', '.join(platforms)}")
            vs = cfg.get("visual_source", {})
            if vs:
                lines.append(f"<b>Visual source:</b> {_esc(vs.get('primary', 'N/A'))}")
            types = cfg.get("content_types_enabled", [])
            if types:
                lines.append(f"<b>Content types:</b> {', '.join(types[:8])}")
        except Exception as e:
            lines.append(f"<i>Error reading config.json: {_esc(str(e))}</i>")

    # Read strategy.md (show first ~500 chars)
    if strategy_path.exists():
        try:
            md = strategy_path.read_text(encoding="utf-8")
            preview = md[:500]
            if len(md) > 500:
                preview += "..."
            lines.append(f"\n<b>Strategy Notes:</b>\n<pre>{_esc(preview)}</pre>")
        except Exception as e:
            lines.append(f"<i>Error reading strategy.md: {_esc(str(e))}</i>")

    # Show calendar if exists
    cal_path = Path(settings.BRAND_FOLDER) / "content_calendar.md"
    if cal_path.exists():
        lines.append("\nContent calendar available — see <code>brand/content_calendar.md</code>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# State for /reset_brand confirmation
_reset_pending: dict[int, float] = {}


async def reset_brand_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset_brand — wipe brand config and start fresh."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    confirm_word = parts[1].strip() if len(parts) > 1 else ""

    user_id = update.effective_user.id

    if confirm_word.upper() == "RESET":
        brand_path = Path(settings.BRAND_FOLDER)

        # Backup guidelines.md
        gl = brand_path / "guidelines.md"
        if gl.exists():
            import shutil
            shutil.copy2(str(gl), str(gl) + ".bak")

        # Delete config files
        deleted = []
        for fname in ("config.json", "strategy.md", "content_calendar.md"):
            p = brand_path / fname
            if p.exists():
                p.unlink()
                deleted.append(fname)

        # Delete onboarding session
        onboarding.delete_session(user_id)

        # Invalidate caches
        compositor_config.invalidate_cache()

        summary = ", ".join(deleted) if deleted else "no config files found"
        await update.message.reply_text(
            f"Brand reset complete.\n"
            f"Deleted: {summary}\n"
            f"guidelines.md backed up to guidelines.md.bak\n\n"
            f"Run /onboard to set up again.",
        )
    else:
        await update.message.reply_text(
            "This will wipe your brand config and start fresh.\n\n"
            "Type <code>/reset_brand RESET</code> to confirm.",
            parse_mode="HTML",
        )


async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /upload — set flag to receive photos as brand assets."""
    if not _authorized(update.effective_user.id):
        return

    context.user_data["awaiting_asset_upload"] = True
    context.user_data["_asset_upload_count"] = 0
    await update.message.reply_text(
        "Send me images to add to your brand library.\n"
        "I'll index them automatically. Send /done when finished.",
    )


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /done — clear the asset upload flag."""
    if not _authorized(update.effective_user.id):
        return

    was_uploading = context.user_data.pop("awaiting_asset_upload", False)
    count = context.user_data.pop("_asset_upload_count", 0)

    if was_uploading:
        await update.message.reply_text(
            f"Asset upload complete. {count} image(s) added to your library.\n"
            f"Use /library to browse your assets.",
        )
    else:
        await update.message.reply_text("Nothing to finish.")


async def preview_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /preview [topic] — generate a sample post without rate limits or history."""
    if not _authorized(update.effective_user.id):
        return

    topic = " ".join(context.args) if context.args else ""
    if not topic:
        await update.message.reply_text(
            "Usage: /preview <topic>\n\n"
            "Example: /preview weekly product update"
        )
        return

    await update.message.chat.send_action("typing")

    try:
        brand_context = guidelines.get_brand_context()
        result = await brain.pipeline_generate(
            request=topic,
            brand_context=brand_context,
        )
        draft = result.draft

        if not draft.get("caption"):
            await update.message.reply_text("Preview generation failed — no caption produced.")
            return

        lines = [
            "<b>Preview</b>\n",
            f"{_esc(draft['caption'])}",
        ]
        hashtags = draft.get("hashtags", [])
        if hashtags:
            lines.append(f"\n{' '.join('#' + h for h in hashtags)}")
        if draft.get("content_type"):
            lines.append(f"\n<i>Type: {_esc(draft['content_type'])}</i>")
        if draft.get("image_prompt"):
            lines.append(f"<i>Image prompt: {_esc(draft['image_prompt'][:150])}</i>")

        lines.append("\n<i>This is a preview — not saved or tracked.</i>")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error("Preview error: %s", e)
        await update.message.reply_text(
            f"Preview failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /template_import <figma_url> [name] — Figma template import (Phase 5)
# ---------------------------------------------------------------------------

async def template_import_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_import <figma_url> [name] — import a template from Figma."""
    if not _authorized(update.effective_user.id):
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/template_import &lt;figma_url&gt; [name]</code>\n\n"
            "Example:\n<code>/template_import https://figma.com/design/abc123/MyFile?node-id=1-2 Hero Card</code>",
            parse_mode="HTML",
        )
        return

    figma_url = args[0]
    name = " ".join(args[1:]) if len(args) > 1 else ""

    from config import settings as _settings
    if not _settings.FIGMA_ACCESS_TOKEN:
        await update.message.reply_text(
            "Figma integration requires <code>FIGMA_ACCESS_TOKEN</code> in .env",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Importing template from Figma...")

    try:
        from agent import template_generator as _tg
        design, screenshot_img = await _tg.import_from_figma(figma_url, name or None)

        context.user_data["tplref_pending"] = {
            "design": _tg.design_to_dict(design),
            "name": name,
        }

        await _send_template_preview(update, context, design, screenshot_img)
    except Exception as e:
        logger.error("Figma template import failed: %s", e)
        await update.message.reply_text(
            f"Figma import failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /font_upload — upload custom TTF/OTF font
# ---------------------------------------------------------------------------

async def font_upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /font_upload — mark next file upload as a font file."""
    if not _authorized(update.effective_user.id):
        return

    context.user_data["awaiting_font_upload"] = True
    await update.message.reply_text(
        "Send me a TTF or OTF font file. It will be saved to the brand fonts directory "
        "and available for use in templates and compositions.",
    )
