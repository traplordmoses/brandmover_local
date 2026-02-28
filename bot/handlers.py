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

from agent import asset_gen, auto_state, brain, compositor, compositor_config, engine, feedback, generation_history, guidelines, image_gen, publisher, scheduler, state
from config import settings

logger = logging.getLogger(__name__)

# Rate limiting — minimum seconds between generation requests per user
_RATE_LIMIT_SECONDS = 10
_last_request_time: dict[int, float] = {}


def _authorized(user_id: int) -> bool:
    """Check if a Telegram user is the authorized operator."""
    return user_id == settings.TELEGRAM_ALLOWED_USER_ID


def _rate_limited(user_id: int) -> bool:
    """Check if user is sending requests too fast. Returns True if blocked."""
    now = time.time()
    last = _last_request_time.get(user_id, 0)
    if now - last < _RATE_LIMIT_SECONDS:
        return True
    _last_request_time[user_id] = now
    return False


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

    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("Nothing to approve. Send me a content request first.")
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

    # If multiple image options exist, select the chosen one
    image_urls = pending.get("image_urls", [])
    if image_urls and 1 <= option_num <= len(image_urls):
        pending["image_url"] = image_urls[option_num - 1]
        logger.info("Approve: selected option %d of %d", option_num, len(image_urls))
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
            await generation_history.async_update_generation_status(ts,"approved")
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
                count = state.add_profile_image(active_profile, composed_path)
                logger.info(
                    "Saved approved image to profile %s (%d images)",
                    active_profile, count,
                )
        except Exception as e:
            logger.warning("Failed to save to style profile: %s", e)

        # Clean up temp composed file
        try:
            Path(composed_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("Composed cleanup failed for %s: %s", composed_path, e)
        state.clear_last_composed()

    # Save approved mascot outputs to grow character reference library
    _mascot_kw = re.compile(r"mascot|character", re.IGNORECASE)
    _is_mascot_draft = (
        _mascot_kw.search(pending.get("original_request", ""))
        or _mascot_kw.search(pending.get("image_prompt", ""))
    )
    if _is_mascot_draft and pending.get("image_url"):
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=20, follow_redirects=True) as _c:
                _r = await _c.get(pending["image_url"])
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

    # Post to X
    tweet_url = None
    try:
        await update.message.chat.send_action("typing")
        tweet_url = await publisher.post_to_x(
            pending.get("caption", ""),
            pending.get("hashtags", []),
            pending.get("image_url"),
        )
    except Exception as e:
        logger.error("Failed to post to X: %s", e)
        await update.message.reply_text(
            f"Approved, but X posting failed: {_esc(str(e))}\n"
            f"Feedback logged ({count} total entries).",
            parse_mode="HTML",
        )
        state.clear_pending()
        state.clear_draft_history()
        return

    # If this draft came from the auto-post scheduler, record it so the slot
    # is marked as fulfilled for today and duplicate detection works.
    auto_slot = pending.get("auto_slot")
    if auto_slot:
        auto_state.record_post(
            slot_name=auto_slot,
            caption=pending.get("caption", ""),
            tweet_url=tweet_url,
            event_ids=pending.get("auto_event_ids"),
        )
        logger.info("Auto-post slot '%s' recorded via /approve", auto_slot)

    state.clear_pending()
    state.clear_draft_history()
    slot_note = f"  (auto-slot: {_esc(auto_slot)})" if auto_slot else ""
    await update.message.reply_text(
        f"Posted to X!{slot_note}\n"
        f"{_esc(tweet_url)}\n\n"
        f"Feedback logged ({count} total entries).",
        parse_mode="HTML",
    )
    logger.info("Draft approved and posted: %s (feedback #%d)", tweet_url, count)

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

        composed = await compositor.compose_branded_image(draft, url, ct)
        photo = composed if composed else url

        # Save composed for archiving
        if composed and isinstance(composed, io.BytesIO):
            try:
                tmp_composed = str(Path(tempfile.gettempdir()) / f"brandmover_edit_composed_{ts}.png")
                with open(tmp_composed, "wb") as f:
                    f.write(composed.getvalue())
                composed.seek(0)
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

    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("Nothing to reject. Send me a content request first.")
        return

    # Extract feedback after "/reject"
    text = update.message.text or ""
    feedback_text = text.partition("/reject")[2].strip()
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
    logger.info("Draft rejected (feedback #%d): %s", count, feedback_text[:100])

    # Update generation history status
    try:
        ts = pending.get("timestamp", 0)
        if ts:
            await generation_history.async_update_generation_status(ts,"rejected")
    except Exception as e:
        logger.debug("Generation history update failed: %s", e)

    # Auto-summarize at threshold
    if count % settings.FEEDBACK_SUMMARIZE_EVERY == 0:
        try:
            await feedback.summarize_preferences()
            logger.info("Auto-summarized preferences after %d entries", count)
        except Exception as e:
            logger.error("Auto-summarize failed: %s", e)

    # Clear the old pending before running revision — prevents any stray message
    # handlers from triggering a second generation during the revision
    state.clear_pending()

    # Branch: agent mode re-runs with revision context, pipeline mode uses revise_draft
    if settings.AGENT_MODE == "agent":
        await _handle_agent_revision(update, pending, feedback_text)
    else:
        await _handle_pipeline_revision(update, pending, feedback_text)


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
            image_url = await image_gen.generate_image(draft["image_prompt"], content_type=content_type)

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

    if not context.user_data.get("awaiting_setup_pdf"):
        await update.message.reply_text(
            "Use /setup first if you want to bootstrap brand guidelines from a PDF.",
        )
        return

    document = update.message.document
    if not document:
        return

    # Check file type
    file_name = document.file_name or ""
    if not file_name.lower().endswith(".pdf"):
        await update.message.reply_text("Please send a PDF file (.pdf).")
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Extracting text from PDF and generating guidelines...")

    try:
        # Download the PDF
        tg_file = await document.get_file()
        import tempfile
        from pathlib import Path

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

    timestamp = int(time.time())
    tmp_path = str(Path(tempfile.gettempdir()) / f"brandmover_upload_{timestamp}.jpg")

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

    # --- Priority flag checks (logo > ingest > brand_check) ---
    user_data = context.user_data if context else {}

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
        await update.message.reply_text(
            "got it. what should i do with this? reply with:\n"
            "reference / mascot / style <name> / background"
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any plain text message as a content request — branches on AGENT_MODE."""
    if not _authorized(update.effective_user.id):
        return

    if not update.message or not update.message.text:
        return

    request = update.message.text.strip()
    if not request:
        return

    # Rate limit — prevent accidental double-sends from burning API credits
    if _rate_limited(update.effective_user.id):
        await update.message.reply_text(
            f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
        )
        return

    # Block if there's already a pending draft
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


async def _handle_agent_mode(update: Update, request: str) -> None:
    """Run the agent tool-use loop for a content request."""
    await update.message.chat.send_action("typing")

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
            await update.message.reply_text(
                f"Agent finished but couldn't produce a structured draft.\n\n"
                f"<pre>{_esc(result.final_text[:1000])}</pre>\n\n"
                f"({result.turns_used} turns, {result.total_time}s, tools: {', '.join(result.tool_calls_made)})",
                parse_mode="HTML",
            )
            return

        image_url = result.image_url
        image_urls = result.image_urls

        # Save pending state
        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft.get("hashtags", []),
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=request,
            image_urls=image_urls if len(image_urls) > 1 else None,
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

        # Generate image with smart model routing
        image_url = None
        if draft.get("image_prompt"):
            await update.message.chat.send_action("upload_photo")
            content_type = draft.get("content_type", "announcement")
            image_url = await image_gen.generate_image(draft["image_prompt"], content_type=content_type)
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
        template_section = (
            f"\n<b>--- Image Template ---</b>\n"
            f"<b>Title:</b> {_esc(draft.get('title', ''))}\n"
            f"<b>Subtitle:</b> {_esc(draft.get('subtitle', ''))}\n"
            f"<b>Platform:</b> {_esc(draft.get('platform', 'WEB'))}\n"
        )

    # --- Multi-option path (N>1 images) ---
    if image_urls and len(image_urls) > 1:
        for idx, url in enumerate(image_urls, 1):
            composed = await compositor.compose_branded_image(draft, url, content_type)
            photo = composed if composed else url

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

    if image_url:
        composed = await compositor.compose_branded_image(draft, image_url, content_type)
        photo = composed if composed else image_url

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
            )
            if len(text_msg) > 1024:
                await update.message.reply_text(text_msg, parse_mode="HTML")
        except Exception as e:
            logger.warning("Failed to send image via Telegram: %s — sending text only", e)
            await update.message.reply_text(text_msg, parse_mode="HTML")
    else:
        await update.message.reply_text(text_msg, parse_mode="HTML")


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
        composed = await compositor.compose_branded_image(draft, image_url, content_type)
        photo = composed if composed else image_url

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

    msg = (
        f"<b>Auto-Post Status: {paused_str}</b>\n\n"
        f"<b>Enabled:</b> {settings.AUTO_POST_ENABLED}\n"
        f"<b>Dry run:</b> {settings.AUTO_POST_DRY_RUN}\n"
        f"<b>Posts today:</b> {status['posts_today']}/{global_cfg.get('max_posts_per_day', 6)}\n"
        f"<b>Last post:</b> {last_str}\n"
        f"<b>Min gap:</b> {global_cfg.get('min_gap_minutes', 120)} min\n\n"
        f"<b>Slots:</b>\n" + "\n".join(slot_lines) + "\n\n"
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
