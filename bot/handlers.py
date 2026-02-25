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
from telegram import Update
from telegram.ext import ContextTypes

from agent import brain, compositor, engine, feedback, guidelines, image_gen, publisher, state
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
        "/setup — Bootstrap guidelines from a PDF upload\n"
        "/cancel — Clear pending draft\n"
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
    msg = (
        f"<b>Pending Draft</b>\n\n"
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
    count = feedback.log_feedback(
        request=pending.get("original_request", ""),
        draft=pending,
        accepted=True,
        resources_used=pending.get("resources_used", []),
    )

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

    # Save approved Finny outputs to grow character reference library
    _finny_kw = re.compile(r"finny|mascot", re.IGNORECASE)
    _is_finny_draft = (
        _finny_kw.search(pending.get("original_request", ""))
        or _finny_kw.search(pending.get("image_prompt", ""))
    )
    if _is_finny_draft and pending.get("image_url"):
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=20, follow_redirects=True) as _c:
                _r = await _c.get(pending["image_url"])
                _r.raise_for_status()
                ts = int(time.time())
                save_path = Path(settings.BRAND_FOLDER) / "assets" / f"finny_approved_{ts}.png"
                _PILImage.open(io.BytesIO(_r.content)).convert("RGB").save(str(save_path), "PNG")
                logger.info("Saved approved Finny output: %s", save_path)
        except Exception as e:
            logger.warning("Failed to save Finny output: %s", e)

    state.clear_pending()
    await update.message.reply_text(
        f"Approved! (X posting disabled for now — would have posted this draft)\n"
        f"Feedback logged ({count} total entries)."
    )
    logger.info("Draft approved (feedback #%d)", count)

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

        # Build edit prompt with BloFin brand constraints
        edit_prompt = (
            f"Edit this image: {feedback_text}. "
            f"Keep everything else identical. Maintain BloFin brand style: "
            f"matte black materials, amber orange accents, pure #000000 background, "
            f"no gradients, no glow."
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
    count = feedback.log_feedback(
        request=pending.get("original_request", ""),
        draft=pending,
        accepted=False,
        feedback_text=feedback_text,
        resources_used=pending.get("resources_used", []),
    )
    logger.info("Draft rejected (feedback #%d): %s", count, feedback_text[:100])

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

        # Save revised draft
        state.save_pending(
            caption=draft["caption"],
            hashtags=draft.get("hashtags", []),
            image_url=image_url,
            alt_text=draft["alt_text"],
            image_prompt=draft["image_prompt"],
            original_request=pending["original_request"],
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

        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft.get("hashtags", []),
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=pending["original_request"],
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

    state.set_reference_image(tmp_path)
    logger.info("Reference image saved to state: %s", tmp_path)

    caption = (update.message.caption or "").strip()

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
            "reference / finny / style <name> / background"
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
