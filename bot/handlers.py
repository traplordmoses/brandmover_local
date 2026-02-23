"""
Telegram command and message handlers.
All handlers are async. Only responds to the authorized user.
"""

import html
import logging
import time

from telegram import Update
from telegram.ext import ContextTypes

from agent import brain, engine, feedback, guidelines, image_gen, publisher, state
from config import settings

logger = logging.getLogger(__name__)


def _authorized(user_id: int) -> bool:
    """Check if a Telegram user is the authorized operator."""
    return user_id == settings.TELEGRAM_ALLOWED_USER_ID


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
        "/approve — Approve the pending draft\n"
        "/reject <i>reason</i> — Revise the draft with feedback\n"
        "/status — Show pending draft details\n"
        "/refs — Show loaded reference materials\n"
        "/feedback — Show approval/rejection stats\n"
        "/learn — Trigger preference learning from feedback history\n"
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
        f"<b>Hashtags:</b> {_esc(' '.join(pending['hashtags']))}\n\n"
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
    """Handle /approve — approve the pending draft and log feedback."""
    if not _authorized(update.effective_user.id):
        return

    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("Nothing to approve. Send me a content request first.")
        return

    await update.message.chat.send_action("typing")

    # Log feedback
    count = feedback.log_feedback(
        request=pending.get("original_request", ""),
        draft=pending,
        accepted=True,
        resources_used=pending.get("resources_used", []),
    )

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
            hashtags=draft["hashtags"],
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
        f"Hashtags: {' '.join(pending.get('hashtags', []))}\n"
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
            hashtags=result.draft["hashtags"],
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any plain text message as a content request — branches on AGENT_MODE."""
    if not _authorized(update.effective_user.id):
        return

    if not update.message or not update.message.text:
        return

    request = update.message.text.strip()
    if not request:
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

        # Save pending state
        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft["hashtags"],
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=request,
        )

        await _send_draft(update, result.draft, image_url, resources=result.resources)

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
            hashtags=draft["hashtags"],
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


async def _send_draft(update: Update, draft: dict, image_url: str | None, resources=None) -> None:
    """Send the generated draft to the user for review."""
    caption = draft["caption"]
    hashtags = " ".join(draft["hashtags"])

    text_msg = (
        f"<b>Draft Ready</b>\n\n"
        f"{_esc(caption)}\n\n"
        f"{_esc(hashtags)}\n\n"
        f"<i>Alt text:</i> {_esc(draft.get('alt_text', ''))}\n\n"
        f"/approve to post to X\n"
        f"/reject <i>feedback</i> to revise"
    )

    # Append resource summary if available
    if resources:
        text_msg += f"\n\n<i>Resources: {_esc(resources.to_summary())}</i>"

    if image_url:
        try:
            await update.message.reply_photo(
                photo=image_url,
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
