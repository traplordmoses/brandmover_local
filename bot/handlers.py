"""
Telegram command and message handlers.
All handlers are async. Only responds to the authorized user.
"""

import html
import logging
import time

from telegram import Update
from telegram.ext import ContextTypes

from agent import brain, guidelines, image_gen, publisher, state
from config import settings

logger = logging.getLogger(__name__)


def _authorized(user_id: int) -> bool:
    """Check if a Telegram user is the authorized operator."""
    return user_id == settings.TELEGRAM_ALLOWED_USER_ID


def _esc(text: str) -> str:
    """HTML-escape text for Telegram messages."""
    return html.escape(str(text))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show available commands."""
    if not _authorized(update.effective_user.id):
        return

    msg = (
        "<b>BrandMover Local</b>\n\n"
        "Send any message to generate a branded post draft.\n\n"
        "<b>Commands:</b>\n"
        "/approve — Post the pending draft to X\n"
        "/reject <i>reason</i> — Revise the draft with feedback\n"
        "/status — Show pending draft details\n"
        "/cancel — Clear pending draft\n"
        "/help — Show this message"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


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
    """Handle /approve — post the pending draft to X."""
    if not _authorized(update.effective_user.id):
        return

    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("Nothing to approve. Send me a content request first.")
        return

    await update.message.chat.send_action("typing")

    # X posting disabled for now
    # try:
    #     tweet_url = await publisher.post_to_x(
    #         caption=pending["caption"],
    #         hashtags=pending["hashtags"],
    #         image_url=pending.get("image_url"),
    #     )
    #     state.clear_pending()
    #     await update.message.reply_text(
    #         f"Posted to X!\n\n{tweet_url}",
    #         parse_mode="HTML",
    #     )
    #     logger.info("Published to X: %s", tweet_url)
    # except Exception as e:
    #     logger.error("Failed to post to X: %s", e)
    #     await update.message.reply_text(
    #         f"Failed to post to X: {_esc(str(e))}\n\n"
    #         "The draft is still pending. Fix the issue and try /approve again, or /cancel.",
    #         parse_mode="HTML",
    #     )

    state.clear_pending()
    await update.message.reply_text(
        "Approved! (X posting disabled for now — would have posted this draft)"
    )
    logger.info("Draft approved (X posting disabled)")


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
    feedback = text.partition("/reject")[2].strip()
    if not feedback:
        await update.message.reply_text(
            "Please include feedback: /reject <i>make it more urgent and add a CTA</i>",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")

    try:
        brand_context = guidelines.get_brand_context()
        draft = await brain.revise_draft(
            original_draft=pending,
            feedback=feedback,
            brand_context=brand_context,
        )

        # Generate new image if prompt changed
        image_url = pending.get("image_url")
        if draft.get("image_prompt") != pending.get("image_prompt"):
            await update.message.chat.send_action("upload_photo")
            image_url = await image_gen.generate_image(draft["image_prompt"])

        # Save revised draft
        state.save_pending(
            caption=draft["caption"],
            hashtags=draft["hashtags"],
            image_url=image_url,
            alt_text=draft["alt_text"],
            image_prompt=draft["image_prompt"],
            original_request=pending["original_request"],
        )

        # Send revised draft to user
        await _send_draft(update, draft, image_url)

    except Exception as e:
        logger.error("Revision failed: %s", e)
        await update.message.reply_text(
            f"Revision failed: {_esc(str(e))}\n\nOriginal draft still pending. Try again or /cancel.",
            parse_mode="HTML",
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any plain text message as a content request."""
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

    await update.message.chat.send_action("typing")

    try:
        # Step 1: Load brand context
        brand_context = guidelines.get_brand_context()
        logger.info("Brand context loaded: %d chars", len(brand_context))

        # Step 2: Generate draft via LLM
        draft = await brain.generate_draft(request, brand_context)
        logger.info("Draft generated: %s", draft.get("caption", "")[:80])

        # Step 3: Generate image
        image_url = None
        if draft.get("image_prompt"):
            await update.message.chat.send_action("upload_photo")
            image_url = await image_gen.generate_image(draft["image_prompt"])
            if not image_url:
                await update.message.reply_text(
                    "Image generation failed — sending text draft only.",
                )

        # Step 4: Save pending state
        state.save_pending(
            caption=draft["caption"],
            hashtags=draft["hashtags"],
            image_url=image_url,
            alt_text=draft["alt_text"],
            image_prompt=draft["image_prompt"],
            original_request=request,
        )

        # Step 5: Send draft to user
        await _send_draft(update, draft, image_url)

    except Exception as e:
        logger.error("Pipeline error: %s", e)
        await update.message.reply_text(
            f"Something went wrong: {_esc(str(e))}\n\nPlease try again.",
            parse_mode="HTML",
        )


async def _send_draft(update: Update, draft: dict, image_url: str | None) -> None:
    """Send the generated draft to the user for review."""
    caption = draft["caption"]
    hashtags = " ".join(draft["hashtags"])

    text_msg = (
        f"<b>Draft Ready</b>\n\n"
        f"{_esc(caption)}\n\n"
        f"{_esc(hashtags)}\n\n"
        f"<i>Alt text:</i> {_esc(draft['alt_text'])}\n\n"
        f"/approve to post to X\n"
        f"/reject <i>feedback</i> to revise"
    )

    if image_url:
        try:
            await update.message.reply_photo(
                photo=image_url,
                caption=text_msg[:1024],  # Telegram photo caption limit
                parse_mode="HTML",
            )
            # If caption was truncated, send the rest as text
            if len(text_msg) > 1024:
                await update.message.reply_text(text_msg, parse_mode="HTML")
        except Exception as e:
            logger.warning("Failed to send image via Telegram: %s — sending text only", e)
            await update.message.reply_text(text_msg, parse_mode="HTML")
    else:
        await update.message.reply_text(text_msg, parse_mode="HTML")
