"""
Main Telegram bot entry point.
Builds the Application, registers handlers, starts polling,
and launches the auto-post scheduler as a background task.
"""

import asyncio
import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from bot import handlers
from config import settings

logger = logging.getLogger(__name__)


def create_bot() -> Application:
    """Build and configure the Telegram bot Application."""
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")

    if not settings.TELEGRAM_ALLOWED_USER_ID:
        raise RuntimeError("TELEGRAM_ALLOWED_USER_ID is not set in .env")

    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("help", handlers.help_command))
    app.add_handler(CommandHandler("start", handlers.help_command))
    app.add_handler(CommandHandler("status", handlers.status_command))
    app.add_handler(CommandHandler("refs", handlers.refs_command))
    app.add_handler(CommandHandler("cancel", handlers.cancel_command))
    app.add_handler(CommandHandler("approve", handlers.approve_command))
    app.add_handler(CommandHandler("reject", handlers.reject_command))
    app.add_handler(CommandHandler("feedback", handlers.feedback_command))
    app.add_handler(CommandHandler("learn", handlers.learn_command))
    app.add_handler(CommandHandler("style", handlers.style_command))
    app.add_handler(CommandHandler("brand", handlers.brand_command))
    app.add_handler(CommandHandler("edit", handlers.edit_command))
    app.add_handler(CommandHandler("setup", handlers.setup_command))
    app.add_handler(CommandHandler("autostatus", handlers.autostatus_command))
    app.add_handler(CommandHandler("autopause", handlers.autopause_command))
    app.add_handler(CommandHandler("autoforce", handlers.autoforce_command))

    # Photo uploads (reference images)
    app.add_handler(
        MessageHandler(filters.PHOTO, handlers.handle_photo)
    )

    # Image documents (user sends image as file, not compressed)
    app.add_handler(
        MessageHandler(filters.Document.IMAGE, handlers.handle_photo)
    )

    # Non-image document uploads (PDF brand bootstrap)
    app.add_handler(
        MessageHandler(
            filters.Document.ALL & ~filters.Document.IMAGE,
            handlers.handle_document,
        )
    )

    # Plain text messages → content request (pipeline or agent mode)
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message)
    )

    logger.info("Bot configured with %d handlers", len(app.handlers[0]))
    return app


async def _start_scheduler(app: Application) -> None:
    """Post-init hook: launch the auto-post scheduler as a background task."""
    from scripts.auto_post import run_scheduler_loop

    bot = app.bot
    task = asyncio.create_task(run_scheduler_loop(bot=bot))
    # Store reference so it doesn't get GC'd
    app.bot_data["_scheduler_task"] = task
    logger.info("Auto-post scheduler background task launched")


def run() -> None:
    """Start the bot polling loop with the auto-post scheduler."""
    logger.info(
        "Starting BrandMover Local bot (user_id=%s, llm=%s, auto_post=%s)",
        settings.TELEGRAM_ALLOWED_USER_ID,
        settings.LLM_PROVIDER,
        settings.AUTO_POST_ENABLED,
    )
    app = create_bot()

    # Register the scheduler as a post-init hook so it starts after the
    # bot's event loop and updater are running.
    app.post_init = _start_scheduler

    app.run_polling(drop_pending_updates=True)
