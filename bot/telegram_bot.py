"""
Main Telegram bot entry point.
Builds the Application, registers handlers, and starts polling.
"""

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
    app.add_handler(CommandHandler("edit", handlers.edit_command))
    app.add_handler(CommandHandler("setup", handlers.setup_command))

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


def run() -> None:
    """Start the bot polling loop."""
    logger.info(
        "Starting BrandMover Local bot (user_id=%s, llm=%s)",
        settings.TELEGRAM_ALLOWED_USER_ID,
        settings.LLM_PROVIDER,
    )
    app = create_bot()
    app.run_polling(drop_pending_updates=True)
