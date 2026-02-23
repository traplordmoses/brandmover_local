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
    app.add_handler(CommandHandler("cancel", handlers.cancel_command))
    app.add_handler(CommandHandler("approve", handlers.approve_command))
    app.add_handler(CommandHandler("reject", handlers.reject_command))

    # Plain text messages → content request pipeline
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
