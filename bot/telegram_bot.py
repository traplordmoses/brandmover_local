"""
Main Telegram bot entry point.
Builds the Application, registers handlers, starts polling,
and launches the auto-post scheduler as a background task.
"""

import asyncio
import logging

from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from bot import handlers
from config import settings

logger = logging.getLogger(__name__)


def create_bot() -> Application:
    """Build and configure the Telegram bot Application."""
    settings.validate(exit_on_error=True)

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
    app.add_handler(CommandHandler("generate", handlers.generate_command))
    app.add_handler(CommandHandler("logo", handlers.logo_command))
    app.add_handler(CommandHandler("ingest", handlers.ingest_command))
    app.add_handler(CommandHandler("apply", handlers.apply_command))
    app.add_handler(CommandHandler("brand_check", handlers.brand_check_command))
    app.add_handler(CommandHandler("train_lora", handlers.train_lora_command))
    app.add_handler(CommandHandler("lora_status", handlers.lora_status_command))
    app.add_handler(CommandHandler("lora_versions", handlers.lora_versions_command))
    app.add_handler(CommandHandler("lora_switch", handlers.lora_switch_command))
    app.add_handler(CommandHandler("lora_rollback", handlers.lora_rollback_command))
    app.add_handler(CommandHandler("history", handlers.history_command))
    app.add_handler(CommandHandler("analytics", handlers.analytics_command))
    app.add_handler(CommandHandler("template", handlers.template_command))
    app.add_handler(CommandHandler("template_upload", handlers.template_upload_command))
    app.add_handler(CommandHandler("template_from_reference", handlers.template_from_reference_command))
    app.add_handler(CommandHandler("onboard", handlers.onboard_command))
    app.add_handler(CommandHandler("onboard_cancel", handlers.onboard_cancel_command))
    app.add_handler(CommandHandler("onboard_skip", handlers.onboard_skip_command))
    app.add_handler(CommandHandler("library", handlers.library_command))
    app.add_handler(CommandHandler("strategy", handlers.strategy_command))
    app.add_handler(CommandHandler("reset_brand", handlers.reset_brand_command))
    app.add_handler(CommandHandler("regen_guidelines", handlers.regen_guidelines_command))
    app.add_handler(CommandHandler("upload", handlers.upload_command))
    app.add_handler(CommandHandler("done", handlers.done_command))
    app.add_handler(CommandHandler("preview", handlers.preview_command))

    # Inline button callbacks (e.g. /generate approve/reject buttons)
    app.add_handler(CallbackQueryHandler(handlers.generate_callback, pattern=r"^gen_"))

    # Draft inline buttons (Approve/Reject/Edit/Reroll)
    app.add_handler(CallbackQueryHandler(handlers.draft_callback, pattern=r"^draft_"))

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
