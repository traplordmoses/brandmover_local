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

    app = (
        Application.builder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(15)
        .build()
    )

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
    app.add_handler(CommandHandler("review", handlers.review_command))
    app.add_handler(CommandHandler("style", handlers.style_command))
    app.add_handler(CommandHandler("brand", handlers.brand_command))
    app.add_handler(CommandHandler("refine", handlers.refine_command))
    app.add_handler(CommandHandler("edit", handlers.edit_command))
    app.add_handler(CommandHandler("setup", handlers.setup_command))
    app.add_handler(CommandHandler("schedule", handlers.schedule_command))
    app.add_handler(CommandHandler("scheduled", handlers.scheduled_command))
    app.add_handler(CommandHandler("unschedule", handlers.unschedule_command))
    app.add_handler(CommandHandler("autostatus", handlers.autostatus_command))
    app.add_handler(CommandHandler("autopause", handlers.autopause_command))
    app.add_handler(CommandHandler("autoforce", handlers.autoforce_command))
    app.add_handler(CommandHandler("generate", handlers.generate_command))
    app.add_handler(CommandHandler("logo", handlers.logo_command))
    app.add_handler(CommandHandler("ingest", handlers.ingest_command))
    app.add_handler(CommandHandler("apply", handlers.apply_command))
    app.add_handler(CommandHandler("brand_edit", handlers.brand_edit_command))
    app.add_handler(CommandHandler("confirm_edit", handlers.confirm_edit_command))
    app.add_handler(CommandHandler("cancel_edit", handlers.cancel_edit_command))
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
    app.add_handler(CommandHandler("template_test", handlers.template_test_command))
    app.add_handler(CommandHandler("template_from_reference", handlers.template_from_reference_command))
    app.add_handler(CommandHandler("onboard", handlers.onboard_command))
    app.add_handler(CommandHandler("onboard_cancel", handlers.onboard_cancel_command))
    app.add_handler(CommandHandler("onboard_skip", handlers.onboard_skip_command))
    app.add_handler(CommandHandler("library", handlers.library_command))
    app.add_handler(CommandHandler("skills", handlers.skills_command))
    app.add_handler(CommandHandler("strategy", handlers.strategy_command))
    app.add_handler(CommandHandler("reset_brand", handlers.reset_brand_command))
    app.add_handler(CommandHandler("regen_guidelines", handlers.regen_guidelines_command))
    app.add_handler(CommandHandler("upload", handlers.upload_command))
    app.add_handler(CommandHandler("done", handlers.done_command))
    app.add_handler(CommandHandler("preview", handlers.preview_command))
    app.add_handler(CommandHandler("template_import", handlers.template_import_command))
    app.add_handler(CommandHandler("font_upload", handlers.font_upload_command))
    app.add_handler(CommandHandler("discord_setup", handlers.discord_setup_command))
    app.add_handler(CommandHandler("platforms", handlers.platforms_command))
    app.add_handler(CommandHandler("pref", handlers.pref_command))
    app.add_handler(CommandHandler("unpref", handlers.unpref_command))
    app.add_handler(CommandHandler("preferences", handlers.preferences_command))
    app.add_handler(CommandHandler("topics", handlers.topics_command))
    app.add_handler(CommandHandler("heartbeat", handlers.heartbeat_command))
    app.add_handler(CommandHandler("campaign", handlers.campaign_command))
    app.add_handler(CommandHandler("campaign_schedule", handlers.campaign_schedule_command))
    app.add_handler(CommandHandler("campaign_preview", handlers.campaign_preview_command))
    app.add_handler(CommandHandler("score", handlers.score_command))
    app.add_handler(CommandHandler("approval_rate", handlers.approval_rate_command))
    app.add_handler(CommandHandler("health", handlers.health_command))
    app.add_handler(CommandHandler("digest", handlers.digest_command))
    app.add_handler(CommandHandler("save_asset", handlers.save_asset_command))
    app.add_handler(CommandHandler("remake", handlers.remake_command))
    app.add_handler(CommandHandler("code", handlers.code_command))

    # Inline button callbacks (e.g. /generate approve/reject buttons)
    app.add_handler(CallbackQueryHandler(handlers.generate_callback, pattern=r"^gen_"))

    # Draft inline buttons (Approve/Reject/Edit/Reroll)
    app.add_handler(CallbackQueryHandler(handlers.draft_callback, pattern=r"^draft_"))

    # Template-from-reference inline buttons (Save/Adjust/Discard)
    app.add_handler(CallbackQueryHandler(handlers.tplref_callback, pattern=r"^tplref_"))

    # Claude Code inline buttons (Reload/Diff/Revert)
    app.add_handler(CallbackQueryHandler(handlers.code_callback, pattern=r"^code_"))

    # Voice and audio messages (transcribe → process as text)
    app.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, handlers.handle_voice)
    )

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

    # Channel/group message logger (lowest priority — logs silently, doesn't block)
    _monitor_ids = settings.TELEGRAM_MONITOR_CHANNELS
    if _monitor_ids:
        try:
            _chat_ids = [int(x.strip()) for x in _monitor_ids.split(",") if x.strip()]
            if _chat_ids:
                from agent.unified_tools import log_channel_message

                async def _log_channel_msg(update, context):
                    msg = update.effective_message
                    if not msg or not msg.text:
                        return
                    chat_id = msg.chat_id
                    author = ""
                    if msg.from_user:
                        author = msg.from_user.first_name or msg.from_user.username or str(msg.from_user.id)
                    elif msg.sender_chat:
                        author = msg.sender_chat.title or str(msg.sender_chat.id)
                    import time as _time
                    timestamp = msg.date.timestamp() if msg.date else _time.time()
                    log_channel_message(chat_id, author, msg.text, timestamp)

                app.add_handler(
                    MessageHandler(
                        filters.Chat(chat_id=_chat_ids) & filters.TEXT & ~filters.COMMAND,
                        _log_channel_msg,
                    ),
                    group=1,  # separate handler group so it doesn't conflict
                )
                logger.info("Channel message logger active for: %s", _chat_ids)
        except ValueError:
            logger.warning("Invalid TELEGRAM_MONITOR_CHANNELS: %s", _monitor_ids)

    logger.info("Bot configured with %d handlers", len(app.handlers[0]))
    return app


async def _start_scheduler(app: Application) -> None:
    """Post-init hook: launch the auto-post scheduler and Discord client as background tasks."""
    from scripts.auto_post import run_scheduler_loop

    # Register the heartbeat notifier so agent/ never imports from bot/
    from agent.heartbeat import set_notifier
    from bot.handlers import send_auto_draft
    set_notifier(send_auto_draft)

    bot = app.bot
    task = asyncio.create_task(run_scheduler_loop(bot=bot))
    # Store reference so it doesn't get GC'd
    app.bot_data["_scheduler_task"] = task
    logger.info("Auto-post scheduler background task launched")

    # Start Discord client if configured
    if settings.DISCORD_BOT_TOKEN:
        from agent import discord_bot
        discord_task = asyncio.create_task(discord_bot.start_client())
        app.bot_data["_discord_task"] = discord_task
        logger.info("Discord client background task launched")


async def _shutdown_cleanup(app: Application) -> None:
    """Post-shutdown hook: close shared API clients."""
    from agent._client import close as close_clients
    try:
        await close_clients()
        logger.info("Shared API clients closed")
    except Exception as e:
        logger.debug("Client close failed: %s", e)


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
    app.post_shutdown = _shutdown_cleanup

    app.run_polling(drop_pending_updates=True)
