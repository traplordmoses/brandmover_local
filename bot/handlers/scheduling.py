"""
Scheduling handlers — auto-post commands, schedule commands, send_auto_draft, campaigns.
"""

__all__ = [
    "send_auto_draft",
    "autostatus_command",
    "autopause_command",
    "autoforce_command",
    "schedule_command",
    "scheduled_command",
    "unschedule_command",
    "campaign_command",
    "campaign_preview_command",
    "campaign_schedule_command",
]

import asyncio as _aio
import io
import logging
import random as _random
import tempfile
import time
from pathlib import Path

from PIL import Image as _PILImage
from telegram import Update
from telegram.ext import ContextTypes

from agent import auto_state, campaigns, schedule_queue, scheduler, state
from config import settings

from bot.handlers.core import (
    _REVIEW_PROMPTS,
    _authorized,
    _esc,
    _maybe_compose,
    _prepare_photo,
)

logger = logging.getLogger(__name__)


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

    review = _random.choice(_REVIEW_PROMPTS)

    if image_url:
        photo, composed = await _maybe_compose(draft, image_url, content_type)

        # Save composed for archiving on approve
        if composed and isinstance(composed, io.BytesIO):
            try:
                tmp_fd_c = tempfile.NamedTemporaryFile(suffix=".png", prefix="brandmover_auto_composed_", delete=False)
                tmp_composed = tmp_fd_c.name
                tmp_fd_c.close()
                _data = composed.getvalue()
                await _aio.to_thread(Path(tmp_composed).write_bytes, _data)
                composed.seek(0)
                state.set_last_composed(tmp_composed, content_type)
            except Exception as e:
                logger.warning("Failed to save auto composed image: %s", e)

        photo = _prepare_photo(photo)
        if photo:
            photo_caption = f"[auto: {_esc(slot_name)}]\n{_esc(caption)}\n\n{review}"
            try:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=photo_caption[:1024],
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("Failed to send auto-draft image: %s \u2014 sending text", e)
                text_msg = f"[auto: {_esc(slot_name)}]\n{_esc(caption)}\n\n<i>(image unavailable)</i>"
                await bot.send_message(chat_id=chat_id, text=text_msg, parse_mode="HTML")
        else:
            text_msg = f"[auto: {_esc(slot_name)}]\n{_esc(caption)}\n\n<i>(image unavailable)</i>"
            await bot.send_message(chat_id=chat_id, text=text_msg, parse_mode="HTML")
    else:
        text_msg = f"[auto: {_esc(slot_name)}]\n{_esc(caption)}\n\n{review}"
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
            sched_lines.append(f"  \u23F0 {dt.strftime('%b %d %H:%M')} \u2014 {item.get('prompt', '')[:40]}")
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
    """Handle /schedule <time> <prompt> — schedule a post for a specific time."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
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
    if item is None:
        await update.message.reply_text(
            "This post is already scheduled around that time. "
            "Use /unschedule to cancel the existing one first.",
            parse_mode="HTML",
        )
        return
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
            f"{icon} <code>{item['id']}</code> \u2014 {_esc(time_str)}{_esc(rec_tag)}\n"
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
# /campaign — multi-day campaign management
# ---------------------------------------------------------------------------


async def campaign_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /campaign — list campaigns or show status of a specific one."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=2)

    # /campaign — list all
    if len(parts) <= 1:
        msg = campaigns.format_campaign_list()
        await update.message.reply_text(msg, parse_mode="HTML")
        return

    subcommand = parts[1].lower()

    if subcommand == "pause" and len(parts) > 2:
        name = parts[2].strip()
        if campaigns.pause_campaign(name):
            await update.message.reply_text(
                f"Campaign <b>{_esc(name)}</b> paused.", parse_mode="HTML",
            )
        else:
            await update.message.reply_text(f"Campaign '{_esc(name)}' not found or not active.", parse_mode="HTML")
        return

    if subcommand == "resume" and len(parts) > 2:
        name = parts[2].strip()
        if campaigns.resume_campaign(name):
            await update.message.reply_text(
                f"Campaign <b>{_esc(name)}</b> resumed.", parse_mode="HTML",
            )
        else:
            await update.message.reply_text(f"Campaign '{_esc(name)}' not found or not paused.", parse_mode="HTML")
        return

    if subcommand == "delete" and len(parts) > 2:
        name = parts[2].strip()
        if campaigns.delete_campaign(name):
            await update.message.reply_text(
                f"Campaign <b>{_esc(name)}</b> deleted. Scheduled posts cancelled.",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text(f"Campaign '{_esc(name)}' not found.", parse_mode="HTML")
        return

    # /campaign <name> — show status
    name = " ".join(parts[1:]).strip()
    campaign = campaigns.get_campaign(name)
    if campaign:
        msg = campaigns.format_campaign_status(name)
        await update.message.reply_text(msg, parse_mode="HTML")
    else:
        msg = campaigns.format_campaign_list()
        msg += f"\n\nCampaign '{_esc(name)}' not found."
        await update.message.reply_text(msg, parse_mode="HTML")


async def campaign_preview_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /campaign_preview [name] — generate an HTML preview of a campaign."""
    if not _authorized(update.effective_user.id):
        return

    from agent import campaign_preview

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)

    if len(parts) > 1:
        name = parts[1].strip()
    else:
        active = campaigns.list_campaigns(status_filter="active")
        if not active:
            all_c = campaigns.list_campaigns()
            if all_c:
                name = all_c[0].get("name", "")
            else:
                await update.message.reply_text("No campaigns found.")
                return
        else:
            name = active[0].get("name", "")

    campaign = campaigns.get_campaign(name)
    if not campaign:
        await update.message.reply_text(f"Campaign '{_esc(name)}' not found.", parse_mode="HTML")
        return

    await update.message.chat.send_action("typing")

    path = campaign_preview.generate_preview_html(name)
    if not path:
        await update.message.reply_text("Failed to generate preview.")
        return

    _html_data = await _aio.to_thread(Path(path).read_bytes)
    await update.message.reply_document(
        document=io.BytesIO(_html_data),
        filename=f"{name}_preview.html",
        caption=f"Campaign preview for <b>{_esc(name)}</b> \u2014 open in your browser",
        parse_mode="HTML",
    )


async def campaign_schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /campaign_schedule <name> — schedule all pending posts for a campaign."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        await update.message.reply_text(
            "Usage: /campaign_schedule <i>campaign_name</i>\n\n"
            "Schedules all pending campaign posts into the post queue.",
            parse_mode="HTML",
        )
        return

    name = parts[1].strip()
    campaign = campaigns.get_campaign(name)
    if not campaign:
        await update.message.reply_text(f"Campaign '{_esc(name)}' not found.", parse_mode="HTML")
        return

    result = campaigns.schedule_campaign_posts(name)
    lines = [f"<b>Campaign '{_esc(name)}' scheduling:</b>", ""]
    lines.append(f"Scheduled: {result['scheduled']} posts")
    if result["skipped"]:
        lines.append(f"Skipped: {result['skipped']}")
    for err in result.get("errors", []):
        lines.append(f"  \u26A0 {_esc(err)}")

    if result["scheduled"] > 0:
        lines.append("")
        lines.append("Posts will fire at their scheduled times.")
        lines.append("Use /scheduled to see the queue.")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")
