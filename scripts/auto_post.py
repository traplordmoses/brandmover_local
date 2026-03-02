#!/usr/bin/env python3
"""
Automated X posting entry point for BrandMover.

Generates content via run_agent(), saves as a pending draft, and sends to
Telegram for human review.  The actual X post happens when the operator
sends /approve in Telegram.

Can run as:
  - In-process background task inside the Telegram bot (preferred — see
    telegram_bot.py which calls run_scheduler_loop)
  - Standalone cron/daemon for --dry-run and --force testing

Usage:
    # Cron mode (default): check schedule, generate due drafts, exit
    python scripts/auto_post.py

    # Daemon mode: long-running, checks every 5 minutes
    python scripts/auto_post.py --daemon

    # Dry run: generate content, log it, don't save or send
    python scripts/auto_post.py --dry-run

    # Force a specific slot (bypasses time window check)
    python scripts/auto_post.py --force engagement_morning

    # Combine: force + dry run
    python scripts/auto_post.py --force onchain_midday --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from agent import auto_state, engine, schedule_queue, scheduler, state
from config import settings

logger = logging.getLogger(__name__)

# Max retries for agent generation failures
_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 300  # 5 minutes

# Daemon / in-process scheduler check interval
SCHEDULER_INTERVAL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Lightweight Telegram notification (standalone mode only — no bot instance)
# ---------------------------------------------------------------------------

async def _notify_telegram(message: str) -> None:
    """Send a plain text notification via raw HTTP (standalone mode)."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_ALLOWED_USER_ID:
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": settings.TELEGRAM_ALLOWED_USER_ID,
            "text": message,
            "parse_mode": "HTML",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.warning("Telegram notification failed: %s", resp.text[:200])
    except Exception as e:
        logger.warning("Telegram notification error: %s", e)


# ---------------------------------------------------------------------------
# Core slot processing
# ---------------------------------------------------------------------------

async def process_slot(
    slot_name: str,
    slot_config: dict,
    global_config: dict,
    dry_run: bool = False,
    bot=None,
) -> bool:
    """Process a single slot: generate content, save as pending draft, notify.

    When bot is provided (in-process mode), sends the draft via the bot's
    Telegram API with the full branded compositor.  When bot is None
    (standalone mode), sends a plain text/photo notification via HTTP.

    The actual X post happens when the operator sends /approve in Telegram.

    Returns True if a draft was successfully generated and queued.
    """
    slot_type = slot_config.get("type", "unknown")
    logger.info("Processing slot: %s (type=%s, dry_run=%s)", slot_name, slot_type, dry_run)

    # Rate limit check
    min_gap = global_config.get("min_gap_minutes", 120)
    max_posts = global_config.get("max_posts_per_day", 6)
    allowed, reason = auto_state.can_post(min_gap, max_posts)
    if not allowed and not dry_run:
        logger.info("Skipping %s: %s", slot_name, reason)
        return False

    # Don't queue if there's already a pending draft awaiting review
    if state.has_pending() and not dry_run:
        logger.info("Skipping %s: a draft is already pending approval", slot_name)
        return False

    # Build the prompt
    prompt, event_ids = await scheduler.build_prompt_for_slot(slot_name, slot_config)
    logger.info("Prompt built for %s (%d chars)", slot_name, len(prompt))

    # Run the agent (same pipeline as manual Telegram requests)
    result = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            result = await engine.run_agent(request=prompt)
            if result.draft:
                break
            logger.warning(
                "Agent returned no draft for %s (attempt %d/%d)",
                slot_name, attempt + 1, _MAX_RETRIES + 1,
            )
        except Exception as e:
            logger.error(
                "Agent failed for %s (attempt %d/%d): %s",
                slot_name, attempt + 1, _MAX_RETRIES + 1, e,
            )

        if attempt < _MAX_RETRIES:
            logger.info("Retrying in %ds...", _RETRY_DELAY_SECONDS)
            await asyncio.sleep(_RETRY_DELAY_SECONDS)

    if not result or not result.draft:
        logger.error("Failed to generate content for %s after %d attempts", slot_name, _MAX_RETRIES + 1)
        await _notify_telegram(
            f"<b>Auto-post failed</b>\n\n"
            f"Slot: {slot_name}\n"
            f"Could not generate content after {_MAX_RETRIES + 1} attempts."
        )
        return False

    caption = result.draft.get("caption", "")
    image_url = result.image_url

    # Duplicate check
    if auto_state.is_duplicate_caption(caption):
        logger.warning("Duplicate caption detected for %s, skipping", slot_name)
        await _notify_telegram(
            f"<b>Auto-post skipped (duplicate)</b>\n\n"
            f"Slot: {slot_name}\n"
            f"Caption was too similar to a recent post."
        )
        return False

    # --- Dry run: log and optionally notify, but don't save or post ---
    if dry_run:
        logger.info(
            "DRY RUN — slot=%s caption=%s image=%s turns=%d time=%.1fs",
            slot_name, caption[:80], bool(image_url), result.turns_used, result.total_time,
        )
        print(f"\n{'='*60}")
        print(f"DRY RUN: {slot_name}")
        print(f"{'='*60}")
        print(f"Type: {slot_config.get('type')}")
        print(f"Caption: {caption}")
        print(f"Image: {image_url or '(none)'}")
        print(f"Alt text: {result.draft.get('alt_text', '')}")
        print(f"Image prompt: {result.draft.get('image_prompt', '')[:200]}")
        print(f"Agent: {result.turns_used} turns, {result.total_time}s")
        print(f"Tools: {', '.join(result.tool_calls_made)}")
        print(f"{'='*60}\n")

        if global_config.get("notify_telegram"):
            await _notify_telegram(
                f"<b>Auto-post dry run</b>\n\n"
                f"Slot: {slot_name}\n"
                f"Caption: {caption}\n"
                f"Image: {'yes' if image_url else 'no'}"
            )
        return True

    # --- Save as pending draft (same state the manual flow uses) ---
    state.save_pending(
        caption=caption,
        hashtags=result.draft.get("hashtags", []),
        image_url=image_url,
        alt_text=result.draft.get("alt_text", ""),
        image_prompt=result.draft.get("image_prompt", ""),
        original_request=prompt,
        image_urls=result.image_urls if len(result.image_urls) > 1 else None,
        auto_slot=slot_name,
        auto_event_ids=event_ids if event_ids else None,
    )

    # Save last generated for /edit support
    if image_url:
        state.save_last_generated(image_url, result.draft.get("content_type", "default"))

    # --- Send draft to Telegram for review ---
    if bot:
        # In-process mode: use the bot instance + full compositor
        from bot.handlers import send_auto_draft
        await send_auto_draft(bot, result.draft, image_url, slot_name)
    else:
        # Standalone mode: send via raw HTTP
        notification = (
            f"<b>Auto-Draft Ready</b>  [slot: <code>{slot_name}</code>]\n\n"
            f"{caption}\n\n"
            f"/approve to post to X\n"
            f"/reject <i>feedback</i> to revise\n"
            f"/cancel to discard"
        )
        await _notify_telegram(notification)

    logger.info("Draft queued for approval: slot=%s", slot_name)
    return True


# ---------------------------------------------------------------------------
# User-scheduled item processing
# ---------------------------------------------------------------------------

async def process_scheduled_item(
    item: dict,
    global_config: dict,
    dry_run: bool = False,
    bot=None,
) -> bool:
    """Process a user-scheduled queue item.

    Similar to process_slot but uses the user's prompt directly
    and tracks status in the schedule queue.

    Returns True if a draft was generated and queued.
    """
    item_id = item["id"]
    prompt = item["prompt"]
    label = item.get("label", prompt[:40])
    slot_name = f"scheduled:{item_id}"

    logger.info("Processing scheduled item: %s (%s)", item_id, label)
    schedule_queue.mark_generating(item_id)

    # Rate limit check
    min_gap = global_config.get("min_gap_minutes", 120)
    max_posts = global_config.get("max_posts_per_day", 6)
    allowed, reason = auto_state.can_post(min_gap, max_posts)
    if not allowed and not dry_run:
        logger.info("Skipping scheduled %s: %s", item_id, reason)
        # Don't mark as failed — leave as generating so it retries next cycle
        # Reset back to pending so it's picked up again
        items = schedule_queue._read_queue()
        for i in items:
            if i["id"] == item_id:
                i["status"] = "pending"
                break
        schedule_queue._write_queue(items)
        return False

    # Don't queue if there's already a pending draft awaiting review
    if state.has_pending() and not dry_run:
        logger.info("Skipping scheduled %s: a draft is already pending approval", item_id)
        items = schedule_queue._read_queue()
        for i in items:
            if i["id"] == item_id:
                i["status"] = "pending"
                break
        schedule_queue._write_queue(items)
        return False

    # Run the agent with the user's prompt
    result = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            result = await engine.run_agent(request=prompt)
            if result.draft:
                break
            logger.warning(
                "Agent returned no draft for scheduled %s (attempt %d/%d)",
                item_id, attempt + 1, _MAX_RETRIES + 1,
            )
        except Exception as e:
            logger.error(
                "Agent failed for scheduled %s (attempt %d/%d): %s",
                item_id, attempt + 1, _MAX_RETRIES + 1, e,
            )
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(60)  # shorter retry delay for user-scheduled

    if not result or not result.draft:
        schedule_queue.mark_failed(item_id, "Could not generate content")
        await _notify_telegram(
            f"<b>Scheduled post failed</b>\n\n"
            f"ID: <code>{item_id}</code>\n"
            f"Prompt: {prompt[:100]}\n"
            f"Could not generate content after {_MAX_RETRIES + 1} attempts."
        )
        return False

    caption = result.draft.get("caption", "")
    image_url = result.image_url

    # Duplicate check
    if auto_state.is_duplicate_caption(caption):
        schedule_queue.mark_failed(item_id, "Duplicate caption")
        await _notify_telegram(
            f"<b>Scheduled post skipped (duplicate)</b>\n\n"
            f"ID: <code>{item_id}</code>\n"
            f"Caption was too similar to a recent post."
        )
        return False

    if dry_run:
        logger.info("DRY RUN — scheduled=%s caption=%s", item_id, caption[:80])
        schedule_queue.mark_done(item_id)
        return True

    # Save as pending draft
    state.save_pending(
        caption=caption,
        hashtags=result.draft.get("hashtags", []),
        image_url=image_url,
        alt_text=result.draft.get("alt_text", ""),
        image_prompt=result.draft.get("image_prompt", ""),
        original_request=prompt,
        image_urls=result.image_urls if len(result.image_urls) > 1 else None,
        auto_slot=slot_name,
    )

    if image_url:
        state.save_last_generated(image_url, result.draft.get("content_type", "default"))

    # Mark as done (recurrence handled inside mark_done)
    schedule_queue.mark_done(item_id)

    # Send draft to Telegram for review
    if bot:
        from bot.handlers import send_auto_draft
        await send_auto_draft(bot, result.draft, image_url, slot_name)
    else:
        notification = (
            f"<b>Scheduled Draft Ready</b>  [<code>{item_id}</code>]\n\n"
            f"{caption}\n\n"
            f"/approve to post to X\n"
            f"/reject <i>feedback</i> to revise\n"
            f"/cancel to discard"
        )
        await _notify_telegram(notification)

    logger.info("Scheduled draft queued for approval: %s", item_id)
    return True


# ---------------------------------------------------------------------------
# Cron / daemon runners
# ---------------------------------------------------------------------------

async def run_cron(
    dry_run: bool = False,
    force_slot: str | None = None,
    bot=None,
) -> int:
    """Single cron run: check schedule + user queue, process due items.

    Returns number of drafts generated.
    """
    schedule = scheduler.load_schedule()
    global_config = schedule.get("global", {})
    slots = schedule.get("slots", {})
    drafts_made = 0

    # --- 1. Process user-scheduled queue items (always, even if auto-post is off) ---
    # User-scheduled posts are explicit user requests, not auto-pilot.
    due_items = schedule_queue.get_due_items(window_seconds=SCHEDULER_INTERVAL_SECONDS)
    if due_items and not force_slot:
        if not auto_state.is_paused():
            for item in due_items:
                if state.has_pending() and not dry_run:
                    logger.info("User queue: pending draft exists, deferring")
                    break
                success = await process_scheduled_item(
                    item, global_config,
                    dry_run=dry_run or settings.AUTO_POST_DRY_RUN,
                    bot=bot,
                )
                if success:
                    drafts_made += 1
                    if not dry_run:
                        break

    # If a user-scheduled draft was generated, skip predefined slots this cycle
    if drafts_made and not dry_run:
        return drafts_made

    # --- 2. Process predefined time slots ---
    if not slots:
        logger.debug("No predefined slots in schedule.json")
    elif force_slot:
        if force_slot not in slots:
            logger.error("Unknown slot: %s (available: %s)", force_slot, list(slots.keys()))
            return drafts_made
        due_slots = [force_slot]
        logger.info("Forcing slot: %s", force_slot)
    else:
        due_slots = scheduler.get_due_slots(schedule)
        if due_slots:
            logger.info("Due slots: %s", due_slots)
        else:
            due_slots = []

    # Check if auto-posting is enabled and not paused (for predefined slots)
    if not settings.AUTO_POST_ENABLED and not dry_run and not force_slot:
        logger.debug("Auto-posting disabled — skipping predefined slots")
        return drafts_made

    if auto_state.is_paused() and not force_slot:
        logger.debug("Auto-posting paused — skipping predefined slots")
        return drafts_made

    for slot_name in due_slots:
        slot_config = slots[slot_name]
        success = await process_slot(
            slot_name, slot_config, global_config,
            dry_run=dry_run or settings.AUTO_POST_DRY_RUN,
            bot=bot,
        )
        if success:
            drafts_made += 1
            if not dry_run:
                break

    # --- 3. Periodic housekeeping ---
    schedule_queue.prune_old()

    return drafts_made


async def run_scheduler_loop(bot=None) -> None:
    """Long-running scheduler loop — meant to run as a background task
    inside the Telegram bot process.

    Checks every SCHEDULER_INTERVAL_SECONDS for due slots and generates
    drafts that get sent to Telegram for approval.
    """
    logger.info(
        "Auto-post scheduler started (interval=%ds, enabled=%s)",
        SCHEDULER_INTERVAL_SECONDS, settings.AUTO_POST_ENABLED,
    )

    while True:
        try:
            drafts = await run_cron(bot=bot)
            if drafts:
                logger.info("Scheduler cycle: %d draft(s) generated", drafts)
        except Exception as e:
            logger.error("Scheduler cycle error: %s", e)

        await asyncio.sleep(SCHEDULER_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BrandMover auto-post scheduler")
    parser.add_argument("--daemon", action="store_true", help="Run as long-running daemon")
    parser.add_argument("--dry-run", action="store_true", help="Generate content without posting")
    parser.add_argument("--force", type=str, help="Force a specific slot (bypass time window)")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/brandmover_auto_post.log"),
        ],
    )

    logger.info(
        "Auto-post starting (daemon=%s, dry_run=%s, force=%s)",
        args.daemon, args.dry_run, args.force,
    )

    if args.daemon:
        asyncio.run(run_scheduler_loop())
    else:
        drafts = asyncio.run(run_cron(dry_run=args.dry_run, force_slot=args.force))
        logger.info("Auto-post complete: %d draft(s)", drafts)
        sys.exit(0 if drafts >= 0 else 1)


if __name__ == "__main__":
    main()
