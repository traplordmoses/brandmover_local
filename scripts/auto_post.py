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

from agent import auto_state, content_planner, context_feed, engine, preference_engine, publisher, schedule_queue, scheduler, state
from config import settings

logger = logging.getLogger(__name__)

# Max retries for agent generation failures
_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 300  # 5 minutes

# Gate daily preference cluster refresh (ISO date string)
_last_cluster_refresh_date: str | None = None

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
                logger.warning("Telegram notification failed: HTTP %s", resp.status_code)
    except Exception as e:
        logger.error("Telegram notification failed: %s", type(e).__name__)


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

    # --- Aggregate real-time context (on-chain events, X mentions) ---
    context_snapshot = await context_feed.aggregate_context()
    context_block = context_snapshot.summary
    if context_snapshot.has_urgent and settings.EVENT_TRIGGER_ENABLED:
        logger.info(
            "Urgent context signal detected for slot %s (%d signals)",
            slot_name, len(context_snapshot.signals),
        )

    # Rate limit check
    min_gap = global_config.get("min_gap_minutes", 120)
    max_posts = global_config.get("max_posts_per_day", 6)
    allowed, reason = auto_state.can_post(min_gap, max_posts)
    if not allowed and not dry_run:
        logger.info("Skipping %s: %s", slot_name, reason)
        return False

    # Don't queue if there's already a pending draft awaiting review
    if await state.async_has_pending() and not dry_run:
        logger.info("Skipping %s: a draft is already pending approval", slot_name)
        return False

    # Build the prompt
    prompt, event_ids = await scheduler.build_prompt_for_slot(slot_name, slot_config)

    # Inject live context into the prompt if available
    if context_block:
        prompt = f"{prompt}\n\n{context_block}"

    logger.info("Prompt built for %s (%d chars, context=%d chars)",
                slot_name, len(prompt), len(context_block))

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

    # --- Preference engine scoring ---
    try:
        score_result = await preference_engine.score_draft(result.draft, prompt)
        logger.info(
            "Preference score for %s: %.1f (%s)",
            slot_name, score_result.score, score_result.reasoning[:80],
        )
        if score_result.should_reject:
            logger.warning(
                "Draft rejected by preference engine (score=%.1f, threshold=%.1f): %s",
                score_result.score, settings.DRAFT_SCORE_THRESHOLD, score_result.reasoning,
            )
            # Retry once with the score reasoning appended to the prompt
            retry_prompt = (
                f"{prompt}\n\n"
                f"[QUALITY NOTE: A previous draft scored {score_result.score}/10. "
                f"Issues: {score_result.reasoning} "
                f"Flags: {', '.join(score_result.flags) if score_result.flags else 'none'}. "
                f"Please address these issues in this attempt.]"
            )
            retry_result = await engine.run_agent(request=retry_prompt)
            if retry_result and retry_result.draft:
                retry_score = await preference_engine.score_draft(retry_result.draft, prompt)
                logger.info(
                    "Retry preference score for %s: %.1f", slot_name, retry_score.score,
                )
                # Use the retry result regardless (we only retry once)
                result = retry_result
                caption = result.draft.get("caption", "")
                image_url = result.image_url
    except Exception as e:
        logger.warning("Preference scoring failed for %s: %s", slot_name, e)

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
    await state.async_save_pending(
        caption=caption,
        hashtags=result.draft.get("hashtags", []),
        image_url=image_url,
        alt_text=result.draft.get("alt_text", ""),
        image_prompt=result.draft.get("image_prompt", ""),
        original_request=prompt,
        image_urls=result.image_urls if len(result.image_urls) > 1 else None,
        auto_slot=slot_name,
        auto_event_ids=event_ids if event_ids else None,
        conversation_history=result.conversation_history,
    )

    # Save last generated for /edit support
    if image_url:
        await asyncio.to_thread(state.save_last_generated, image_url, result.draft.get("content_type", "default"))

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

    # --- Pre-approved draft: post directly, skip generation ---
    pre_draft = item.get("draft")
    if pre_draft:
        logger.info("Scheduled item %s has pre-approved draft, posting directly", item_id)
        schedule_queue.mark_generating(item_id)

        caption = pre_draft.get("caption", "")
        hashtags = pre_draft.get("hashtags", [])
        publish_image = pre_draft.get("composed_path") or pre_draft.get("image_url")

        if dry_run:
            logger.info("DRY RUN — scheduled=%s (pre-approved) caption=%s", item_id, caption[:80])
            schedule_queue.mark_done(item_id)
            return True

        try:
            publish_results = await publisher.publish_to_all(
                draft=pre_draft,
                image_url=publish_image,
                composed_path=pre_draft.get("composed_path"),
            )
            tweet_url = publish_results.get("x")
            schedule_queue.mark_done(item_id, tweet_url=tweet_url)
            auto_state.record_post(
                slot_name=slot_name, caption=caption, tweet_url=tweet_url,
            )
            platform_summary = ", ".join(
                f"{p}: {u or 'failed'}" for p, u in publish_results.items()
            )
            await _notify_telegram(
                f"<b>Scheduled post published</b>\n\n"
                f"{caption[:200]}\n\n"
                f"{platform_summary}"
            )
            logger.info("Pre-approved scheduled post published: %s -> %s", item_id, publish_results)
            return True
        except Exception as e:
            logger.error("Pre-approved scheduled post failed: %s", e)
            schedule_queue.mark_failed(item_id, str(e)[:200])
            await _notify_telegram(
                f"<b>Scheduled post failed</b>\n\n"
                f"ID: <code>{item_id}</code>\n"
                f"Error: {str(e)[:200]}"
            )
            return False

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
    if await state.async_has_pending() and not dry_run:
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
    await state.async_save_pending(
        caption=caption,
        hashtags=result.draft.get("hashtags", []),
        image_url=image_url,
        alt_text=result.draft.get("alt_text", ""),
        image_prompt=result.draft.get("image_prompt", ""),
        original_request=prompt,
        image_urls=result.image_urls if len(result.image_urls) > 1 else None,
        auto_slot=slot_name,
        conversation_history=result.conversation_history,
    )

    if image_url:
        await asyncio.to_thread(state.save_last_generated, image_url, result.draft.get("content_type", "default"))

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

    # --- 1. Process user-scheduled queue items (always, even if auto-post is paused) ---
    # User-scheduled posts are explicit user requests, not auto-pilot.
    # They fire regardless of pause state — pausing only affects predefined slots.
    due_items = schedule_queue.get_due_items(window_seconds=SCHEDULER_INTERVAL_SECONDS)
    if due_items and not force_slot:
        for item in due_items:
            if await state.async_has_pending() and not dry_run:
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

    # --- 2. Process predefined time slots (or content planner) ---
    # When CONTENT_PLANNER_ENABLED, use the planner instead of slot-based scheduling.
    if settings.CONTENT_PLANNER_ENABLED and not force_slot:
        planned = content_planner.get_next_planned_post()
        if planned and not auto_state.is_paused():
            logger.info(
                "Content planner: next post %s/%s (%s)",
                planned.date, planned.time_slot, planned.content_type,
            )
            # Build a slot_config compatible dict from the planned post
            planner_slot_name = f"planner:{planned.content_type}_{planned.time_slot}"
            planner_slot_config = {
                "type": planned.content_type,
                "prompt_hint": planned.prompt_hint,
            }
            content_planner.mark_post_status(planned.date, planned.time_slot, "generating")
            success = await process_slot(
                planner_slot_name, planner_slot_config, global_config,
                dry_run=dry_run or settings.AUTO_POST_DRY_RUN,
                bot=bot,
            )
            if success:
                content_planner.mark_post_status(planned.date, planned.time_slot, "posted")
                drafts_made += 1
            else:
                content_planner.mark_post_status(planned.date, planned.time_slot, "skipped")
        elif not planned:
            logger.debug("Content planner: no posts due right now")
    else:
        # Fallback: original slot-based scheduling
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

    # --- 3a. Daily content plan update ---
    if settings.CONTENT_PLANNER_ENABLED:
        try:
            from datetime import datetime, timezone as _tz
            _plan_today = datetime.now(_tz.utc).strftime("%Y-%m-%d")
            plan = content_planner.load_plan()
            if plan.week_start != _plan_today:
                await content_planner.async_update_plan_daily()
                logger.info("Content plan updated for %s", _plan_today)
        except Exception as e:
            logger.debug("Content plan daily update failed: %s", e)

    # --- 3b. Daily preference cluster refresh ---
    global _last_cluster_refresh_date
    try:
        from datetime import datetime, timezone
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if _last_cluster_refresh_date != today_str:
            preference_engine.refresh_clusters()
            _last_cluster_refresh_date = today_str
            logger.info("Preference clusters refreshed for %s", today_str)
    except Exception as e:
        logger.debug("Preference cluster refresh failed: %s", e)

    # --- 4. Daily self-review check ---
    try:
        from agent.self_review_scheduler import maybe_trigger_daily_review
        await maybe_trigger_daily_review()
    except Exception as e:
        logger.debug("Self-review daily check failed: %s", e)

    # --- 4b. Daily digest ---
    try:
        from agent.digest import maybe_trigger_daily_digest
        daily_sent = await maybe_trigger_daily_digest(bot=bot)
        if daily_sent:
            logger.info("Daily digest sent")
    except Exception as e:
        logger.debug("Daily digest check failed: %s", e)

    # --- 4c. Weekly digest (Sundays) ---
    try:
        from agent.digest import maybe_trigger_weekly_digest
        digest_sent = await maybe_trigger_weekly_digest(bot=bot)
        if digest_sent:
            logger.info("Weekly digest sent")
    except Exception as e:
        logger.debug("Weekly digest check failed: %s", e)

    # --- 4d. Health check ---
    try:
        from agent.health_monitor import maybe_run_health_check
        await maybe_run_health_check(bot=bot)
    except Exception as e:
        logger.debug("Health check failed: %s", e)

    # --- 5. Topic bank refresh (every TOPIC_BANK_REFRESH_INTERVAL_HOURS) ---
    try:
        import time as _time
        from agent.topic_bank import load_bank, seed_bank_if_empty
        seed_bank_if_empty()
        bank = load_bank()
        hours_since_refresh = (_time.time() - (bank.last_refreshed or 0)) / 3600
        if hours_since_refresh > settings.TOPIC_BANK_REFRESH_INTERVAL_HOURS:
            from agent.topic_refresh import refresh_topic_bank
            result = await refresh_topic_bank()
            logger.info("Topic bank refreshed: %s", result)
    except Exception as e:
        logger.debug("Topic bank refresh failed: %s", e)

    # --- 6. Auto preference extraction (every PREF_EXTRACTION_INTERVAL_HOURS) ---
    if settings.PREF_EXTRACTION_ENABLED:
        try:
            from agent.pref_extractor import extract_preferences
            new_prefs = await extract_preferences()
            if new_prefs:
                logger.info("Auto-extracted %d new preferences: %s", len(new_prefs), new_prefs)
                # Notify via Telegram
                msg = "<b>Auto-learned preferences</b>\n\n"
                for p in new_prefs:
                    msg += f"\u2022 {p}\n"
                msg += "\nUse /preferences to view all. /unpref <number> to remove any."
                await _notify_telegram(msg)
        except Exception as e:
            logger.debug("Preference extraction failed: %s", e)

    return drafts_made


async def run_scheduler_loop(bot=None) -> None:
    """Long-running scheduler loop — meant to run as a background task
    inside the Telegram bot process.

    When HEARTBEAT_ENABLED=true, uses the heartbeat reasoning layer
    (assess → reason → dispatch). Otherwise falls back to the original
    cron-based loop.
    """
    use_heartbeat = settings.HEARTBEAT_ENABLED
    logger.info(
        "Scheduler started (interval=%ds, enabled=%s, heartbeat=%s)",
        SCHEDULER_INTERVAL_SECONDS, settings.AUTO_POST_ENABLED, use_heartbeat,
    )

    while True:
        try:
            if use_heartbeat:
                from agent.heartbeat import heartbeat_tick
                action_taken = await heartbeat_tick(bot=bot)
                if action_taken:
                    logger.info("Heartbeat: action taken this cycle")
            else:
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
