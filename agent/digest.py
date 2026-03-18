"""
Digest generator -- daily and weekly summaries of bot performance.

Replaces the former weekly_digest.py with both daily and weekly digest support.
Daily digests collect metrics from generation_history, feedback, and auto_state.
Weekly digests reuse the report_generator HTML pipeline.

State lives in state/digest_state.json via FileStore.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.paths import STATE_DIR
from agent.state_manager import FileStore

logger = logging.getLogger(__name__)

_STATE_FILE = STATE_DIR / "digest_state.json"
_store = FileStore(_STATE_FILE, default_factory=lambda: {
    "last_daily_at": 0.0,
    "last_weekly_at": 0.0,
    "total_daily": 0,
    "total_weekly": 0,
})

# Only trigger weekly on Sundays (6 = Sunday in weekday())
_DIGEST_WEEKDAY = 6
# Minimum hours between weekly digests (prevents double-trigger)
_MIN_WEEKLY_INTERVAL_HOURS = 144  # ~6 days


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def collect_daily_metrics(date_key: str | None = None) -> dict:
    """Aggregate metrics from generation_history, feedback, and auto_state for one day.

    Args:
        date_key: ISO date string (YYYY-MM-DD). Defaults to today (UTC).

    Returns:
        Dict with posts_published, approval_rate, rejections, failures,
        and content_type_breakdown.
    """
    if date_key is None:
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- Generation history metrics ---
    from agent.generation_history import _read_history
    entries = _read_history()

    day_entries = []
    for e in entries:
        ts = e.get("timestamp", 0)
        entry_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if entry_date == date_key:
            day_entries.append(e)

    posts_published = sum(
        1 for e in day_entries if e.get("status") == "approved"
    )
    failures = sum(
        1 for e in day_entries if e.get("status") in ("failed", "error")
    )

    # Content type breakdown
    content_type_breakdown: dict[str, int] = {}
    for e in day_entries:
        ct = e.get("content_type", "unknown")
        content_type_breakdown[ct] = content_type_breakdown.get(ct, 0) + 1

    # --- Feedback / approval metrics ---
    from agent.preference_engine import get_daily_approval_rate
    approval_data = get_daily_approval_rate(date_key)
    approval_rate = approval_data.get("rate", 0.0)
    rejections = approval_data.get("rejected", 0)

    return {
        "date": date_key,
        "posts_published": posts_published,
        "total_generations": len(day_entries),
        "approval_rate": approval_rate,
        "rejections": rejections,
        "failures": failures,
        "content_type_breakdown": content_type_breakdown,
    }


# ---------------------------------------------------------------------------
# Daily digest
# ---------------------------------------------------------------------------

async def generate_daily_digest() -> str | None:
    """Generate an HTML report for today and return the file path.

    Returns None on failure.
    """
    from agent.report_generator import generate_report

    now = datetime.now(timezone.utc)
    date_key = now.strftime("%Y-%m-%d")
    metrics = await asyncio.to_thread(collect_daily_metrics, date_key)

    try:
        # Build custom sections from daily metrics
        sections = [
            {
                "title": "Daily Summary",
                "items": [
                    f"Posts published: {metrics['posts_published']}",
                    f"Total generations: {metrics['total_generations']}",
                    f"Approval rate: {metrics['approval_rate']:.0f}%",
                    f"Rejections: {metrics['rejections']}",
                    f"Failures: {metrics['failures']}",
                ],
            },
        ]
        if metrics["content_type_breakdown"]:
            ct_items = [
                f"{ct}: {count}" for ct, count in
                sorted(metrics["content_type_breakdown"].items(), key=lambda x: -x[1])
            ]
            sections.append({"title": "Content Types", "items": ct_items})

        report_path = await asyncio.to_thread(
            generate_report,
            report_type="custom",
            title="Daily Digest",
            subtitle=now.strftime("%B %d, %Y"),
            custom_sections=sections,
        )

        if report_path:
            state = await asyncio.to_thread(_store.read)
            state["last_daily_at"] = time.time()
            state["total_daily"] = state.get("total_daily", 0) + 1
            await asyncio.to_thread(_store.write, state)
            logger.info("Daily digest generated: %s", report_path)
        return report_path
    except Exception as e:
        logger.error("Daily digest generation failed: %s", e)
        return None


async def maybe_trigger_daily_digest(bot=None) -> bool:
    """Trigger daily digest once per day at DAILY_DIGEST_HOUR.

    Call from the scheduler loop. Returns True if a digest was generated and sent.
    """
    from config import settings

    if not settings.DAILY_DIGEST_ENABLED:
        return False

    now = datetime.now()
    if now.hour != settings.DAILY_DIGEST_HOUR:
        return False

    state = _store.read()
    last = state.get("last_daily_at", 0.0)
    hours_since = (time.time() - last) / 3600
    if hours_since < 20:  # At least 20 hours between daily digests
        return False

    logger.info("Daily digest: triggering for today")
    report_path = await generate_daily_digest()
    if not report_path:
        return False

    # Send via Telegram if bot is available
    if bot:
        try:
            with open(report_path, "rb") as f:
                await bot.send_document(
                    chat_id=settings.TELEGRAM_ALLOWED_USER_ID,
                    document=f,
                    filename=Path(report_path).name,
                    caption="<b>Daily Digest</b> -- your performance summary is ready.",
                    parse_mode="HTML",
                )
            logger.info("Daily digest sent to Telegram")
        except Exception as e:
            logger.error("Failed to send daily digest via Telegram: %s", e)
    else:
        try:
            from scripts.auto_post import _notify_telegram
            await _notify_telegram(
                f"<b>Daily Digest Ready</b>\n\n"
                f"Report saved to: <code>{report_path}</code>"
            )
        except Exception as e:
            logger.error("Failed to notify about daily digest: %s", e)

    return True


# ---------------------------------------------------------------------------
# Weekly digest (migrated from weekly_digest.py)
# ---------------------------------------------------------------------------

def _should_run_weekly() -> bool:
    """Check if it's Sunday and enough time has passed since the last weekly digest."""
    now = datetime.now()
    if now.weekday() != _DIGEST_WEEKDAY:
        return False

    state = _store.read()
    hours_since = (time.time() - state.get("last_weekly_at", 0)) / 3600
    return hours_since >= _MIN_WEEKLY_INTERVAL_HOURS


async def generate_weekly_digest() -> str | None:
    """Generate a weekly performance report and return the HTML file path.

    Returns None on failure.
    """
    from agent.report_generator import generate_report

    now = datetime.now()
    week_start = (now - timedelta(days=7)).strftime("%b %d")
    week_end = now.strftime("%b %d, %Y")

    try:
        report_path = await asyncio.to_thread(
            generate_report,
            report_type="performance",
            title="Weekly Digest",
            subtitle=f"{week_start} -- {week_end}",
        )
        if report_path:
            state = await asyncio.to_thread(_store.read)
            state["last_weekly_at"] = time.time()
            state["total_weekly"] = state.get("total_weekly", 0) + 1
            await asyncio.to_thread(_store.write, state)
            logger.info("Weekly digest generated: %s", report_path)
        return report_path
    except Exception as e:
        logger.error("Weekly digest generation failed: %s", e)
        return None


async def maybe_trigger_weekly_digest(bot=None) -> bool:
    """Check if a weekly digest should be sent. Call from scheduler loop.

    Returns True if a digest was generated and sent.
    """
    if not _should_run_weekly():
        return False

    logger.info("Weekly digest: triggering for this week")
    report_path = await generate_weekly_digest()
    if not report_path:
        return False

    # Send via Telegram if bot is available
    if bot:
        try:
            from config import settings
            with open(report_path, "rb") as f:
                await bot.send_document(
                    chat_id=settings.TELEGRAM_ALLOWED_USER_ID,
                    document=f,
                    filename=Path(report_path).name,
                    caption="<b>Weekly Digest</b> -- your performance summary is ready.",
                    parse_mode="HTML",
                )
            logger.info("Weekly digest sent to Telegram")
        except Exception as e:
            logger.error("Failed to send weekly digest via Telegram: %s", e)
    else:
        try:
            from scripts.auto_post import _notify_telegram
            await _notify_telegram(
                f"<b>Weekly Digest Ready</b>\n\n"
                f"Report saved to: <code>{report_path}</code>"
            )
        except Exception as e:
            logger.error("Failed to notify about weekly digest: %s", e)

    return True
