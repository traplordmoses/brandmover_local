"""
Weekly digest automation — triggers every Sunday via the scheduler loop.

Generates a branded performance report for the past week and sends it
to the operator via Telegram. Uses report_generator.py for HTML output.

State lives in state/weekly_digest_state.json.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_STATE_FILE = _project_root / "state" / "weekly_digest_state.json"

# Only trigger on Sundays (6 = Sunday in weekday())
_DIGEST_WEEKDAY = 6
# Minimum hours between digests (prevents double-trigger)
_MIN_INTERVAL_HOURS = 144  # ~6 days


def _read_state() -> dict:
    if not _STATE_FILE.exists():
        return {"last_digest_at": 0.0, "total_digests": 0}
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"last_digest_at": 0.0, "total_digests": 0}


def _write_state(data: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(_STATE_FILE))


def should_run_digest() -> bool:
    """Check if it's Sunday and enough time has passed since the last digest."""
    now = datetime.now()
    if now.weekday() != _DIGEST_WEEKDAY:
        return False

    state = _read_state()
    hours_since = (time.time() - state.get("last_digest_at", 0)) / 3600
    return hours_since >= _MIN_INTERVAL_HOURS


async def generate_weekly_digest() -> str | None:
    """Generate a weekly performance report and return the HTML file path.

    Returns None on failure.
    """
    from agent.report_generator import generate_report

    now = datetime.now()
    week_start = (now - timedelta(days=7)).strftime("%b %d")
    week_end = now.strftime("%b %d, %Y")

    try:
        report_path = generate_report(
            report_type="performance",
            title=f"Weekly Digest",
            subtitle=f"{week_start} — {week_end}",
        )
        if report_path:
            state = _read_state()
            state["last_digest_at"] = time.time()
            state["total_digests"] = state.get("total_digests", 0) + 1
            _write_state(state)
            logger.info("Weekly digest generated: %s", report_path)
        return report_path
    except Exception as e:
        logger.error("Weekly digest generation failed: %s", e)
        return None


async def maybe_trigger_weekly_digest(bot=None) -> bool:
    """Check if a weekly digest should be sent. Call from scheduler loop.

    Returns True if a digest was generated and sent.
    """
    if not should_run_digest():
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
                    caption="<b>Weekly Digest</b> — your performance summary is ready.",
                    parse_mode="HTML",
                )
            logger.info("Weekly digest sent to Telegram")
        except Exception as e:
            logger.error("Failed to send weekly digest via Telegram: %s", e)
    else:
        # Standalone mode — notify via HTTP
        try:
            from scripts.auto_post import _notify_telegram
            await _notify_telegram(
                f"<b>Weekly Digest Ready</b>\n\n"
                f"Report saved to: <code>{report_path}</code>"
            )
        except Exception as e:
            logger.error("Failed to notify about weekly digest: %s", e)

    return True
