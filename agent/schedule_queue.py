"""
User-driven schedule queue for BrandMover.

Stores scheduled post requests with target times. The background scheduler
loop processes due items alongside predefined time slots.

Storage: state/schedule_queue.json
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_QUEUE_FILE = Path(settings.AUTO_POST_STATE_FILE).parent / "schedule_queue.json"


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _read_queue() -> list[dict]:
    """Read the schedule queue from disk."""
    if not _QUEUE_FILE.exists():
        return []
    try:
        data = json.loads(_QUEUE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return data.get("items", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read schedule_queue.json: %s", e)
        return []


def _write_queue(items: list[dict]) -> None:
    """Write the queue to disk."""
    _QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _QUEUE_FILE.write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def add_scheduled(
    prompt: str,
    scheduled_utc: float,
    recurrence: str = "once",
    label: str | None = None,
) -> dict:
    """Add a new scheduled post to the queue.

    Args:
        prompt: The content generation prompt.
        scheduled_utc: Unix timestamp (UTC) for when to generate.
        recurrence: "once", "daily", or "weekly".
        label: Optional short label for display.

    Returns the created queue item dict.
    """
    item = {
        "id": uuid.uuid4().hex[:8],
        "prompt": prompt,
        "scheduled_utc": scheduled_utc,
        "created_at": time.time(),
        "status": "pending",  # pending → generating → posted | failed | cancelled
        "recurrence": recurrence,
        "label": label or prompt[:40],
    }
    items = _read_queue()
    items.append(item)
    _write_queue(items)
    logger.info(
        "Scheduled post %s for %s: %s",
        item["id"],
        datetime.fromtimestamp(scheduled_utc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        prompt[:60],
    )
    return item


def list_scheduled(include_done: bool = False) -> list[dict]:
    """List scheduled items. By default only pending/generating items."""
    items = _read_queue()
    if include_done:
        return items
    return [i for i in items if i.get("status") in ("pending", "generating")]


def cancel_scheduled(item_id: str) -> bool:
    """Cancel a scheduled item by ID. Returns True if found and cancelled."""
    items = _read_queue()
    for item in items:
        if item["id"] == item_id and item["status"] in ("pending", "generating"):
            item["status"] = "cancelled"
            item["cancelled_at"] = time.time()
            _write_queue(items)
            logger.info("Cancelled scheduled post: %s", item_id)
            return True
    return False


def get_due_items(window_seconds: int = 300) -> list[dict]:
    """Return items that are due now (within window).

    An item is due if:
    1. status == "pending"
    2. scheduled_utc <= now + window_seconds (so we catch items within
       the next scheduler cycle)
    """
    now = time.time()
    items = _read_queue()
    due = []
    for item in items:
        if item.get("status") != "pending":
            continue
        if item.get("scheduled_utc", float("inf")) <= now + window_seconds:
            due.append(item)
    # Sort by scheduled time (earliest first)
    due.sort(key=lambda x: x.get("scheduled_utc", 0))
    return due


def mark_generating(item_id: str) -> None:
    """Mark an item as currently being processed."""
    items = _read_queue()
    for item in items:
        if item["id"] == item_id:
            item["status"] = "generating"
            _write_queue(items)
            return


def mark_done(item_id: str, tweet_url: str | None = None) -> None:
    """Mark an item as successfully posted."""
    items = _read_queue()
    for item in items:
        if item["id"] == item_id:
            item["status"] = "posted"
            item["posted_at"] = time.time()
            if tweet_url:
                item["tweet_url"] = tweet_url
            _write_queue(items)

            # Handle recurrence — create next occurrence
            if item.get("recurrence") == "daily":
                next_time = item["scheduled_utc"] + 86400
                add_scheduled(item["prompt"], next_time, "daily", item.get("label"))
            elif item.get("recurrence") == "weekly":
                next_time = item["scheduled_utc"] + 604800
                add_scheduled(item["prompt"], next_time, "weekly", item.get("label"))
            return


def mark_failed(item_id: str, reason: str = "") -> None:
    """Mark an item as failed."""
    items = _read_queue()
    for item in items:
        if item["id"] == item_id:
            item["status"] = "failed"
            item["failed_at"] = time.time()
            item["failure_reason"] = reason
            _write_queue(items)
            return


def prune_old(max_age_hours: int = 72) -> int:
    """Remove completed/cancelled/failed items older than max_age_hours.

    Uses the completion timestamp (posted_at/failed_at/cancelled_at) for
    finished items, falling back to created_at.

    Returns the number of pruned items.
    """
    cutoff = time.time() - (max_age_hours * 3600)
    items = _read_queue()
    original = len(items)

    def _keep(i: dict) -> bool:
        status = i.get("status")
        if status in ("pending", "generating"):
            return True
        # For finished items, use the completion timestamp
        finished_at = (
            i.get("posted_at")
            or i.get("failed_at")
            or i.get("cancelled_at")
            or i.get("created_at", 0)
        )
        return finished_at > cutoff

    items = [i for i in items if _keep(i)]
    pruned = original - len(items)
    if pruned:
        _write_queue(items)
        logger.info("Pruned %d old scheduled items", pruned)
    return pruned


# ---------------------------------------------------------------------------
# Natural language time parser
# ---------------------------------------------------------------------------

# Day-of-week name mapping
_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def parse_time(text: str, now: datetime | None = None) -> tuple[float | None, str]:
    """Parse a natural language time expression into a UTC timestamp.

    Supports:
    - "in 2 hours", "in 30 minutes", "in 1 hour 30 min"
    - "3pm", "3:30pm", "15:00"
    - "tomorrow 3pm", "tomorrow at 15:00"
    - "monday 9am", "friday at 3:30pm"
    - "2026-03-05 14:00", "2026-03-05T14:00"
    - "today 5pm"

    Args:
        text: The time expression to parse.
        now: Override current time for testing.

    Returns (utc_timestamp, human_readable_string) or (None, error_message).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    text = text.strip().lower()

    # --- Pattern 1: "in X hours/minutes" ---
    m = re.match(
        r"in\s+(?:(\d+)\s*h(?:ours?)?)?[\s,]*(?:(\d+)\s*m(?:in(?:utes?)?)?)?$",
        text,
    )
    if m and (m.group(1) or m.group(2)):
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        target = now + timedelta(hours=hours, minutes=minutes)
        return target.timestamp(), target.strftime("%Y-%m-%d %H:%M UTC")

    # --- Pattern 2: ISO datetime "2026-03-05 14:00" or "2026-03-05T14:00" ---
    m = re.match(r"(\d{4}-\d{2}-\d{2})[T\s](\d{1,2}:\d{2})", text)
    if m:
        try:
            target = datetime.strptime(
                f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone.utc)
            if target <= now:
                return None, "That time is in the past"
            return target.timestamp(), target.strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            return None, f"Could not parse date: {text}"

    # --- Extract time-of-day component ---
    time_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not time_match:
        return None, (
            "Could not parse time. Try: \"3pm\", \"in 2 hours\", "
            "\"tomorrow 9am\", \"monday 3:30pm\""
        )

    # Reject if there's unrecognized text after the time portion
    trailing = text[time_match.end():].strip()
    if trailing:
        return None, f"Could not parse: unexpected text after time"

    hour = int(time_match.group(1))
    minute = int(time_match.group(2) or 0)
    ampm = time_match.group(3)

    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif not ampm and hour < 8:
        # Assume PM for ambiguous single-digit hours (e.g., "3" → 3pm not 3am)
        hour += 12

    if hour > 23 or minute > 59:
        return None, f"Invalid time: {hour}:{minute:02d}"

    # --- Determine the target date ---
    # Strip the time portion to check for date keywords
    date_text = text[:time_match.start()].strip().rstrip("at").strip()

    if not date_text or date_text == "today":
        # Today
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            # Time already passed today → push to tomorrow
            target += timedelta(days=1)

    elif date_text == "tomorrow":
        target = (now + timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )

    elif date_text in _WEEKDAYS:
        # Next occurrence of the named weekday
        target_weekday = _WEEKDAYS[date_text]
        days_ahead = (target_weekday - now.weekday()) % 7
        if days_ahead == 0:
            # Same weekday — check if time has passed
            candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate <= now:
                days_ahead = 7
        target = (now + timedelta(days=days_ahead)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )

    else:
        return None, (
            f"Could not understand date \"{date_text}\". "
            "Try: \"today\", \"tomorrow\", or a weekday name."
        )

    return target.timestamp(), target.strftime("%Y-%m-%d %H:%M UTC")


def parse_schedule_command(text: str) -> tuple[str | None, float | None, str | None, str]:
    """Parse a /schedule command into (prompt, utc_timestamp, recurrence, error).

    Supported formats:
        /schedule 3pm tomorrow post about our new product
        /schedule in 2 hours something cool
        /schedule daily 9am morning engagement post
        /schedule weekly monday 3pm community update

    Returns (prompt, timestamp, recurrence, error_or_time_display).
    On failure, prompt and timestamp are None, and the last string is an error message.
    """
    text = text.strip()

    # Detect recurrence prefix
    recurrence = "once"
    if text.startswith("daily "):
        recurrence = "daily"
        text = text[6:].strip()
    elif text.startswith("weekly "):
        recurrence = "weekly"
        text = text[7:].strip()

    # Strategy: try progressively longer prefixes as the time expression,
    # and the rest as the prompt. We test 1-word, 2-word, ..., up to 5-word
    # prefixes.
    words = text.split()
    if not words:
        return None, None, None, "Usage: /schedule <time> <prompt>"

    best_ts = None
    best_display = ""
    best_split = 0

    for n in range(1, min(6, len(words)) + 1):
        time_part = " ".join(words[:n])
        ts, display = parse_time(time_part)
        if ts is not None:
            best_ts = ts
            best_display = display
            best_split = n

    if best_ts is None:
        return None, None, None, (
            "Could not parse time. Examples:\n"
            "  /schedule 3pm post about our launch\n"
            "  /schedule tomorrow 9am morning engagement\n"
            "  /schedule in 2 hours something cool\n"
            "  /schedule daily 3pm afternoon update"
        )

    prompt = " ".join(words[best_split:]).strip()
    if not prompt:
        return None, None, None, "Please include what to post after the time."

    return prompt, best_ts, recurrence, best_display
