"""
State management for the auto-posting scheduler.
Separate from Telegram's state.json to avoid race conditions.
Tracks daily post counts, posted event IDs, prompt rotation, and recent captions.
"""

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from config import settings
from agent.state_manager import FileStore

logger = logging.getLogger(__name__)

_auto_lock = threading.Lock()

_STATE_FILE = Path(settings.AUTO_POST_STATE_FILE)

# Migrate from old location if needed
from agent.paths import migrate_state_file
migrate_state_file(
    Path(__file__).resolve().parent.parent / "auto_post_state.json",
    _STATE_FILE,
)

# ---------------------------------------------------------------------------
# File I/O — delegated to FileStore
# ---------------------------------------------------------------------------

_store = FileStore(_STATE_FILE, default_factory=lambda: _default_state())


def _get_store() -> FileStore:
    """Return the active FileStore, respecting any monkey-patching of _STATE_FILE."""
    global _store
    if _store.path != _STATE_FILE:
        _store = FileStore(_STATE_FILE, default_factory=lambda: _default_state())
    return _store


def _read_state() -> dict:
    """Read auto_post_state.json, return empty structure if missing or corrupt."""
    data = _get_store().read()
    # Ensure all required keys exist
    for key, default in _default_state().items():
        data.setdefault(key, default)
    return data


def _write_state(data: dict) -> None:
    """Write state dict to auto_post_state.json."""
    _get_store().write(data)


def _default_state() -> dict:
    return {
        "posts_today": [],
        "posted_event_ids": [],
        "rotation_indices": {},
        "recent_captions": [],
        "paused": False,
        "last_post_timestamp": 0,
        "dead_letters": [],
    }


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def can_post(min_gap_minutes: int = 120, max_posts_per_day: int = 6, ignore_paused: bool = False) -> tuple[bool, str]:
    """Check if a post is allowed right now.

    Returns (allowed, reason) where reason explains why posting is blocked.
    When ignore_paused=True, skips the pause check (for user-scheduled posts).
    """
    state = _read_state()

    if state.get("paused") and not ignore_paused:
        return False, "Auto-posting is paused"

    # Check daily limit
    today = _today_key()
    today_posts = [p for p in state["posts_today"] if p.get("date") == today]
    if len(today_posts) >= max_posts_per_day:
        return False, f"Daily limit reached ({len(today_posts)}/{max_posts_per_day})"

    # Check minimum gap
    last_ts = state.get("last_post_timestamp", 0)
    elapsed = time.time() - last_ts
    gap_seconds = min_gap_minutes * 60
    if elapsed < gap_seconds:
        remaining = int((gap_seconds - elapsed) / 60)
        return False, f"Too soon since last post ({remaining}min remaining of {min_gap_minutes}min gap)"

    return True, "OK"


def is_slot_posted(slot_name: str) -> bool:
    """Check if a slot has already been posted today."""
    state = _read_state()
    today = _today_key()
    return any(
        p.get("slot") == slot_name and p.get("date") == today
        for p in state["posts_today"]
    )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def is_duplicate_caption(caption: str, threshold: float = 0.6) -> bool:
    """Check if caption is too similar to recent captions using word overlap.

    Returns True if similarity exceeds threshold against any of the last 20 captions.
    """
    state = _read_state()
    recent = state.get("recent_captions", [])
    if not recent:
        return False

    caption_words = set(_normalize_words(caption))
    if not caption_words:
        return False

    for prev in recent[-20:]:
        prev_words = set(_normalize_words(prev))
        if not prev_words:
            continue
        overlap = len(caption_words & prev_words)
        union = len(caption_words | prev_words)
        similarity = overlap / union if union else 0
        if similarity > threshold:
            logger.warning(
                "Duplicate caption detected (similarity=%.2f): %s",
                similarity, caption[:80],
            )
            return True

    return False


def _normalize_words(text: str) -> list[str]:
    """Extract lowercase words, strip punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Recording posts
# ---------------------------------------------------------------------------


def record_post(
    slot_name: str,
    caption: str,
    tweet_url: str | None = None,
    event_ids: list[str] | None = None,
) -> None:
    """Record a successful post."""
    with _auto_lock:
        state = _read_state()
        now = time.time()
        today = _today_key()

        entry = {
            "slot": slot_name,
            "date": today,
            "timestamp": now,
            "caption": caption[:200],
            "tweet_url": tweet_url,
        }
        state["posts_today"].append(entry)
        state["last_post_timestamp"] = now

        # Track recent captions for duplicate detection (keep last 20)
        state["recent_captions"].append(caption)
        state["recent_captions"] = state["recent_captions"][-20:]

        # Track posted event IDs to avoid re-commenting on the same on-chain events
        if event_ids:
            state["posted_event_ids"].extend(event_ids)
            state["posted_event_ids"] = state["posted_event_ids"][-100:]

        # Auto-prune entries older than 48h
        cutoff = now - (48 * 3600)
        state["posts_today"] = [
            p for p in state["posts_today"] if p.get("timestamp", 0) > cutoff
        ]

        _write_state(state)
    logger.info("Recorded auto-post: slot=%s, tweet=%s", slot_name, tweet_url)


# ---------------------------------------------------------------------------
# Rotation indices for prompt variety
# ---------------------------------------------------------------------------


def get_rotation_index(category: str) -> int:
    """Get the current rotation index for a prompt category."""
    state = _read_state()
    return state.get("rotation_indices", {}).get(category, 0)


def advance_rotation(category: str, pool_size: int) -> int:
    """Advance rotation index and return the new value."""
    with _auto_lock:
        state = _read_state()
        indices = state.get("rotation_indices", {})
        current = indices.get(category, 0)
        next_idx = (current + 1) % pool_size
        indices[category] = next_idx
        state["rotation_indices"] = indices
        _write_state(state)
    return next_idx


# ---------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------


def is_paused() -> bool:
    """Check if auto-posting is paused."""
    return _read_state().get("paused", False)


def set_paused(paused: bool) -> None:
    """Set the paused state."""
    with _auto_lock:
        state = _read_state()
        state["paused"] = paused
        _write_state(state)
    logger.info("Auto-posting %s", "paused" if paused else "resumed")


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------


def get_status_summary() -> dict:
    """Return a summary of auto-post state for /autostatus."""
    state = _read_state()
    today = _today_key()
    today_posts = [p for p in state["posts_today"] if p.get("date") == today]
    dead_letters = state.get("dead_letters", [])
    today_failures = [d for d in dead_letters if d.get("date") == today]
    return {
        "paused": state.get("paused", False),
        "posts_today": len(today_posts),
        "last_post_timestamp": state.get("last_post_timestamp", 0),
        "recent_slots": [p.get("slot") for p in today_posts],
        "recent_captions": [p.get("caption", "")[:60] for p in today_posts[-3:]],
        "dead_letters_total": len(dead_letters),
        "dead_letters_today": len(today_failures),
    }


# ---------------------------------------------------------------------------
# Dead-letter queue — failed generation attempts for inspection and retry
# ---------------------------------------------------------------------------

_MAX_DEAD_LETTERS = 50  # Keep at most this many entries


def record_failure(slot_name: str, error: str, retry_count: int = 0) -> None:
    """Record a failed generation attempt in the dead-letter queue.

    Stores the slot name, error message, retry count, and timestamp so
    operators can inspect failures via /autostatus and optionally retry.
    """
    with _auto_lock:
        state = _read_state()
        dead_letters = state.get("dead_letters", [])
        dead_letters.append({
            "slot_name": slot_name,
            "error": str(error)[:500],
            "retry_count": retry_count,
            "timestamp": time.time(),
            "date": _today_key(),
        })
        # Prune oldest entries if over the limit
        if len(dead_letters) > _MAX_DEAD_LETTERS:
            dead_letters = dead_letters[-_MAX_DEAD_LETTERS:]
        state["dead_letters"] = dead_letters
        _write_state(state)
    logger.warning(
        "Dead-letter recorded: slot=%s error=%s retries=%d",
        slot_name, str(error)[:100], retry_count,
    )


def get_dead_letters(limit: int = 10) -> list[dict]:
    """Return the most recent dead-letter entries (newest first)."""
    state = _read_state()
    dead_letters = state.get("dead_letters", [])
    return list(reversed(dead_letters[-limit:]))


def retry_dead_letter(index: int) -> dict | None:
    """Remove a dead-letter entry by index (0-based, newest first) and return it.

    Returns the removed entry so the caller can re-queue it, or None if the
    index is out of range.
    """
    with _auto_lock:
        state = _read_state()
        dead_letters = state.get("dead_letters", [])
        if not dead_letters:
            return None
        # index is newest-first, so convert to list index
        reversed_idx = len(dead_letters) - 1 - index
        if reversed_idx < 0 or reversed_idx >= len(dead_letters):
            return None
        entry = dead_letters.pop(reversed_idx)
        state["dead_letters"] = dead_letters
        _write_state(state)
    logger.info("Dead-letter retried: slot=%s", entry.get("slot_name"))
    return entry


def clear_dead_letters() -> int:
    """Clear all dead-letter entries. Returns the number removed."""
    with _auto_lock:
        state = _read_state()
        count = len(state.get("dead_letters", []))
        state["dead_letters"] = []
        _write_state(state)
    logger.info("Dead-letter queue cleared (%d entries)", count)
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today_key() -> str:
    """Return today's date as YYYY-MM-DD in UTC."""
    return time.strftime("%Y-%m-%d", time.gmtime())
