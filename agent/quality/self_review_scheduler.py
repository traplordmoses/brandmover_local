"""
Self-review trigger logic.

Tracks approvals and triggers run_self_review() automatically:
- After every 10 approvals
- Or daily (>24h since last review)
- Whichever comes first

State lives in state/self_review_state.json.
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_STATE_FILE = _project_root / "state" / "self_review_state.json"

_APPROVAL_THRESHOLD = 10
_DAILY_INTERVAL = 24 * 60 * 60  # 24 hours


def _read_state() -> dict:
    """Read self_review_state.json."""
    if not _STATE_FILE.exists():
        return _default_state()
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        for key, default in _default_state().items():
            data.setdefault(key, default)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read self_review_state.json: %s", e)
        return _default_state()


def _write_state(data: dict) -> None:
    """Write state to self_review_state.json (atomic write)."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _STATE_FILE.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}")
    tmp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(str(tmp_path), str(_STATE_FILE))


def _default_state() -> dict:
    return {
        "approvals_since_last_review": 0,
        "last_review_at": 0.0,
        "total_reviews": 0,
    }


def record_approval() -> bool:
    """Increment approval counter. Returns True if a self-review should be triggered."""
    state = _read_state()
    state["approvals_since_last_review"] += 1
    _write_state(state)

    count = state["approvals_since_last_review"]
    logger.debug("Self-review scheduler: %d/%d approvals", count, _APPROVAL_THRESHOLD)
    return count >= _APPROVAL_THRESHOLD


def should_run_daily() -> bool:
    """Check if it's been >24h since the last review."""
    state = _read_state()
    last_review = state.get("last_review_at", 0)
    return (time.time() - last_review) > _DAILY_INTERVAL


def mark_review_complete() -> None:
    """Reset counter and update last_review_at after a review completes."""
    state = _read_state()
    state["approvals_since_last_review"] = 0
    state["last_review_at"] = time.time()
    state["total_reviews"] = state.get("total_reviews", 0) + 1
    _write_state(state)
    logger.info("Self-review scheduler: review recorded, counter reset")


async def maybe_trigger_review() -> bool:
    """Run a self-review if the approval threshold has been reached.

    Call this from _do_approve() after incrementing the counter.
    Runs in the background — never blocks the caller.
    Returns True if a review was triggered.
    """
    state = _read_state()
    if state["approvals_since_last_review"] < _APPROVAL_THRESHOLD:
        return False

    logger.info("Self-review: approval threshold reached (%d), triggering background review",
                state["approvals_since_last_review"])
    asyncio.create_task(_run_review_background())
    return True


async def maybe_trigger_daily_review() -> bool:
    """Run a self-review if >24h since last one.

    Call this from the auto_post scheduler loop.
    Returns True if a review was triggered.
    """
    if not should_run_daily():
        return False

    # Also skip if there's no meaningful data (fewer than 3 approvals total)
    state = _read_state()
    if state["approvals_since_last_review"] < 1 and state["last_review_at"] > 0:
        return False

    logger.info("Self-review: daily interval exceeded, triggering background review")
    asyncio.create_task(_run_review_background())
    return True


async def _run_review_background() -> None:
    """Run self-review in the background. Never raises — logs errors."""
    try:
        from agent.self_review import run_self_review
        result = await run_self_review()
        if result.get("error"):
            logger.warning("Background self-review completed with error: %s", result["error"])
        else:
            mark_review_complete()
            logger.info("Background self-review completed successfully")
    except Exception as e:
        logger.error("Background self-review failed: %s", e)
