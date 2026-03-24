"""
Growth engine — target account management, follower tracking, and reply drafting.

State is persisted in state/growth_state.json with the following shape:
{
    "targets": [{"username": str, "reason": str, "added_at": float}],
    "follower_history": [{"count": int, "timestamp": float}],
    "last_follower_check": float
}
"""

import json
import logging
import time
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROWTH_STATE_PATH = _PROJECT_ROOT / "state" / "growth_state.json"

# Feature flag — set GROWTH_ENGINE_ENABLED=true in .env to enable
GROWTH_ENGINE_ENABLED: bool = getattr(settings, "GROWTH_ENGINE_ENABLED", False)


def _read_state() -> dict:
    """Load growth state from disk."""
    if not GROWTH_STATE_PATH.exists():
        return {"targets": [], "follower_history": [], "last_follower_check": 0}
    try:
        return json.loads(GROWTH_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read growth state: %s", e)
        return {"targets": [], "follower_history": [], "last_follower_check": 0}


def _write_state(data: dict) -> None:
    """Persist growth state to disk."""
    GROWTH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROWTH_STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Target account management
# ---------------------------------------------------------------------------


def list_targets() -> list[dict]:
    """Return list of target accounts."""
    return _read_state().get("targets", [])


def add_target(username: str, reason: str = "") -> dict:
    """Add a target account. Returns the new entry."""
    username = username.lstrip("@").lower()
    st = _read_state()
    targets = st.get("targets", [])

    # Check for duplicates
    for t in targets:
        if t["username"] == username:
            return {"error": f"@{username} is already a target", "existing": t}

    entry = {"username": username, "reason": reason, "added_at": time.time()}
    targets.append(entry)
    st["targets"] = targets
    _write_state(st)
    return entry


def remove_target(username: str) -> bool:
    """Remove a target account. Returns True if found and removed."""
    username = username.lstrip("@").lower()
    st = _read_state()
    targets = st.get("targets", [])
    original_len = len(targets)
    targets = [t for t in targets if t["username"] != username]
    if len(targets) == original_len:
        return False
    st["targets"] = targets
    _write_state(st)
    return True


# ---------------------------------------------------------------------------
# Follower tracking
# ---------------------------------------------------------------------------


def record_follower_count(count: int) -> None:
    """Record a follower count data point."""
    st = _read_state()
    history = st.get("follower_history", [])
    history.append({"count": count, "timestamp": time.time()})
    # Keep last 365 data points
    if len(history) > 365:
        history = history[-365:]
    st["follower_history"] = history
    st["last_follower_check"] = time.time()
    _write_state(st)


def get_follower_history(days: int = 30) -> list[dict]:
    """Get follower count history for the last N days."""
    cutoff = time.time() - (days * 86400)
    st = _read_state()
    return [h for h in st.get("follower_history", []) if h["timestamp"] > cutoff]


def get_growth_stats(days: int = 7) -> dict:
    """Calculate growth stats for the last N days."""
    history = get_follower_history(days=days)
    if len(history) < 2:
        return {
            "current": history[-1]["count"] if history else None,
            "change": None,
            "pct_change": None,
            "data_points": len(history),
        }

    current = history[-1]["count"]
    start = history[0]["count"]
    change = current - start
    pct_change = (change / start * 100) if start > 0 else 0.0

    return {
        "current": current,
        "start": start,
        "change": change,
        "pct_change": round(pct_change, 2),
        "data_points": len(history),
        "period_days": days,
    }


def is_growth_stalling(threshold_pct: float = 1.0) -> bool:
    """Check if weekly growth is below the threshold percentage."""
    stats = get_growth_stats(days=7)
    if stats["pct_change"] is None:
        return False  # Not enough data to determine
    return stats["pct_change"] < threshold_pct


# ---------------------------------------------------------------------------
# Follower tracking via Twitter API
# ---------------------------------------------------------------------------


async def track_follower_growth() -> dict | None:
    """Fetch current follower count from Twitter API and record it.

    Returns the growth stats dict or None if the API call fails.
    """
    try:
        from agent.publishing.publisher import _get_client_v2
        client = _get_client_v2()
        me = client.get_me(user_fields=["public_metrics"])
        if me and me.data:
            metrics = me.data.get("public_metrics", {})
            count = metrics.get("followers_count", 0)
            record_follower_count(count)
            logger.info("Growth engine: recorded follower count %d", count)
            return get_growth_stats(days=7)
    except Exception as e:
        logger.debug("Growth engine: failed to fetch follower count: %s", e)
    return None


# ---------------------------------------------------------------------------
# Growth dashboard summary
# ---------------------------------------------------------------------------


def get_growth_dashboard() -> dict:
    """Assemble a growth dashboard summary."""
    weekly = get_growth_stats(days=7)
    monthly = get_growth_stats(days=30)
    targets = list_targets()

    # Determine suggested next action
    suggestion = "Add target accounts with /target_add to start monitoring."
    if targets and weekly.get("pct_change") is not None:
        if weekly["pct_change"] < 1.0:
            suggestion = "Growth is slow. Try a growth thread (/growth_thread) or engage targets with /replies."
        elif weekly["pct_change"] < 5.0:
            suggestion = "Steady growth. Keep engaging targets and posting threads."
        else:
            suggestion = "Strong growth! Double down on what is working."
    elif not targets:
        suggestion = "Add target accounts with /target_add @username to start monitoring."

    return {
        "weekly": weekly,
        "monthly": monthly,
        "target_count": len(targets),
        "suggestion": suggestion,
    }
