"""
Session transcript logger — JSONL per-user logs of all agent interactions.

Every agent interaction (user message, agent response, tool call, approval,
rejection) is logged to a JSONL file in state/transcripts/{user_id}.jsonl.
This provides:
- Full debugging capability
- Training data for potential fine-tuning
- Analytics on agent usage patterns

Each line is a JSON object with: timestamp, event_type, and event-specific data.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from agent.paths import STATE_DIR as _STATE_DIR
_TRANSCRIPTS_DIR = _STATE_DIR / "transcripts"


def _get_file(user_id: int | str) -> Path:
    """Get transcript file path for a user. Forces integer ID to prevent path traversal."""
    _TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = str(int(user_id))  # Force integer — rejects path-like strings
    return _TRANSCRIPTS_DIR / f"{safe_id}.jsonl"


def log_event(
    user_id: int | str,
    event_type: str,
    data: dict | None = None,
) -> None:
    """Append a single event to the user's transcript."""
    entry = {
        "timestamp": time.time(),
        "event_type": event_type,
        **(data or {}),
    }
    try:
        with open(_get_file(user_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning("Failed to write transcript for user %s: %s", user_id, e)


def log_user_message(user_id: int | str, text: str, **kwargs) -> None:
    """Log an incoming user message."""
    log_event(user_id, "user_message", {"text": text[:2000], **kwargs})


def log_agent_response(user_id: int | str, text: str, **kwargs) -> None:
    """Log an agent response."""
    log_event(user_id, "agent_response", {"text": text[:2000], **kwargs})


def log_tool_call(user_id: int | str, tool_name: str, duration_ms: float = 0, **kwargs) -> None:
    """Log a tool call during agent execution."""
    log_event(user_id, "tool_call", {"tool": tool_name, "duration_ms": duration_ms, **kwargs})


def log_draft_action(user_id: int | str, action: str, caption: str = "", feedback: str = "", **kwargs) -> None:
    """Log a draft approval/rejection."""
    log_event(user_id, "draft_action", {"action": action, "caption": caption[:500], "feedback": feedback[:500], **kwargs})


def log_publish(user_id: int | str, platform: str, url: str = "", **kwargs) -> None:
    """Log a post publication."""
    log_event(user_id, "publish", {"platform": platform, "url": url, **kwargs})


def get_recent_events(user_id: int | str, n: int = 50, event_type: str | None = None) -> list[dict]:
    """Read the last N events from a user's transcript. Optionally filter by type."""
    path = _get_file(user_id)
    if not path.exists():
        return []

    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if event_type is None or entry.get("event_type") == event_type:
                        events.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    return events[-n:]


def get_transcript_stats(user_id: int | str) -> dict:
    """Get summary stats for a user's transcript."""
    path = _get_file(user_id)
    if not path.exists():
        return {"total_events": 0, "file_size_kb": 0}

    counts: dict[str, int] = {}
    total = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    et = entry.get("event_type", "unknown")
                    counts[et] = counts.get(et, 0) + 1
                    total += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass

    return {
        "total_events": total,
        "by_type": counts,
        "file_size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
    }
