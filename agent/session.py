"""
Persistent session memory for the agent.
Stores recent posts, rejections, learned preferences, and last run metadata.
Loaded at the start of each agent run and injected as context.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from agent.paths import STATE_DIR as _STATE_DIR
SESSION_PATH = _STATE_DIR / "agent_session.json"

# Limits to keep context window lean
MAX_RECENT_POSTS = 7
MAX_REJECTED_DRAFTS = 5
MAX_LEARNED_PREFERENCES = 15


@dataclass
class AgentSession:
    recent_posts: list[dict] = field(default_factory=list)
    # Each: {caption: str, slot: str, timestamp: float, tweet_url: str|None}

    rejected_drafts: list[dict] = field(default_factory=list)
    # Each: {caption: str, feedback: str, slot: str, timestamp: float}

    learned_preferences: list[str | dict] = field(default_factory=list)
    # Each entry: str (legacy) or {"text": str, "added": float, "source": str}
    # Temporal decay: newer preferences are weighted higher in context assembly

    last_run: dict = field(default_factory=dict)
    # {slot: str, turns_used: int, tools_called: list, finished_via: str, timestamp: float}

    last_preference_extraction: float | None = None
    # Timestamp of last auto preference extraction run


# In-memory cache for session file to avoid re-reading on every call
_cached_session: AgentSession | None = None
_cache_mtime: float = 0.0


def load_session() -> AgentSession:
    """Read session from disk. Returns fresh session if missing or corrupt.

    Uses mtime-based in-memory caching to avoid re-reading on every call.
    """
    global _cached_session, _cache_mtime
    try:
        if SESSION_PATH.exists():
            mtime = os.stat(SESSION_PATH).st_mtime
            if _cached_session is not None and mtime == _cache_mtime:
                return _cached_session
            data = json.loads(SESSION_PATH.read_text())
            session = AgentSession(
                recent_posts=data.get("recent_posts", []),
                rejected_drafts=data.get("rejected_drafts", []),
                learned_preferences=data.get("learned_preferences", []),
                last_run=data.get("last_run", {}),
                last_preference_extraction=data.get("last_preference_extraction"),
            )
            _cached_session = session
            _cache_mtime = mtime
            return session
    except Exception as e:
        logger.warning("Failed to load session from %s: %s", SESSION_PATH, e)
    return AgentSession()


def save_session(session: AgentSession) -> None:
    """Write session to disk, pruning to stay within limits. Invalidates cache."""
    global _cached_session, _cache_mtime
    # Prune to caps (keep most recent)
    session.recent_posts = session.recent_posts[-MAX_RECENT_POSTS:]
    session.rejected_drafts = session.rejected_drafts[-MAX_REJECTED_DRAFTS:]
    session.learned_preferences = session.learned_preferences[-MAX_LEARNED_PREFERENCES:]

    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp_path = SESSION_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(asdict(session), indent=2, default=str))
        os.replace(str(tmp_path), str(SESSION_PATH))
        _cached_session = session
        _cache_mtime = os.stat(SESSION_PATH).st_mtime
    except Exception as e:
        logger.error("Failed to save session to %s: %s", SESSION_PATH, e)


def record_approved_post(
    caption: str, slot: str = "", tweet_url: str | None = None
) -> None:
    """Record an approved post in session memory."""
    session = load_session()
    session.recent_posts.append({
        "caption": caption,
        "slot": slot,
        "timestamp": time.time(),
        "tweet_url": tweet_url,
    })
    save_session(session)
    logger.info("Session: recorded approved post (slot=%s)", slot)


def record_rejected_draft(
    caption: str, feedback: str, slot: str = ""
) -> None:
    """Record a rejected draft in session memory."""
    session = load_session()
    session.rejected_drafts.append({
        "caption": caption,
        "feedback": feedback,
        "slot": slot,
        "timestamp": time.time(),
    })
    save_session(session)
    logger.info("Session: recorded rejected draft (slot=%s)", slot)


def record_run(
    slot: str,
    turns_used: int,
    tools_called: list[str],
    finished_via: str,
) -> None:
    """Record metadata about the latest agent run."""
    session = load_session()
    session.last_run = {
        "slot": slot,
        "turns_used": turns_used,
        "tools_called": tools_called,
        "finished_via": finished_via,
        "timestamp": time.time(),
    }
    save_session(session)


def _pref_text(p) -> str:
    """Extract text from a preference entry (str or dict)."""
    if isinstance(p, dict):
        return p.get("text", "")
    return str(p)


def add_learned_preference(preference: str, source: str = "operator") -> bool:
    """Add a preference (deduped). Returns True if added, False if duplicate."""
    session = load_session()
    stripped = preference.strip()
    # Dedupe against both legacy strings and new dicts
    existing_texts = {_pref_text(p) for p in session.learned_preferences}
    if stripped in existing_texts:
        return False
    session.learned_preferences.append({
        "text": stripped,
        "added": time.time(),
        "source": source,
    })
    save_session(session)
    logger.info("Session: added preference %r (source=%s)", stripped, source)
    return True


def remove_learned_preference(index: int) -> str | None:
    """Remove a preference by index. Returns the removed text or None."""
    session = load_session()
    if 0 <= index < len(session.learned_preferences):
        removed = session.learned_preferences.pop(index)
        save_session(session)
        removed_text = _pref_text(removed)
        logger.info("Session: removed preference #%d %r", index, removed_text)
        return removed_text
    return None


# ---------------------------------------------------------------------------
# Context builder — formats session for agent injection
# ---------------------------------------------------------------------------

def _relative_time(ts: float) -> str:
    """Convert a unix timestamp to a human-readable relative time string."""
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = int(delta / 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if delta < 86400:
        h = int(delta / 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = int(delta / 86400)
    return f"{d} day{'s' if d != 1 else ''} ago"


def build_session_context() -> str:
    """Build formatted session context for injection into agent messages.

    Returns empty string if session has no meaningful data.
    """
    session = load_session()
    sections: list[str] = []

    # Recent approved posts (newest first, max 5 shown)
    if session.recent_posts:
        lines = []
        for post in reversed(session.recent_posts[-5:]):
            caption = post.get("caption", "")[:120]
            slot = post.get("slot", "unknown")
            ts = post.get("timestamp", 0)
            lines.append(f'- "{caption}" (slot: {slot}, {_relative_time(ts)})')
        sections.append(
            "Recent approved posts (newest first):\n" + "\n".join(lines)
        )

    # Recent rejections (max 3 shown)
    if session.rejected_drafts:
        lines = []
        for draft in reversed(session.rejected_drafts[-3:]):
            fb = draft.get("feedback", "")[:200]
            ts = draft.get("timestamp", 0)
            lines.append(f'- Draft rejected {_relative_time(ts)} — Feedback: "{fb}"')
        sections.append("Recent rejections:\n" + "\n".join(lines))

    # Learned preferences (sorted by recency, with decay indicator)
    if session.learned_preferences:
        prefs_with_age = []
        for p in session.learned_preferences:
            text = _pref_text(p)
            if isinstance(p, dict) and p.get("added"):
                age_days = (time.time() - p["added"]) / 86400
                if age_days > 60:
                    text += " (older)"  # Signal to agent that this may be stale
                prefs_with_age.append((p.get("added", 0), text))
            else:
                prefs_with_age.append((0, text))
        # Sort newest first
        prefs_with_age.sort(key=lambda x: x[0], reverse=True)
        lines = [f"- {text}" for _, text in prefs_with_age]
        sections.append("Learned preferences:\n" + "\n".join(lines))

    # Last run
    if session.last_run:
        lr = session.last_run
        slot = lr.get("slot", "unknown")
        turns = lr.get("turns_used", "?")
        via = lr.get("finished_via", "?")
        ts = lr.get("timestamp", 0)
        sections.append(
            f"Last run: {slot}, {turns} turns, {_relative_time(ts)}, finished via {via}"
        )

    if not sections:
        return ""

    return "CONTEXT FROM RECENT ACTIVITY:\n\n" + "\n\n".join(sections)
