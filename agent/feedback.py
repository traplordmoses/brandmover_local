"""
Feedback system — logs approvals/rejections and learns brand preferences over time.

ARCHITECTURE:
This implements a CLOSED-LOOP LEARNING system. Every time the operator approves
or rejects a draft, we log the full context (what was requested, what was generated,
whether it was approved, and any feedback text). Periodically, Claude analyzes
all recent feedback and generates a condensed "learned preferences" guide.

This guide is then injected into the system prompt for ALL future generations,
so the agent naturally adapts its style over time — without any model fine-tuning
or retraining. It's pure prompt engineering with a feedback loop.

STORAGE:
- feedback.json: Append-only log of all feedback entries (never deleted, only grows).
  Each entry: {request, draft, accepted, feedback_text, resources_used, timestamp}
- learned_preferences.md: LLM-generated summary of patterns (overwritten each time).
  This is what gets injected into the system prompt.

INTERVIEW TALKING POINT:
"We built a self-improving content agent using a feedback summarization loop.
Every approval/rejection is logged, then periodically Claude analyzes the patterns
and writes a concise preference guide. This guide is injected into every future
system prompt, so the agent gets better at matching the operator's taste over time —
no fine-tuning needed. It's a closed-loop system that improves through prompt context."
"""

import asyncio
import copy
import json
import logging
import os
import threading
import time
from pathlib import Path

import anthropic

from config import settings

logger = logging.getLogger(__name__)

# In-memory cache for feedback.json to avoid re-reading on every call
_cached_feedback: list[dict] | None = None
_feedback_cache_mtime: float = 0.0

# Threading lock for sync read-modify-write functions
_feedback_lock = threading.Lock()

# File paths — all state lives in state/ directory
_project_root = Path(__file__).resolve().parent.parent
from agent.paths import STATE_DIR as _STATE_DIR
_FEEDBACK_FILE = _STATE_DIR / "feedback.json"        # Append-only feedback log
_PREFERENCES_FILE = _STATE_DIR / "learned_preferences.md"  # Claude-generated summary

# ── Migrate from old locations (pre-refactor) ──
# Early versions stored these files in the project root. This block
# automatically moves them to state/ on first run after upgrade.
for _old, _new in [
    (_project_root / "feedback.json", _FEEDBACK_FILE),
    (_project_root / "learned_preferences.md", _PREFERENCES_FILE),
]:
    if _old.exists() and not _new.exists():
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        _shutil.move(str(_old), str(_new))


def _read_feedback() -> list[dict]:
    """Read the feedback log from disk. Returns empty list if missing or corrupt.

    Uses mtime-based in-memory caching to avoid re-reading on every call.
    Defensive: catches JSON parse errors and file I/O errors rather than crashing.
    The bot should keep working even if the feedback file is corrupted.
    """
    global _cached_feedback, _feedback_cache_mtime
    if not _FEEDBACK_FILE.exists():
        return []
    try:
        mtime = os.stat(_FEEDBACK_FILE).st_mtime
        if _cached_feedback is not None and mtime == _feedback_cache_mtime:
            return copy.deepcopy(_cached_feedback)
        data = json.loads(_FEEDBACK_FILE.read_text(encoding="utf-8"))
        _cached_feedback = data
        _feedback_cache_mtime = mtime
        return copy.deepcopy(data)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read feedback.json: %s", e)
        return []


def _write_feedback(entries: list[dict]) -> None:
    """Write the full feedback log to disk and update in-memory cache."""
    global _cached_feedback, _feedback_cache_mtime
    _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _FEEDBACK_FILE.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}")
    tmp_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(str(tmp_path), str(_FEEDBACK_FILE))
    _cached_feedback = copy.deepcopy(entries)
    _feedback_cache_mtime = os.stat(_FEEDBACK_FILE).st_mtime


def log_feedback(
    request: str,
    draft: dict,
    accepted: bool,
    feedback_text: str = "",
    resources_used: list[str] | None = None,
    tags: list[str] | None = None,
) -> int:
    """Append a feedback entry to the log. Returns the new total count.

    Called by the /approve and /reject handlers in bot/handlers.py.
    Each entry captures the full context of the decision:
    - request: What the user originally asked for
    - draft: The full generated draft (caption, image_prompt, etc.)
    - accepted: True = approved, False = rejected
    - feedback_text: Optional reason (e.g., "too formal", "love the image")
    - resources_used: Which APIs/files were consulted during generation
    - tags: Structured tags for retrieval (e.g., ["content_type:meme", "mood:casual"])
    """
    with _feedback_lock:
        entries = _read_feedback()
        entry: dict = {
            "request": request,
            "draft": draft,
            "accepted": accepted,
            "feedback_text": feedback_text,
            "resources_used": resources_used or [],
            "timestamp": time.time(),
        }
        if tags:
            entry["tags"] = tags
        entries.append(entry)
        _write_feedback(entries)
        count = len(entries)
        logger.info("Logged feedback #%d (accepted=%s)", count, accepted)
        return count


def get_feedback_context() -> str:
    """Return recent feedback history + learned preferences as a formatted string.

    This is injected into the system prompt so the agent knows:
    1. What patterns the user has approved/rejected (from learned_preferences.md)
    2. The raw last 10 entries (so the agent can see recent decisions)

    The combination gives the agent both high-level patterns AND recent specifics.
    """
    parts = []

    # Learned preferences — Claude-generated summary of all past feedback
    if _PREFERENCES_FILE.exists():
        prefs = _PREFERENCES_FILE.read_text(encoding="utf-8").strip()
        if prefs:
            parts.append(f"--- LEARNED PREFERENCES ---\n{prefs}")

    # Recent feedback — last 10 raw entries for immediate context.
    # Shows the agent exactly what was approved/rejected recently.
    entries = _read_feedback()
    if entries:
        recent = entries[-10:]
        lines = []
        for e in recent:
            status = "APPROVED" if e["accepted"] else "REJECTED"
            fb = f" — Feedback: {e['feedback_text']}" if e.get("feedback_text") else ""
            caption = e.get("draft", {}).get("caption", "")[:100]
            lines.append(f"[{status}] Request: {e['request'][:80]} | Caption: {caption}{fb}")
        parts.append("--- RECENT FEEDBACK (last 10) ---\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "No feedback history yet."


def get_feedback_stats() -> str:
    """Human-readable stats for the /feedback command.

    Shows: total drafts reviewed, approval rate, recent rejection reasons,
    and whether learned preferences have been generated.
    """
    entries = _read_feedback()
    if not entries:
        return "No feedback recorded yet. Generate content and use /approve or /reject to start building preferences."

    total = len(entries)
    approved = sum(1 for e in entries if e["accepted"])
    rejected = total - approved
    rate = (approved / total * 100) if total else 0

    lines = [
        f"Total drafts reviewed: {total}",
        f"Approved: {approved} | Rejected: {rejected}",
        f"Approval rate: {rate:.0f}%",
    ]

    # Show recent rejection reasons — these are the most actionable feedback
    rejections = [e for e in entries if not e["accepted"] and e.get("feedback_text")]
    if rejections:
        lines.append("\nRecent rejection reasons:")
        for e in rejections[-5:]:
            lines.append(f"  - {e['feedback_text'][:100]}")

    # Preferences status — preferences are now auto-extracted daily via pref_extractor.py
    try:
        from agent.session import load_session
        session = load_session()
        pref_count = len(session.learned_preferences)
        if pref_count:
            lines.append(f"\nLearned preferences: {pref_count} active (auto-extracted daily from approval/rejection patterns, see /preferences)")
        else:
            lines.append("\nLearned preferences: None yet (auto-extracted daily from approval/rejection patterns, see /preferences)")
    except Exception:
        lines.append("\nLearned preferences: Use /preferences to view")

    return "\n".join(lines)


async def summarize_preferences() -> str:
    """Call Claude to analyze feedback.json and generate learned_preferences.md.

    THIS IS THE LEARNING STEP. It:
    1. Reads the last 50 feedback entries
    2. Sends them to Claude with a prompt asking to extract patterns
    3. Claude writes a concise guide of what the user likes/dislikes
    4. We save this guide to learned_preferences.md
    5. On every future brain call, this guide is loaded into the system prompt

    Triggered automatically every FEEDBACK_SUMMARIZE_EVERY entries (default: 10),
    or manually via the /learn command.
    """
    entries = _read_feedback()
    if not entries:
        return "No feedback to analyze yet."

    # Only analyze last 50 entries — enough to find patterns without
    # overwhelming Claude's context window.
    feedback_text = json.dumps(entries[-50:], indent=2)

    from agent._client import get_anthropic
    client = get_anthropic()
    response = await client.messages.create(
        model=settings.HAIKU_MODEL,
        max_tokens=1500,
        system="You analyze content feedback logs and extract patterns about brand preferences.",
        messages=[{
            "role": "user",
            "content": (
                f"Analyze this feedback log for {settings.BRAND_NAME} social media content. "
                "Extract patterns about what the user approves vs rejects. "
                "Write a concise guide (under 500 words) of learned preferences that a content agent should follow. "
                "Focus on: tone preferences, length preferences, hashtag preferences, image style preferences, "
                "common rejection reasons, and any specific dos/don'ts.\n\n"
                f"FEEDBACK LOG:\n{feedback_text}"
            ),
        }],
    )

    # Save the generated summary — this overwrites any previous version.
    # The file is loaded by unified_prompt.py on every brain call.
    summary = response.content[0].text
    with _feedback_lock:
        tmp_path = _PREFERENCES_FILE.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}")
        tmp_path.write_text(summary, encoding="utf-8")
        os.replace(str(tmp_path), str(_PREFERENCES_FILE))
    logger.info("Updated learned_preferences.md (%d chars)", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Async wrappers — non-blocking versions for use in bot handlers.
# Blocking file I/O (read/write JSON) is wrapped in asyncio.to_thread()
# so it doesn't block the Telegram event loop while other users' messages
# are being processed.
# ---------------------------------------------------------------------------

async def async_log_feedback(*args, **kwargs) -> int:
    return await asyncio.to_thread(log_feedback, *args, **kwargs)
