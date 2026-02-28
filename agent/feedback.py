"""
Feedback system — logs approvals/rejections and learns brand preferences over time.

Storage:
- feedback.json: append-only log of all feedback entries
- learned_preferences.md: LLM-generated summary of patterns (overwritten each time)
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import anthropic

from config import settings

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_STATE_DIR = _project_root / "state"
_FEEDBACK_FILE = _STATE_DIR / "feedback.json"
_PREFERENCES_FILE = _STATE_DIR / "learned_preferences.md"

# Migrate from old locations if needed
for _old, _new in [
    (_project_root / "feedback.json", _FEEDBACK_FILE),
    (_project_root / "learned_preferences.md", _PREFERENCES_FILE),
]:
    if _old.exists() and not _new.exists():
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        _shutil.move(str(_old), str(_new))


def _read_feedback() -> list[dict]:
    """Read the feedback log. Returns empty list if missing or corrupt."""
    if not _FEEDBACK_FILE.exists():
        return []
    try:
        return json.loads(_FEEDBACK_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read feedback.json: %s", e)
        return []


def _write_feedback(entries: list[dict]) -> None:
    """Write the full feedback log."""
    _FEEDBACK_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def log_feedback(
    request: str,
    draft: dict,
    accepted: bool,
    feedback_text: str = "",
    resources_used: list[str] | None = None,
) -> int:
    """
    Append a feedback entry. Returns the new total count.
    """
    entries = _read_feedback()
    entries.append({
        "request": request,
        "draft": draft,
        "accepted": accepted,
        "feedback_text": feedback_text,
        "resources_used": resources_used or [],
        "timestamp": time.time(),
    })
    _write_feedback(entries)
    count = len(entries)
    logger.info("Logged feedback #%d (accepted=%s)", count, accepted)
    return count


def get_feedback_context() -> str:
    """
    Return recent feedback history + learned preferences as a string
    for the agent system prompt.
    """
    parts = []

    # Learned preferences
    if _PREFERENCES_FILE.exists():
        prefs = _PREFERENCES_FILE.read_text(encoding="utf-8").strip()
        if prefs:
            parts.append(f"--- LEARNED PREFERENCES ---\n{prefs}")

    # Recent feedback (last 10)
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
    """Human-readable stats for the /feedback command."""
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

    # Last 5 rejections with feedback
    rejections = [e for e in entries if not e["accepted"] and e.get("feedback_text")]
    if rejections:
        lines.append("\nRecent rejection reasons:")
        for e in rejections[-5:]:
            lines.append(f"  - {e['feedback_text'][:100]}")

    # Preferences status
    if _PREFERENCES_FILE.exists():
        lines.append(f"\nLearned preferences: Yes (updated at {_PREFERENCES_FILE.stat().st_size} bytes)")
    else:
        lines.append(f"\nLearned preferences: Not yet generated (auto-generates every {settings.FEEDBACK_SUMMARIZE_EVERY} entries, or use /learn)")

    return "\n".join(lines)


async def summarize_preferences() -> str:
    """
    Call Claude to analyze feedback.json and write learned_preferences.md.
    Returns the generated summary.
    """
    entries = _read_feedback()
    if not entries:
        return "No feedback to analyze yet."

    # Build analysis prompt
    feedback_text = json.dumps(entries[-50:], indent=2)  # last 50 max

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-6",
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

    summary = response.content[0].text
    _PREFERENCES_FILE.write_text(summary, encoding="utf-8")
    logger.info("Updated learned_preferences.md (%d chars)", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Async wrappers — non-blocking versions for use in bot handlers
# ---------------------------------------------------------------------------

async def async_log_feedback(*args, **kwargs) -> int:
    return await asyncio.to_thread(log_feedback, *args, **kwargs)
