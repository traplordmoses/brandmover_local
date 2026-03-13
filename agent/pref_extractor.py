"""
Automatic preference extraction from session history.
Analyzes patterns in approved and rejected drafts to distill learned preferences.
"""

import json
import logging
import time

import anthropic

from agent.session import load_session, save_session, add_learned_preference
from config import settings

logger = logging.getLogger(__name__)


async def extract_preferences() -> list[str]:
    """
    Analyze session history and extract/update learned preferences.
    Returns list of newly added preferences.

    Only runs if:
    - At least PREF_EXTRACTION_MIN_EVENTS total events (approvals + rejections)
    - At least PREF_EXTRACTION_INTERVAL_HOURS since last extraction
    - PREF_EXTRACTION_ENABLED is True
    """
    if not settings.PREF_EXTRACTION_ENABLED:
        return []

    session = load_session()

    total_events = len(session.recent_posts) + len(session.rejected_drafts)
    if total_events < settings.PREF_EXTRACTION_MIN_EVENTS:
        return []

    # Check interval
    last_extraction = session.last_preference_extraction
    if last_extraction:
        hours_since = (time.time() - last_extraction) / 3600
        if hours_since < settings.PREF_EXTRACTION_INTERVAL_HOURS:
            return []

    from agent._client import get_anthropic
    client = get_anthropic()

    # Load raw feedback log for richer signal (beyond session's capped lists)
    raw_feedback_block = ""
    try:
        from agent.feedback import _read_feedback
        raw_entries = _read_feedback()
        if raw_entries:
            recent_raw = raw_entries[-20:]  # Last 20 to keep context manageable
            raw_feedback_block = (
                f"\n\nFull feedback history (raw log, last {len(recent_raw)} entries):\n"
                f"{json.dumps(recent_raw, indent=2, default=str)}"
            )
    except Exception:
        pass

    prompt = (
        f"Analyze this content approval/rejection history and extract actionable "
        f"preferences for a social media content agent.\n\n"
        f"Approved posts (these worked):\n"
        f"{json.dumps(session.recent_posts, indent=2, default=str)}\n\n"
        f"Rejected drafts (these didn't work) with feedback:\n"
        f"{json.dumps(session.rejected_drafts, indent=2, default=str)}\n\n"
        f"Current learned preferences:\n"
        f"{json.dumps(session.learned_preferences)}"
        f"{raw_feedback_block}\n\n"
        f"Tasks:\n"
        f"1. Identify 1-3 NEW patterns from the data that aren't already captured "
        f"in current preferences.\n"
        f"2. Identify any current preferences that should be REMOVED (contradicted "
        f"by recent approvals).\n"
        f"3. Be specific and actionable. Bad: \"make better content\". "
        f"Good: \"keep captions under 180 characters — all 3 rejections cited length\"\n\n"
        f"Use the update_preferences tool to submit."
    )

    update_prefs_tool = {
        "name": "update_preferences",
        "description": "Submit preference updates",
        "input_schema": {
            "type": "object",
            "properties": {
                "add": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New preferences to add",
                },
                "remove": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Existing preferences to remove (exact match)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of what patterns you found",
                },
            },
            "required": ["add", "remove", "reasoning"],
        },
    }

    try:
        response = await client.messages.create(
            model=settings.AGENT_MODEL,
            max_tokens=800,
            system="You are a concise preference analyst. Extract actionable patterns.",
            tools=[update_prefs_tool],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        logger.error("Preference extraction failed: %s", e)
        return []

    new_prefs: list[str] = []

    for block in response.content:
        if block.type == "tool_use" and block.name == "update_preferences":
            inp = block.input

            reasoning = inp.get("reasoning", "")
            if reasoning:
                logger.info("Pref extraction reasoning: %s", reasoning[:300])

            # Remove contradicted preferences
            for to_remove in inp.get("remove", []):
                session = load_session()  # reload in case of concurrent edits
                for i, existing in enumerate(session.learned_preferences):
                    if existing == to_remove:
                        session.learned_preferences.pop(i)
                        save_session(session)
                        logger.info("Pref extraction: removed %r", to_remove)
                        break

            # Add new preferences
            for to_add in inp.get("add", []):
                if add_learned_preference(to_add):
                    new_prefs.append(to_add)

            break

    # Update extraction timestamp
    session = load_session()
    session.last_preference_extraction = time.time()
    save_session(session)

    if new_prefs:
        logger.info(
            "Pref extraction: added %d new preferences: %s",
            len(new_prefs), new_prefs,
        )

    return new_prefs
