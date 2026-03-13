"""
Self-review — the agent periodically analyzes its own performance and
updates learned_preferences.md automatically.

Loop: generate → get feedback → accumulate → self-review → update preferences → generate better

The self-review LLM call is separate from the main conversation —
it's a background process, not a user-facing interaction.
"""

import json
import logging
import re
import time
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_STATE_DIR = _project_root / "state"
_FEEDBACK_FILE = _STATE_DIR / "feedback.json"
_HISTORY_FILE = _STATE_DIR / "generation_history.json"
_PREFERENCES_FILE = _STATE_DIR / "learned_preferences.md"
_CONVERSATION_FILE = _STATE_DIR / "conversation.json"
_REVIEW_FILE = _STATE_DIR / "last_self_review.json"


# ---------------------------------------------------------------------------
# Default-FAIL quality gate — drafts must prove readiness with evidence
# ---------------------------------------------------------------------------

# Words that should never appear in captions (AI-sounding language)
_AI_WORDS = re.compile(
    r"\b(revolutioniz|leverag|cutting.?edge|seamless|dive into|unlock|empower|"
    r"game.?chang|elevat|supercharg|turbo.?charg|next.?gen)\w*\b",
    re.IGNORECASE,
)

# Hashtag pattern
_HASHTAG = re.compile(r"#\w+")


def draft_quality_gate(draft: dict) -> dict:
    """Check a draft against hard rules before sending to user.

    Philosophy: DEFAULT FAIL. The draft must pass every check with evidence.
    A single hard-rule violation means the draft is not ready.

    Args:
        draft: Dict with keys like caption, title, subtitle, image_prompt, content_type.

    Returns:
        {
            "passed": bool,
            "verdict": "READY" | "NEEDS WORK",
            "checks": [{"rule": str, "passed": bool, "detail": str}, ...],
            "auto_fixed": [str, ...]  # descriptions of auto-applied fixes
        }
    """
    checks: list[dict] = []
    auto_fixed: list[str] = []

    caption = draft.get("caption", "")
    title = draft.get("title", "")
    subtitle = draft.get("subtitle", "")
    image_prompt = draft.get("image_prompt", "")
    all_text = f"{caption} {title} {subtitle}"

    # 1. HASHTAG CHECK — zero tolerance
    hashtags_found = _HASHTAG.findall(all_text)
    if hashtags_found:
        # Auto-fix: strip hashtags
        draft["caption"] = _HASHTAG.sub("", caption).strip()
        draft["title"] = _HASHTAG.sub("", title).strip()
        draft["subtitle"] = _HASHTAG.sub("", subtitle).strip()
        auto_fixed.append(f"Stripped hashtags: {', '.join(hashtags_found)}")
        checks.append({"rule": "no_hashtags", "passed": True, "detail": f"Auto-fixed: removed {hashtags_found}"})
    else:
        checks.append({"rule": "no_hashtags", "passed": True, "detail": "Clean"})

    # 2. AI WORDS CHECK
    ai_matches = _AI_WORDS.findall(all_text)
    if ai_matches:
        checks.append({"rule": "no_ai_words", "passed": False, "detail": f"Found AI words: {ai_matches}"})
    else:
        checks.append({"rule": "no_ai_words", "passed": True, "detail": "Clean"})

    # 3. CAPTION LENGTH CHECK
    caption_len = len(draft.get("caption", ""))
    if caption_len < 10:
        checks.append({"rule": "caption_length", "passed": False, "detail": f"Too short: {caption_len} chars (min 10)"})
    elif caption_len > 280:
        checks.append({"rule": "caption_length", "passed": False, "detail": f"Too long: {caption_len} chars (max 280)"})
    else:
        checks.append({"rule": "caption_length", "passed": True, "detail": f"{caption_len} chars"})

    # 4. EMOJI CHECK — max 1
    import unicodedata
    emoji_count = sum(1 for c in caption if unicodedata.category(c) == "So")
    if emoji_count > 1:
        checks.append({"rule": "max_1_emoji", "passed": False, "detail": f"Found {emoji_count} emojis (max 1)"})
    else:
        checks.append({"rule": "max_1_emoji", "passed": True, "detail": f"{emoji_count} emoji(s)"})

    # 5. IMAGE PROMPT EXISTS (if content expects image)
    if not image_prompt:
        checks.append({"rule": "has_image_prompt", "passed": False, "detail": "Missing image_prompt"})
    elif len(image_prompt) < 20:
        checks.append({"rule": "has_image_prompt", "passed": False, "detail": f"Image prompt too vague: {len(image_prompt)} chars"})
    else:
        checks.append({"rule": "has_image_prompt", "passed": True, "detail": f"{len(image_prompt)} chars"})

    # 6. CONTENT TYPE VALID
    from agent.content_types import ALL_CONTENT_TYPES
    ct = draft.get("content_type", "")
    if ct and ct in ALL_CONTENT_TYPES:
        checks.append({"rule": "valid_content_type", "passed": True, "detail": ct})
    elif ct:
        checks.append({"rule": "valid_content_type", "passed": False, "detail": f"Unknown type: {ct}"})
    else:
        checks.append({"rule": "valid_content_type", "passed": True, "detail": "Not specified (OK)"})

    # Verdict: ALL checks must pass
    all_passed = all(c["passed"] for c in checks)

    return {
        "passed": all_passed,
        "verdict": "READY" if all_passed else "NEEDS WORK",
        "checks": checks,
        "auto_fixed": auto_fixed,
    }

_REVIEW_SYSTEM_PROMPT = (
    "You are reviewing your own performance as a marketing agent. "
    "Analyze the data and produce:\n"
    "1. PATTERNS: What content types get approved vs rejected? What tones work? What doesn't?\n"
    "2. FRICTION: Where did the user get frustrated, reject multiple times, "
    "or give the same feedback repeatedly?\n"
    "3. IMPROVEMENTS: Specific, actionable rules for your future self. Not vague — concrete. "
    "Example: 'Use casual degen tone for community posts, max 180 chars caption' or "
    "'Stop using formal announcement framing — rejected 4 out of 5 times.'\n"
    "4. STATS: approval rate, average rejections before approval, most common rejection reasons, "
    "best performing content type.\n\n"
    "Output a JSON object with exactly these keys:\n"
    "- \"patterns\": list of pattern strings\n"
    "- \"friction\": list of friction point strings\n"
    "- \"improvements\": a markdown document that will replace learned_preferences.md "
    "(concrete rules, not vague advice)\n"
    "- \"stats\": object with approval_rate (float 0-1), avg_rejections_before_approval (float), "
    "common_rejection_reasons (list of strings), best_content_type (string or null)\n\n"
    "Return ONLY valid JSON — no markdown fences, no commentary outside the JSON."
)


def _read_json_file(path: Path, max_entries: int = 50) -> list | dict | None:
    """Read a JSON file, returning None if missing/corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[-max_entries:]
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path.name, e)
        return None


def _read_text_file(path: Path) -> str:
    """Read a text file, returning empty string if missing."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


async def run_self_review() -> dict:
    """Analyze recent agent performance and generate updated learnings.

    Returns:
        dict with keys: insights (list[str]), updated_preferences (str),
        stats (dict), error (str|None)
    """
    logger.info("Self-review: starting analysis")
    t_start = time.time()

    # a) Read data sources
    feedback_entries = _read_json_file(_FEEDBACK_FILE, max_entries=50) or []
    history_entries = _read_json_file(_HISTORY_FILE, max_entries=50) or []
    current_prefs = _read_text_file(_PREFERENCES_FILE)
    conversation_data = _read_json_file(_CONVERSATION_FILE) or {}

    if not feedback_entries:
        result = {
            "insights": [],
            "updated_preferences": current_prefs,
            "stats": {"approval_rate": 0, "total_reviewed": 0},
            "error": "No feedback data to analyze",
        }
        logger.info("Self-review: skipped — no feedback data")
        return result

    # Build the analysis payload
    analysis_input = {
        "feedback_entries": feedback_entries,
        "generation_history": history_entries[:30],  # trim history more aggressively
        "current_preferences": current_prefs or "(none set yet)",
    }

    # Include recent conversation patterns if available
    if isinstance(conversation_data, dict):
        # Only include relevant fields, not the full blob
        conv_summary = {}
        for uid, ctx in conversation_data.items():
            if isinstance(ctx, dict):
                conv_summary[uid] = {
                    "message_count": ctx.get("message_count", 0),
                    "last_command": ctx.get("last_command"),
                    "last_bot_action": ctx.get("last_bot_action"),
                }
        if conv_summary:
            analysis_input["conversation_patterns"] = conv_summary

    # e) Send to Claude
    from agent._client import get_anthropic
    client = get_anthropic()

    try:
        response = await client.messages.create(
            model=settings.SONNET_MODEL,
            max_tokens=3000,
            system=_REVIEW_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Analyze this performance data for the {settings.BRAND_NAME} marketing agent.\n\n"
                    f"DATA:\n{json.dumps(analysis_input, indent=2, ensure_ascii=False)}"
                ),
            }],
        )
    except Exception as e:
        logger.error("Self-review: Claude API call failed: %s", e)
        return {
            "insights": [],
            "updated_preferences": current_prefs,
            "stats": {},
            "error": f"API call failed: {e}",
        }

    raw_text = response.content[0].text.strip()

    # Parse the JSON response
    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

    try:
        review_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("Self-review: failed to parse Claude response as JSON: %s", e)
        return {
            "insights": [],
            "updated_preferences": current_prefs,
            "stats": {},
            "error": f"Failed to parse review response: {e}",
            "raw_response": raw_text[:2000],
        }

    # Extract sections
    patterns = review_data.get("patterns", [])
    friction = review_data.get("friction", [])
    improvements = review_data.get("improvements", "")
    stats = review_data.get("stats", {})

    # f) Write updated preferences → session.learned_preferences (single source of truth)
    added_prefs = []
    if improvements and isinstance(improvements, str):
        from agent.session import add_learned_preference
        # Extract actionable lines from the improvements markdown
        for line in improvements.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if line and len(line) > 10 and not line.startswith("#"):
                if add_learned_preference(line):
                    added_prefs.append(line)
        if added_prefs:
            logger.info("Self-review: added %d preferences to session", len(added_prefs))

    # g) Write full review for debugging
    review_record = {
        "timestamp": time.time(),
        "patterns": patterns,
        "friction": friction,
        "improvements_length": len(improvements) if improvements else 0,
        "stats": stats,
        "feedback_entries_analyzed": len(feedback_entries),
        "duration_seconds": round(time.time() - t_start, 1),
    }
    try:
        _REVIEW_FILE.write_text(
            json.dumps(review_record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Failed to write last_self_review.json: %s", e)

    logger.info(
        "Self-review: complete in %.1fs — approval_rate=%.0f%%, %d patterns, %d friction points",
        time.time() - t_start,
        stats.get("approval_rate", 0) * 100,
        len(patterns),
        len(friction),
    )

    # h) Return results
    insights = patterns + friction
    return {
        "insights": insights,
        "updated_preferences": improvements,
        "stats": stats,
        "error": None,
    }


def get_last_review_summary() -> str:
    """Return a one-line summary of the last self-review for the system prompt."""
    if not _REVIEW_FILE.exists():
        return ""
    try:
        data = json.loads(_REVIEW_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    ts = data.get("timestamp", 0)
    if not ts:
        return ""

    stats = data.get("stats", {})
    approval_rate = stats.get("approval_rate", 0)
    patterns = data.get("patterns", [])

    date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
    top_insight = patterns[0][:80] if patterns else "no patterns identified"

    return (
        f"Last self-review: {date_str}. "
        f"Approval rate: {approval_rate * 100:.0f}%. "
        f"Top insight: {top_insight}"
    )
