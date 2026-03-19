"""
Composable weighted assertion scoring framework for draft quality.

Instead of binary pass/fail (see self_review.draft_quality_gate), this provides
a numeric 0-100 quality score across weighted dimensions. Each assertion has a
name, weight, and scoring function that returns 0.0-1.0.

Inspired by promptfoo's assertion system.

Usage:
    from agent.scoring import score_draft, format_score_report

    result = score_draft(draft)
    print(result["total_score"])   # 0-100
    print(result["grade"])         # "A", "B", "C", "D", or "F"

    # Custom assertions:
    my_assertions = [Assertion("custom", 1.0, my_scorer, "My check")]
    result = score_draft(draft, assertions=my_assertions)
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from agent.content_types import ALL_CONTENT_TYPES
from agent.self_review import _AI_WORDS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Assertion:
    """A single scoring dimension."""
    name: str
    weight: float          # 0.0 - 1.0
    scorer: Callable[[dict], float]  # takes draft dict, returns 0.0 - 1.0
    description: str


@dataclass(frozen=True)
class ScoreResult:
    """Result of running one assertion against a draft."""
    name: str
    score: float           # 0.0 - 1.0
    weight: float
    detail: str


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------

def _score_caption_quality(draft: dict) -> float:
    """Sweet spot is 80-200 chars. <30 or >280 = 0."""
    caption = draft.get("caption", "")
    length = len(caption)
    if length < 30:
        return 0.0
    if 80 <= length <= 200:
        return 1.0
    if length > 280:
        return 0.0
    # 30-80: scale linearly
    if length < 80:
        return (length - 30) / 50.0
    # 200-280: scale linearly downward
    return (280 - length) / 80.0


def _score_no_ai_language(draft: dict) -> float:
    """1.0 if clean, 0.0 if >3 AI words, linear between."""
    caption = draft.get("caption", "")
    title = draft.get("title", "")
    subtitle = draft.get("subtitle", "")
    all_text = f"{caption} {title} {subtitle}"
    matches = _AI_WORDS.findall(all_text)
    count = len(matches)
    if count == 0:
        return 1.0
    if count > 3:
        return 0.0
    # 1-3 matches: linear scale
    return 1.0 - (count / 3.0)


def _score_hashtag_free(draft: dict) -> float:
    """1.0 if no hashtags, 0.0 if any."""
    caption = draft.get("caption", "")
    title = draft.get("title", "")
    subtitle = draft.get("subtitle", "")
    all_text = f"{caption} {title} {subtitle}"
    if re.search(r"#\w+", all_text):
        return 0.0
    return 1.0


def _score_emoji_restraint(draft: dict) -> float:
    """1.0 if 0-1 emoji, 0.5 if 2, 0.0 if >2."""
    caption = draft.get("caption", "")
    count = sum(1 for c in caption if unicodedata.category(c) == "So")
    if count <= 1:
        return 1.0
    if count == 2:
        return 0.5
    return 0.0


def _score_image_prompt_depth(draft: dict) -> float:
    """Score based on image prompt length. >100 = 1.0, <20 = 0.0, linear between."""
    prompt = draft.get("image_prompt", "")
    length = len(prompt)
    if length >= 100:
        return 1.0
    if length < 20:
        return 0.0
    return (length - 20) / 80.0


def _score_has_content_type(draft: dict) -> float:
    """1.0 if valid type, 0.5 if any type set, 0.0 if missing."""
    ct = draft.get("content_type", "")
    if not ct:
        return 0.0
    if ct in ALL_CONTENT_TYPES:
        return 1.0
    return 0.5


# ---------------------------------------------------------------------------
# Default assertion set
# ---------------------------------------------------------------------------

DEFAULT_ASSERTIONS: list[Assertion] = [
    Assertion("caption_quality", 0.25, _score_caption_quality, "Caption length sweet spot (80-200 chars)"),
    Assertion("no_ai_language", 0.20, _score_no_ai_language, "No AI-sounding language"),
    Assertion("hashtag_free", 0.10, _score_hashtag_free, "No hashtags in text"),
    Assertion("emoji_restraint", 0.10, _score_emoji_restraint, "Max 1 emoji in caption"),
    Assertion("image_prompt_depth", 0.20, _score_image_prompt_depth, "Detailed image prompt (>100 chars)"),
    Assertion("has_content_type", 0.15, _score_has_content_type, "Valid content type set"),
]


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def _grade_from_score(score: float) -> str:
    """Map a 0-100 score to a letter grade."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def score_draft(draft: dict, assertions: list[Assertion] | None = None, threshold: float = 60.0) -> dict:
    """Score a draft against weighted assertions.

    Args:
        draft: Dict with keys like caption, title, subtitle, image_prompt, content_type.
        assertions: Custom assertion list, or None to use DEFAULT_ASSERTIONS.
        threshold: Minimum score (0-100) to pass. Default 60 (C grade).

    Returns:
        {
            "total_score": float 0-100,
            "grade": "A"|"B"|"C"|"D"|"F",
            "results": list of ScoreResult dicts,
            "passed_threshold": bool
        }
    """
    if assertions is None:
        assertions = DEFAULT_ASSERTIONS

    results: list[ScoreResult] = []
    total_weight = sum(a.weight for a in assertions)

    for assertion in assertions:
        raw_score = assertion.scorer(draft)
        # Clamp to 0-1
        raw_score = max(0.0, min(1.0, raw_score))

        detail = f"{raw_score:.0%}"
        results.append(ScoreResult(
            name=assertion.name,
            score=raw_score,
            weight=assertion.weight,
            detail=detail,
        ))

    # Weighted average, normalized to 0-100
    if total_weight > 0:
        weighted_sum = sum(r.score * r.weight for r in results)
        total_score = round((weighted_sum / total_weight) * 100, 1)
    else:
        total_score = 0.0

    grade = _grade_from_score(total_score)

    return {
        "total_score": total_score,
        "grade": grade,
        "results": [
            {"name": r.name, "score": r.score, "weight": r.weight, "detail": r.detail}
            for r in results
        ],
        "passed_threshold": total_score >= threshold,
    }


# ---------------------------------------------------------------------------
# Telegram-friendly report formatter
# ---------------------------------------------------------------------------

# Bar characters for visual score display
_BAR_FULL = "\u2588"   # █
_BAR_EMPTY = "\u2591"  # ░


def _score_bar(score: float, width: int = 10) -> str:
    """Render a small bar chart for a 0-1 score."""
    filled = round(score * width)
    return _BAR_FULL * filled + _BAR_EMPTY * (width - filled)


def format_score_report(result: dict) -> str:
    """Format a score_draft result as an HTML string for Telegram.

    Args:
        result: Output from score_draft().

    Returns:
        HTML-formatted string suitable for parse_mode="HTML".
    """
    total = result["total_score"]
    grade = result["grade"]
    passed = result["passed_threshold"]

    lines = [
        f"<b>Quality Score: {total:.0f}/100 (Grade {grade})</b>",
        "",
    ]

    for r in result["results"]:
        bar = _score_bar(r["score"])
        pct = f'{r["score"]:.0%}'
        weight_pct = f'{r["weight"]:.0%}'
        lines.append(f"<code>{bar}</code> {r['name']} {pct} (w:{weight_pct})")

    lines.append("")
    verdict = "PASSED" if passed else "BELOW THRESHOLD"
    lines.append(f"<b>Verdict:</b> {verdict}")

    return "\n".join(lines)
