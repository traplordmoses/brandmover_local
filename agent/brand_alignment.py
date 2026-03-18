"""
Brand alignment scoring -- detects drift between generated content and BrandConfig.

Pure Python scorer (no LLM calls) that checks drafts against the brand guidelines
parsed by compositor_config.py. Provides both per-draft scoring and longitudinal
drift detection across recent outputs.

Public API:
    result = score_brand_alignment(draft)     # Per-draft check, returns 0-100
    drift = detect_brand_drift(days=7)        # Longitudinal drift analysis
    context = get_alignment_context()          # Formatted string for prompt injection

    # Plugs into scoring.py assertion framework:
    brand_alignment_assertion = Assertion(...)
"""

import logging
import re
import time
from datetime import datetime, timezone, timedelta

from agent.compositor_config import get_config, BrandConfig
from agent.scoring import Assertion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Caption length sweet spot per brand guidelines
_CAPTION_MIN = 50
_CAPTION_MAX = 150

# Filler phrases that dilute brand voice
_FILLER_PHRASES = [
    "excited to announce",
    "thrilled to share",
    "we're proud to",
    "stay tuned",
    "don't miss out",
    "game changer",
    "game-changer",
    "buckle up",
    "let that sink in",
    "this is huge",
    "mark your calendars",
]

# Patterns that indicate tone violations
_ALL_CAPS_SENTENCE = re.compile(r"\b[A-Z]{4,}(?:\s+[A-Z]{4,})+\b")
_EXCLAMATION_EXCESS = re.compile(r"!{2,}")  # Two or more consecutive exclamation marks
_HASHTAG_IN_BODY = re.compile(r"#\w+")


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_voice_match(text: str, config: BrandConfig) -> tuple[float, list[str]]:
    """Check if text uses vocabulary consistent with voice_traits and brand_phrases.

    Returns (score 0-1, list of drift flags).
    """
    if not text:
        return 0.0, ["empty_text"]

    text_lower = text.lower()
    flags: list[str] = []
    hits = 0
    total_checks = 0

    # Check voice traits -- each trait keyword should feel present in the text
    for trait in config.voice_traits:
        trait_lower = trait.lower()
        total_checks += 1
        # Check if any word from the trait appears in the text
        trait_words = [w for w in trait_lower.split() if len(w) > 3]
        if any(w in text_lower for w in trait_words):
            hits += 1

    # Check brand phrases -- bonus for using established phrases
    for phrase in config.brand_phrases:
        phrase_lower = phrase.lower()
        total_checks += 1
        if phrase_lower in text_lower:
            hits += 1

    if total_checks == 0:
        # No voice traits or brand phrases configured -- pass by default
        return 1.0, []

    score = hits / total_checks
    if score < 0.2:
        flags.append("low_voice_match: caption lacks brand vocabulary")
    if config.brand_phrases and not any(p.lower() in text_lower for p in config.brand_phrases):
        flags.append("no_brand_phrases_used")

    return min(1.0, score), flags


def _check_avoid_terms(text: str, config: BrandConfig) -> tuple[float, list[str]]:
    """Check text against avoid_terms. 1.0 if clean, reduce per violation.

    Returns (score 0-1, list of drift flags).
    """
    if not config.avoid_terms:
        return 1.0, []

    text_lower = text.lower()
    violations: list[str] = []

    for term in config.avoid_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            violations.append(f"avoid_term_found: '{term}'")

    if not violations:
        return 1.0, []

    # Each violation reduces score by 0.3, minimum 0.0
    penalty = len(violations) * 0.3
    score = max(0.0, 1.0 - penalty)
    return score, violations


def _check_tone_consistency(text: str) -> tuple[float, list[str]]:
    """Check for forbidden tone patterns: ALL CAPS, excessive exclamation, hashtags, filler.

    Returns (score 0-1, list of drift flags).
    """
    if not text:
        return 1.0, []

    flags: list[str] = []
    violations = 0

    # ALL CAPS sentences (4+ letter words in all caps, at least 2 consecutive)
    if _ALL_CAPS_SENTENCE.search(text):
        violations += 1
        flags.append("tone_violation: ALL CAPS sentence detected")

    # Excessive exclamation marks
    if _EXCLAMATION_EXCESS.search(text):
        violations += 1
        flags.append("tone_violation: excessive exclamation marks")

    # Single exclamation marks -- mild penalty if more than one
    excl_count = text.count("!")
    if excl_count > 1:
        violations += 1
        flags.append(f"tone_violation: {excl_count} exclamation marks")

    # Hashtags in body text
    if _HASHTAG_IN_BODY.search(text):
        violations += 1
        flags.append("tone_violation: hashtag in body text")

    # Filler phrases
    text_lower = text.lower()
    for phrase in _FILLER_PHRASES:
        if phrase in text_lower:
            violations += 1
            flags.append(f"tone_violation: filler phrase '{phrase}'")

    if violations == 0:
        return 1.0, []

    # Each violation costs 0.25
    score = max(0.0, 1.0 - violations * 0.25)
    return score, flags


def _check_length_compliance(text: str) -> tuple[float, list[str]]:
    """Check caption length against brand guidelines sweet spot (50-150 chars).

    Returns (score 0-1, list of drift flags).
    """
    length = len(text)
    flags: list[str] = []

    if _CAPTION_MIN <= length <= _CAPTION_MAX:
        return 1.0, []

    if length == 0:
        return 0.0, ["length_violation: empty caption"]

    if length < _CAPTION_MIN:
        # Linear degradation from 50 down to 0
        score = max(0.0, length / _CAPTION_MIN)
        flags.append(f"length_violation: too short ({length} chars, min {_CAPTION_MIN})")
        return score, flags

    # length > _CAPTION_MAX
    # Linear degradation from 150 up to 300 (where it hits 0)
    overshoot = length - _CAPTION_MAX
    max_overshoot = _CAPTION_MAX  # At 300 chars, score = 0
    score = max(0.0, 1.0 - overshoot / max_overshoot)
    flags.append(f"length_violation: too long ({length} chars, max {_CAPTION_MAX})")
    return score, flags


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_brand_alignment(draft: dict) -> dict:
    """Pure Python scorer that checks a draft against BrandConfig.

    Args:
        draft: Dict with keys like caption, title, subtitle, image_prompt, content_type.

    Returns:
        {
            "alignment_score": float 0-100,
            "checks": [{"name": str, "score": float 0-1, "weight": float}],
            "drift_flags": [str]
        }
    """
    config = get_config()

    caption = draft.get("caption", "")
    title = draft.get("title", "")
    subtitle = draft.get("subtitle", "")
    all_text = f"{caption} {title} {subtitle}".strip()

    all_flags: list[str] = []
    checks: list[dict] = []

    # 1. Voice match (weight 0.30)
    voice_score, voice_flags = _check_voice_match(all_text, config)
    checks.append({"name": "voice_match", "score": voice_score, "weight": 0.30})
    all_flags.extend(voice_flags)

    # 2. Avoid terms clean (weight 0.25)
    avoid_score, avoid_flags = _check_avoid_terms(all_text, config)
    checks.append({"name": "avoid_terms_clean", "score": avoid_score, "weight": 0.25})
    all_flags.extend(avoid_flags)

    # 3. Tone consistency (weight 0.25)
    tone_score, tone_flags = _check_tone_consistency(all_text)
    checks.append({"name": "tone_consistency", "score": tone_score, "weight": 0.25})
    all_flags.extend(tone_flags)

    # 4. Length compliance (weight 0.20)
    length_score, length_flags = _check_length_compliance(caption)
    checks.append({"name": "length_compliance", "score": length_score, "weight": 0.20})
    all_flags.extend(length_flags)

    # Weighted average -> 0-100
    total_weight = sum(c["weight"] for c in checks)
    if total_weight > 0:
        weighted_sum = sum(c["score"] * c["weight"] for c in checks)
        alignment_score = round((weighted_sum / total_weight) * 100, 1)
    else:
        alignment_score = 0.0

    return {
        "alignment_score": alignment_score,
        "checks": checks,
        "drift_flags": all_flags,
    }


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def detect_brand_drift(days: int = 7) -> dict:
    """Analyze recent outputs for systematic brand drift.

    Loads recent posts from agent/session.py and recent feedback from
    agent/feedback.py, then checks for recurring violations over the
    specified time window.

    Returns:
        {
            "drift_detected": bool,
            "drift_score": float 0-100 (100 = no drift),
            "issues": [str],
            "recommendations": [str]
        }
    """
    config = get_config()
    cutoff = time.time() - (days * 86400)
    issues: list[str] = []
    recommendations: list[str] = []

    # Load recent posts from session
    try:
        from agent.session import load_session
        session = load_session()
        recent_posts = [
            p for p in session.recent_posts
            if p.get("timestamp", 0) >= cutoff
        ]
    except Exception as e:
        logger.warning("Failed to load session for drift detection: %s", e)
        recent_posts = []

    # Load recent feedback
    try:
        from agent.feedback import _read_feedback
        all_feedback = _read_feedback()
        recent_feedback = [
            f for f in all_feedback
            if f.get("timestamp", 0) >= cutoff
        ]
    except Exception as e:
        logger.warning("Failed to load feedback for drift detection: %s", e)
        recent_feedback = []

    if not recent_posts and not recent_feedback:
        return {
            "drift_detected": False,
            "drift_score": 100.0,
            "issues": [],
            "recommendations": ["No recent data available for drift analysis."],
        }

    # Collect all caption texts for analysis
    captions: list[str] = []
    for post in recent_posts:
        cap = post.get("caption", "")
        if cap:
            captions.append(cap)
    for fb in recent_feedback:
        cap = fb.get("draft", {}).get("caption", "")
        if cap:
            captions.append(cap)

    if not captions:
        return {
            "drift_detected": False,
            "drift_score": 100.0,
            "issues": [],
            "recommendations": ["No caption data found in recent history."],
        }

    penalty_points = 0.0

    # Check 1: Are avoid_terms appearing more frequently?
    if config.avoid_terms:
        avoid_violations = 0
        for caption in captions:
            caption_lower = caption.lower()
            for term in config.avoid_terms:
                if term.lower() in caption_lower:
                    avoid_violations += 1
        if avoid_violations > 0:
            violation_rate = avoid_violations / len(captions)
            if violation_rate > 0.3:
                issues.append(
                    f"Avoid terms appearing in {violation_rate:.0%} of recent posts "
                    f"({avoid_violations} violations across {len(captions)} posts)"
                )
                recommendations.append(
                    "Reinforce avoid-terms list in generation prompts. "
                    "Consider adding explicit negative examples."
                )
                penalty_points += 20.0
            elif violation_rate > 0.1:
                issues.append(
                    f"Avoid terms appearing occasionally ({avoid_violations} violations)"
                )
                penalty_points += 10.0

    # Check 2: Is average caption length drifting outside the sweet spot?
    lengths = [len(c) for c in captions]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    if avg_length < _CAPTION_MIN:
        issues.append(
            f"Average caption length ({avg_length:.0f} chars) is below "
            f"minimum ({_CAPTION_MIN} chars)"
        )
        recommendations.append(
            "Captions are trending too short. Encourage more descriptive content."
        )
        penalty_points += 15.0
    elif avg_length > _CAPTION_MAX:
        issues.append(
            f"Average caption length ({avg_length:.0f} chars) exceeds "
            f"maximum ({_CAPTION_MAX} chars)"
        )
        recommendations.append(
            "Captions are trending too long. Enforce conciseness in prompts."
        )
        penalty_points += 15.0

    # Check 3: Are brand_phrases being used (or forgotten)?
    if config.brand_phrases:
        phrase_usage_count = 0
        for caption in captions:
            caption_lower = caption.lower()
            if any(p.lower() in caption_lower for p in config.brand_phrases):
                phrase_usage_count += 1
        usage_rate = phrase_usage_count / len(captions) if captions else 0
        if usage_rate < 0.1:
            issues.append(
                f"Brand phrases used in only {usage_rate:.0%} of recent posts. "
                f"Phrases: {', '.join(config.brand_phrases[:3])}"
            )
            recommendations.append(
                "Brand phrases are underused. Inject established phrases "
                "more frequently into generation prompts."
            )
            penalty_points += 15.0
        elif usage_rate < 0.3:
            issues.append(
                f"Brand phrase usage is low ({usage_rate:.0%} of posts)"
            )
            penalty_points += 5.0

    # Check 4: Tone violations across recent posts
    tone_violations = 0
    for caption in captions:
        _, tone_flags = _check_tone_consistency(caption)
        tone_violations += len(tone_flags)
    if tone_violations > 0:
        violation_rate = tone_violations / len(captions)
        if violation_rate > 1.0:
            issues.append(
                f"High tone violation rate: {tone_violations} violations "
                f"across {len(captions)} posts (avg {violation_rate:.1f} per post)"
            )
            recommendations.append(
                "Tone is drifting. Review filler phrases and punctuation "
                "patterns in recent outputs."
            )
            penalty_points += 20.0
        elif violation_rate > 0.5:
            issues.append(
                f"Moderate tone violations: {tone_violations} across "
                f"{len(captions)} posts"
            )
            penalty_points += 10.0

    # Compute drift score (100 = no drift, 0 = severe drift)
    drift_score = max(0.0, min(100.0, 100.0 - penalty_points))
    drift_detected = drift_score < 70.0

    return {
        "drift_detected": drift_detected,
        "drift_score": round(drift_score, 1),
        "issues": issues,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Context builder for prompt injection
# ---------------------------------------------------------------------------

def get_alignment_context() -> str:
    """Format alignment data for injection into the generation prompt.

    Similar to performance.get_performance_context(), returns a formatted
    string that can be appended to the system prompt to make the agent
    aware of brand alignment status.
    """
    config = get_config()
    drift = detect_brand_drift(days=7)

    sections: list[str] = []

    # Brand alignment summary
    lines = ["BRAND ALIGNMENT STATUS:"]

    if config.voice_traits:
        lines.append(f"- Voice traits to embody: {', '.join(config.voice_traits[:5])}")
    if config.brand_phrases:
        lines.append(f"- Use brand phrases: {', '.join(config.brand_phrases[:5])}")
    if config.avoid_terms:
        lines.append(f"- Avoid terms: {', '.join(config.avoid_terms[:5])}")
    lines.append(f"- Caption sweet spot: {_CAPTION_MIN}-{_CAPTION_MAX} chars")

    sections.append("\n".join(lines))

    # Drift status
    if drift["issues"]:
        drift_lines = [
            f"BRAND DRIFT ALERT (score: {drift['drift_score']}/100):"
        ]
        for issue in drift["issues"]:
            drift_lines.append(f"- {issue}")
        if drift["recommendations"]:
            drift_lines.append("Corrections needed:")
            for rec in drift["recommendations"]:
                drift_lines.append(f"  - {rec}")
        sections.append("\n".join(drift_lines))
    else:
        sections.append(
            f"Brand alignment: ON TRACK (drift score: {drift['drift_score']}/100)"
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Assertion adapter for scoring.py framework
# ---------------------------------------------------------------------------

def _score_brand_alignment_for_assertion(draft: dict) -> float:
    """Adapter that returns 0.0-1.0 for the scoring.py Assertion interface."""
    result = score_brand_alignment(draft)
    return result["alignment_score"] / 100.0


brand_alignment_assertion = Assertion(
    "brand_alignment",
    0.20,
    _score_brand_alignment_for_assertion,
    "Brand voice and guideline compliance",
)
