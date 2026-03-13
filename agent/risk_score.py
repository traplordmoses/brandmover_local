"""
Content safety risk scoring — lightweight, local-only.

Scans generated text (caption, title, subtitle) against pattern-based
risk categories before content reaches the user or auto-post.  No API
calls; designed to be fast and deterministic.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Risk categories — each maps to a list of patterns matched with word-boundary
# regex.  Patterns are matched case-insensitively.
# ---------------------------------------------------------------------------

RISK_PATTERNS: dict[str, list[str]] = {
    "profanity": [
        "fuck", "shit", "ass", "asshole", "bitch", "bastard",
        "damn", "crap", "dick", "piss", "slut", "whore",
        "cunt", "cock", "wanker",
    ],
    "controversial": [
        "politics", "religion", "racist", "sexist", "homophob",
        "transphob", "nazi", "terrorist", "kill", "murder", "suicide",
    ],
    "financial_risk": [
        "guaranteed returns", "get rich", "financial advice",
        "invest now", "100x", "moon", "pump", "rug pull",
        "not financial advice",
    ],
    "spam_signals": [
        "click here", "buy now", "limited time", "act fast",
        "don't miss", "exclusive offer", "free money", "wire transfer",
    ],
    "competitor_mentions": [],  # populated from brand guidelines avoid_terms
}

# Severity weight added to the cumulative score for each match in a category.
SEVERITY_WEIGHTS: dict[str, float] = {
    "profanity":            0.4,
    "controversial":        0.3,
    "financial_risk":       0.2,
    "spam_signals":         0.15,
    "competitor_mentions":  0.1,
}

# Severity label per category (mirrors relative weight).
SEVERITY_LABELS: dict[str, str] = {
    "profanity":            "high",
    "controversial":        "medium",
    "financial_risk":       "medium",
    "spam_signals":         "low",
    "competitor_mentions":  "low",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compile_patterns(
    terms: list[str],
) -> list[re.Pattern[str]]:
    """Compile a list of literal terms into word-boundary regexes."""
    compiled: list[re.Pattern[str]] = []
    for term in terms:
        # Escape so multi-word phrases and special chars are safe.
        escaped = re.escape(term)
        compiled.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
    return compiled


def _load_competitor_terms() -> list[str]:
    """Try to pull avoid_terms from the brand config (best-effort)."""
    try:
        from agent.compositor_config import get_config

        cfg = get_config()
        return list(cfg.avoid_terms) if cfg.avoid_terms else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_risk(
    text: str,
    load_avoid_terms: bool = True,
) -> dict[str, Any]:
    """Score content text for safety risks.

    Parameters
    ----------
    text:
        Combined text to scan (caption + title + subtitle, etc.).
    load_avoid_terms:
        If True, attempt to load the brand's ``avoid_terms`` from
        ``compositor_config`` and add them to the competitor_mentions
        category for this run.

    Returns
    -------
    dict with keys:
        risk_level   – "low" | "medium" | "high"
        risk_score   – float 0.0–1.0 (clamped)
        flags        – list of {"category", "matched", "severity"} dicts
        safe_to_post – bool (True unless risk_level is "high")
    """
    if not text:
        return {
            "risk_level": "low",
            "risk_score": 0.0,
            "flags": [],
            "safe_to_post": True,
        }

    # Build per-run patterns dict (avoid mutating the module-level dict).
    patterns: dict[str, list[str]] = {k: list(v) for k, v in RISK_PATTERNS.items()}

    if load_avoid_terms:
        extra = _load_competitor_terms()
        if extra:
            patterns["competitor_mentions"] = patterns["competitor_mentions"] + extra

    # Scan text against every category.
    flags: list[dict[str, str]] = []
    raw_score: float = 0.0

    for category, terms in patterns.items():
        if not terms:
            continue
        compiled = _compile_patterns(terms)
        weight = SEVERITY_WEIGHTS[category]
        severity = SEVERITY_LABELS[category]

        for pattern, original_term in zip(compiled, terms):
            if pattern.search(text):
                flags.append({
                    "category": category,
                    "matched": original_term,
                    "severity": severity,
                })
                raw_score += weight

    # Clamp to 0.0–1.0 range.
    risk_score = min(raw_score, 1.0)

    # Determine level from thresholds.
    if risk_score > 0.5:
        risk_level = "high"
    elif risk_score >= 0.2:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 3),
        "flags": flags,
        "safe_to_post": risk_level != "high",
    }
