"""
Structural diversity tracker -- prevents content repetition across posts.

Maintains a rolling log of recently used content structures (skeleton IDs,
hook types, body patterns, CTA types) and scores new drafts for structural
similarity against recent output.
"""

import logging
import time
from dataclasses import asdict, dataclass, field

from agent.paths import STATE_DIR
from agent.state_manager import FileStore

logger = logging.getLogger(__name__)

_DIVERSITY_FILE = STATE_DIR / "diversity_tracker.json"
_store = FileStore(_DIVERSITY_FILE, default_factory=dict)

# How many recent entries to keep in the rolling log
_MAX_ENTRIES = 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StructureEntry:
    """Metadata about a single post's structure."""

    skeleton_id: str
    hook_type: str
    body_structure: list[str]
    cta_type: str
    tone: str
    content_type: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StructureEntry":
        return cls(
            skeleton_id=data.get("skeleton_id", "unknown"),
            hook_type=data.get("hook_type", ""),
            body_structure=data.get("body_structure", []),
            cta_type=data.get("cta_type", ""),
            tone=data.get("tone", ""),
            content_type=data.get("content_type", ""),
            timestamp=data.get("timestamp", 0.0),
        )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def log_structure(entry: StructureEntry) -> None:
    """Append a structure entry to the rolling log.

    Keeps only the most recent _MAX_ENTRIES entries.
    """
    data = _store.read()
    entries = data.get("entries", [])
    entries.append(entry.to_dict())
    # Trim to max size
    if len(entries) > _MAX_ENTRIES:
        entries = entries[-_MAX_ENTRIES:]
    data["entries"] = entries
    _store.write(data)
    logger.debug("Logged structure: skeleton=%s, hook=%s", entry.skeleton_id, entry.hook_type)


def get_recent_structures(n: int = 20) -> list[StructureEntry]:
    """Return the last N structure entries, newest first."""
    data = _store.read()
    raw_entries = data.get("entries", [])
    entries = [StructureEntry.from_dict(e) for e in raw_entries]
    entries.sort(key=lambda e: e.timestamp, reverse=True)
    return entries[:n]


def get_recent_skeleton_ids(n: int = 10) -> list[str]:
    """Return just the skeleton IDs of recent posts, newest first."""
    return [e.skeleton_id for e in get_recent_structures(n)]


# ---------------------------------------------------------------------------
# Diversity scoring
# ---------------------------------------------------------------------------

def check_structural_diversity(
    skeleton_id: str,
    hook_type: str,
    body_structure: list[str],
    cta_type: str,
    variation_aggressiveness: float = 0.6,
) -> dict:
    """Check a proposed structure against recent posts for repetition.

    Returns a dict with:
        - diversity_score: float 0-10 (10 = very diverse)
        - should_reject: bool (True if too similar)
        - reasons: list[str] (specific overlap reasons)
    """
    recent = get_recent_structures(10)

    if not recent:
        return {"diversity_score": 10.0, "should_reject": False, "reasons": []}

    score = 10.0
    reasons: list[str] = []

    # Check 1: Same skeleton used recently
    recent_ids = [e.skeleton_id for e in recent]
    if skeleton_id in recent_ids[:2]:
        penalty = 3.0 * variation_aggressiveness
        score -= penalty
        position = recent_ids.index(skeleton_id) + 1
        reasons.append(f"skeleton '{skeleton_id}' was used {position} post(s) ago")
    elif skeleton_id in recent_ids[:5]:
        penalty = 1.5 * variation_aggressiveness
        score -= penalty
        position = recent_ids.index(skeleton_id) + 1
        reasons.append(f"skeleton '{skeleton_id}' was used {position} post(s) ago")

    # Check 2: Same hook type in last 2 posts
    recent_hooks = [e.hook_type for e in recent[:2]]
    if hook_type in recent_hooks:
        penalty = 2.0 * variation_aggressiveness
        score -= penalty
        reasons.append(f"hook type '{hook_type}' used in last 2 posts")

    # Check 3: Same hook type 3+ times in last 5
    hook_count = sum(1 for e in recent[:5] if e.hook_type == hook_type)
    if hook_count >= 3:
        penalty = 2.5 * variation_aggressiveness
        score -= penalty
        reasons.append(f"hook type '{hook_type}' used {hook_count}/5 recent posts")

    # Check 4: Same CTA type consecutively
    if recent and recent[0].cta_type == cta_type and cta_type != "none":
        penalty = 1.0 * variation_aggressiveness
        score -= penalty
        reasons.append(f"CTA type '{cta_type}' used in previous post")

    # Check 5: Same body structure opening in last 3 posts
    body_key = tuple(body_structure[:2]) if body_structure else ()
    recent_body_keys = [tuple(e.body_structure[:2]) for e in recent[:3]]
    if body_key and body_key in recent_body_keys:
        penalty = 1.5 * variation_aggressiveness
        score -= penalty
        reasons.append(f"body structure opening {list(body_key)} repeated in last 3 posts")

    score = max(score, 0.0)

    # Rejection threshold scales with aggressiveness
    # At 0.6 aggressiveness, reject below 4.0
    # At 1.0, reject below 5.0
    # At 0.0, never reject (threshold = 0)
    rejection_threshold = 5.0 * variation_aggressiveness
    should_reject = score < rejection_threshold

    if reasons:
        logger.info(
            "Diversity check: score=%.1f, reject=%s, reasons=%s",
            score, should_reject, reasons,
        )

    return {
        "diversity_score": round(score, 1),
        "should_reject": should_reject,
        "reasons": reasons,
    }


def get_diversity_summary(days: int = 7) -> dict:
    """Return a summary of structural diversity over recent posts.

    Useful for monitoring and the /diversity command.
    """
    cutoff = time.time() - (days * 86400)
    recent = get_recent_structures(_MAX_ENTRIES)
    in_range = [e for e in recent if e.timestamp >= cutoff]

    if not in_range:
        return {
            "period_days": days,
            "total_posts": 0,
            "unique_skeletons": 0,
            "unique_hooks": 0,
            "unique_ctas": 0,
            "skeleton_distribution": {},
            "hook_distribution": {},
        }

    skeleton_counts: dict[str, int] = {}
    hook_counts: dict[str, int] = {}
    cta_counts: dict[str, int] = {}

    for entry in in_range:
        skeleton_counts[entry.skeleton_id] = skeleton_counts.get(entry.skeleton_id, 0) + 1
        hook_counts[entry.hook_type] = hook_counts.get(entry.hook_type, 0) + 1
        cta_counts[entry.cta_type] = cta_counts.get(entry.cta_type, 0) + 1

    return {
        "period_days": days,
        "total_posts": len(in_range),
        "unique_skeletons": len(skeleton_counts),
        "unique_hooks": len(hook_counts),
        "unique_ctas": len(cta_counts),
        "skeleton_distribution": skeleton_counts,
        "hook_distribution": hook_counts,
        "cta_distribution": cta_counts,
    }
