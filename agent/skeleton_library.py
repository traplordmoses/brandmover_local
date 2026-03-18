"""
Structural skeleton library -- content structure templates for variety enforcement.

Provides a library of content skeletons (format, hook, body structure, CTA, tone)
that the content planner selects from during generation. Works with the diversity
tracker to prevent structural repetition across posts.
"""

import logging
import random
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Skeleton:
    """A content structure template defining the shape of a post."""

    id: str
    format: str          # "single_post" | "thread" | "long_form"
    hook: str            # Hook type (question, bold_claim, statistic, etc.)
    body: list[str]      # Body section flow
    cta: str             # CTA type (question_to_audience, none, soft_ask, etc.)
    tone: str            # Tone modifier (educational, provocative, understated, etc.)
    description: str = ""  # Human-readable summary

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Skeleton":
        return cls(
            id=data.get("id", "unknown"),
            format=data.get("format", "single_post"),
            hook=data.get("hook", "cold_open"),
            body=data.get("body", []),
            cta=data.get("cta", "none"),
            tone=data.get("tone", "neutral"),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Hook types
# ---------------------------------------------------------------------------

HOOK_TYPES = (
    "question",           # Open with a question
    "bold_claim",         # Lead with a strong assertion
    "statistic",          # Lead with a surprising number
    "contrarian_take",    # "Most people think X. They're wrong."
    "cold_open",          # Start mid-thought, no preamble
    "pattern_interrupt",  # Break expectations with something unexpected
    "observation",        # "Have you noticed that..."
    "story_opener",       # Start with a specific moment or scene
)

# ---------------------------------------------------------------------------
# Body structures
# ---------------------------------------------------------------------------

BODY_STRUCTURES = (
    "single_insight",       # One idea, briefly expanded
    "listicle",             # Numbered or bulleted points
    "problem_solution",     # State problem, provide answer
    "narrative_arc",        # Setup, tension, resolution
    "compare_contrast",     # Two things side by side
    "building_argument",    # Premise, evidence, conclusion
    "chronological",        # Past to present or step by step
    "reframe",              # Show familiar thing from new angle
)

# ---------------------------------------------------------------------------
# CTA types
# ---------------------------------------------------------------------------

CTA_TYPES = (
    "question_to_audience",  # Ask readers something
    "soft_ask",              # Gentle nudge toward action
    "hard_ask",              # Direct call to action
    "link_drop",             # Share a link
    "save_this",             # "Bookmark for later"
    "none",                  # Pure value, no ask
    "engagement_prompt",     # Invite replies or interaction
)

# ---------------------------------------------------------------------------
# Seed skeleton library (15 proven structures)
# ---------------------------------------------------------------------------

SEED_SKELETONS: list[Skeleton] = [
    # --- Single post formats ---
    Skeleton(
        id="quiet_value",
        format="single_post",
        hook="cold_open",
        body=["single_insight", "brief_context", "practical_takeaway"],
        cta="none",
        tone="understated + educational",
        description="Start mid-thought. One insight with brief context. No ask.",
    ),
    Skeleton(
        id="data_punch",
        format="single_post",
        hook="statistic",
        body=["context_for_stat", "why_it_matters", "what_to_do"],
        cta="save_this",
        tone="educational + urgent",
        description="Lead with a surprising number. Explain why it matters.",
    ),
    Skeleton(
        id="bold_declaration",
        format="single_post",
        hook="bold_claim",
        body=["single_insight", "brief_evidence"],
        cta="none",
        tone="authoritative + declarative",
        description="State a truth. One line of support. Period.",
    ),
    Skeleton(
        id="question_flip",
        format="single_post",
        hook="question",
        body=["reframe", "practical_takeaway"],
        cta="question_to_audience",
        tone="curious + provocative",
        description="Ask a question. Flip the expected answer. Ask back.",
    ),
    Skeleton(
        id="observation_post",
        format="single_post",
        hook="observation",
        body=["single_insight", "brief_context"],
        cta="none",
        tone="dry_wit + observational",
        description="Notice something others missed. Say it plainly.",
    ),
    Skeleton(
        id="pattern_break",
        format="single_post",
        hook="pattern_interrupt",
        body=["reframe", "single_insight"],
        cta="engagement_prompt",
        tone="playful + surprising",
        description="Break expectations. Reframe the familiar. Invite reaction.",
    ),
    Skeleton(
        id="micro_lesson",
        format="single_post",
        hook="cold_open",
        body=["problem_solution"],
        cta="soft_ask",
        tone="educational + concise",
        description="Jump straight into a small lesson. Problem, then answer.",
    ),
    Skeleton(
        id="cultural_commentary",
        format="single_post",
        hook="observation",
        body=["compare_contrast"],
        cta="none",
        tone="dry_wit + philosophical",
        description="Observe a cultural pattern. Compare two things. Let it sit.",
    ),

    # --- Thread formats ---
    Skeleton(
        id="contrarian_thread",
        format="thread",
        hook="contrarian_take",
        body=["bold_claim", "evidence_1", "evidence_2", "reframe", "nuance"],
        cta="question_to_audience",
        tone="provocative + authoritative",
        description="Challenge conventional wisdom. Build the case. Add nuance.",
    ),
    Skeleton(
        id="listicle_thread",
        format="thread",
        hook="bold_claim",
        body=["listicle"],
        cta="save_this",
        tone="educational + structured",
        description="Strong opener. Numbered points. Save for later.",
    ),
    Skeleton(
        id="story_thread",
        format="thread",
        hook="story_opener",
        body=["narrative_arc"],
        cta="soft_ask",
        tone="vulnerable + authentic",
        description="Start with a specific moment. Build tension. Extract the lesson.",
    ),
    Skeleton(
        id="building_case_thread",
        format="thread",
        hook="question",
        body=["building_argument"],
        cta="engagement_prompt",
        tone="analytical + persuasive",
        description="Pose a question. Build the argument piece by piece. Invite debate.",
    ),

    # --- Long form formats ---
    Skeleton(
        id="story_to_lesson",
        format="long_form",
        hook="story_opener",
        body=["scene_setting", "tension", "turning_point", "lesson_extracted", "broader_principle"],
        cta="soft_ask",
        tone="vulnerable + authoritative",
        description="Open with a scene. Build to turning point. Extract the principle.",
    ),
    Skeleton(
        id="deep_analysis",
        format="long_form",
        hook="statistic",
        body=["context_for_stat", "building_argument", "compare_contrast", "practical_takeaway"],
        cta="question_to_audience",
        tone="analytical + educational",
        description="Lead with data. Build the analysis. Compare approaches. Takeaway.",
    ),
    Skeleton(
        id="chronological_breakdown",
        format="long_form",
        hook="cold_open",
        body=["chronological", "reframe", "practical_takeaway"],
        cta="engagement_prompt",
        tone="educational + narrative",
        description="Walk through a timeline. Reframe what it means. Call to action.",
    ),
]

# Index for fast lookup
_SKELETON_INDEX: dict[str, Skeleton] = {s.id: s for s in SEED_SKELETONS}


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def get_all_skeletons() -> list[Skeleton]:
    """Return the full skeleton library."""
    return list(SEED_SKELETONS)


def get_skeleton(skeleton_id: str) -> Skeleton | None:
    """Look up a skeleton by ID."""
    return _SKELETON_INDEX.get(skeleton_id)


def get_skeletons_for_format(format_type: str) -> list[Skeleton]:
    """Return all skeletons matching a format (single_post, thread, long_form)."""
    return [s for s in SEED_SKELETONS if s.format == format_type]


def _map_content_type_to_format(content_type: str) -> str:
    """Map a BrandMover content type to a skeleton format."""
    thread_types = {"thread", "educational"}
    long_form_types = {"report", "deep_dive", "analysis"}
    if content_type in thread_types:
        return "thread"
    if content_type in long_form_types:
        return "long_form"
    return "single_post"


def get_skeleton_performance(last_n: int = 50) -> dict[str, float]:
    """Get average engagement rate per skeleton_id from performance + diversity data.

    Cross-references performance.py metrics with diversity_tracker.py structure logs
    to compute which skeletons perform best.

    Returns: {skeleton_id: avg_engagement_rate}
    """
    try:
        from agent.performance import _load_performance
        from agent.diversity_tracker import get_recent_structures
    except ImportError:
        return {}

    perf_data = _load_performance()
    if not perf_data:
        return {}

    structures = get_recent_structures(last_n)
    if not structures:
        return {}

    # Build a time-indexed list of performance entries
    # Match posts by timestamp (within 60 seconds)
    match_window = 60.0
    skeleton_engagements: dict[str, list[float]] = {}

    for perf in perf_data[-last_n:]:
        perf_time = perf.get("posted_at", 0)
        engagement = perf.get("engagement_rate", 0.0)
        if not perf_time:
            continue

        # Find the closest structure entry within the match window
        best_match = None
        best_delta = match_window + 1
        for struct in structures:
            delta = abs(struct.timestamp - perf_time)
            if delta < best_delta:
                best_delta = delta
                best_match = struct

        if best_match and best_delta <= match_window:
            sid = best_match.skeleton_id
            if sid not in skeleton_engagements:
                skeleton_engagements[sid] = []
            skeleton_engagements[sid].append(engagement)

    # Compute averages
    result: dict[str, float] = {}
    for sid, rates in skeleton_engagements.items():
        if rates:
            result[sid] = sum(rates) / len(rates)

    return result


def select_skeleton(
    content_type: str,
    recent_skeleton_ids: list[str],
    variation_aggressiveness: float = 0.6,
    preferred: list[str] | None = None,
    excluded: list[str] | None = None,
    performance_weight: float = 0.1,
) -> Skeleton:
    """Select a skeleton that maximizes structural distance from recent posts.

    Args:
        content_type: The content type being generated.
        recent_skeleton_ids: IDs of skeletons used in recent posts (newest first).
        variation_aggressiveness: 0.0 (prefer consistency) to 1.0 (max variety).
        preferred: Skeleton IDs this brand prefers (boosted weight).
        excluded: Skeleton IDs to never use for this brand.
        performance_weight: Weight for boosting high-performing skeletons (0.0-1.0).
            When > 0 and performance data exists, high-performing skeletons get
            a score boost. Default 0.1 keeps it subtle so diversity still dominates.

    Returns:
        The selected Skeleton.
    """
    format_type = _map_content_type_to_format(content_type)
    candidates = get_skeletons_for_format(format_type)

    # Fall back to all skeletons if no format-specific ones
    if not candidates:
        candidates = list(SEED_SKELETONS)

    # Apply exclusions
    if excluded:
        excluded_set = set(excluded)
        candidates = [s for s in candidates if s.id not in excluded_set]

    # If all excluded, fall back to full library for this format
    if not candidates:
        candidates = get_skeletons_for_format(format_type) or list(SEED_SKELETONS)

    # Load performance data for score boosting
    perf_scores: dict[str, float] = {}
    if performance_weight > 0:
        try:
            perf_scores = get_skeleton_performance()
        except Exception as e:
            logger.debug("Failed to load skeleton performance data: %s", e)

    # Score each candidate by diversity distance
    scored: list[tuple[Skeleton, float]] = []
    for skeleton in candidates:
        score = _compute_diversity_score(
            skeleton, recent_skeleton_ids, variation_aggressiveness,
        )

        # Boost preferred skeletons
        if preferred and skeleton.id in preferred:
            score += 0.15

        # Boost high-performing skeletons (subtle, scaled by performance_weight)
        if perf_scores and skeleton.id in perf_scores:
            # Normalize engagement rate to a 0-1 boost range
            # Cap at 10% engagement rate to avoid extreme outliers
            engagement = min(perf_scores[skeleton.id], 10.0)
            score += (engagement / 10.0) * performance_weight

        scored.append((skeleton, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Weighted random from top candidates to avoid perfect determinism
    top_n = max(2, len(scored) // 3)
    top_candidates = scored[:top_n]

    weights = [max(s, 0.01) for _, s in top_candidates]
    selected = random.choices(
        [s for s, _ in top_candidates],
        weights=weights,
        k=1,
    )[0]

    logger.info(
        "Selected skeleton '%s' for %s (format=%s, candidates=%d, aggressiveness=%.1f)",
        selected.id, content_type, format_type, len(candidates), variation_aggressiveness,
    )
    return selected


def _compute_diversity_score(
    skeleton: Skeleton,
    recent_ids: list[str],
    aggressiveness: float,
) -> float:
    """Score a skeleton candidate based on distance from recent usage.

    Higher score = more diverse from recent posts.

    Penalties:
    - Same skeleton ID used recently: strong penalty (scaled by recency)
    - Same hook type in last 2 posts: moderate penalty
    - Same CTA type used consecutively: light penalty
    """
    score = 1.0

    if not recent_ids:
        return score

    # Recent skeleton lookup for hook/cta checks
    recent_skeletons = [_SKELETON_INDEX.get(rid) for rid in recent_ids]
    recent_skeletons = [s for s in recent_skeletons if s is not None]

    # Penalty 1: Same skeleton ID used recently
    for i, rid in enumerate(recent_ids):
        if rid == skeleton.id:
            recency_weight = 1.0 / (i + 1)  # More recent = stronger penalty
            penalty = 0.5 * recency_weight * aggressiveness
            score -= penalty

    # Penalty 2: Same hook type in last 2 posts
    recent_hooks = [s.hook for s in recent_skeletons[:2]]
    if skeleton.hook in recent_hooks:
        score -= 0.3 * aggressiveness

    # Penalty 3: Same CTA type used consecutively
    if recent_skeletons and recent_skeletons[0].cta == skeleton.cta:
        score -= 0.15 * aggressiveness

    # Penalty 4: Same body structure pattern in last 3 posts
    recent_body_keys = [tuple(s.body[:2]) for s in recent_skeletons[:3]]
    if tuple(skeleton.body[:2]) in recent_body_keys:
        score -= 0.2 * aggressiveness

    return max(score, 0.0)


def format_skeleton_for_prompt(skeleton: Skeleton) -> str:
    """Format a skeleton as instructions for the LLM generation prompt.

    This gets injected into the system prompt or think step so the LLM
    follows the structural template when generating content.
    """
    body_flow = " -> ".join(skeleton.body)
    lines = [
        f"Structure template: {skeleton.id}",
        f"Format: {skeleton.format}",
        f"Hook style: {skeleton.hook} (open with this type of hook)",
        f"Body flow: {body_flow}",
        f"CTA: {skeleton.cta}",
        f"Tone: {skeleton.tone}",
    ]
    if skeleton.description:
        lines.append(f"Guide: {skeleton.description}")
    return "\n".join(lines)
