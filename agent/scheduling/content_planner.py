"""
Content planner -- rolling 7-day content plan with automatic balancing.

Generates and maintains a content plan that balances content types,
fills gaps, and accommodates event-triggered posts.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from agent._client import get_anthropic
from agent.content_types import AGENT_SELECTABLE_TYPES
from agent.generation_history import get_recent_generations
from agent.paths import STATE_DIR
from agent.publishing.analytics import PERFORMANCE_DATA_FILE
from agent.state_manager import FileStore
from config import settings

logger = logging.getLogger(__name__)

_PLAN_FILE = STATE_DIR / "content_plan.json"
_store = FileStore(_PLAN_FILE, default_factory=dict)
_sync_lock = threading.Lock()

# Time slots available for scheduling posts
TIME_SLOTS = ("morning", "midday", "afternoon", "evening")

# Default weekly content mix (posts per week by type).
# Override via CONTENT_PLANNER_MIX env var or pass target_mix to identify_gaps.
DEFAULT_CONTENT_MIX: dict[str, int] = {
    "announcement": 2,
    "meme": 3,
    "community": 2,
    "market_commentary": 2,
    "educational": 1,
    "brand_3d": 1,
    "lifestyle": 1,
}


def _effective_mix() -> dict[str, int]:
    """Return the active content mix: settings override if set, else default."""
    return settings.CONTENT_MIX_RATIOS or DEFAULT_CONTENT_MIX


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlannedPost:
    """A single planned post within the content calendar."""

    date: str  # ISO date, e.g. "2026-03-18"
    time_slot: str  # "morning" | "midday" | "afternoon" | "evening"
    content_type: str
    prompt_hint: str = ""
    skeleton_id: str = ""  # Structural skeleton template ID
    status: str = "planned"  # planned | generating | posted | skipped | event_override
    event_source: str | None = None
    created_at: float = field(default_factory=time.time)
    posted_at: float | None = None


@dataclass
class ContentPlan:
    """A rolling content plan covering a configurable horizon."""

    week_start: str  # ISO date of the plan's start
    posts: list[PlannedPost] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    version: int = 1


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _plan_to_dict(plan: ContentPlan) -> dict:
    """Convert a ContentPlan to a JSON-safe dict."""
    return {
        "week_start": plan.week_start,
        "posts": [asdict(p) for p in plan.posts],
        "last_updated": plan.last_updated,
        "version": plan.version,
    }


def _dict_to_plan(data: dict) -> ContentPlan:
    """Reconstruct a ContentPlan from a dict."""
    posts = [PlannedPost(**p) for p in data.get("posts", [])]
    return ContentPlan(
        week_start=data.get("week_start", _today_iso()),
        posts=posts,
        last_updated=data.get("last_updated", time.time()),
        version=data.get("version", 1),
    )


def _get_tz() -> timezone | ZoneInfo:
    """Return the user-configured timezone, falling back to UTC."""
    if settings.TIMEZONE:
        try:
            return ZoneInfo(settings.TIMEZONE)
        except (KeyError, Exception):
            logger.warning("Invalid TIMEZONE '%s', falling back to UTC", settings.TIMEZONE)
    return timezone.utc


def _now() -> datetime:
    """Return the current datetime in the user-configured timezone."""
    return datetime.now(_get_tz())


def _today_iso() -> str:
    return _now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Performance-driven weighting
# ---------------------------------------------------------------------------

def _load_performance_weights() -> dict[str, float]:
    """Read analytics data and return engagement multipliers per content type.

    Reads ``state/performance_data.json`` (written by
    ``agent.publishing.analytics``), computes the average engagement rate for
    each content type, and returns a multiplier dict where 1.0 means average
    performance.  Types above average get a multiplier >1, types below get <1.

    Returns an empty dict when no measured data is available.
    """
    try:
        if not PERFORMANCE_DATA_FILE.exists():
            return {}
        data = json.loads(PERFORMANCE_DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError, OSError):
        return {}

    # Only consider posts that have been checked (have real metrics)
    measured = [p for p in data if p.get("last_checked", 0) > 0]
    if not measured:
        return {}

    # Aggregate engagement rates by content_type
    type_engagement: dict[str, list[float]] = {}
    for p in measured:
        ct = p.get("content_type") or "unknown"
        type_engagement.setdefault(ct, []).append(p.get("engagement_rate", 0.0))

    # Compute per-type average
    type_avg: dict[str, float] = {}
    for ct, rates in type_engagement.items():
        type_avg[ct] = sum(rates) / len(rates) if rates else 0.0

    # Global average across all measured posts
    all_rates = [p.get("engagement_rate", 0.0) for p in measured]
    global_avg = sum(all_rates) / len(all_rates) if all_rates else 0.0

    if global_avg <= 0:
        return {}

    # Compute multiplier: type_avg / global_avg, clamped to [0.3, 3.0]
    weights: dict[str, float] = {}
    for ct, avg in type_avg.items():
        raw = avg / global_avg
        weights[ct] = max(0.3, min(3.0, round(raw, 2)))

    logger.debug("Performance weights: %s (global avg %.2f%%)", weights, global_avg)
    return weights


def _build_performance_insights(weights: dict[str, float]) -> str:
    """Build a human-readable performance insight string for the planning prompt.

    Returns an empty string if no weights are available.
    """
    if not weights:
        return ""

    try:
        if not PERFORMANCE_DATA_FILE.exists():
            return ""
        data = json.loads(PERFORMANCE_DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError, OSError):
        return ""

    measured = [p for p in data if p.get("last_checked", 0) > 0]
    if not measured:
        return ""

    # Compute per-type stats
    type_stats: dict[str, dict] = {}
    for p in measured:
        ct = p.get("content_type") or "unknown"
        stats = type_stats.setdefault(ct, {"count": 0, "total_eng": 0.0})
        stats["count"] += 1
        stats["total_eng"] += p.get("engagement_rate", 0.0)

    lines = ["PERFORMANCE DATA (from real X/Twitter analytics):"]
    # Sort by average engagement descending
    ranked = sorted(
        type_stats.items(),
        key=lambda x: x[1]["total_eng"] / x[1]["count"] if x[1]["count"] else 0,
        reverse=True,
    )
    best_avg = ranked[0][1]["total_eng"] / ranked[0][1]["count"] if ranked else 0
    for ct, stats in ranked:
        avg = stats["total_eng"] / stats["count"] if stats["count"] else 0
        label = "your best" if avg == best_avg else (
            "below average" if weights.get(ct, 1.0) < 0.9 else "solid"
        )
        lines.append(
            f"  - '{ct}' has {avg:.1f}% avg engagement ({label}, {stats['count']} posts measured)"
        )

    lines.append("Weight the plan toward higher-performing content types.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_plan() -> ContentPlan:
    """Read the content plan from state/content_plan.json via FileStore."""
    with _sync_lock:
        data = _store.read()
        if not data or "week_start" not in data:
            return ContentPlan(week_start=_today_iso())
        return _dict_to_plan(data)


def save_plan(plan: ContentPlan) -> None:
    """Write the content plan to disk."""
    plan.last_updated = time.time()
    with _sync_lock:
        _store.write(_plan_to_dict(plan))
    logger.info(
        "Content plan saved: %d posts, week_start=%s",
        len(plan.posts), plan.week_start,
    )


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

def get_content_type_distribution(days: int = 7) -> dict[str, int]:
    """Count recent posts by content_type from generation_history.

    Looks back *days* days and counts entries with status "approved" or "posted".
    """
    cutoff = time.time() - (days * 86400)
    entries = get_recent_generations(n=200)
    counts: dict[str, int] = {}
    for entry in entries:
        ts = entry.get("timestamp", 0)
        status = entry.get("status", "")
        if ts >= cutoff and status in ("approved", "posted", "draft"):
            ct = entry.get("content_type", "default")
            counts[ct] = counts.get(ct, 0) + 1
    return counts


def identify_gaps(
    distribution: dict[str, int],
    target_mix: dict[str, int] | None = None,
    performance_weights: dict[str, float] | None = None,
) -> list[str]:
    """Find underrepresented content types relative to the target mix.

    Returns a list of content types sorted by how far below target they are
    (most underrepresented first).  When *performance_weights* are provided
    (from ``_load_performance_weights()``), the raw deficit is multiplied by
    the engagement multiplier so high-performing types bubble up.
    """
    mix = target_mix or _effective_mix()
    if performance_weights is None:
        performance_weights = _load_performance_weights()
    gaps: list[tuple[str, float]] = []
    for content_type, target_count in mix.items():
        # Only consider types the agent can actually produce
        if content_type not in AGENT_SELECTABLE_TYPES:
            continue
        actual = distribution.get(content_type, 0)
        deficit = target_count - actual
        if deficit > 0:
            # Multiply deficit by engagement weight so proven types get priority
            weight = performance_weights.get(content_type, 1.0)
            gaps.append((content_type, deficit * weight))
    # Sort by weighted deficit (largest first)
    gaps.sort(key=lambda x: x[1], reverse=True)
    return [ct for ct, _ in gaps]


# ---------------------------------------------------------------------------
# Plan generation via Claude Haiku
# ---------------------------------------------------------------------------

async def generate_plan(days_ahead: int = 7) -> ContentPlan:
    """Use Claude Haiku to generate a content plan based on gaps and history.

    Builds a prompt with the current distribution, identified gaps, and
    available content types, then asks Haiku to produce a structured plan.
    """
    horizon = days_ahead if days_ahead > 0 else settings.PLAN_HORIZON_DAYS
    distribution = get_content_type_distribution(days=7)

    # Load engagement-based performance weights from analytics data
    perf_weights = _load_performance_weights()

    gaps = identify_gaps(distribution, performance_weights=perf_weights)

    # Load brand context for flavor
    brand_context = ""
    try:
        from agent.guidelines import get_brand_context
        brand_context = get_brand_context()[:2000]
    except Exception:
        pass

    today = _today_iso()
    dates = [
        (_now() + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]

    prompt = (
        f"You are a social media content planner. Generate a {horizon}-day content plan.\n\n"
        f"Today: {today}\n"
        f"Dates to fill: {', '.join(dates)}\n"
        f"Available time slots: {', '.join(TIME_SLOTS)}\n"
        f"Available content types: {', '.join(AGENT_SELECTABLE_TYPES)}\n\n"
        f"Recent 7-day distribution: {json.dumps(distribution)}\n"
        f"Underrepresented types (fill these first): {', '.join(gaps) if gaps else 'none'}\n\n"
        f"Target weekly mix: {json.dumps(_effective_mix())}\n\n"
    )
    if brand_context:
        prompt += f"Brand context (brief):\n{brand_context[:1000]}\n\n"

    # Inject detailed analytics-based performance insights
    perf_insights = _build_performance_insights(perf_weights)
    if perf_insights:
        prompt += f"\n{perf_insights}\n\n"

    # Also inject the legacy performance context as a fallback
    try:
        from agent.performance import get_performance_context
        perf_context = get_performance_context()
        if perf_context and not perf_insights:
            prompt += f"\nPerformance insight: {perf_context}\n\n"
    except Exception:
        pass

    prompt += (
        "Generate 2-3 posts per day spread across time slots. "
        "Favor content types with proven high engagement. "
        "Return ONLY a JSON array of objects with keys: "
        '"date", "time_slot", "content_type", "prompt_hint" (a brief topic/angle). '
        "No markdown, no explanation -- just the JSON array."
    )

    client = get_anthropic()
    try:
        response = await client.messages.create(
            model=settings.HAIKU_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

        items = json.loads(raw)
        posts = []
        for item in items:
            ct = item.get("content_type", "announcement")
            if ct not in AGENT_SELECTABLE_TYPES:
                ct = "announcement"
            slot = item.get("time_slot", "morning")
            if slot not in TIME_SLOTS:
                slot = "morning"
            date = item.get("date", today)
            posts.append(PlannedPost(
                date=date,
                time_slot=slot,
                content_type=ct,
                prompt_hint=item.get("prompt_hint", ""),
            ))

        # Assign structural skeletons if enabled
        _assign_skeletons_to_posts(posts)

        plan = ContentPlan(week_start=today, posts=posts)
        save_plan(plan)
        logger.info("Generated content plan: %d posts over %d days", len(posts), horizon)
        return plan

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("Failed to parse Haiku plan response: %s", e)
        # Fall back to a basic algorithmic plan
        return _generate_fallback_plan(dates, gaps)
    except Exception as e:
        logger.error("Haiku plan generation failed: %s", e)
        return _generate_fallback_plan(dates, gaps)


def _generate_fallback_plan(dates: list[str], gaps: list[str]) -> ContentPlan:
    """Build a simple round-robin plan without LLM assistance."""
    types_pool = gaps if gaps else list(_effective_mix().keys())
    posts = []
    idx = 0
    for date in dates:
        # 2 posts per day in different slots
        for slot in ("morning", "afternoon"):
            ct = types_pool[idx % len(types_pool)]
            posts.append(PlannedPost(
                date=date,
                time_slot=slot,
                content_type=ct,
                prompt_hint="",
            ))
            idx += 1

    # Assign structural skeletons if enabled
    _assign_skeletons_to_posts(posts)

    plan = ContentPlan(week_start=dates[0] if dates else _today_iso(), posts=posts)
    save_plan(plan)
    logger.info("Fallback plan generated: %d posts", len(posts))
    return plan


def _assign_skeletons_to_posts(posts: list[PlannedPost]) -> None:
    """Assign structural skeletons to planned posts using diversity-aware selection.

    Modifies posts in place, setting skeleton_id on each.
    Skipped if SKELETON_LIBRARY_ENABLED is False.
    """
    if not settings.SKELETON_LIBRARY_ENABLED:
        return

    try:
        from agent.skeleton_library import select_skeleton
        from agent.diversity_tracker import get_recent_skeleton_ids
        from agent.compositor_config import get_config

        brand_config = get_config()
        recent_ids = get_recent_skeleton_ids(10)

        for post in posts:
            skeleton = select_skeleton(
                content_type=post.content_type,
                recent_skeleton_ids=recent_ids,
                variation_aggressiveness=brand_config.variation_aggressiveness,
                preferred=brand_config.preferred_skeletons or None,
                excluded=brand_config.excluded_skeletons or None,
                performance_weight=0.3,
            )
            post.skeleton_id = skeleton.id
            # Track this assignment so the next post considers it
            recent_ids.insert(0, skeleton.id)
    except Exception as e:
        logger.warning("Skeleton assignment failed, posts will generate without structure: %s", e)


# ---------------------------------------------------------------------------
# Daily plan maintenance
# ---------------------------------------------------------------------------

async def update_plan_daily() -> ContentPlan:
    """Drop past posts, extend the plan to maintain the configured horizon.

    Called once per day from the scheduler housekeeping section.
    """
    plan = load_plan()
    today = _today_iso()
    horizon = settings.PLAN_HORIZON_DAYS

    # Remove posts before today
    original_count = len(plan.posts)
    plan.posts = [p for p in plan.posts if p.date >= today]
    dropped = original_count - len(plan.posts)
    if dropped:
        logger.info("Dropped %d past posts from content plan", dropped)

    # Check how many days are covered
    covered_dates = {p.date for p in plan.posts}
    end_date = _now() + timedelta(days=horizon)
    needed_dates = []
    for i in range(horizon):
        d = (datetime.now(timezone.utc) + timedelta(days=i)).strftime("%Y-%m-%d")
        if d not in covered_dates:
            needed_dates.append(d)

    if needed_dates:
        logger.info("Extending plan to cover %d new dates", len(needed_dates))
        distribution = get_content_type_distribution(days=7)
        perf_weights = _load_performance_weights()
        gaps = identify_gaps(distribution, performance_weights=perf_weights)
        types_pool = gaps if gaps else list(_effective_mix().keys())
        idx = 0
        for date in needed_dates:
            for slot in ("morning", "afternoon"):
                ct = types_pool[idx % len(types_pool)]
                plan.posts.append(PlannedPost(
                    date=date,
                    time_slot=slot,
                    content_type=ct,
                    prompt_hint="",
                ))
                idx += 1

    plan.week_start = today
    plan.version += 1
    save_plan(plan)
    return plan


# ---------------------------------------------------------------------------
# Post retrieval and status management
# ---------------------------------------------------------------------------

def get_next_planned_post() -> PlannedPost | None:
    """Return the next due post (status='planned', date <= today, earliest slot).

    Slot ordering: morning < midday < afternoon < evening.
    """
    plan = load_plan()
    today = _today_iso()
    slot_order = {s: i for i, s in enumerate(TIME_SLOTS)}

    candidates = [
        p for p in plan.posts
        if p.status == "planned" and p.date <= today
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda p: (p.date, slot_order.get(p.time_slot, 99)))
    return candidates[0]


def mark_post_status(date: str, time_slot: str, status: str) -> None:
    """Update a post's status by date and time_slot."""
    plan = load_plan()
    for post in plan.posts:
        if post.date == date and post.time_slot == time_slot:
            post.status = status
            if status == "posted":
                post.posted_at = time.time()
            save_plan(plan)
            logger.info("Marked %s/%s as %s", date, time_slot, status)
            return
    logger.warning("Post not found: %s/%s", date, time_slot)


def insert_event_post(
    title: str,
    content_type: str,
    prompt_hint: str,
) -> PlannedPost:
    """Insert an event-triggered post into today's plan.

    Event posts get status='planned' with event_source set, and are
    prioritized by get_next_planned_post (today + earliest available slot).
    """
    plan = load_plan()
    today = _today_iso()

    # Find the first unused slot today
    used_slots = {
        p.time_slot for p in plan.posts
        if p.date == today and p.status not in ("skipped",)
    }
    chosen_slot = "morning"
    for slot in TIME_SLOTS:
        if slot not in used_slots:
            chosen_slot = slot
            break

    # Validate content_type
    ct = content_type if content_type in AGENT_SELECTABLE_TYPES else "announcement"

    post = PlannedPost(
        date=today,
        time_slot=chosen_slot,
        content_type=ct,
        prompt_hint=prompt_hint or title,
        status="planned",
        event_source=title,
    )
    plan.posts.append(post)
    save_plan(plan)
    logger.info("Inserted event post: %s (%s) at %s/%s", title, ct, today, chosen_slot)
    return post


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

_plan_lock = asyncio.Lock()


async def async_generate_plan(days_ahead: int = 7) -> ContentPlan:
    async with _plan_lock:
        return await generate_plan(days_ahead)


async def async_update_plan_daily() -> ContentPlan:
    async with _plan_lock:
        return await update_plan_daily()
