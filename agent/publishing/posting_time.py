"""
Optimal posting time analysis — learns the best times to post from engagement data.

Reads performance_data.json and auto_post_state.json to build a heatmap of
engagement by hour-of-day and day-of-week. Recommends optimal posting slots
based on simple statistics (averages, standard deviations).

Public API:
    analysis = analyze_posting_times()
    slot = get_optimal_slot("community")
    context = format_timing_context()
"""

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agent.paths import STATE_DIR

logger = logging.getLogger(__name__)

_PERFORMANCE_FILE = STATE_DIR / "performance_data.json"
_AUTO_POST_FILE = STATE_DIR / "auto_post_state.json"

_MIN_POSTS_FOR_RECOMMENDATIONS = 10

_DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_DEFAULT_BEST_HOURS = [9, 12, 17]
_DEFAULT_BEST_DAYS = ["tuesday", "wednesday", "thursday"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_performance_data() -> list[dict]:
    """Load performance_data.json."""
    if not _PERFORMANCE_FILE.exists():
        return []
    try:
        data = json.loads(_PERFORMANCE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _load_auto_post_state() -> dict:
    """Load auto_post_state.json."""
    if not _AUTO_POST_FILE.exists():
        return {}
    try:
        return json.loads(_AUTO_POST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_post_records() -> list[dict]:
    """Merge performance data and auto-post state into a unified list of
    post records with ``posted_at``, ``engagement_rate``, and optional
    ``content_type``.
    """
    records: list[dict] = []

    # Source 1: performance_data.json (has engagement metrics)
    for entry in _load_performance_data():
        posted_at = entry.get("posted_at", 0)
        if not posted_at:
            continue
        records.append({
            "posted_at": posted_at,
            "engagement_rate": entry.get("engagement_rate", 0.0),
            "likes": entry.get("likes", 0),
            "retweets": entry.get("retweets", 0),
            "replies": entry.get("replies", 0),
            "impressions": entry.get("impressions", 0),
            "content_type": entry.get("content_type", ""),
        })

    # Source 2: auto_post_state.json (may have posts not yet in perf data)
    state = _load_auto_post_state()
    existing_times = {r["posted_at"] for r in records}
    for post in state.get("posts_today", []):
        ts = post.get("timestamp", 0)
        if ts and ts not in existing_times:
            records.append({
                "posted_at": ts,
                "engagement_rate": 0.0,
                "likes": 0,
                "retweets": 0,
                "replies": 0,
                "impressions": 0,
                "content_type": "",
            })

    return records


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_posting_times() -> dict:
    """Analyze engagement data to find optimal posting times.

    Groups posts by hour-of-day and day-of-week. Calculates average
    engagement rate for each bucket.

    Returns:
        {
            "best_hours": [int, ...],       # Top 3 hours by avg engagement
            "best_days": [str, ...],        # Top 3 days by avg engagement
            "heatmap": {day: {hour: rate}}, # Full heatmap
            "recommendations": [str, ...],  # Human-readable tips
            "total_posts": int,
            "sufficient_data": bool,
        }
    """
    records = _extract_post_records()
    total_posts = len(records)

    if total_posts < _MIN_POSTS_FOR_RECOMMENDATIONS:
        return {
            "best_hours": _DEFAULT_BEST_HOURS,
            "best_days": _DEFAULT_BEST_DAYS,
            "heatmap": {},
            "recommendations": [
                f"Only {total_posts} posts tracked so far (need {_MIN_POSTS_FOR_RECOMMENDATIONS}+). "
                "Using industry-standard defaults: Tue-Thu, 9am/12pm/5pm.",
                "Recommendations will improve as more posts are published and engagement data is collected.",
            ],
            "total_posts": total_posts,
            "sufficient_data": False,
        }

    # Group by (day_of_week, hour)
    day_hour_rates: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    day_rates: dict[str, list[float]] = defaultdict(list)
    hour_rates: dict[int, list[float]] = defaultdict(list)

    for record in records:
        posted_at = record["posted_at"]
        engagement = record.get("engagement_rate", 0.0)

        try:
            dt = datetime.fromtimestamp(posted_at)
        except (OSError, ValueError, OverflowError):
            continue

        day_name = _DAY_NAMES[dt.weekday()]
        hour = dt.hour

        day_hour_rates[day_name][hour].append(engagement)
        day_rates[day_name].append(engagement)
        hour_rates[hour].append(engagement)

    # Calculate averages
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Best hours (top 3 by avg engagement)
    hour_avgs = {h: _avg(rates) for h, rates in hour_rates.items()}
    best_hours = sorted(hour_avgs, key=lambda h: hour_avgs[h], reverse=True)[:3]

    # Best days (top 3 by avg engagement)
    day_avgs = {d: _avg(rates) for d, rates in day_rates.items()}
    best_days = sorted(day_avgs, key=lambda d: day_avgs[d], reverse=True)[:3]

    # Build full heatmap
    heatmap: dict[str, dict[str, float]] = {}
    for day_name in _DAY_NAMES:
        heatmap[day_name] = {}
        for hour in range(24):
            rates = day_hour_rates[day_name][hour]
            heatmap[day_name][str(hour)] = round(_avg(rates), 4) if rates else 0.0

    # Generate recommendations
    recommendations = _build_recommendations(
        best_hours, best_days, hour_avgs, day_avgs, total_posts,
    )

    return {
        "best_hours": best_hours,
        "best_days": best_days,
        "heatmap": heatmap,
        "recommendations": recommendations,
        "total_posts": total_posts,
        "sufficient_data": True,
    }


def _build_recommendations(
    best_hours: list[int],
    best_days: list[str],
    hour_avgs: dict[int, float],
    day_avgs: dict[str, float],
    total_posts: int,
) -> list[str]:
    """Build human-readable recommendation strings."""
    tips: list[str] = []

    if best_hours:
        formatted_hours = ", ".join(
            f"{h}:00 ({hour_avgs[h]:.2f}%)" for h in best_hours
        )
        tips.append(f"Best posting hours: {formatted_hours}")

    if best_days:
        formatted_days = ", ".join(
            f"{d.capitalize()} ({day_avgs[d]:.2f}%)" for d in best_days
        )
        tips.append(f"Best posting days: {formatted_days}")

    # Worst day/hour warnings
    if day_avgs:
        worst_day = min(day_avgs, key=lambda d: day_avgs[d])
        if day_avgs.get(worst_day, 0) > 0:
            tips.append(
                f"Lowest engagement on {worst_day.capitalize()} "
                f"({day_avgs[worst_day]:.2f}%) — consider skipping or reducing posts."
            )

    # Weekend vs weekday comparison
    weekday_rates = [r for d, r in day_avgs.items() if d not in ("saturday", "sunday") and r > 0]
    weekend_rates = [r for d, r in day_avgs.items() if d in ("saturday", "sunday") and r > 0]
    if weekday_rates and weekend_rates:
        weekday_avg = sum(weekday_rates) / len(weekday_rates)
        weekend_avg = sum(weekend_rates) / len(weekend_rates)
        if weekday_avg > weekend_avg * 1.3:
            tips.append("Weekdays outperform weekends by 30%+ — focus content on weekdays.")
        elif weekend_avg > weekday_avg * 1.3:
            tips.append("Weekends outperform weekdays by 30%+ — consider weekend posting.")

    tips.append(f"Based on {total_posts} tracked posts.")
    return tips


# ---------------------------------------------------------------------------
# Optimal slot recommendation
# ---------------------------------------------------------------------------

def get_optimal_slot(content_type: str = "") -> dict:
    """Recommend the best next slot for a given content type.

    Args:
        content_type: Optional content type for type-specific analysis.

    Returns:
        {
            "day": "monday",
            "hour": 14,
            "reason": "Community posts get 3.2x more engagement on Monday afternoons",
        }
    """
    records = _extract_post_records()
    total_posts = len(records)

    if total_posts < _MIN_POSTS_FOR_RECOMMENDATIONS:
        return {
            "day": "wednesday",
            "hour": 12,
            "reason": (
                f"Insufficient data ({total_posts} posts). "
                "Defaulting to Wednesday noon — a strong general-purpose slot."
            ),
        }

    # Filter by content type if provided
    if content_type:
        type_records = [r for r in records if r.get("content_type") == content_type]
        # Fall back to all records if fewer than 5 of this type
        if len(type_records) >= 5:
            records = type_records

    # Find the best (day, hour) combination
    day_hour_rates: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        try:
            dt = datetime.fromtimestamp(record["posted_at"])
        except (OSError, ValueError, OverflowError):
            continue
        day_name = _DAY_NAMES[dt.weekday()]
        hour = dt.hour
        day_hour_rates[(day_name, hour)].append(record.get("engagement_rate", 0.0))

    if not day_hour_rates:
        return {
            "day": "wednesday",
            "hour": 12,
            "reason": "No valid timestamps found. Defaulting to Wednesday noon.",
        }

    # Pick the slot with the highest average engagement (min 2 data points)
    best_slot = None
    best_avg = -1.0
    for (day, hour), rates in day_hour_rates.items():
        if len(rates) < 2:
            continue
        avg = sum(rates) / len(rates)
        if avg > best_avg:
            best_avg = avg
            best_slot = (day, hour)

    # If no slot has 2+ data points, just pick the overall best
    if best_slot is None:
        best_slot = max(
            day_hour_rates,
            key=lambda k: sum(day_hour_rates[k]) / len(day_hour_rates[k]),
        )
        rates = day_hour_rates[best_slot]
        best_avg = sum(rates) / len(rates)

    day, hour = best_slot

    # Build reason
    type_qualifier = f"{content_type} posts" if content_type else "Posts"
    period = "mornings" if hour < 12 else "afternoons" if hour < 17 else "evenings"
    reason = (
        f"{type_qualifier} get {best_avg:.2f}% avg engagement on "
        f"{day.capitalize()} {period} ({hour}:00)."
    )

    return {
        "day": day,
        "hour": hour,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Context formatting for heartbeat / system prompt
# ---------------------------------------------------------------------------

def format_timing_context() -> str:
    """Human-readable summary for injection into the heartbeat's decision prompt.

    Returns a compact multi-line string summarizing optimal posting times
    and key engagement patterns.
    """
    analysis = analyze_posting_times()

    if not analysis["sufficient_data"]:
        return (
            f"Timing data: {analysis['total_posts']} posts tracked "
            f"(need {_MIN_POSTS_FOR_RECOMMENDATIONS}+ for recommendations). "
            "Using defaults: Tue-Thu at 9am, 12pm, 5pm."
        )

    lines = [f"Timing analysis ({analysis['total_posts']} posts):"]

    # Best hours
    best_hours = analysis["best_hours"]
    if best_hours:
        hour_strs = [f"{h}:00" for h in best_hours]
        lines.append(f"  Peak hours: {', '.join(hour_strs)}")

    # Best days
    best_days = analysis["best_days"]
    if best_days:
        day_strs = [d.capitalize() for d in best_days]
        lines.append(f"  Peak days: {', '.join(day_strs)}")

    # Add top 2 recommendations
    for rec in analysis["recommendations"][:2]:
        lines.append(f"  - {rec}")

    return "\n".join(lines)
