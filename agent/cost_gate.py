"""
Cost gate — daily budget enforcement for generation spend.

Reads generation_history.json entries, sums today's costs, and compares
against a configurable daily budget (DAILY_COST_BUDGET_USD env var).

Used as a quality gate before triggering image generation: if the daily
budget is exhausted, the caller can skip or warn before spending more.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agent.paths import STATE_DIR
from config import settings

logger = logging.getLogger(__name__)

_HISTORY_FILE = STATE_DIR / "generation_history.json"


def _read_history() -> list[dict]:
    """Read generation history entries. Returns empty list on error."""
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("cost_gate: failed to read generation_history.json: %s", e)
        return []


def _date_from_timestamp(ts: float) -> str:
    """Convert a Unix timestamp to a YYYY-MM-DD date string (UTC)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _today_utc() -> str:
    """Return today's date as YYYY-MM-DD in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_cost_budget(estimated_cost: float = 0.0) -> dict:
    """Check whether the daily cost budget allows another generation.

    Args:
        estimated_cost: The estimated cost of the next generation. If the
            remaining budget is less than this amount, ``allowed`` is False.

    Returns:
        Dictionary with keys: allowed, spent_today_usd, budget_usd,
        remaining_usd, entry_count_today.
    """
    budget = settings.DAILY_COST_BUDGET_USD
    today = _today_utc()
    entries = _read_history()

    spent = 0.0
    count = 0
    for entry in entries:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        if _date_from_timestamp(ts) == today:
            spent += entry.get("estimated_cost_usd", 0.0)
            count += 1

    spent = round(spent, 4)
    remaining = round(max(budget - spent, 0.0), 4)
    allowed = (spent + estimated_cost) <= budget

    return {
        "allowed": allowed,
        "spent_today_usd": spent,
        "budget_usd": budget,
        "remaining_usd": remaining,
        "entry_count_today": count,
    }


def get_cost_summary(days: int = 7) -> dict:
    """Return a cost summary over the last *days* days.

    Returns:
        Dictionary with keys: daily_costs (list of {date, cost_usd, count}),
        total_usd, avg_daily_usd.
    """
    entries = _read_history()

    # Bucket costs by date
    daily: dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "count": 0})
    for entry in entries:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        date_str = _date_from_timestamp(ts)
        daily[date_str]["cost_usd"] += entry.get("estimated_cost_usd", 0.0)
        daily[date_str]["count"] += 1

    # Sort by date descending, take the last N days
    sorted_dates = sorted(daily.keys(), reverse=True)[:days]
    daily_costs = [
        {
            "date": d,
            "cost_usd": round(daily[d]["cost_usd"], 4),
            "count": daily[d]["count"],
        }
        for d in sorted_dates
    ]

    total = round(sum(dc["cost_usd"] for dc in daily_costs), 4)
    avg = round(total / len(daily_costs), 4) if daily_costs else 0.0

    return {
        "daily_costs": daily_costs,
        "total_usd": total,
        "avg_daily_usd": avg,
    }
