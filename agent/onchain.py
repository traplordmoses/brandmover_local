"""
FOID loreboard data fetcher — uses OpenClaw scripts for on-chain data.
Fetches board state, classifies events, and formats summaries for agent prompts.
"""

import asyncio
import json
import logging
import re
import shlex
import subprocess  # nosec B404 — mitigated by allowlist + shlex
from dataclasses import dataclass, field
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class LoreboardEvent:
    """A single loreboard event."""
    event_type: str  # proposal, vote, canonization, epoch, vote_surge
    title: str
    description: str
    event_id: str = ""
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class LoreboardState:
    """Parsed state of the FOID loreboard."""
    events: list[LoreboardEvent] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    is_quiet: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

_EVENT_PATTERNS = {
    "proposal": re.compile(r"propos|submit|new\s+entry|create.*lore", re.IGNORECASE),
    "vote": re.compile(r"vote|ballot|support|endorse", re.IGNORECASE),
    "canonization": re.compile(r"canon|accepted|approved|enshrined|finalized\s+lore", re.IGNORECASE),
    "epoch": re.compile(r"epoch|cycle|round|phase\s+\d+", re.IGNORECASE),
    "vote_surge": re.compile(r"surge|spike|flurry|burst|wave\s+of\s+vote", re.IGNORECASE),
}


def classify_events(raw_data: dict) -> list[LoreboardEvent]:
    """Classify raw loreboard data into typed events.

    Handles various data shapes from OpenClaw scripts.
    """
    events: list[LoreboardEvent] = []

    # Handle array of items (tasks, activities, vault entries)
    items = []
    if isinstance(raw_data, list):
        items = raw_data
    elif isinstance(raw_data, dict):
        # Try common keys
        for key in ("events", "items", "tasks", "activities", "entries", "data"):
            if key in raw_data and isinstance(raw_data[key], list):
                items = raw_data[key]
                break
        if not items and "vault" in raw_data and isinstance(raw_data["vault"], dict):
            # read_vault.js returns vault object
            items = list(raw_data["vault"].values()) if raw_data["vault"] else []

    for item in items:
        if not isinstance(item, dict):
            continue

        title = item.get("title", item.get("name", item.get("description", "")))
        description = item.get("description", item.get("content", item.get("body", "")))
        text = f"{title} {description}"
        event_id = str(item.get("id", item.get("event_id", item.get("hash", ""))))
        timestamp = item.get("timestamp", item.get("created_at", 0))

        # Classify by pattern matching
        event_type = "unknown"
        for etype, pattern in _EVENT_PATTERNS.items():
            if pattern.search(text):
                event_type = etype
                break

        if event_type == "unknown":
            # Default: if it has a title, treat as a proposal-like entry
            event_type = "proposal" if title else "unknown"

        events.append(LoreboardEvent(
            event_type=event_type,
            title=str(title)[:200],
            description=str(description)[:500],
            event_id=event_id,
            timestamp=float(timestamp) if timestamp else 0.0,
            metadata=item,
        ))

    return events


# ---------------------------------------------------------------------------
# OpenClaw script execution
# ---------------------------------------------------------------------------


async def _run_openclaw_script(script_name: str, args: str = "") -> dict | list | None:
    """Execute an OpenClaw script and return parsed JSON output."""
    script_path = Path(settings.OPENCLAW_SCRIPTS_DIR) / script_name
    if not script_path.exists():
        logger.warning("OpenClaw script not found: %s", script_path)
        return None

    cmd = ["node", str(script_path)]
    if args:
        try:
            cmd.extend(shlex.split(args))
        except ValueError as e:
            logger.error("Invalid args for %s: %s", script_name, e)
            return None

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(script_path.parent),
        )

        if result.returncode != 0:
            logger.warning(
                "OpenClaw %s failed (exit %d): %s",
                script_name, result.returncode, result.stderr[:200],
            )
            return None

        output = result.stdout.strip()
        if not output:
            return None

        return json.loads(output)

    except subprocess.TimeoutExpired:
        logger.error("OpenClaw %s timed out", script_name)
        return None
    except json.JSONDecodeError:
        logger.warning("OpenClaw %s returned non-JSON: %s", script_name, result.stdout[:200])
        return None
    except Exception as e:
        logger.error("OpenClaw %s error: %s", script_name, e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_board_state() -> LoreboardState:
    """Fetch loreboard state via OpenClaw scripts.

    Tries read_vault.js first, then browse_tasks.js as fallback.
    """
    state = LoreboardState()

    # Primary: read_vault.js for loreboard data
    data = await _run_openclaw_script("read_vault.js")
    if data:
        state.raw_data = data if isinstance(data, dict) else {"data": data}
        state.events = classify_events(state.raw_data)
        state.is_quiet = len(state.events) == 0
        logger.info("Fetched loreboard state: %d events", len(state.events))
        return state

    # Fallback: browse_tasks.js for any activity
    data = await _run_openclaw_script("browse_tasks.js")
    if data:
        state.raw_data = data if isinstance(data, dict) else {"data": data}
        state.events = classify_events(state.raw_data)
        state.is_quiet = len(state.events) == 0
        logger.info("Fetched board state via browse_tasks: %d events", len(state.events))
        return state

    state.is_quiet = True
    state.error = "Could not fetch on-chain data from OpenClaw"
    logger.warning("Failed to fetch loreboard state from OpenClaw")
    return state


def format_onchain_summary(state: LoreboardState, posted_event_ids: list[str] | None = None) -> str:
    """Turn loreboard state into natural language for agent prompts.

    Filters out already-posted events if posted_event_ids is provided.
    """
    if state.error:
        return f"On-chain data unavailable: {state.error}"

    if state.is_quiet or not state.events:
        return (
            "The loreboard is quiet right now. No new proposals, votes, or canonizations. "
            "Consider a reflective or philosophical post about lore, identity, or culture on chain."
        )

    # Filter out already-posted events
    events = state.events
    if posted_event_ids:
        skip = set(posted_event_ids)
        events = [e for e in events if e.event_id not in skip]
        if not events:
            return (
                "All recent loreboard events have already been covered. "
                "Consider a reflective post about the loreboard's broader themes."
            )

    # Group by type
    by_type: dict[str, list[LoreboardEvent]] = {}
    for e in events:
        by_type.setdefault(e.event_type, []).append(e)

    lines = ["Here's what's happening on the FOID loreboard:\n"]

    type_labels = {
        "proposal": "New Proposals",
        "vote": "Voting Activity",
        "canonization": "Canonizations",
        "epoch": "Epoch Updates",
        "vote_surge": "Vote Surges",
    }

    for etype, label in type_labels.items():
        group = by_type.get(etype, [])
        if group:
            lines.append(f"**{label}** ({len(group)}):")
            for e in group[:3]:  # Cap at 3 per type
                lines.append(f"  - {e.title}")
                if e.description:
                    lines.append(f"    {e.description[:120]}")
            if len(group) > 3:
                lines.append(f"  ... and {len(group) - 3} more")
            lines.append("")

    # Include any unknown types
    for etype, group in by_type.items():
        if etype not in type_labels and group:
            lines.append(f"**Other Activity** ({len(group)}):")
            for e in group[:2]:
                lines.append(f"  - {e.title}")

    return "\n".join(lines)


def get_new_event_ids(state: LoreboardState, posted_event_ids: list[str] | None = None) -> list[str]:
    """Return event IDs from the state that haven't been posted yet."""
    if not state.events:
        return []
    skip = set(posted_event_ids or [])
    return [e.event_id for e in state.events if e.event_id and e.event_id not in skip]
