"""
Campaign runbooks — multi-day content campaign orchestration.

A campaign is a named, time-bounded sequence of posts with:
- A theme/brief that ties all posts together
- Daily content slots with specific content types and angles
- Content mix enforcement (e.g., 30% educational, 20% lifestyle)
- Progress tracking and completion status

Campaigns live in state/campaigns.json and are managed via /campaign commands.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from agent.paths import STATE_DIR

logger = logging.getLogger(__name__)

_CAMPAIGNS_FILE = STATE_DIR / "campaigns.json"


@dataclass
class CampaignSlot:
    """A single content slot within a campaign day."""
    day: int                          # Day number (1-indexed)
    content_type: str                 # e.g., "announcement", "educational"
    angle: str                        # Specific angle/topic for this slot
    status: str = "pending"           # "pending" | "drafted" | "approved" | "posted"
    draft_timestamp: float = 0.0
    post_url: str = ""


@dataclass
class Campaign:
    """A multi-day content campaign."""
    name: str                         # Unique campaign identifier
    brief: str                        # Campaign theme/objective
    start_date: str                   # ISO date string (YYYY-MM-DD)
    duration_days: int                # Total campaign length
    content_mix: dict[str, int] = field(default_factory=dict)  # type → weight
    slots: list[dict] = field(default_factory=list)            # List of CampaignSlot dicts
    status: str = "active"            # "active" | "paused" | "completed"
    created_at: float = 0.0
    kpis: dict[str, str] = field(default_factory=dict)  # KPI name → target


def _read_campaigns() -> dict:
    """Read campaigns from disk."""
    if not _CAMPAIGNS_FILE.exists():
        return {"campaigns": []}
    try:
        return json.loads(_CAMPAIGNS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read campaigns.json: %s", e)
        return {"campaigns": []}


def _write_campaigns(data: dict) -> None:
    """Write campaigns to disk."""
    _CAMPAIGNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CAMPAIGNS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def create_campaign(
    name: str,
    brief: str,
    duration_days: int,
    content_mix: dict[str, int] | None = None,
    start_date: str = "",
    kpis: dict[str, str] | None = None,
) -> dict:
    """Create a new campaign with auto-generated daily slots.

    Slots are distributed across days using the content mix ratios.
    Default: 1 post per day, content type selected by mix weights.

    Returns:
        {"success": bool, "campaign": dict, "message": str}
    """
    data = _read_campaigns()

    # Check for duplicate name
    for c in data["campaigns"]:
        if c.get("name") == name:
            return {"success": False, "campaign": None, "message": f"Campaign '{name}' already exists."}

    if not start_date:
        start_date = time.strftime("%Y-%m-%d")

    if not content_mix:
        from config import settings
        content_mix = settings.CONTENT_MIX_RATIOS or {
            "educational": 25, "community": 20, "announcement": 20,
            "engagement": 15, "lifestyle": 10, "meme": 10,
        }

    # Generate slots using weighted distribution
    import random
    types = list(content_mix.keys())
    weights = list(content_mix.values())
    slots = []
    for day in range(1, duration_days + 1):
        selected_type = random.choices(types, weights=weights, k=1)[0]
        slots.append({
            "day": day,
            "content_type": selected_type,
            "angle": "",  # Agent fills this in based on the brief
            "status": "pending",
            "draft_timestamp": 0.0,
            "post_url": "",
        })

    campaign = Campaign(
        name=name,
        brief=brief,
        start_date=start_date,
        duration_days=duration_days,
        content_mix=content_mix,
        slots=slots,
        status="active",
        created_at=time.time(),
        kpis=kpis or {},
    )

    data["campaigns"].append(asdict(campaign))
    _write_campaigns(data)
    logger.info("Created campaign '%s': %d days, %d slots", name, duration_days, len(slots))

    return {"success": True, "campaign": asdict(campaign), "message": f"Campaign '{name}' created with {len(slots)} slots."}


def get_campaign(name: str) -> dict | None:
    """Get a campaign by name."""
    data = _read_campaigns()
    for c in data["campaigns"]:
        if c.get("name") == name:
            return c
    return None


def list_campaigns(status_filter: str | None = None) -> list[dict]:
    """List all campaigns, optionally filtered by status."""
    data = _read_campaigns()
    campaigns = data.get("campaigns", [])
    if status_filter:
        campaigns = [c for c in campaigns if c.get("status") == status_filter]
    return campaigns


def update_slot_status(campaign_name: str, day: int, new_status: str,
                       post_url: str = "") -> bool:
    """Update a slot's status within a campaign. Returns True if found."""
    data = _read_campaigns()
    for campaign in data["campaigns"]:
        if campaign.get("name") != campaign_name:
            continue
        for slot in campaign.get("slots", []):
            if slot.get("day") == day:
                slot["status"] = new_status
                if new_status == "drafted":
                    slot["draft_timestamp"] = time.time()
                if post_url:
                    slot["post_url"] = post_url
                # Check if campaign is completed (all slots posted)
                all_done = all(s.get("status") == "posted" for s in campaign["slots"])
                if all_done:
                    campaign["status"] = "completed"
                _write_campaigns(data)
                logger.info("Campaign '%s' day %d → %s", campaign_name, day, new_status)
                return True
    return False


def get_next_pending_slot(campaign_name: str) -> dict | None:
    """Get the next pending slot in a campaign."""
    campaign = get_campaign(campaign_name)
    if not campaign:
        return None
    for slot in campaign.get("slots", []):
        if slot.get("status") == "pending":
            return slot
    return None


def get_campaign_progress(campaign_name: str) -> dict:
    """Get progress summary for a campaign."""
    campaign = get_campaign(campaign_name)
    if not campaign:
        return {"error": "Campaign not found"}

    slots = campaign.get("slots", [])
    by_status: dict[str, int] = {}
    for s in slots:
        st = s.get("status", "pending")
        by_status[st] = by_status.get(st, 0) + 1

    total = len(slots)
    posted = by_status.get("posted", 0)
    progress_pct = round(posted / total * 100) if total else 0

    return {
        "name": campaign.get("name"),
        "brief": campaign.get("brief", "")[:200],
        "status": campaign.get("status"),
        "total_slots": total,
        "by_status": by_status,
        "progress_pct": progress_pct,
        "kpis": campaign.get("kpis", {}),
    }


def format_campaign_status(campaign_name: str) -> str:
    """Format a campaign's status for display in Telegram."""
    progress = get_campaign_progress(campaign_name)
    if "error" in progress:
        return progress["error"]

    campaign = get_campaign(campaign_name)
    lines = [
        f"<b>{progress['name']}</b> — {progress['status'].upper()}",
        f"{progress['brief'][:100]}",
        "",
        f"Progress: {progress['progress_pct']}% ({progress['by_status'].get('posted', 0)}/{progress['total_slots']} posted)",
    ]

    # Show per-status breakdown
    for status, count in progress["by_status"].items():
        icon = {"pending": "⏳", "drafted": "📝", "approved": "✅", "posted": "📤"}.get(status, "•")
        lines.append(f"  {icon} {status}: {count}")

    # Show KPIs if defined
    kpis = progress.get("kpis", {})
    if kpis:
        lines.append("")
        lines.append("<b>KPIs:</b>")
        for k, v in kpis.items():
            lines.append(f"  • {k}: {v}")

    # Show next pending slot
    if campaign:
        next_slot = get_next_pending_slot(campaign_name)
        if next_slot:
            lines.append("")
            lines.append(f"Next: Day {next_slot['day']} — {next_slot['content_type']}")
            if next_slot.get("angle"):
                lines.append(f"  Angle: {next_slot['angle']}")

    return "\n".join(lines)
