"""
Campaign runbooks — multi-day content campaign orchestration.

A campaign is a named, time-bounded sequence of posts with:
- A theme/brief that ties all posts together
- Explicit posts with pre-written copy OR generated posts with angles
- Multiple posts per day (morning/evening)
- Auto-scheduling into the schedule queue
- Progress tracking and completion status

Campaigns live in state/campaigns.json and are managed via /campaign commands.

Usage flow:
    1. User describes campaign to the bot (NL or structured)
    2. Bot calls create_campaign() with parsed posts
    3. schedule_campaign_posts() queues each post into schedule_queue
    4. Scheduler fires each post at the scheduled time
    5. On approval/post, update_slot_status() marks progress
    6. Campaign auto-completes when all slots are posted
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.paths import STATE_DIR

logger = logging.getLogger(__name__)

_CAMPAIGNS_FILE = STATE_DIR / "campaigns.json"


def _get_local_tz():
    """Get the configured local timezone."""
    from config import settings
    tz_name = getattr(settings, "TIMEZONE", "")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except (KeyError, Exception):
            pass
    return datetime.now().astimezone().tzinfo


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CampaignSlot:
    """A single post within a campaign."""
    day: int                          # Day number (1-indexed)
    slot_label: str = ""              # e.g., "morning", "evening", "post_1"
    content_type: str = ""            # e.g., "announcement", "engagement"
    prompt: str = ""                  # Generation prompt (if no pre-written copy)
    copy: str = ""                    # Pre-written post copy (posted as-is)
    angle: str = ""                   # Brief description of this post's angle
    media_note: str = ""              # e.g., "[screenshot of swipe UI]"
    status: str = "pending"           # pending | scheduled | drafted | approved | posted | skipped
    schedule_queue_id: str = ""       # Links to schedule_queue item
    draft_timestamp: float = 0.0
    post_url: str = ""


@dataclass
class Campaign:
    """A multi-day content campaign."""
    name: str
    brief: str                        # Campaign theme/objective
    start_date: str                   # ISO date string (YYYY-MM-DD)
    duration_days: int
    slots: list[dict] = field(default_factory=list)
    status: str = "active"            # active | paused | completed
    created_at: float = 0.0
    post_times: dict[str, str] = field(default_factory=dict)  # label → "HH:MM" local
    kpis: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Campaign CRUD
# ---------------------------------------------------------------------------

def create_campaign(
    name: str,
    brief: str,
    slots: list[dict],
    start_date: str = "",
    post_times: dict[str, str] | None = None,
    kpis: dict[str, str] | None = None,
) -> dict:
    """Create a campaign from explicit slot definitions.

    Args:
        name: Unique campaign name.
        brief: Campaign theme/objective.
        slots: List of slot dicts with keys: day, slot_label, copy/prompt,
               content_type, angle, media_note.
        start_date: ISO date (YYYY-MM-DD). Defaults to today.
        post_times: Mapping of slot_label → "HH:MM" in local time.
                    e.g., {"morning": "09:00", "evening": "18:00"}
                    Defaults to morning=9:00, evening=18:00.
        kpis: Optional KPI targets.

    Returns:
        {"success": bool, "campaign": dict | None, "message": str}
    """
    data = _read_campaigns()

    for c in data["campaigns"]:
        if c.get("name") == name:
            return {"success": False, "campaign": None,
                    "message": f"Campaign '{name}' already exists."}

    if not start_date:
        start_date = time.strftime("%Y-%m-%d")

    if not post_times:
        post_times = {"morning": "09:00", "evening": "18:00"}

    # Ensure all slots have required defaults
    duration_days = 0
    for slot in slots:
        slot.setdefault("status", "pending")
        slot.setdefault("schedule_queue_id", "")
        slot.setdefault("draft_timestamp", 0.0)
        slot.setdefault("post_url", "")
        slot.setdefault("slot_label", "morning")
        slot.setdefault("copy", "")
        slot.setdefault("prompt", "")
        slot.setdefault("angle", "")
        slot.setdefault("media_note", "")
        slot.setdefault("content_type", "")
        if slot.get("day", 0) > duration_days:
            duration_days = slot["day"]

    campaign = Campaign(
        name=name,
        brief=brief,
        start_date=start_date,
        duration_days=duration_days,
        slots=slots,
        status="active",
        created_at=time.time(),
        post_times=post_times,
        kpis=kpis or {},
    )

    data["campaigns"].append(asdict(campaign))
    _write_campaigns(data)
    logger.info("Created campaign '%s': %d days, %d slots", name, duration_days, len(slots))

    return {
        "success": True,
        "campaign": asdict(campaign),
        "message": f"Campaign '{name}' created with {len(slots)} posts over {duration_days} days.",
    }


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


def update_slot_status(
    campaign_name: str,
    day: int,
    slot_label: str,
    new_status: str,
    post_url: str = "",
) -> bool:
    """Update a slot's status within a campaign. Returns True if found."""
    data = _read_campaigns()
    for campaign in data["campaigns"]:
        if campaign.get("name") != campaign_name:
            continue
        for slot in campaign.get("slots", []):
            if slot.get("day") == day and slot.get("slot_label", "") == slot_label:
                slot["status"] = new_status
                if new_status == "drafted":
                    slot["draft_timestamp"] = time.time()
                if post_url:
                    slot["post_url"] = post_url
                # Auto-complete campaign when all slots are posted/skipped
                terminal = {"posted", "skipped"}
                all_done = all(
                    s.get("status") in terminal for s in campaign["slots"]
                )
                if all_done:
                    campaign["status"] = "completed"
                    logger.info("Campaign '%s' completed!", campaign_name)
                _write_campaigns(data)
                logger.info(
                    "Campaign '%s' day %d/%s → %s",
                    campaign_name, day, slot_label, new_status,
                )
                return True
    return False


def update_slot_by_queue_id(queue_id: str, new_status: str, post_url: str = "") -> bool:
    """Update a campaign slot by its schedule_queue_id. Returns True if found."""
    data = _read_campaigns()
    for campaign in data["campaigns"]:
        for slot in campaign.get("slots", []):
            if slot.get("schedule_queue_id") == queue_id:
                slot["status"] = new_status
                if new_status == "drafted":
                    slot["draft_timestamp"] = time.time()
                if post_url:
                    slot["post_url"] = post_url
                terminal = {"posted", "skipped"}
                all_done = all(
                    s.get("status") in terminal for s in campaign["slots"]
                )
                if all_done:
                    campaign["status"] = "completed"
                    logger.info("Campaign '%s' completed!", campaign["name"])
                _write_campaigns(data)
                logger.info(
                    "Campaign slot (queue_id=%s) → %s", queue_id, new_status,
                )
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


def pause_campaign(campaign_name: str) -> bool:
    """Pause a campaign. Returns True if found."""
    data = _read_campaigns()
    for campaign in data["campaigns"]:
        if campaign.get("name") == campaign_name:
            campaign["status"] = "paused"
            _write_campaigns(data)
            return True
    return False


def resume_campaign(campaign_name: str) -> bool:
    """Resume a paused campaign. Returns True if found."""
    data = _read_campaigns()
    for campaign in data["campaigns"]:
        if campaign.get("name") == campaign_name and campaign.get("status") == "paused":
            campaign["status"] = "active"
            _write_campaigns(data)
            return True
    return False


def delete_campaign(campaign_name: str) -> bool:
    """Delete a campaign and cancel its scheduled posts. Returns True if found."""
    from agent import schedule_queue

    data = _read_campaigns()
    original_len = len(data["campaigns"])
    campaign = None
    for c in data["campaigns"]:
        if c.get("name") == campaign_name:
            campaign = c
            break

    if not campaign:
        return False

    # Cancel all linked schedule queue items
    for slot in campaign.get("slots", []):
        qid = slot.get("schedule_queue_id")
        if qid:
            schedule_queue.cancel_scheduled(qid)

    data["campaigns"] = [c for c in data["campaigns"] if c.get("name") != campaign_name]
    _write_campaigns(data)
    logger.info("Deleted campaign '%s' (%d slots)", campaign_name, len(campaign.get("slots", [])))
    return len(data["campaigns"]) < original_len


# ---------------------------------------------------------------------------
# Schedule campaign posts into the queue
# ---------------------------------------------------------------------------

def schedule_campaign_posts(campaign_name: str) -> dict:
    """Schedule all pending slots of a campaign into the schedule queue.

    Calculates UTC timestamps from start_date + day offset + post_times,
    then calls schedule_queue.add_scheduled() for each.

    Returns:
        {"scheduled": int, "skipped": int, "errors": list[str]}
    """
    from agent import schedule_queue

    campaign = get_campaign(campaign_name)
    if not campaign:
        return {"scheduled": 0, "skipped": 0, "errors": ["Campaign not found"]}

    start_date = campaign.get("start_date", "")
    post_times = campaign.get("post_times", {"morning": "09:00", "evening": "18:00"})
    local_tz = _get_local_tz()

    try:
        base_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=local_tz)
    except ValueError:
        return {"scheduled": 0, "skipped": 0, "errors": [f"Invalid start_date: {start_date}"]}

    scheduled = 0
    skipped = 0
    errors: list[str] = []

    data = _read_campaigns()
    # Find the campaign in data to update schedule_queue_ids
    campaign_data = None
    for c in data["campaigns"]:
        if c.get("name") == campaign_name:
            campaign_data = c
            break

    if not campaign_data:
        return {"scheduled": 0, "skipped": 0, "errors": ["Campaign data not found"]}

    for slot in campaign_data.get("slots", []):
        if slot.get("status") != "pending":
            skipped += 1
            continue

        # Already scheduled
        if slot.get("schedule_queue_id"):
            skipped += 1
            continue

        day = slot.get("day", 1)
        label = slot.get("slot_label", "morning")

        # Calculate target time
        time_str = post_times.get(label, post_times.get("morning", "09:00"))
        try:
            hour, minute = map(int, time_str.split(":"))
        except (ValueError, AttributeError):
            hour, minute = 9, 0

        target_dt = (base_date + timedelta(days=day - 1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0,
        )
        target_utc = target_dt.timestamp()

        # Skip if target time is in the past
        if target_utc < time.time() - 300:
            errors.append(f"Day {day}/{label}: target time is in the past")
            skipped += 1
            continue

        # Build the prompt — use pre-written copy or generation prompt
        copy = slot.get("copy", "")
        prompt = slot.get("prompt", "")
        angle = slot.get("angle", "")
        media_note = slot.get("media_note", "")
        brief = campaign_data.get("brief", "")

        if copy:
            # Pre-written copy: wrap in a directive so the agent posts it as-is
            full_prompt = (
                f"[CAMPAIGN: {campaign_name}]\n"
                f"Post this exact copy (do not change the wording):\n\n"
                f"{copy}"
            )
            if media_note:
                full_prompt += f"\n\nMedia note: {media_note}"
        elif prompt:
            full_prompt = (
                f"[CAMPAIGN: {campaign_name}]\n"
                f"Campaign brief: {brief}\n"
                f"Generate a post: {prompt}"
            )
            if angle:
                full_prompt += f"\nAngle: {angle}"
            if media_note:
                full_prompt += f"\nMedia note: {media_note}"
        else:
            full_prompt = (
                f"[CAMPAIGN: {campaign_name}]\n"
                f"Campaign brief: {brief}\n"
                f"Day {day} ({label}) — {slot.get('content_type', 'engagement')}"
            )
            if angle:
                full_prompt += f"\nAngle: {angle}"

        queue_label = f"{campaign_name} D{day}/{label}"
        item = schedule_queue.add_scheduled(
            prompt=full_prompt,
            scheduled_utc=target_utc,
            label=queue_label,
        )
        if item is None:
            errors.append(f"Day {day}/{label}: duplicate detected, skipped")
            skipped += 1
            continue

        slot["schedule_queue_id"] = item["id"]
        slot["status"] = "scheduled"
        scheduled += 1

    _write_campaigns(data)
    logger.info(
        "Campaign '%s' scheduling: %d scheduled, %d skipped, %d errors",
        campaign_name, scheduled, skipped, len(errors),
    )
    return {"scheduled": scheduled, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

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
        "start_date": campaign.get("start_date"),
        "duration_days": campaign.get("duration_days"),
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
        f"({progress['start_date']} — {progress['duration_days']} days)",
        "",
        f"Progress: {progress['progress_pct']}% "
        f"({progress['by_status'].get('posted', 0)}/{progress['total_slots']} posted)",
    ]

    for status, count in progress["by_status"].items():
        icon = {
            "pending": "⏳", "scheduled": "📅", "drafted": "📝",
            "approved": "✅", "posted": "📤", "skipped": "⏭",
        }.get(status, "•")
        lines.append(f"  {icon} {status}: {count}")

    kpis = progress.get("kpis", {})
    if kpis:
        lines.append("")
        lines.append("<b>KPIs:</b>")
        for k, v in kpis.items():
            lines.append(f"  • {k}: {v}")

    if campaign:
        next_slot = get_next_pending_slot(campaign_name)
        if next_slot:
            lines.append("")
            day = next_slot.get("day", "?")
            label = next_slot.get("slot_label", "")
            lines.append(f"<b>Next:</b> Day {day} ({label})")
            if next_slot.get("angle"):
                lines.append(f"  {next_slot['angle']}")
            elif next_slot.get("copy"):
                lines.append(f"  {next_slot['copy'][:80]}...")

    return "\n".join(lines)


def format_campaign_list() -> str:
    """Format all campaigns for Telegram display."""
    campaigns = list_campaigns()
    if not campaigns:
        return "No campaigns. Send me a campaign plan and I'll set it up."

    lines = ["<b>Campaigns:</b>", ""]
    for c in campaigns:
        status = c.get("status", "?")
        icon = {"active": "🟢", "paused": "⏸", "completed": "✅"}.get(status, "•")
        slots = c.get("slots", [])
        posted = sum(1 for s in slots if s.get("status") == "posted")
        total = len(slots)
        lines.append(
            f"{icon} <b>{c['name']}</b> — {posted}/{total} posted ({status})"
        )
    return "\n".join(lines)
