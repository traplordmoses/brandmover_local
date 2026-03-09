"""
Session plan state management — multi-post content sessions.

Plans are optional and ephemeral (24h expiry). They let the operator and agent
agree on a list of content pieces to work through one by one.

State lives in state/session_plan.json, separate from per-user state.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_PLAN_FILE = _project_root / "state" / "session_plan.json"
_DRAFT_QUEUE_FILE = _project_root / "state" / "draft_queue.json"
_EXPIRY_SECONDS = 24 * 60 * 60  # 24 hours

_VALID_STATUSES = {"pending", "generating", "review", "approved", "rejected", "skipped"}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _read_plan() -> dict | None:
    """Read session_plan.json, return None if missing or corrupt."""
    if not _PLAN_FILE.exists():
        return None
    try:
        return json.loads(_PLAN_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read session_plan.json: %s", e)
        return None


def _write_plan(data: dict) -> None:
    """Write plan dict to session_plan.json."""
    _PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PLAN_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_plan() -> dict | None:
    """Return the current plan, or None if missing/expired.

    Auto-deletes if updated_at is older than 24h.
    """
    plan = _read_plan()
    if plan is None:
        return None

    updated_at = plan.get("updated_at", 0)
    if time.time() - updated_at > _EXPIRY_SECONDS:
        logger.info("Session plan expired (%.0fh old), removing", (time.time() - updated_at) / 3600)
        clear_plan()
        return None

    return plan


def save_plan(plan_name: str, items: list[dict]) -> dict:
    """Create a new plan. All items start as pending, current_item=1.

    Args:
        plan_name: Human-readable plan name.
        items: List of dicts with 'description', 'tone', and optional 'notes'.

    Returns:
        The saved plan dict.
    """
    now = time.time()
    plan_items = []
    for i, item in enumerate(items, start=1):
        plan_items.append({
            "id": i,
            "description": item.get("description", ""),
            "tone": item.get("tone", ""),
            "status": "pending",
            "notes": item.get("notes"),
        })

    plan = {
        "created_at": now,
        "updated_at": now,
        "plan_name": plan_name,
        "items": plan_items,
        "current_item": 1,
        "autonomous": False,
    }
    _write_plan(plan)
    logger.info("Session plan created: '%s' with %d items", plan_name, len(plan_items))
    return plan


def get_plan_summary() -> str:
    """Return a compact text summary for the system prompt.

    e.g. "Active plan: 'Product launch' — 2/5 done. Next: #3 hype post"
    """
    plan = get_plan()
    if not plan:
        return ""

    items = plan.get("items", [])
    total = len(items)
    done = sum(1 for it in items if it["status"] in ("approved", "skipped"))
    current_id = plan.get("current_item")

    current_item = None
    if current_id:
        current_item = next((it for it in items if it["id"] == current_id), None)

    mode = " [AUTONOMOUS]" if plan.get("autonomous") else ""
    summary = f"SESSION PLAN{mode}: '{plan['plan_name']}' — {done}/{total} done."
    if plan.get("autonomous") and done < total:
        queued = len(get_queued_drafts())
        summary += f" {queued} drafts queued for review."
    elif current_item:
        desc = current_item["description"][:60]
        summary += f" Next: #{current_item['id']} {desc}"
        if current_item.get("tone"):
            summary += f" (tone: {current_item['tone']})"
    elif done == total:
        summary += " All items complete!"

    return summary


def update_item(item_id: int, status: str | None = None, notes: str | None = None) -> dict | None:
    """Update a specific item's status/notes, bump updated_at.

    If status is 'approved' or 'skipped', auto-advances current_item.

    Returns:
        The updated item dict, or None if plan/item not found.
    """
    plan = get_plan()
    if not plan:
        return None

    item = next((it for it in plan["items"] if it["id"] == item_id), None)
    if not item:
        logger.warning("Session plan: item #%d not found", item_id)
        return None

    if status:
        if status not in _VALID_STATUSES:
            logger.warning("Session plan: invalid status '%s'", status)
            return None
        item["status"] = status

    if notes is not None:
        item["notes"] = notes

    plan["updated_at"] = time.time()
    _write_plan(plan)

    # Auto-advance if completed/skipped
    if status in ("approved", "skipped"):
        advance_current()

    logger.info("Session plan: item #%d → %s", item_id, status or "(notes updated)")
    return item


def advance_current() -> dict | None:
    """Find the next pending item and set current_item to it.

    Returns:
        The new current item dict, or None if all done.
    """
    plan = _read_plan()
    if not plan:
        return None

    for item in plan["items"]:
        if item["status"] == "pending":
            plan["current_item"] = item["id"]
            plan["updated_at"] = time.time()
            _write_plan(plan)
            logger.info("Session plan: advanced to item #%d", item["id"])
            return item

    # All done — set current_item to None
    plan["current_item"] = None
    plan["updated_at"] = time.time()
    _write_plan(plan)
    logger.info("Session plan: all items complete")
    return None


def clear_plan() -> None:
    """Delete session_plan.json and draft_queue.json."""
    try:
        _PLAN_FILE.unlink(missing_ok=True)
        _DRAFT_QUEUE_FILE.unlink(missing_ok=True)
        logger.info("Session plan cleared")
    except OSError as e:
        logger.warning("Failed to clear session plan: %s", e)


# ---------------------------------------------------------------------------
# Autonomous mode
# ---------------------------------------------------------------------------


def set_autonomous(enabled: bool) -> dict | None:
    """Toggle autonomous mode on the current plan.

    Returns the updated plan, or None if no plan exists.
    """
    plan = get_plan()
    if not plan:
        return None

    plan["autonomous"] = enabled
    plan["updated_at"] = time.time()
    _write_plan(plan)
    logger.info("Session plan: autonomous mode %s", "enabled" if enabled else "disabled")
    return plan


def is_autonomous() -> bool:
    """Check if the current plan is in autonomous mode."""
    plan = get_plan()
    return bool(plan and plan.get("autonomous"))


# ---------------------------------------------------------------------------
# Draft queue — stores completed drafts from autonomous runs
# ---------------------------------------------------------------------------


def _read_draft_queue() -> list[dict]:
    """Read draft_queue.json."""
    if not _DRAFT_QUEUE_FILE.exists():
        return []
    try:
        return json.loads(_DRAFT_QUEUE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read draft_queue.json: %s", e)
        return []


def _write_draft_queue(queue: list[dict]) -> None:
    """Write draft queue to draft_queue.json."""
    _DRAFT_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DRAFT_QUEUE_FILE.write_text(
        json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_queued_draft(item_id: int, draft: dict, image_url: str | None = None) -> int:
    """Save a completed draft to the queue.

    Returns the total number of queued drafts.
    """
    queue = _read_draft_queue()
    queue.append({
        "item_id": item_id,
        "draft": draft,
        "image_url": image_url,
        "created_at": time.time(),
        "reviewed": False,
    })
    _write_draft_queue(queue)
    logger.info("Draft queued for item #%d (%d total)", item_id, len(queue))
    return len(queue)


def get_queued_drafts() -> list[dict]:
    """Return all queued drafts."""
    return _read_draft_queue()


def get_queued_draft_by_item(item_id: int) -> dict | None:
    """Return the queued draft for a specific plan item."""
    for entry in _read_draft_queue():
        if entry.get("item_id") == item_id:
            return entry
    return None


def mark_draft_reviewed(item_id: int) -> None:
    """Mark a queued draft as reviewed."""
    queue = _read_draft_queue()
    for entry in queue:
        if entry.get("item_id") == item_id:
            entry["reviewed"] = True
            break
    _write_draft_queue(queue)


def clear_draft_queue() -> None:
    """Delete all queued drafts."""
    try:
        _DRAFT_QUEUE_FILE.unlink(missing_ok=True)
    except OSError:
        pass
