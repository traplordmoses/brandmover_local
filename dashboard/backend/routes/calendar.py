"""Content calendar endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dashboard.backend.services import data_bridge as db

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


@router.get("")
def get_calendar(
    days_back: int = 30,
    days_forward: int = 30,
    start: str | None = None,
    end: str | None = None,
):
    import time
    from datetime import datetime

    if start and end:
        start_ts = datetime.fromisoformat(start + "T00:00:00").timestamp()
        end_ts = datetime.fromisoformat(end + "T23:59:59").timestamp()
        d_back = int((time.time() - start_ts) / 86400) + 1
        d_fwd = int((end_ts - time.time()) / 86400) + 1
    else:
        d_back = days_back
        d_fwd = days_forward

    entries = db.get_calendar_entries(days_back=d_back, days_forward=d_fwd)

    posts = []
    stats = {"total": 0, "posted": 0, "pending": 0, "failed": 0, "cancelled": 0}
    for e in entries:
        ts = e.get("timestamp", 0)
        posted_at = e.get("posted_at")
        date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else None
        time_str = datetime.utcfromtimestamp(ts).strftime("%H:%M") if ts else None

        posts.append({
            "id": e.get("id", ""),
            "date": date_str,
            "time": time_str,
            "caption": e.get("caption", ""),
            "status": e.get("status", "unknown"),
            "campaign": e.get("campaign"),
            "label": e.get("label", ""),
            "has_image": e.get("has_media", False),
            "has_media": e.get("has_media", False),
            "tweet_url": e.get("tweet_url"),
            "recurrence": e.get("recurrence", "once"),
            "content_type": e.get("content_type"),
            "image_url": e.get("image_url"),
        })

        stats["total"] += 1
        s = e.get("status", "")
        if s in stats:
            stats[s] += 1

    return {"posts": posts, "stats": stats}


@router.get("/item/{item_id}")
def get_calendar_item(item_id: str):
    for item in db.get_schedule_queue():
        if item.get("id") == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")


@router.post("/item/{item_id}/cancel")
def cancel_item(item_id: str):
    if db.cancel_scheduled_item(item_id):
        return {"ok": True}
    raise HTTPException(status_code=404, detail="Item not found or not cancellable")


class EditItemBody(BaseModel):
    caption: str | None = None
    scheduled_utc: float | None = None


@router.post("/item/{item_id}/edit")
def edit_item(item_id: str, body: EditItemBody):
    updates = {}
    if body.caption is not None:
        updates["draft"] = {}
        # Preserve existing draft fields, update caption
        for item in db.get_schedule_queue():
            if item.get("id") == item_id:
                updates["draft"] = item.get("draft", {})
                break
        updates["draft"]["caption"] = body.caption
    if body.scheduled_utc is not None:
        updates["scheduled_utc"] = body.scheduled_utc
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    if db.update_scheduled_item(item_id, updates):
        return {"ok": True}
    raise HTTPException(status_code=404, detail="Item not found")


@router.get("/pending-draft")
def get_pending_draft():
    draft = db.get_pending_draft()
    if draft is None:
        return {"draft": None}
    return {"draft": draft}
