"""Bot status and heartbeat endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fastapi import APIRouter
from pydantic import BaseModel

from config import settings as cfg
from dashboard.backend.services import data_bridge as db

router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("")
def get_status():
    state = db.get_auto_post_state()
    heartbeat = db.get_heartbeat_log(1)
    pending = db.get_pending_scheduled()
    next_item = None
    if pending:
        n = pending[0]
        next_item = {
            "label": n.get("label", n.get("id", "")),
            "timestamp": n.get("scheduled_utc", 0),
        }

    last_hb_ts = heartbeat[-1].get("timestamp", 0) if heartbeat else 0

    return {
        "paused": state.get("paused", False),
        "last_heartbeat": last_hb_ts,
        "last_post_timestamp": state.get("last_post_timestamp", 0),
        "next_scheduled": next_item,
        "agent_mode": cfg.AGENT_MODE,
    }


@router.get("/heartbeat-log")
def get_heartbeat_log(limit: int = 20):
    entries = db.get_heartbeat_log(limit)
    # Reverse so newest is first
    entries.reverse()
    return {"entries": entries}


class PauseBody(BaseModel):
    paused: bool | None = None


@router.post("/pause")
def toggle_pause(body: PauseBody | None = None):
    state = db.get_auto_post_state()
    current = state.get("paused", False)
    new_state = body.paused if (body and body.paused is not None) else (not current)
    db.set_paused(new_state)
    return {"paused": new_state}


@router.get("/activity")
def get_activity():
    state = db.get_auto_post_state()
    recent_posts = state.get("posts_today", [])[-20:]

    queue_items = db.get_posted_scheduled()[-20:]

    activity = []
    for p in recent_posts:
        activity.append({
            "source": "auto_post",
            "timestamp": p.get("timestamp", 0),
            "caption": p.get("caption", ""),
            "tweet_url": p.get("tweet_url"),
            "slot": p.get("slot", ""),
        })
    for item in queue_items:
        activity.append({
            "source": "schedule_queue",
            "timestamp": item.get("posted_at", item.get("scheduled_utc", 0)),
            "caption": item.get("draft", {}).get("caption", ""),
            "status": item.get("status"),
            "id": item.get("id"),
        })

    activity.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {"entries": activity[:20]}
