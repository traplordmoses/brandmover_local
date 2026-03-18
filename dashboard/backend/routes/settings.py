"""Settings and config endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fastapi import APIRouter
from pydantic import BaseModel

from config import settings as cfg
from dashboard.backend.services import data_bridge as db

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_settings():
    return {
        "schedule": db.get_schedule_config(),
        "agent_mode": cfg.AGENT_MODE,
        "brand_name": cfg.BRAND_NAME,
        "publish_platforms": cfg.PUBLISH_PLATFORMS,
        "auto_post_enabled": cfg.AUTO_POST_ENABLED,
        "heartbeat_enabled": cfg.HEARTBEAT_ENABLED,
        "content_planner_enabled": cfg.CONTENT_PLANNER_ENABLED,
        "skeleton_library_enabled": cfg.SKELETON_LIBRARY_ENABLED,
        "diversity_tracker_enabled": cfg.DIVERSITY_TRACKER_ENABLED,
    }


class ScheduleConfigBody(BaseModel):
    config: dict


@router.put("/schedule")
def update_schedule(body: ScheduleConfigBody):
    db.update_schedule_config(body.config)
    return {"ok": True}


@router.get("/stats")
def get_stats():
    stats = db.get_generation_stats()
    # Add approval rate from feedback
    feedback = db.get_feedback_log(500)
    if feedback:
        approvals = sum(1 for f in feedback if f.get("action") == "approve")
        rejections = sum(1 for f in feedback if f.get("action") == "reject")
        total = approvals + rejections
        stats["approval_rate"] = round(approvals / total, 2) if total else None
        stats["total_feedback"] = len(feedback)
    return stats


@router.get("/topic-bank")
def get_topic_bank():
    return db.get_topic_bank()


@router.get("/preferences")
def get_preferences():
    return {
        "learned_preferences": db.get_learned_preferences(),
        "clusters": db.get_preference_clusters(),
    }
