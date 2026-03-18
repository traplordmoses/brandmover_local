"""Campaign overview endpoints."""

from fastapi import APIRouter, HTTPException

from dashboard.backend.services import data_bridge as db

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])


@router.get("")
def list_campaigns():
    campaigns = db.get_campaigns()
    summaries = []
    for c in campaigns:
        slots = c.get("slots", [])
        posted = sum(1 for s in slots if s.get("status") == "posted")
        total = len(slots)
        # Compute date range from start_date + duration
        start_date = c.get("start_date", "")
        duration = c.get("duration_days", 7)
        end_date = ""
        if start_date:
            from datetime import datetime, timedelta
            try:
                sd = datetime.strptime(start_date, "%Y-%m-%d")
                end_date = (sd + timedelta(days=duration)).strftime("%Y-%m-%d")
            except ValueError:
                pass

        summaries.append({
            "name": c.get("name", ""),
            "brief": c.get("brief", ""),
            "description": c.get("brief", "")[:120],
            "status": c.get("status", "active"),
            "start_date": start_date,
            "end_date": end_date,
            "total_posts": total,
            "delivered": posted,
            "slots": slots,
            "created_at": c.get("created_at"),
        })
    return {"campaigns": summaries}


@router.get("/{name}")
def get_campaign(name: str):
    campaign = db.get_campaign(name)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


@router.get("/{name}/posts")
def get_campaign_posts(name: str):
    queue = db.get_schedule_queue()
    posts = []
    for item in queue:
        prompt = item.get("prompt", "")
        if f"[CAMPAIGN: {name}]" in prompt:
            posts.append(item)
    return posts
