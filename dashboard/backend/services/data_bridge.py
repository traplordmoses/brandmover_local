"""
Data bridge — reads/writes from BrandMover's actual data stores.

All dashboard endpoints go through this layer so there's a single
place to manage file access, caching, and write safety.
"""

import json
import os
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

STATE_DIR = _PROJECT_ROOT / "state"
BRAND_DIR = _PROJECT_ROOT / "brand"
CONFIG_DIR = _PROJECT_ROOT / "config"


def _read_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp_{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Schedule Queue
# ---------------------------------------------------------------------------

def get_schedule_queue() -> list[dict]:
    data = _read_json(STATE_DIR / "schedule_queue.json", [])
    return data if isinstance(data, list) else data.get("items", [])


def get_pending_scheduled() -> list[dict]:
    return [i for i in get_schedule_queue() if i.get("status") in ("pending", "generating")]


def get_posted_scheduled() -> list[dict]:
    return [i for i in get_schedule_queue() if i.get("status") == "posted"]


def update_scheduled_item(item_id: str, updates: dict) -> bool:
    items = get_schedule_queue()
    for item in items:
        if item["id"] == item_id:
            item.update(updates)
            _write_json(STATE_DIR / "schedule_queue.json", items)
            return True
    return False


def cancel_scheduled_item(item_id: str) -> bool:
    items = get_schedule_queue()
    for item in items:
        if item["id"] == item_id and item.get("status") in ("pending", "generating"):
            item["status"] = "cancelled"
            item["cancelled_at"] = time.time()
            _write_json(STATE_DIR / "schedule_queue.json", items)
            return True
    return False


# ---------------------------------------------------------------------------
# Auto Post State
# ---------------------------------------------------------------------------

def get_auto_post_state() -> dict:
    state = _read_json(STATE_DIR / "auto_post_state.json")
    defaults = {
        "posts_today": [],
        "posted_event_ids": [],
        "rotation_indices": {},
        "recent_captions": [],
        "paused": False,
        "last_post_timestamp": 0,
    }
    for k, v in defaults.items():
        state.setdefault(k, v)
    return state


def set_paused(paused: bool) -> dict:
    state = get_auto_post_state()
    state["paused"] = paused
    _write_json(STATE_DIR / "auto_post_state.json", state)
    return state


# ---------------------------------------------------------------------------
# Pending Draft
# ---------------------------------------------------------------------------

def get_pending_draft() -> dict | None:
    state = _read_json(STATE_DIR / "state.json")
    return state.get("pending")


def clear_pending_draft() -> bool:
    state = _read_json(STATE_DIR / "state.json")
    if "pending" in state:
        del state["pending"]
        _write_json(STATE_DIR / "state.json", state)
        return True
    return False


# ---------------------------------------------------------------------------
# Heartbeat Log
# ---------------------------------------------------------------------------

def get_heartbeat_log(n: int = 50) -> list[dict]:
    path = STATE_DIR / "heartbeat_log.jsonl"
    if not path.exists():
        return []
    try:
        lines = path.read_text().strip().split("\n")
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------

def get_campaigns() -> list[dict]:
    data = _read_json(STATE_DIR / "campaigns.json")
    return data.get("campaigns", []) if isinstance(data, dict) else []


def get_campaign(name: str) -> dict | None:
    for c in get_campaigns():
        if c.get("name") == name:
            return c
    return None


# ---------------------------------------------------------------------------
# Feedback & Preferences
# ---------------------------------------------------------------------------

def get_feedback_log(limit: int = 100) -> list[dict]:
    data = _read_json(STATE_DIR / "feedback.json", [])
    if isinstance(data, list):
        return data[-limit:]
    return []


def get_learned_preferences() -> str:
    path = STATE_DIR / "learned_preferences.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def get_preference_clusters() -> dict:
    return _read_json(STATE_DIR / "preference_clusters.json")


# ---------------------------------------------------------------------------
# Generation History
# ---------------------------------------------------------------------------

def get_generation_history(limit: int = 100) -> list[dict]:
    data = _read_json(STATE_DIR / "generation_history.json", [])
    if isinstance(data, list):
        return data[-limit:]
    return []


def get_generation_stats() -> dict:
    history = get_generation_history(500)
    if not history:
        return {"total": 0, "total_cost": 0, "by_model": {}, "by_status": {}}

    total_cost = sum(e.get("estimated_cost_usd", 0) for e in history)
    by_model: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for e in history:
        model = e.get("model_id", "unknown")
        by_model[model] = by_model.get(model, 0) + 1
        status = e.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    return {
        "total": len(history),
        "total_cost": round(total_cost, 2),
        "by_model": by_model,
        "by_status": by_status,
    }


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def get_session() -> dict:
    return _read_json(STATE_DIR / "agent_session.json")


# ---------------------------------------------------------------------------
# Brand Documents
# ---------------------------------------------------------------------------

def list_brand_documents() -> list[dict]:
    docs = []
    if BRAND_DIR.exists():
        for f in sorted(BRAND_DIR.iterdir()):
            if f.is_file() and f.suffix in (".md", ".txt", ".json"):
                docs.append({
                    "name": f.name,
                    "path": str(f.relative_to(_PROJECT_ROOT)),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                    "type": f.suffix,
                })
        # Also check subdirectories for key files
        for subdir in ("prompts", "personality", "skills", "templates"):
            sub = BRAND_DIR / subdir
            if sub.exists() and sub.is_dir():
                for f in sorted(sub.iterdir()):
                    if f.is_file() and f.suffix in (".md", ".txt", ".json"):
                        docs.append({
                            "name": f"{subdir}/{f.name}",
                            "path": str(f.relative_to(_PROJECT_ROOT)),
                            "size": f.stat().st_size,
                            "modified": f.stat().st_mtime,
                            "type": f.suffix,
                        })
    return docs


def read_brand_document(rel_path: str) -> str | None:
    full = (_PROJECT_ROOT / rel_path).resolve()
    if not full.exists() or not full.is_relative_to(BRAND_DIR.resolve()):
        return None
    return full.read_text(encoding="utf-8")


def write_brand_document(rel_path: str, content: str) -> bool:
    full = (_PROJECT_ROOT / rel_path).resolve()
    if not full.is_relative_to(BRAND_DIR.resolve()):
        return False
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Schedule Config
# ---------------------------------------------------------------------------

def get_schedule_config() -> dict:
    return _read_json(CONFIG_DIR / "schedule.json")


def update_schedule_config(config: dict) -> bool:
    _write_json(CONFIG_DIR / "schedule.json", config)
    return True


# ---------------------------------------------------------------------------
# Topic Bank
# ---------------------------------------------------------------------------

def get_topic_bank() -> dict:
    return _read_json(STATE_DIR / "topic_bank.json")


# ---------------------------------------------------------------------------
# Diversity Tracker
# ---------------------------------------------------------------------------

def get_diversity_entries() -> list[dict]:
    data = _read_json(STATE_DIR / "diversity_tracker.json")
    return data.get("entries", []) if isinstance(data, dict) else []


# ---------------------------------------------------------------------------
# Publish Queue
# ---------------------------------------------------------------------------

def get_publish_queue() -> list[dict]:
    return _read_json(STATE_DIR / "publish_queue.json", [])


# ---------------------------------------------------------------------------
# Aggregated Calendar View
# ---------------------------------------------------------------------------

def get_calendar_entries(days_back: int = 30, days_forward: int = 30) -> list[dict]:
    """Aggregate all schedule_queue items + auto_post posts into calendar entries."""
    now = time.time()
    start = now - (days_back * 86400)
    end = now + (days_forward * 86400)

    entries = []

    # From schedule queue (all items within range)
    for item in get_schedule_queue():
        ts = item.get("scheduled_utc", 0)
        posted_at = item.get("posted_at", 0)
        relevant_ts = posted_at or ts
        if start <= relevant_ts <= end or start <= ts <= end:
            entries.append({
                "id": item.get("id", ""),
                "source": "schedule_queue",
                "timestamp": ts,
                "posted_at": posted_at or None,
                "status": item.get("status", "unknown"),
                "caption": _extract_caption(item),
                "label": item.get("label", ""),
                "campaign": _extract_campaign(item.get("prompt", "")),
                "has_media": bool(item.get("draft", {}).get("image_url")),
                "recurrence": item.get("recurrence", "once"),
            })

    # From auto_post_state (posted items)
    state = get_auto_post_state()
    for post in state.get("posts_today", []):
        ts = post.get("timestamp", 0)
        if start <= ts <= end:
            # Avoid duplicates with schedule_queue
            existing_ids = {e["id"] for e in entries}
            slot = post.get("slot", "")
            post_id = slot.split(":")[-1] if ":" in slot else slot
            if post_id not in existing_ids:
                entries.append({
                    "id": post_id,
                    "source": "auto_post",
                    "timestamp": ts,
                    "posted_at": ts,
                    "status": "posted",
                    "caption": post.get("caption", ""),
                    "label": post.get("slot", ""),
                    "campaign": None,
                    "has_media": False,
                    "tweet_url": post.get("tweet_url"),
                    "recurrence": "once",
                })

    entries.sort(key=lambda e: e.get("timestamp", 0))
    return entries


def _extract_caption(item: dict) -> str:
    """Extract the display caption from a queue item."""
    draft = item.get("draft", {})
    if draft and draft.get("caption"):
        return draft["caption"]
    prompt = item.get("prompt", "")
    # Extract from exact-copy prompts
    import re
    m = re.search(r"post this exact copy[^:]*:\s*\n+(.*)", prompt, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()
        media_split = re.split(r"\n\s*MEDIA TASK:", text, flags=re.IGNORECASE)
        return media_split[0].strip()
    return prompt[:200]


def _extract_campaign(prompt: str) -> str | None:
    import re
    m = re.search(r"\[CAMPAIGN:\s*([^\]]+)\]", prompt)
    return m.group(1).strip() if m else None
