"""
Generation history — append-only log of all image generations.

Tracks asset type, content type, prompt, model, image URLs, and status
(draft → approved/rejected). Follows the same pattern as feedback.py.
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# In-memory cache for generation_history.json to avoid re-reading on every call
_cached_history: list[dict] | None = None
_history_cache_mtime: float = 0.0

# Threading lock for sync read/write functions (protects cache + file I/O)
_sync_lock = threading.Lock()

_project_root = Path(__file__).resolve().parent.parent
from agent.paths import STATE_DIR as _STATE_DIR
_HISTORY_FILE = _STATE_DIR / "generation_history.json"

# Migrate from old location if needed
_OLD_HISTORY = _project_root / "generation_history.json"
if _OLD_HISTORY.exists() and not _HISTORY_FILE.exists():
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    _shutil.move(str(_OLD_HISTORY), str(_HISTORY_FILE))


def _read_history() -> list[dict]:
    """Read the history log. Returns empty list if missing or corrupt.

    Uses mtime-based in-memory caching to avoid re-reading on every call.
    """
    global _cached_history, _history_cache_mtime
    if not _HISTORY_FILE.exists():
        return []
    try:
        mtime = os.stat(_HISTORY_FILE).st_mtime
        if _cached_history is not None and mtime == _history_cache_mtime:
            return _cached_history
        data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
        _cached_history = data
        _history_cache_mtime = mtime
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read generation_history.json: %s", e)
        return []


def _write_history(entries: list[dict]) -> None:
    """Write the full history log and update in-memory cache."""
    global _cached_history, _history_cache_mtime
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _HISTORY_FILE.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(str(tmp_path), str(_HISTORY_FILE))
    _cached_history = entries
    _history_cache_mtime = os.stat(_HISTORY_FILE).st_mtime


# Estimated cost per prediction by model (USD). Based on Replicate pricing.
_MODEL_COSTS: dict[str, float] = {
    "flux-1.1-pro": 0.04,
    "nano-banana-pro": 0.02,
    "recraft-v3-svg": 0.04,
    "seedream-3.0": 0.03,
}


def _estimate_cost(model_id: str, image_count: int = 1) -> float:
    """Estimate cost for a generation based on model and image count."""
    short_name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
    per_image = _MODEL_COSTS.get(short_name, 0.04)
    return round(per_image * image_count, 4)


_MAX_HISTORY_ENTRIES = 500
_history_lock = asyncio.Lock()


def log_generation(
    asset_type: str,
    content_type: str,
    prompt: str,
    model_id: str,
    image_urls: list[str],
    original_request: str,
    status: str = "draft",
    tags: list[str] | None = None,
) -> int:
    """Append a generation entry. Returns the new total count.

    Tags enable structured retrieval — e.g. ["campaign:launch", "mood:urgent",
    "topic:feature-release"]. Memory search includes tags for better matching.
    """
    cost = _estimate_cost(model_id, max(len(image_urls), 1))
    entries = _read_history()
    entry: dict = {
        "asset_type": asset_type,
        "content_type": content_type,
        "prompt": prompt,
        "model_id": model_id,
        "image_urls": image_urls,
        "original_request": original_request,
        "status": status,
        "estimated_cost_usd": cost,
        "timestamp": time.time(),
    }
    if tags:
        entry["tags"] = tags
    entries.append(entry)
    # Cap history size to prevent unbounded growth
    if len(entries) > _MAX_HISTORY_ENTRIES:
        entries = entries[-_MAX_HISTORY_ENTRIES:]
    _write_history(entries)
    count = len(entries)
    logger.info("Logged generation #%d (%s/%s, status=%s)", count, asset_type, content_type, status)
    # Invalidate the search index so the next search_memory call picks up the new entry
    try:
        from agent.memory import invalidate_search_index
        invalidate_search_index()
    except ImportError:
        pass
    return count


def update_generation_status(timestamp: float, new_status: str) -> bool:
    """Find entry by timestamp and update its status. Returns True if found."""
    entries = _read_history()
    for entry in reversed(entries):
        if abs(entry.get("timestamp", 0) - timestamp) < 1.0:
            entry["status"] = new_status
            entry["status_updated_at"] = time.time()
            _write_history(entries)
            logger.info("Updated generation status: %.0f → %s", timestamp, new_status)
            return True
    logger.warning("Generation entry not found for timestamp %.0f", timestamp)
    return False


def get_generation_stats() -> dict:
    """Return summary stats: totals by type, status, model, and cost."""
    with _sync_lock:
        entries = _read_history()
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_model: dict[str, int] = {}
    total_cost = 0.0

    for e in entries:
        at = e.get("asset_type", "unknown")
        st = e.get("status", "unknown")
        model = e.get("model_id", "unknown").split("/")[-1]

        by_type[at] = by_type.get(at, 0) + 1
        by_status[st] = by_status.get(st, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        total_cost += e.get("estimated_cost_usd", 0.0)

    return {
        "total": len(entries),
        "by_type": by_type,
        "by_status": by_status,
        "by_model": by_model,
        "estimated_total_cost_usd": round(total_cost, 2),
    }


def get_recent_generations(n: int = 10) -> list[dict]:
    """Return the last N generation entries."""
    with _sync_lock:
        entries = _read_history()
    return entries[-n:]


def get_approval_analytics() -> dict:
    """Return approval/rejection rates broken down by content_type and model."""
    with _sync_lock:
        entries = _read_history()

    by_content_type: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}

    for e in entries:
        status = e.get("status", "draft")
        if status not in ("approved", "rejected"):
            continue

        ct = e.get("content_type", "unknown")
        model = e.get("model_id", "unknown").rsplit("/", 1)[-1]

        ct_stats = by_content_type.setdefault(ct, {"approved": 0, "rejected": 0})
        ct_stats[status] += 1

        m_stats = by_model.setdefault(model, {"approved": 0, "rejected": 0})
        m_stats[status] += 1

    def _rate(d: dict[str, int]) -> float:
        total = d["approved"] + d["rejected"]
        return round(d["approved"] / total * 100, 1) if total else 0.0

    return {
        "by_content_type": {
            k: {**v, "rate": _rate(v)} for k, v in sorted(by_content_type.items())
        },
        "by_model": {
            k: {**v, "rate": _rate(v)} for k, v in sorted(by_model.items())
        },
    }


# ---------------------------------------------------------------------------
# Async wrappers — non-blocking versions for use in bot handlers
# ---------------------------------------------------------------------------

async def async_log_generation(*args, **kwargs) -> int:
    async with _history_lock:
        return await asyncio.to_thread(log_generation, *args, **kwargs)

async def async_update_generation_status(timestamp: float, new_status: str) -> bool:
    async with _history_lock:
        return await asyncio.to_thread(update_generation_status, timestamp, new_status)
