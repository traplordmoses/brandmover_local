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
from datetime import datetime, timezone
from pathlib import Path

from agent.state_manager import FileStore

logger = logging.getLogger(__name__)

# Threading lock for sync read/write functions (protects read-modify-write sequences)
_sync_lock = threading.Lock()

_project_root = Path(__file__).resolve().parent.parent
from agent.paths import STATE_DIR as _STATE_DIR, migrate_state_file
_HISTORY_FILE = _STATE_DIR / "generation_history.json"

# Migrate from old location if needed
migrate_state_file(_project_root / "generation_history.json", _HISTORY_FILE)

# ---------------------------------------------------------------------------
# File I/O — delegated to FileStore
# ---------------------------------------------------------------------------

_store = FileStore(_HISTORY_FILE, default_factory=list)


def _get_store() -> FileStore:
    """Return the active FileStore, respecting any monkey-patching of _HISTORY_FILE."""
    global _store
    if _store.path != _HISTORY_FILE:
        _store = FileStore(_HISTORY_FILE, default_factory=list)
    return _store


def _read_history() -> list[dict]:
    """Read the history log. Returns empty list if missing or corrupt."""
    return _get_store().read()


def _write_history(entries: list[dict]) -> None:
    """Write the full history log."""
    _get_store().write(entries)


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


def _maybe_rotate_on_load() -> None:
    """Rotate on import to clean up if a previous process crashed before rotation."""
    with _sync_lock:
        entries = _read_history()
        if len(entries) > _MAX_HISTORY_ENTRIES:
            rotated = _maybe_rotate(entries)
            _write_history(rotated)


def _maybe_rotate(entries: list[dict]) -> list[dict]:
    """If entries exceed _MAX_HISTORY_ENTRIES, archive older ones and keep the most recent.

    Archive file goes to state/generation_history_archive_{timestamp}.json.
    Must be called under _sync_lock.
    Returns the (possibly trimmed) entries list.
    """
    if len(entries) <= _MAX_HISTORY_ENTRIES:
        return entries
    archive_count = len(entries) - _MAX_HISTORY_ENTRIES
    archived = entries[:archive_count]
    kept = entries[archive_count:]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = _STATE_DIR / f"generation_history_archive_{ts}.json"
    archive_path.write_text(
        json.dumps(archived, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Rotated %d generation history entries to %s (keeping %d)",
        archive_count, archive_path.name, len(kept),
    )
    return kept


# Rotate on import to clean up if previous process crashed before rotation
try:
    _maybe_rotate_on_load()
except Exception:
    pass


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
    with _sync_lock:
        return _log_generation_locked(
            asset_type, content_type, prompt, model_id, image_urls,
            original_request, status, tags,
        )


def _log_generation_locked(
    asset_type, content_type, prompt, model_id, image_urls,
    original_request, status, tags,
) -> int:
    """Inner implementation of log_generation, called under _sync_lock."""
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
    entries = _maybe_rotate(entries)
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
    with _sync_lock:
        return _update_generation_status_locked(timestamp, new_status)


def _update_generation_status_locked(timestamp: float, new_status: str) -> bool:
    """Inner implementation of update_generation_status, called under _sync_lock."""
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
    return await asyncio.to_thread(log_generation, *args, **kwargs)

async def async_update_generation_status(timestamp: float, new_status: str) -> bool:
    return await asyncio.to_thread(update_generation_status, timestamp, new_status)
