"""
Retry queue for failed X/Twitter posts.
Failed posts are saved to state/publish_queue.json and retried periodically.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.paths import STATE_DIR

logger = logging.getLogger(__name__)

_QUEUE_FILE = STATE_DIR / "publish_queue.json"
_MAX_ATTEMPTS = 3
_lock = threading.Lock()


def _read_queue() -> list[dict]:
    """Read the publish queue from disk. Returns empty list if missing or corrupt."""
    if not _QUEUE_FILE.exists():
        return []
    try:
        data = json.loads(_QUEUE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read publish queue: %s", e)
        return []


def _write_queue(entries: list[dict]) -> None:
    """Write the publish queue to disk using atomic rename."""
    _QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _QUEUE_FILE.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}")
    tmp_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(str(tmp_path), str(_QUEUE_FILE))


def enqueue_failed(
    caption: str,
    image_path: str | None,
    content_type: str,
    error: str,
) -> int:
    """Save a failed post to the retry queue.

    Args:
        caption: Tweet text that failed to post.
        image_path: Image URL/path, or None for text-only.
        content_type: Content type of the post.
        error: Error message from the failed attempt.

    Returns:
        Current queue length after adding the entry.
    """
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "caption": caption,
        "image_path": image_path,
        "content_type": content_type,
        "error": error,
        "attempts": 1,
        "created_at": now,
        "last_attempt": now,
    }
    with _lock:
        queue = _read_queue()
        queue.append(entry)
        _write_queue(queue)
        logger.info("Enqueued failed post for retry (queue size: %d)", len(queue))
        return len(queue)


def get_pending() -> list[dict]:
    """Return all pending retries (not yet abandoned)."""
    with _lock:
        queue = _read_queue()
        return [e for e in queue if e.get("attempts", 0) < _MAX_ATTEMPTS]


async def retry_pending() -> list[dict]:
    """Attempt to publish each pending queued item.

    Successful posts are removed from the queue. Failed posts have their
    attempt count incremented. Posts exceeding MAX_ATTEMPTS are marked
    as abandoned.

    Note: The lock is released during each network call (post_to_x) so that
    enqueue_failed() is not blocked for the duration of all retries. A
    snapshot-then-overwrite approach is used: the queue is read under lock,
    retries run without the lock, and the updated queue is written back under
    lock. Because retry_pending is only called from the single scheduler loop,
    concurrent calls to retry_pending do not occur. Concurrent enqueue_failed
    calls between the read and write are safe because the write replaces only
    the entries that were in the snapshot; any newly enqueued items will be
    picked up on the next retry cycle.

    Returns:
        List of result dicts with keys: caption, status ("success", "failed",
        "abandoned"), error (if failed/abandoned), tweet_url (if success).
    """
    from agent.publisher import post_to_x  # late import to avoid circular

    with _lock:
        queue = _read_queue()
        # Capture the snapshot length so we can merge on write-back.
        snapshot_len = len(queue)

    if not queue:
        return []

    results = []
    updated_queue = []

    for entry in queue:
        if entry.get("attempts", 0) >= _MAX_ATTEMPTS:
            # Already abandoned -- keep in queue as-is
            updated_queue.append(entry)
            continue

        try:
            tweet_url = await post_to_x(
                caption=entry["caption"],
                hashtags=[],
                image_url=entry.get("image_path"),
                _from_retry=True,
            )
            results.append({
                "caption": entry["caption"][:80],
                "status": "success",
                "tweet_url": tweet_url,
            })
            logger.info("Retry succeeded for: %s", entry["caption"][:80])
            # Do not add to updated_queue -- removal on success

        except Exception as e:
            now = datetime.now(timezone.utc).isoformat()
            entry["attempts"] = entry.get("attempts", 0) + 1
            entry["error"] = str(e)
            entry["last_attempt"] = now

            if entry["attempts"] >= _MAX_ATTEMPTS:
                status = "abandoned"
                logger.warning(
                    "Post abandoned after %d attempts: %s",
                    entry["attempts"],
                    entry["caption"][:80],
                )
            else:
                status = "failed"
                logger.info(
                    "Retry %d failed for: %s -- %s",
                    entry["attempts"],
                    entry["caption"][:80],
                    e,
                )

            updated_queue.append(entry)
            results.append({
                "caption": entry["caption"][:80],
                "status": status,
                "error": str(e),
            })

    with _lock:
        # Merge: preserve any entries enqueued while retries were running.
        current_queue = _read_queue()
        newly_enqueued = current_queue[snapshot_len:]
        _write_queue(updated_queue + newly_enqueued)

    return results


def clear_queue() -> int:
    """Clear all entries from the retry queue.

    Returns:
        Number of entries cleared.
    """
    with _lock:
        queue = _read_queue()
        count = len(queue)
        _write_queue([])
        if count:
            logger.info("Cleared %d entries from publish queue", count)
        return count
