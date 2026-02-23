"""
Simple JSON-file state management for pending approvals.
One pending draft at a time. Stored in state.json at project root.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).resolve().parent.parent / "state.json"


def _read_state() -> dict:
    """Read state.json, return empty dict if missing or corrupt."""
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read state.json: %s", e)
        return {}


def _write_state(data: dict) -> None:
    """Write state dict to state.json."""
    _STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def has_pending() -> bool:
    """Check if there is a draft pending approval."""
    return "pending" in _read_state()


def get_pending() -> dict | None:
    """Return the pending draft dict, or None if nothing is pending."""
    state = _read_state()
    return state.get("pending")


def save_pending(
    caption: str,
    hashtags: list[str],
    image_url: str | None,
    alt_text: str,
    image_prompt: str,
    original_request: str,
) -> None:
    """
    Save a draft as pending approval.

    Args:
        caption: Generated post caption.
        hashtags: List of hashtags.
        image_url: Generated image URL or None.
        alt_text: Image alt text.
        image_prompt: The prompt used for image generation.
        original_request: The user's original message.
    """
    state = {
        "pending": {
            "caption": caption,
            "hashtags": hashtags,
            "image_url": image_url,
            "alt_text": alt_text,
            "image_prompt": image_prompt,
            "original_request": original_request,
            "timestamp": time.time(),
        }
    }
    _write_state(state)
    logger.info("Saved pending draft for: %s", original_request[:80])


def clear_pending() -> None:
    """Clear any pending draft."""
    _write_state({})
    logger.info("Cleared pending state")
