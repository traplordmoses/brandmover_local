"""
Core handler utilities — auth helpers, rate limiting, state variables, shared helpers.
"""

__all__: list[str] = []  # No public command handlers; all names are internal helpers.

import asyncio as _aio
import html
import io
import logging
import random as _random
import time
from pathlib import Path

from PIL import Image as _PILImage
_PILImage.MAX_IMAGE_PIXELS = 50_000_000  # 50MP limit — protect against image bombs
from telegram import Update
from telegram.ext import ContextTypes

from agent import compositor_config as _cc
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state variables
# ---------------------------------------------------------------------------

# Rate limiting — minimum seconds between generation requests per user
_RATE_LIMIT_SECONDS = 10
_DAILY_REQUEST_LIMIT = 200
_last_request_time: dict[int, float] = {}
_daily_request_count: dict[int, tuple[int, float]] = {}  # (count, reset_time)

# Tools restricted to admin-only — operators cannot use these
_ADMIN_ONLY_TOOLS = {"execute_code", "post_approved", "create_skill", "git_info"}

# Bulk upload batch tracking — maps user_id to pending asyncio task
_bulk_upload_tasks: dict[int, _aio.Task] = {}

# Per-user lock to prevent double-post race (e.g., tapping "post" twice quickly)
_approve_locks: dict[int, _aio.Lock] = {}

# State for /reset_brand confirmation — moved here so cleanup can reference it
_reset_pending: dict[int, float] = {}

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_approve_lock(user_id: int) -> _aio.Lock:
    if user_id not in _approve_locks:
        _approve_locks[user_id] = _aio.Lock()
    return _approve_locks[user_id]


def _authorized(user_id: int) -> bool:
    """Check if a Telegram user is the admin."""
    return user_id == settings.TELEGRAM_ALLOWED_USER_ID


def _can_operate(user_id: int) -> bool:
    """Check if a Telegram user can generate/approve/reject content.

    Returns True for admin and any user in TELEGRAM_OPERATOR_IDS.
    """
    if user_id == settings.TELEGRAM_ALLOWED_USER_ID:
        return True
    return user_id in settings.TELEGRAM_OPERATOR_IDS


# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

# Patterns that indicate the user wants to generate a template from their reference image
_TEMPLATE_FROM_REF_PATTERNS = [
    "make a template",
    "make template",
    "create a template",
    "create template",
    "generate a template",
    "generate template",
    "use this layout",
    "use this as a template",
    "use this as template",
    "turn this into a template",
    "template from this",
    "template this",
    "copy this layout",
    "replicate this layout",
    "recreate this layout",
    "use this format",
    "copy this format",
]


def _is_template_from_ref_intent(caption: str) -> bool:
    """Check if a photo caption expresses intent to generate a template from reference."""
    lower = caption.lower().strip()
    return any(p in lower for p in _TEMPLATE_FROM_REF_PATTERNS)


# Patterns indicating the user wants to use their uploaded photo directly (no AI generation)
_DIRECT_PHOTO_PATTERNS = [
    "use this", "post this", "announce this", "use this photo",
    "use this image", "publish this", "tweet this", "share this",
    "put this in", "use my photo", "use my image",
]


# Keywords that suggest the user is describing template region positions
_REGION_POSITION_KEYWORDS = [
    "top", "bottom", "left", "right", "centered", "center",
    "full canvas", "full width", "entire background", "background",
    "text goes", "text across", "image zone", "image area",
    "title", "subtitle", "headline",
]


def _is_template_region_update(message: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if a message describes template region positions after a recent upload.

    Returns True if:
    - A template was uploaded within the last 2 messages (user_data has last_uploaded_template_id)
    - The message contains position keywords + percentage or layout terms
    """
    user_data = context.user_data if context else {}
    template_id = user_data.get("last_uploaded_template_id")
    if not template_id:
        return False

    lower = message.lower()
    keyword_hits = sum(1 for kw in _REGION_POSITION_KEYWORDS if kw in lower)
    has_percentage = "%" in lower
    has_region_type = any(w in lower for w in ("text", "image", "logo"))

    # Need at least 2 keyword hits AND (a percentage or a region type word)
    return keyword_hits >= 2 and (has_percentage or has_region_type)


def _is_direct_photo_intent(caption: str) -> bool:
    """Check if a photo caption means 'use this photo directly, don't regenerate it'."""
    lower = caption.lower().strip()
    return any(p in lower for p in _DIRECT_PHOTO_PATTERNS)


# ---------------------------------------------------------------------------
# Cleanup and rate limiting
# ---------------------------------------------------------------------------

_STALE_ENTRY_SECONDS = 3600  # 1 hour
_MAX_APPROVE_LOCKS = 100


def _cleanup_stale_entries() -> None:
    """Remove stale entries from module-level dicts to prevent unbounded growth."""
    now = time.time()

    # Remove rate-limit entries older than 1 hour
    stale = [uid for uid, ts in _last_request_time.items() if now - ts > _STALE_ENTRY_SECONDS]
    for uid in stale:
        del _last_request_time[uid]

    # Remove expired daily counters
    expired = [uid for uid, (_, reset) in _daily_request_count.items() if now > reset]
    for uid in expired:
        del _daily_request_count[uid]

    # Remove finished bulk upload tasks
    done = [uid for uid, task in _bulk_upload_tasks.items() if task.done()]
    for uid in done:
        del _bulk_upload_tasks[uid]

    # Cap approve locks at 100 — drop unlocked entries first
    if len(_approve_locks) > _MAX_APPROVE_LOCKS:
        unlocked = [uid for uid, lock in _approve_locks.items() if not lock.locked()]
        for uid in unlocked:
            del _approve_locks[uid]
            if len(_approve_locks) <= _MAX_APPROVE_LOCKS:
                break

    # Remove stale reset_brand confirmation entries
    stale_reset = [uid for uid, ts in _reset_pending.items() if now - ts > _STALE_ENTRY_SECONDS]
    for uid in stale_reset:
        del _reset_pending[uid]


def _rate_limited(user_id: int) -> bool:
    """Check if user is sending requests too fast. Returns True if blocked."""
    _cleanup_stale_entries()
    now = time.time()

    # Per-request cooldown
    last = _last_request_time.get(user_id, 0)
    if now - last < _RATE_LIMIT_SECONDS:
        return True
    _last_request_time[user_id] = now

    # Daily request cap
    count, reset_time = _daily_request_count.get(user_id, (0, now + 86400))
    if now > reset_time:
        _daily_request_count[user_id] = (1, now + 86400)
    elif count >= _DAILY_REQUEST_LIMIT:
        return True
    else:
        _daily_request_count[user_id] = (count + 1, reset_time)

    return False


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape text for Telegram messages."""
    return html.escape(str(text))


def _truncate_reasoning(text: str, max_len: int = 100) -> str:
    """Truncate Claude's reasoning text to a short summary line for the status message."""
    # Take the first sentence or up to max_len chars
    text = text.replace("\n", " ").strip()
    # Find first sentence end
    for end in (".", "!", "?"):
        idx = text.find(end)
        if 0 < idx < max_len:
            return text[: idx + 1]
    if len(text) > max_len:
        # Cut at last word boundary
        cut = text[:max_len].rsplit(" ", 1)[0]
        return cut + "..."
    return text


_REVIEW_PROMPTS = [
    "how does this look?",
    "what do you think?",
    "want any changes?",
    "ready to go, or need tweaks?",
]


def _prepare_photo(photo) -> io.BytesIO | str | None:
    """Convert any photo source to a Telegram-compatible format.

    - BytesIO -> pass through
    - HTTP URL string -> pass through
    - Local file path string -> BytesIO(read_bytes())
    - Missing file -> None
    """
    if isinstance(photo, io.BytesIO):
        return photo
    if isinstance(photo, str):
        if photo.startswith("http"):
            return photo
        # Local file path from cache_image()
        p = Path(photo)
        if p.exists():
            buf = io.BytesIO(p.read_bytes())
            buf.name = p.name  # Telegram uses .name for format detection
            return buf
        return None
    return photo  # other file-like objects


_STEP_ICONS = {
    "Analyze": "\U0001F50D",          # magnifying glass
    "Plan": "\U0001F4DD",             # memo
    "Verify": "\u2705",               # check mark
    "Plan & Verify": "\U0001F4DD\u2705",
    "Generate": "\u2728",             # sparkles
}

_TOOL_ICONS = {
    "read_brand_guidelines": "\U0001F4DA",   # books
    "read_references": "\U0001F4C2",         # folder
    "check_figma_design": "\U0001F3A8",      # palette
    "generate_image": "\U0001F5BC",          # framed picture
    "img2img": "\U0001F5BC",                 # framed picture
    "read_feedback_history": "\U0001F4AC",   # speech bubble
    "log_resource_usage": "\U0001F4CB",      # clipboard
    "execute_openclaw_script": "\u26D3",     # chain
}


# ---------------------------------------------------------------------------
# Compositor helper
# ---------------------------------------------------------------------------


async def _maybe_compose(draft: dict, image_url: str, content_type: str):
    """Compositor guard. Returns (photo_to_send, composed_bytes_or_None).

    Priority chain: template > compositor > raw.
    /template off disables both templates and compositor.
    """
    from agent import compositor

    cfg = _cc.get_config()
    if not cfg.compositor_enabled:
        return image_url, None

    from agent import template_memory as _tm

    # Priority 1: Template
    try:
        memory = _tm.TemplateMemory()
        template = memory.get_template_for_content_type(content_type)
        if template:
            composed = await _tm.apply_template(template, image_url, draft)
            if composed:
                return composed, composed
    except Exception as e:
        logger.debug("Template composition failed, falling through: %s", e)

    # Priority 2: Compositor
    composed = await compositor.compose_branded_image(draft, image_url, content_type)
    return (composed if composed else image_url), composed


# ---------------------------------------------------------------------------
# Commentary extraction
# ---------------------------------------------------------------------------


def _extract_commentary(response_text: str) -> str:
    """Extract personality commentary from before the JSON draft block.

    Returns the text before the ```json fence, stripped and cleaned.
    """
    import re

    # Find the start of the JSON fence
    fence_idx = response_text.find("```json")
    if fence_idx == -1:
        fence_idx = response_text.find("```\n{")
    if fence_idx == -1:
        return ""

    commentary = response_text[:fence_idx].strip()
    # Remove trailing markers like "Here's your draft:" that are just filler
    commentary = re.sub(r"\s*(here'?s?\s+(the|your)\s+draft:?\s*)$", "", commentary, flags=re.IGNORECASE).strip()
    return commentary
