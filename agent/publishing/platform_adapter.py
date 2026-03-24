"""
Platform adapter -- transforms drafts for different social platforms.

Each platform has different character limits, formatting rules, and conventions.
This module provides a clean abstraction over those differences so the publishing
pipeline can work with a uniform PlatformPost dataclass regardless of destination.

ARCHITECTURE:
- Platform configs are hardcoded dicts (no external files needed).
- adapt_for_platform() transforms a single draft for one platform.
- adapt_for_all_platforms() transforms for all enabled platforms in one call.
- The adapter never touches the network -- it only reshapes text/metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlatformPost:
    """A draft transformed for a specific platform."""
    platform: str
    text: str
    image_url: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Platform configs
# ---------------------------------------------------------------------------

PLATFORM_CONFIGS: dict[str, dict] = {
    "x": {
        "max_chars": 280,
        "strip_hashtags": True,
        "image_upload": True,
    },
    "discord": {
        "max_chars": 2000,
        "strip_hashtags": False,
        "hashtags_in_footer": True,
        "embed_format": True,
        "image_upload": True,
    },
    "telegram": {
        "max_chars": 4096,
        "strip_hashtags": False,
        "keep_full_length": True,
        "image_upload": True,
    },
    "linkedin": {
        "max_chars": 3000,
        "strip_hashtags": False,
        "hashtags_inline": True,
        "image_upload": True,
    },
    "instagram": {
        "max_chars": 2200,
        "strip_hashtags": False,
        "hashtags_inline": True,
        "image_upload": True,
        "image_required": True,
    },
}

SUPPORTED_PLATFORMS = list(PLATFORM_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Adaptation logic
# ---------------------------------------------------------------------------

def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def adapt_for_platform(
    draft: dict,
    platform: str,
    image_url: str | None = None,
) -> PlatformPost:
    """Transform a draft dict into a PlatformPost for the given platform.

    Args:
        draft: Standard draft dict with 'caption', 'hashtags', etc.
        platform: One of 'x', 'discord', 'telegram'.
        image_url: Optional image URL or local path to attach.

    Returns:
        PlatformPost ready for publishing.
    """
    config = PLATFORM_CONFIGS.get(platform)
    if config is None:
        raise ValueError(f"Unsupported platform: {platform!r} (supported: {SUPPORTED_PLATFORMS})")

    caption = draft.get("caption", "")
    hashtags = draft.get("hashtags", [])
    max_chars = config["max_chars"]

    if platform == "x":
        # X: build full text with hashtags, then truncate to 280
        hashtag_str = " ".join(hashtags)
        if hashtag_str:
            full_text = f"{caption}\n\n{hashtag_str}".strip()
        else:
            full_text = caption.strip()
        text = _truncate_text(full_text, max_chars)

    elif platform == "discord":
        # Discord: caption as main text, hashtags moved to metadata for embed footer
        text = _truncate_text(caption.strip(), max_chars)

    elif platform == "telegram":
        # Telegram: keep full caption + hashtags inline
        hashtag_str = " ".join(hashtags)
        if hashtag_str:
            full_text = f"{caption}\n\n{hashtag_str}".strip()
        else:
            full_text = caption.strip()
        text = _truncate_text(full_text, max_chars)

    elif platform == "linkedin":
        # LinkedIn: full caption + hashtags inline at the end, 3000 char limit
        hashtag_str = " ".join(hashtags)
        if hashtag_str:
            full_text = f"{caption}\n\n{hashtag_str}".strip()
        else:
            full_text = caption.strip()
        text = _truncate_text(full_text, max_chars)

    elif platform == "instagram":
        # Instagram: caption + hashtags inline (often in a separate paragraph), 2200 char limit
        hashtag_str = " ".join(hashtags)
        if hashtag_str:
            full_text = f"{caption}\n\n.\n.\n.\n{hashtag_str}".strip()
        else:
            full_text = caption.strip()
        text = _truncate_text(full_text, max_chars)

    else:
        text = _truncate_text(caption.strip(), max_chars)

    metadata = {
        "content_type": draft.get("content_type", ""),
        "alt_text": draft.get("alt_text", ""),
        "auto_slot": draft.get("auto_slot"),
    }

    # Discord-specific: hashtags go in footer
    if platform == "discord" and hashtags:
        metadata["footer_hashtags"] = " ".join(hashtags)

    return PlatformPost(
        platform=platform,
        text=text,
        image_url=image_url,
        metadata=metadata,
    )


def adapt_for_all_platforms(
    draft: dict,
    image_url: str | None = None,
    platforms: list[str] | None = None,
) -> dict[str, PlatformPost]:
    """Transform a draft for all specified platforms.

    Args:
        draft: Standard draft dict.
        image_url: Optional image URL or local path.
        platforms: List of platform names. If None, uses all supported platforms.

    Returns:
        Dict mapping platform name to PlatformPost.
    """
    if platforms is None:
        platforms = list(SUPPORTED_PLATFORMS)

    results: dict[str, PlatformPost] = {}
    for platform in platforms:
        try:
            results[platform] = adapt_for_platform(draft, platform, image_url)
        except ValueError:
            logger.warning("Skipping unsupported platform: %s", platform)
    return results
