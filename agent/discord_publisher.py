"""
Publish branded embeds to Discord channels.

Maps content to the correct channel, builds rich embeds with brand colors,
and handles both URL-based and local file images.
Fire-and-forget — failures are logged but never block X posting.
"""

import logging
from pathlib import Path

import discord

from agent import discord_bot
from agent import compositor_config as cc
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel routing
# ---------------------------------------------------------------------------

# Auto-post slot → channel key
_SLOT_CHANNEL_MAP: dict[str, str] = {
    "engagement_morning": "brand-updates",
    "onchain_midday": "onchain-updates",
    "onchain_afternoon": "onchain-updates",
    "brand_meme": "memes",
}

# Content type → channel key (for manual posts)
_CONTENT_TYPE_CHANNEL_MAP: dict[str, str] = {
    "announcement": "announcements",
    "meme": "memes",
    "onchain": "onchain-updates",
    "onchain_update": "onchain-updates",
    "brand_update": "brand-updates",
    "engagement": "brand-updates",
}

_DEFAULT_CHANNEL = "brand-updates"


def _route_channel(
    auto_slot: str | None = None,
    content_type: str | None = None,
) -> str:
    """Determine which Discord channel key to post to.

    Priority: scheduled: prefix → announcements, slot map, content type map, fallback.
    """
    if auto_slot:
        if auto_slot.startswith("scheduled:"):
            return "announcements"
        mapped = _SLOT_CHANNEL_MAP.get(auto_slot)
        if mapped:
            return mapped

    if content_type:
        mapped = _CONTENT_TYPE_CHANNEL_MAP.get(content_type)
        if mapped:
            return mapped

    return _DEFAULT_CHANNEL


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------


def _build_embed(
    caption: str,
    hashtags: list[str] | None = None,
    image_url: str | None = None,
    local_file: str | None = None,
) -> tuple[discord.Embed, discord.File | None]:
    """Build a branded Discord embed.

    Returns (embed, file_or_none). If a local file is used, the caller
    must pass the File object to channel.send().
    """
    cfg = cc.get_config()
    brand_name = cfg.brand_name or settings.BRAND_NAME
    primary_hex = cc.get_color_hex("primary", "#5865F2")

    embed = discord.Embed(
        description=caption,
        color=discord.Color.from_str(primary_hex),
    )

    # Author line with X profile link
    author_kwargs: dict = {"name": brand_name}
    if cfg.x_handle:
        handle = cfg.x_handle.lstrip("@")
        author_kwargs["url"] = f"https://x.com/{handle}"
    embed.set_author(**author_kwargs)

    # Hashtags as footer
    if hashtags:
        embed.set_footer(text=" ".join(hashtags))

    # Image handling
    file = None
    if local_file and Path(local_file).exists():
        file = discord.File(local_file, filename="post.png")
        embed.set_image(url="attachment://post.png")
    elif image_url and image_url.startswith(("http://", "https://")):
        embed.set_image(url=image_url)
    else:
        # No main image — use logo as thumbnail
        logo_path = Path(settings.BRAND_FOLDER) / "assets" / "logo.png"
        if logo_path.exists():
            file = discord.File(str(logo_path), filename="logo.png")
            embed.set_thumbnail(url="attachment://logo.png")

    return embed, file


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def post_to_discord(
    caption: str,
    hashtags: list[str] | None = None,
    image_url: str | None = None,
    auto_slot: str | None = None,
    content_type: str | None = None,
) -> str | None:
    """Post a branded embed to the appropriate Discord channel.

    Returns the message jump URL on success, or None on failure/skip.
    """
    if not discord_bot.is_ready():
        logger.debug("Discord not ready — skipping post")
        return None

    channel_key = _route_channel(auto_slot=auto_slot, content_type=content_type)
    channel_id = discord_bot.get_channel_id(channel_key)
    if not channel_id:
        logger.debug("No channel ID for '%s' — skipping Discord post", channel_key)
        return None

    client = discord_bot.get_client()
    if not client:
        return None

    channel = client.get_channel(channel_id)
    if not channel:
        logger.warning("Discord channel %d not found in cache", channel_id)
        return None

    # Determine if we have a local composed file to use
    local_file = None
    if image_url and not image_url.startswith(("http://", "https://")):
        # It's a local path
        if Path(image_url).exists():
            local_file = image_url
            image_url = None

    embed, file = _build_embed(
        caption=caption,
        hashtags=hashtags,
        image_url=image_url,
        local_file=local_file,
    )

    kwargs: dict = {"embed": embed}
    if file:
        kwargs["file"] = file

    msg = await channel.send(**kwargs)
    logger.info("Posted to Discord #%s: %s", channel_key, msg.jump_url)
    return msg.jump_url
