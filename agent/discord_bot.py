"""
Discord client lifecycle, channel ID storage, and server structure builder.

Designed to run as a long-lived background task alongside the Telegram bot.
The client only sends messages — it never reads user messages (minimal intents).
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

import discord

from config import settings

logger = logging.getLogger(__name__)

_STATE_DIR = Path(__file__).resolve().parent.parent / "state"
_CHANNELS_FILE = _STATE_DIR / "discord_channels.json"

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: discord.Client | None = None
_ready_event = asyncio.Event()


def _make_client() -> discord.Client:
    intents = discord.Intents.default()
    intents.guilds = True
    return discord.Client(intents=intents)


async def start_client() -> None:
    """Start the Discord client. Meant to be wrapped in asyncio.create_task()."""
    global _client
    if not settings.DISCORD_BOT_TOKEN:
        logger.info("DISCORD_BOT_TOKEN not set — Discord client not started")
        return

    _client = _make_client()

    @_client.event
    async def on_ready():
        logger.info("Discord client ready as %s", _client.user)
        _ready_event.set()

    try:
        await _client.start(settings.DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.error("Discord client failed: %s", e)
        _ready_event.clear()


def get_client() -> discord.Client | None:
    return _client


def is_ready() -> bool:
    return _client is not None and _client.is_ready()


# ---------------------------------------------------------------------------
# Channel ID storage (state/discord_channels.json)
# ---------------------------------------------------------------------------


def _read_channels() -> dict[str, int]:
    if not _CHANNELS_FILE.exists():
        return {}
    try:
        data = json.loads(_CHANNELS_FILE.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("Failed to read discord_channels.json: %s", e)
        return {}


def _write_channels(data: dict[str, int]) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _CHANNELS_FILE.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def get_channel_id(key: str) -> int | None:
    """Look up a channel ID by key (e.g. 'announcements', 'memes')."""
    return _read_channels().get(key)


# ---------------------------------------------------------------------------
# Server structure definition
# ---------------------------------------------------------------------------

_CATEGORIES = {
    "INFORMATION": [
        {"name": "welcome", "read_only": True},
        {"name": "announcements", "read_only": True},
        {"name": "links", "read_only": True},
    ],
    "CONTENT": [
        {"name": "brand-updates", "read_only": True},
        {"name": "memes", "read_only": True},
        {"name": "onchain-updates", "read_only": True},
    ],
    "COMMUNITY": [
        {"name": "general", "read_only": False},
        {"name": "discussion", "read_only": False},
        {"name": "suggestions", "read_only": False},
    ],
    "GOVERNANCE": [
        {"name": "proposals", "read_only": True},
        {"name": "governance-chat", "read_only": False},
    ],
}

_ROLES = [
    {"name": "Admin", "color": discord.Color.red()},
    {"name": "Moderator", "color": discord.Color.orange()},
    {"name": "Community", "color": discord.Color.blue()},
    {"name": "Holder", "color": discord.Color.gold()},
]


# ---------------------------------------------------------------------------
# Server builder
# ---------------------------------------------------------------------------


async def setup_server(
    progress_callback: Callable[[str], object] | None = None,
) -> dict[str, str]:
    """Create categories, channels, and roles in the configured guild.

    Idempotent — checks for existing channels/roles before creating.
    Populates #welcome and #links with branded content.

    Returns a summary dict with counts and any errors.
    """
    if not is_ready():
        return {"error": "Discord client not ready"}

    guild = _client.get_guild(settings.DISCORD_GUILD_ID)
    if guild is None:
        return {"error": f"Guild {settings.DISCORD_GUILD_ID} not found"}

    async def _progress(msg: str):
        logger.info("Discord setup: %s", msg)
        if progress_callback:
            result = progress_callback(msg)
            if asyncio.iscoroutine(result):
                await result

    await _progress("Starting server setup...")

    channel_map: dict[str, int] = _read_channels()
    created_channels = 0
    created_roles = 0

    # --- Roles ---
    existing_roles = {r.name.lower(): r for r in guild.roles}
    for role_def in _ROLES:
        if role_def["name"].lower() in existing_roles:
            await _progress(f"Role '{role_def['name']}' already exists")
            continue
        await guild.create_role(
            name=role_def["name"],
            color=role_def["color"],
            mentionable=True,
        )
        created_roles += 1
        await _progress(f"Created role: {role_def['name']}")

    # --- Categories and Channels ---
    existing_categories = {c.name.upper(): c for c in guild.categories}
    existing_channels_by_name = {c.name: c for c in guild.text_channels}

    bot_member = guild.me

    for cat_name, channels in _CATEGORIES.items():
        # Get or create category
        category = existing_categories.get(cat_name)
        if category is None:
            category = await guild.create_category(cat_name)
            await _progress(f"Created category: {cat_name}")
        else:
            await _progress(f"Category '{cat_name}' already exists")

        for ch_def in channels:
            ch_name = ch_def["name"]

            # Check if channel already exists
            existing_ch = existing_channels_by_name.get(ch_name)
            if existing_ch is not None:
                channel_map[ch_name] = existing_ch.id
                await _progress(f"Channel #{ch_name} already exists")
                continue

            # Build permission overwrites
            overwrites = {}
            if ch_def["read_only"]:
                overwrites[guild.default_role] = discord.PermissionOverwrite(
                    send_messages=False,
                    add_reactions=True,
                    read_messages=True,
                )
            # Bot always has full send/embed/attach
            if bot_member:
                overwrites[bot_member] = discord.PermissionOverwrite(
                    send_messages=True,
                    embed_links=True,
                    attach_files=True,
                    read_messages=True,
                )

            channel = await guild.create_text_channel(
                ch_name,
                category=category,
                overwrites=overwrites,
            )
            channel_map[ch_name] = channel.id
            created_channels += 1
            await _progress(f"Created channel: #{ch_name}")

    # Save channel map
    _write_channels(channel_map)
    await _progress("Channel map saved")

    # --- Populate #welcome and #links ---
    await _populate_welcome(guild, channel_map)
    await _populate_links(guild, channel_map)
    await _progress("Info channels populated")

    summary = {
        "status": "ok",
        "created_channels": str(created_channels),
        "created_roles": str(created_roles),
        "total_channels": str(len(channel_map)),
    }
    await _progress(
        f"Setup complete: {created_channels} channels, {created_roles} roles created"
    )
    return summary


async def _populate_welcome(guild: discord.Guild, channel_map: dict[str, int]) -> None:
    """Send a branded welcome embed to #welcome."""
    ch_id = channel_map.get("welcome")
    if not ch_id:
        return
    channel = guild.get_channel(ch_id)
    if not channel:
        return

    # Skip if channel already has messages (idempotent)
    try:
        async for _ in channel.history(limit=1):
            return
    except discord.Forbidden:
        logger.warning("No permission to read #welcome history")

    from agent import compositor_config as cc
    cfg = cc.get_config()
    brand_name = cfg.brand_name or settings.BRAND_NAME
    tagline = cfg.tagline or ""
    primary_hex = cc.get_color_hex("primary", "#5865F2")

    embed = discord.Embed(
        title=f"Welcome to {brand_name}",
        description=(
            f"{tagline}\n\n"
            "**Community Rules:**\n"
            "1. Be respectful\n"
            "2. No spam or self-promotion\n"
            "3. Stay on topic\n"
            "4. Have fun!\n\n"
            "Check out the other channels to get started."
        ),
        color=discord.Color.from_str(primary_hex),
    )

    # Add logo as thumbnail if available
    logo_path = Path(settings.BRAND_FOLDER) / "assets" / "logo.png"
    file = None
    if logo_path.exists():
        file = discord.File(str(logo_path), filename="logo.png")
        embed.set_thumbnail(url="attachment://logo.png")

    kwargs = {"embed": embed}
    if file:
        kwargs["file"] = file
    await channel.send(**kwargs)


async def _populate_links(guild: discord.Guild, channel_map: dict[str, int]) -> None:
    """Send links embed to #links."""
    ch_id = channel_map.get("links")
    if not ch_id:
        return
    channel = guild.get_channel(ch_id)
    if not channel:
        return

    try:
        async for _ in channel.history(limit=1):
            return
    except discord.Forbidden:
        logger.warning("No permission to read #links history")

    from agent import compositor_config as cc
    cfg = cc.get_config()
    brand_name = cfg.brand_name or settings.BRAND_NAME
    primary_hex = cc.get_color_hex("primary", "#5865F2")

    lines = [f"**{brand_name} Links**\n"]
    if cfg.website:
        lines.append(f"Website: {cfg.website}")
    if cfg.x_handle:
        handle = cfg.x_handle.lstrip("@")
        lines.append(f"X/Twitter: https://x.com/{handle}")

    embed = discord.Embed(
        title="Official Links",
        description="\n".join(lines),
        color=discord.Color.from_str(primary_hex),
    )
    await channel.send(embed=embed)
