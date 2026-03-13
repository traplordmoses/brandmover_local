"""
Channel registry — manages available publishing channels.
"""

import logging
from agent.channels.base import Channel

logger = logging.getLogger(__name__)

_channels: dict[str, Channel] = {}


def register_channel(channel: Channel) -> None:
    """Register a channel adapter."""
    _channels[channel.name] = channel
    logger.info("Registered channel: %s (configured=%s)", channel.name, channel.is_configured())


def get_channel(name: str) -> Channel | None:
    """Get a channel by name."""
    return _channels.get(name)


def list_channels() -> list[dict]:
    """List all registered channels with their status."""
    return [
        {
            "name": ch.name,
            "configured": ch.is_configured(),
            "max_text_length": ch.max_text_length,
        }
        for ch in _channels.values()
    ]


def get_configured_channels() -> list[Channel]:
    """Return all channels that have valid credentials."""
    return [ch for ch in _channels.values() if ch.is_configured()]


def _auto_register() -> None:
    """Auto-register built-in channels."""
    from agent.channels.twitter import TwitterChannel
    register_channel(TwitterChannel())


# Auto-register on import
_auto_register()
