"""
Channel registry — manages available publishing channels and multi-channel dispatch.
"""

import asyncio
import logging
from agent.channels.base import Channel, MessageEnvelope, PublishResult

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


async def publish_to_all(
    envelope: MessageEnvelope,
    channels: list[str] | None = None,
) -> list[PublishResult]:
    """Publish to all configured channels (or specific ones).

    Args:
        envelope: The message to publish.
        channels: Optional list of channel names to publish to.
                  If None, publishes to all configured channels.

    Returns:
        List of PublishResult, one per channel attempted.
    """
    if channels:
        targets = [_channels[name] for name in channels if name in _channels and _channels[name].is_configured()]
    else:
        targets = get_configured_channels()

    if not targets:
        logger.warning("No configured channels to publish to")
        return []

    # Publish to all channels concurrently
    tasks = [ch.publish(envelope) for ch in targets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    publish_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Channel %s failed: %s", targets[i].name, result)
            publish_results.append(PublishResult(
                success=False,
                platform=targets[i].name,
                error=str(result),
            ))
        else:
            publish_results.append(result)

    succeeded = sum(1 for r in publish_results if r.success)
    logger.info("Published to %d/%d channels", succeeded, len(publish_results))

    return publish_results


def _auto_register() -> None:
    """Auto-register built-in channels.

    Registers Twitter and Discord unconditionally (they check credentials
    at publish time). LinkedIn and Instagram are registered when their
    credentials are present in settings, since they are opt-in platforms.
    """
    from agent.channels.twitter import TwitterChannel
    from agent.channels.discord import DiscordChannel
    register_channel(TwitterChannel())
    register_channel(DiscordChannel())

    # LinkedIn — register when credentials are configured
    try:
        from config import settings as _settings
        if getattr(_settings, "LINKEDIN_ACCESS_TOKEN", "") and getattr(_settings, "LINKEDIN_AUTHOR_URN", ""):
            from agent.publishing.channels.linkedin import LinkedInChannel
            register_channel(LinkedInChannel())
    except Exception as e:
        logger.debug("LinkedIn channel registration skipped: %s", e)

    # Instagram — register when credentials are configured
    try:
        from config import settings as _settings
        if getattr(_settings, "INSTAGRAM_ACCESS_TOKEN", "") and getattr(_settings, "INSTAGRAM_BUSINESS_ACCOUNT_ID", ""):
            from agent.publishing.channels.instagram import InstagramChannel
            register_channel(InstagramChannel())
    except Exception as e:
        logger.debug("Instagram channel registration skipped: %s", e)


# Auto-register on import
_auto_register()
