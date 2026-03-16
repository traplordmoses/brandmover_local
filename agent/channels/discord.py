"""
Discord channel adapter — wraps discord_publisher into the Channel interface.
"""

import logging
from agent.channels.base import Channel, MessageEnvelope, PublishResult

logger = logging.getLogger(__name__)


class DiscordChannel(Channel):
    """Publishes content to Discord via the existing discord_publisher module."""

    @property
    def name(self) -> str:
        return "discord"

    @property
    def max_text_length(self) -> int:
        return 2000

    def is_configured(self) -> bool:
        from config import settings
        return bool(settings.DISCORD_BOT_TOKEN and settings.DISCORD_GUILD_ID)

    async def publish(self, envelope: MessageEnvelope) -> PublishResult:
        """Publish via the existing discord_publisher module."""
        if not self.is_configured():
            return PublishResult(
                success=False,
                platform=self.name,
                error="Discord credentials not configured",
            )

        try:
            from agent.discord_publisher import post_to_discord

            result = await post_to_discord(
                caption=envelope.text,
                image_url=envelope.image_url,
                content_type=envelope.content_type or None,
                hashtags=envelope.hashtags or None,
            )
            logger.info("Published to Discord: %s", result)
            return PublishResult(
                success=True,
                url=result or "",
                platform=self.name,
            )
        except Exception as e:
            logger.error("Failed to publish to Discord: %s", e)
            return PublishResult(
                success=False,
                platform=self.name,
                error=str(e),
            )
