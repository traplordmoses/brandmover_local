"""
X/Twitter channel adapter.
"""

import logging
from agent.channels.base import Channel, MessageEnvelope, PublishResult

logger = logging.getLogger(__name__)


class TwitterChannel(Channel):
    """Publishes content to X/Twitter via the existing publisher module."""

    @property
    def name(self) -> str:
        return "twitter"

    @property
    def max_text_length(self) -> int:
        return 280

    def is_configured(self) -> bool:
        from config import settings
        return bool(settings.X_API_KEY and settings.X_ACCESS_TOKEN)

    async def publish(self, envelope: MessageEnvelope) -> PublishResult:
        """Publish via the existing publisher.post_to_x function."""
        if not self.is_configured():
            return PublishResult(
                success=False,
                platform=self.name,
                error="X/Twitter credentials not configured",
            )

        try:
            from agent.publisher import post_to_x
            tweet_url = await post_to_x(
                caption=envelope.text,
                hashtags=envelope.hashtags,
                image_url=envelope.image_url,
            )
            logger.info("Published to X/Twitter: %s", tweet_url)
            return PublishResult(
                success=True,
                url=tweet_url,
                platform=self.name,
            )
        except Exception as e:
            logger.error("Failed to publish to X/Twitter: %s", e)
            return PublishResult(
                success=False,
                platform=self.name,
                error=str(e),
            )
