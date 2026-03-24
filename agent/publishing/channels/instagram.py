"""
Instagram channel adapter — publishes content via the Facebook Graph API.

Uses the Instagram Graph API (via Facebook) to publish image posts and carousel
posts for Instagram Business / Creator accounts. Requires:
    - INSTAGRAM_ACCESS_TOKEN: Long-lived Facebook Page access token with
      instagram_basic, instagram_content_publish, pages_read_engagement
    - INSTAGRAM_BUSINESS_ACCOUNT_ID: The Instagram Business Account ID
      (numeric ID from the Graph API, NOT the @username)

Image flow (single image):
    1. POST /ig-user/media with image_url + caption to create a media container
    2. POST /ig-user/media_publish with the container ID to publish

Carousel flow (up to 10 images):
    1. POST /ig-user/media for each image (is_carousel_item=true) to create item containers
    2. POST /ig-user/media with media_type=CAROUSEL + children IDs to create carousel container
    3. POST /ig-user/media_publish with the carousel container ID to publish

NOTE: Instagram requires images to be publicly accessible URLs. Local file
paths will cause failures. The caller should ensure images are hosted.
"""

import asyncio
import logging

import httpx

from agent.publishing.channels.base import Channel, MessageEnvelope, PublishResult

logger = logging.getLogger(__name__)

_GRAPH_API_BASE = "https://graph.facebook.com/v19.0"
_MAX_TEXT_LENGTH = 2200
_MAX_CAROUSEL_ITEMS = 10


class InstagramChannel(Channel):
    """Publishes content to Instagram via the Facebook Graph API."""

    @property
    def name(self) -> str:
        return "instagram"

    @property
    def max_text_length(self) -> int:
        return _MAX_TEXT_LENGTH

    def is_configured(self) -> bool:
        from config import settings
        return bool(
            getattr(settings, "INSTAGRAM_ACCESS_TOKEN", "")
            and getattr(settings, "INSTAGRAM_BUSINESS_ACCOUNT_ID", "")
        )

    def _get_credentials(self) -> tuple[str, str]:
        """Return (access_token, ig_user_id)."""
        from config import settings
        return settings.INSTAGRAM_ACCESS_TOKEN, settings.INSTAGRAM_BUSINESS_ACCOUNT_ID

    async def publish(self, envelope: MessageEnvelope) -> PublishResult:
        """Publish a single image or carousel post to Instagram."""
        if not self.is_configured():
            return PublishResult(
                success=False,
                platform=self.name,
                error="Instagram credentials not configured",
            )

        access_token, ig_user_id = self._get_credentials()
        caption = envelope.truncate_text(_MAX_TEXT_LENGTH)

        # Collect all available image URLs
        image_urls: list[str] = []
        if envelope.image_urls:
            image_urls = list(envelope.image_urls)
        elif envelope.image_url:
            image_urls = [envelope.image_url]

        if not image_urls:
            return PublishResult(
                success=False,
                platform=self.name,
                error="Instagram requires at least one image (no text-only posts)",
            )

        # Validate URLs are publicly accessible (Instagram requires https URLs)
        for url in image_urls:
            if not url.startswith(("http://", "https://")):
                return PublishResult(
                    success=False,
                    platform=self.name,
                    error=f"Instagram requires publicly accessible image URLs, got: {url[:50]}",
                )

        try:
            if len(image_urls) == 1:
                return await self._publish_single_image(
                    access_token, ig_user_id, caption, image_urls[0],
                )
            else:
                return await self._publish_carousel(
                    access_token, ig_user_id, caption, image_urls[:_MAX_CAROUSEL_ITEMS],
                )
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:300] if e.response else str(e)
            logger.error("Instagram API error %d: %s", e.response.status_code, error_detail)
            return PublishResult(
                success=False,
                platform=self.name,
                error=f"Instagram API {e.response.status_code}: {error_detail}",
            )
        except Exception as e:
            logger.error("Failed to publish to Instagram: %s", e)
            return PublishResult(
                success=False,
                platform=self.name,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Single image post
    # ------------------------------------------------------------------

    async def _publish_single_image(
        self,
        access_token: str,
        ig_user_id: str,
        caption: str,
        image_url: str,
    ) -> PublishResult:
        """Publish a single image post."""
        # Step 1: Create media container
        container_id = await self._create_media_container(
            access_token, ig_user_id, image_url=image_url, caption=caption,
        )

        # Step 2: Publish the container
        media_id = await self._publish_media_container(
            access_token, ig_user_id, container_id,
        )

        post_url = f"https://www.instagram.com/p/{media_id}/" if media_id else ""
        logger.info("Published single image to Instagram: %s", media_id)

        return PublishResult(
            success=True,
            url=post_url,
            platform=self.name,
            metadata={"ig_media_id": media_id},
        )

    # ------------------------------------------------------------------
    # Carousel post
    # ------------------------------------------------------------------

    async def _publish_carousel(
        self,
        access_token: str,
        ig_user_id: str,
        caption: str,
        image_urls: list[str],
    ) -> PublishResult:
        """Publish a carousel post with multiple images (up to 10)."""
        # Step 1: Create individual item containers (can be done concurrently)
        item_tasks = [
            self._create_carousel_item(access_token, ig_user_id, url)
            for url in image_urls
        ]
        item_container_ids = await asyncio.gather(*item_tasks)

        # Step 2: Create carousel container
        carousel_container_id = await self._create_carousel_container(
            access_token, ig_user_id, caption, list(item_container_ids),
        )

        # Step 3: Publish the carousel
        media_id = await self._publish_media_container(
            access_token, ig_user_id, carousel_container_id,
        )

        post_url = f"https://www.instagram.com/p/{media_id}/" if media_id else ""
        logger.info(
            "Published carousel (%d images) to Instagram: %s",
            len(image_urls), media_id,
        )

        return PublishResult(
            success=True,
            url=post_url,
            platform=self.name,
            metadata={
                "ig_media_id": media_id,
                "carousel_items": len(image_urls),
            },
        )

    # ------------------------------------------------------------------
    # Graph API helpers
    # ------------------------------------------------------------------

    async def _create_media_container(
        self,
        access_token: str,
        ig_user_id: str,
        image_url: str,
        caption: str = "",
    ) -> str:
        """Create a single-image media container. Returns container ID."""
        params: dict = {
            "image_url": image_url,
            "access_token": access_token,
        }
        if caption:
            params["caption"] = caption

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_GRAPH_API_BASE}/{ig_user_id}/media",
                data=params,
            )
            resp.raise_for_status()
            data = resp.json()

        container_id = data.get("id", "")
        if not container_id:
            raise ValueError(f"Instagram API did not return container ID: {data}")
        return container_id

    async def _create_carousel_item(
        self,
        access_token: str,
        ig_user_id: str,
        image_url: str,
    ) -> str:
        """Create a carousel item container (no caption). Returns container ID."""
        params = {
            "image_url": image_url,
            "is_carousel_item": "true",
            "access_token": access_token,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_GRAPH_API_BASE}/{ig_user_id}/media",
                data=params,
            )
            resp.raise_for_status()
            data = resp.json()

        container_id = data.get("id", "")
        if not container_id:
            raise ValueError(f"Instagram API did not return carousel item ID: {data}")
        return container_id

    async def _create_carousel_container(
        self,
        access_token: str,
        ig_user_id: str,
        caption: str,
        children_ids: list[str],
    ) -> str:
        """Create the carousel parent container. Returns container ID."""
        params: dict = {
            "media_type": "CAROUSEL",
            "children": ",".join(children_ids),
            "access_token": access_token,
        }
        if caption:
            params["caption"] = caption

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_GRAPH_API_BASE}/{ig_user_id}/media",
                data=params,
            )
            resp.raise_for_status()
            data = resp.json()

        container_id = data.get("id", "")
        if not container_id:
            raise ValueError(f"Instagram API did not return carousel container ID: {data}")
        return container_id

    async def _publish_media_container(
        self,
        access_token: str,
        ig_user_id: str,
        container_id: str,
    ) -> str:
        """Publish a media container. Returns the published media ID."""
        params = {
            "creation_id": container_id,
            "access_token": access_token,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_GRAPH_API_BASE}/{ig_user_id}/media_publish",
                data=params,
            )
            resp.raise_for_status()
            data = resp.json()

        media_id = data.get("id", "")
        if not media_id:
            raise ValueError(f"Instagram API did not return media ID: {data}")
        return media_id
