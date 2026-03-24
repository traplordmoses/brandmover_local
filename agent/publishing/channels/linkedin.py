"""
LinkedIn channel adapter — publishes content via the LinkedIn Share API (UGC Posts).

Uses LinkedIn's REST API v2 to create text and image posts on behalf of an
authenticated user or organization. Requires:
    - LINKEDIN_ACCESS_TOKEN: OAuth 2.0 access token with w_member_social scope
    - LINKEDIN_AUTHOR_URN: URN of the posting entity (e.g., "urn:li:person:xxx"
      or "urn:li:organization:xxx")

Image flow:
    1. Register an upload with LinkedIn's asset API
    2. PUT the binary image to the upload URL
    3. Reference the asset URN in the UGC post
"""

import io
import logging
from pathlib import Path

import httpx

from agent.publishing.channels.base import Channel, MessageEnvelope, PublishResult

logger = logging.getLogger(__name__)

_LINKEDIN_API_BASE = "https://api.linkedin.com/v2"
_MAX_TEXT_LENGTH = 3000


class LinkedInChannel(Channel):
    """Publishes content to LinkedIn via the UGC Posts API."""

    @property
    def name(self) -> str:
        return "linkedin"

    @property
    def max_text_length(self) -> int:
        return _MAX_TEXT_LENGTH

    def is_configured(self) -> bool:
        from config import settings
        return bool(
            getattr(settings, "LINKEDIN_ACCESS_TOKEN", "")
            and getattr(settings, "LINKEDIN_AUTHOR_URN", "")
        )

    def _get_credentials(self) -> tuple[str, str]:
        """Return (access_token, author_urn)."""
        from config import settings
        return settings.LINKEDIN_ACCESS_TOKEN, settings.LINKEDIN_AUTHOR_URN

    def _headers(self, access_token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }

    async def publish(self, envelope: MessageEnvelope) -> PublishResult:
        """Publish a text or image post to LinkedIn."""
        if not self.is_configured():
            return PublishResult(
                success=False,
                platform=self.name,
                error="LinkedIn credentials not configured",
            )

        access_token, author_urn = self._get_credentials()
        text = envelope.truncate_text(_MAX_TEXT_LENGTH)

        try:
            # Determine the best image to use
            image_url = envelope.image_url
            if not image_url and envelope.image_urls:
                image_url = envelope.image_urls[0]

            if image_url:
                return await self._publish_image_post(
                    access_token, author_urn, text, image_url,
                )
            else:
                return await self._publish_text_post(
                    access_token, author_urn, text,
                )
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:300] if e.response else str(e)
            logger.error("LinkedIn API error %d: %s", e.response.status_code, error_detail)
            return PublishResult(
                success=False,
                platform=self.name,
                error=f"LinkedIn API {e.response.status_code}: {error_detail}",
            )
        except Exception as e:
            logger.error("Failed to publish to LinkedIn: %s", e)
            return PublishResult(
                success=False,
                platform=self.name,
                error=str(e),
            )

    async def _publish_text_post(
        self, access_token: str, author_urn: str, text: str,
    ) -> PublishResult:
        """Create a text-only UGC post."""
        payload = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                },
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_LINKEDIN_API_BASE}/ugcPosts",
                json=payload,
                headers=self._headers(access_token),
            )
            resp.raise_for_status()
            data = resp.json()

        post_id = data.get("id", "")
        post_url = f"https://www.linkedin.com/feed/update/{post_id}" if post_id else ""
        logger.info("Published text post to LinkedIn: %s", post_url)

        return PublishResult(
            success=True,
            url=post_url,
            platform=self.name,
            metadata={"linkedin_post_id": post_id},
        )

    async def _publish_image_post(
        self,
        access_token: str,
        author_urn: str,
        text: str,
        image_url: str,
    ) -> PublishResult:
        """Upload an image to LinkedIn and create a post with it."""
        # Step 1: Register the upload
        asset_urn, upload_url = await self._register_image_upload(
            access_token, author_urn,
        )

        # Step 2: Download the image
        image_bytes = await self._download_image(image_url)

        # Step 3: Upload the image binary to LinkedIn
        await self._upload_image_binary(access_token, upload_url, image_bytes)

        # Step 4: Create the post with the uploaded image
        payload = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "IMAGE",
                    "media": [
                        {
                            "status": "READY",
                            "media": asset_urn,
                        }
                    ],
                },
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_LINKEDIN_API_BASE}/ugcPosts",
                json=payload,
                headers=self._headers(access_token),
            )
            resp.raise_for_status()
            data = resp.json()

        post_id = data.get("id", "")
        post_url = f"https://www.linkedin.com/feed/update/{post_id}" if post_id else ""
        logger.info("Published image post to LinkedIn: %s", post_url)

        return PublishResult(
            success=True,
            url=post_url,
            platform=self.name,
            metadata={"linkedin_post_id": post_id, "asset_urn": asset_urn},
        )

    async def _register_image_upload(
        self, access_token: str, author_urn: str,
    ) -> tuple[str, str]:
        """Register an image upload with LinkedIn's asset API.

        Returns (asset_urn, upload_url).
        """
        payload = {
            "registerUploadRequest": {
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                "owner": author_urn,
                "serviceRelationships": [
                    {
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent",
                    }
                ],
            }
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_LINKEDIN_API_BASE}/assets?action=registerUpload",
                json=payload,
                headers=self._headers(access_token),
            )
            resp.raise_for_status()
            data = resp.json()

        value = data["value"]
        asset_urn = value["asset"]
        upload_url = value["uploadMechanism"][
            "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
        ]["uploadUrl"]

        return asset_urn, upload_url

    async def _upload_image_binary(
        self, access_token: str, upload_url: str, image_bytes: bytes,
    ) -> None:
        """PUT image binary to LinkedIn's upload URL."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.put(
                upload_url,
                content=image_bytes,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/octet-stream",
                },
            )
            resp.raise_for_status()

    async def _download_image(self, url_or_path: str) -> bytes:
        """Download image from URL or read from local path."""
        if url_or_path.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url_or_path)
                resp.raise_for_status()
                return resp.content

        p = Path(url_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Local image not found: {url_or_path}")
        return p.read_bytes()
