"""
Publish posts to X (Twitter) using Tweepy v4+.
"""

import io
import logging
from pathlib import Path

import httpx
import tweepy

from config import settings

logger = logging.getLogger(__name__)


async def _download_image(url_or_path: str) -> bytes:
    """Download an image from a URL or read from a local file path."""
    if url_or_path.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url_or_path)
            resp.raise_for_status()
            return resp.content
    # Local file path
    return Path(url_or_path).read_bytes()


def _get_api_v1() -> tweepy.API:
    """Create Tweepy v1.1 API client for media upload."""
    auth = tweepy.OAuth1UserHandler(
        settings.X_API_KEY,
        settings.X_API_SECRET,
        settings.X_ACCESS_TOKEN,
        settings.X_ACCESS_SECRET,
    )
    return tweepy.API(auth)


def _get_client_v2() -> tweepy.Client:
    """Create Tweepy v2 Client for posting tweets."""
    return tweepy.Client(
        bearer_token=settings.X_BEARER_TOKEN,
        consumer_key=settings.X_API_KEY,
        consumer_secret=settings.X_API_SECRET,
        access_token=settings.X_ACCESS_TOKEN,
        access_token_secret=settings.X_ACCESS_SECRET,
    )


async def post_to_x(
    caption: str, hashtags: list[str], image_url: str | None
) -> str:
    """
    Post a tweet to X with an optional image.

    Args:
        caption: Tweet text.
        hashtags: List of hashtags to append.
        image_url: URL of the image to attach, or None for text-only.

    Returns:
        URL of the published tweet.

    Raises:
        tweepy.TweepyException: On API errors.
    """
    hashtag_str = " ".join(hashtags)
    full_text = f"{caption}\n\n{hashtag_str}".strip()

    # Truncate to 280 chars if needed
    if len(full_text) > 280:
        available = 280 - len(hashtag_str) - 3  # 3 for "\n\n" + "..."
        full_text = f"{caption[:available]}...\n\n{hashtag_str}"

    media_id = None
    if image_url:
        try:
            logger.info("Downloading image for X upload: %s", image_url[:100])
            image_bytes = await _download_image(image_url)
            api_v1 = _get_api_v1()
            media = api_v1.media_upload(
                filename="brandmover_post.webp",
                file=io.BytesIO(image_bytes),
            )
            media_id = media.media_id
            logger.info("Media uploaded to X: media_id=%s", media_id)
        except Exception as e:
            logger.error("Failed to upload image to X: %s — posting text only", e)

    client_v2 = _get_client_v2()
    kwargs = {"text": full_text}
    if media_id:
        kwargs["media_ids"] = [media_id]

    response = client_v2.create_tweet(**kwargs)
    tweet_id = response.data["id"]
    # Resolve username for URL
    me = client_v2.get_me()
    username = me.data.username
    tweet_url = f"https://x.com/{username}/status/{tweet_id}"

    logger.info("Tweet posted: %s", tweet_url)
    return tweet_url
