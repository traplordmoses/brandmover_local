"""
Publish posts to X (Twitter) using Tweepy v4+.
"""

import asyncio
import io
import logging
from pathlib import Path

import httpx
import tweepy

from config import settings

from agent import publish_queue

logger = logging.getLogger(__name__)


MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_IMAGES_PER_TWEET = 4  # X supports up to 4 images per tweet


async def _download_image(url_or_path: str) -> bytes:
    """Download an image from a URL or read from a local file path."""
    if url_or_path.startswith(("http://", "https://")):
        from agent._client import get_httpx
        resp = await get_httpx().get(url_or_path)
        resp.raise_for_status()
        data = resp.content
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image too large: {len(data)} bytes (max {MAX_IMAGE_BYTES})")
        return data
    # Local file path
    p = Path(url_or_path)
    if not p.exists():
        raise FileNotFoundError(f"Local image not found: {url_or_path}")
    data = p.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large: {len(data)} bytes (max {MAX_IMAGE_BYTES})")
    return data


# Lazy-initialized singleton Tweepy clients
_api_v1: tweepy.API | None = None
_client_v2: tweepy.Client | None = None


def _get_api_v1() -> tweepy.API:
    """Return singleton Tweepy v1.1 API client for media upload."""
    global _api_v1
    if _api_v1 is None:
        auth = tweepy.OAuth1UserHandler(
            settings.X_API_KEY,
            settings.X_API_SECRET,
            settings.X_ACCESS_TOKEN,
            settings.X_ACCESS_SECRET,
        )
        _api_v1 = tweepy.API(auth)
    return _api_v1


def _get_client_v2() -> tweepy.Client:
    """Return singleton Tweepy v2 Client for posting tweets."""
    global _client_v2
    if _client_v2 is None:
        _client_v2 = tweepy.Client(
            bearer_token=settings.X_BEARER_TOKEN,
            consumer_key=settings.X_API_KEY,
            consumer_secret=settings.X_API_SECRET,
            access_token=settings.X_ACCESS_TOKEN,
            access_token_secret=settings.X_ACCESS_SECRET,
        )
    return _client_v2


async def _upload_single_image(api_v1: tweepy.API, url_or_path: str, filename: str = "brandmover_post.webp") -> int | None:
    """Download and upload a single image to X. Returns media_id or None on failure."""
    try:
        image_bytes = await _download_image(url_or_path)
        media = await asyncio.to_thread(
            api_v1.media_upload,
            filename=filename,
            file=io.BytesIO(image_bytes),
        )
        logger.info("Media uploaded to X: media_id=%s", media.media_id)
        return media.media_id
    except (tweepy.TweepyException, httpx.HTTPError, OSError) as e:
        logger.error("Failed to upload image to X (%s): %s", url_or_path[:80], e)
        return None


async def post_to_x(
    caption: str, hashtags: list[str], image_url: str | None,
    image_urls: list[str] | None = None,
    _from_retry: bool = False,
) -> str:
    """
    Post a tweet to X with optional images (up to 4).

    Args:
        caption: Tweet text.
        hashtags: List of hashtags to append.
        image_url: URL of a single image to attach, or None for text-only.
        image_urls: List of image URLs to attach (up to 4). Takes precedence
                    over image_url when provided.
        _from_retry: Internal flag — True when called from retry_pending()
                     to prevent re-enqueuing on failure.

    Returns:
        URL of the published tweet.

    Raises:
        tweepy.TweepyException: On API errors.
    """
    hashtag_str = " ".join(hashtags)
    full_text = f"{caption}\n\n{hashtag_str}".strip()

    # Truncate to 280 chars if needed
    if len(full_text) > 280:
        available = 280 - len(hashtag_str) - 5  # 2 for "\n\n" + 3 for "..."
        full_text = f"{caption[:available]}...\n\n{hashtag_str}"

    # Consolidate image sources: image_urls takes precedence over image_url
    all_image_urls: list[str] = []
    if image_urls:
        all_image_urls = list(image_urls[:MAX_IMAGES_PER_TWEET])
    elif image_url:
        all_image_urls = [image_url]

    # Upload all images
    media_ids: list[int] = []
    if all_image_urls:
        api_v1 = _get_api_v1()
        for idx, url in enumerate(all_image_urls):
            logger.info("Downloading image %d/%d for X upload: %s", idx + 1, len(all_image_urls), url[:100])
            mid = await _upload_single_image(api_v1, url, filename=f"brandmover_post_{idx}.webp")
            if mid is not None:
                media_ids.append(mid)

    try:
        client_v2 = _get_client_v2()
        kwargs = {"text": full_text}
        if media_ids:
            kwargs["media_ids"] = media_ids

        response = await asyncio.to_thread(client_v2.create_tweet, **kwargs)
        tweet_id = response.data["id"]
        # Resolve username — cached to avoid repeated API calls
        username = await _get_cached_username(client_v2)
        tweet_url = f"https://x.com/{username}/status/{tweet_id}"

        logger.info("Tweet posted: %s", tweet_url)
        return tweet_url
    except Exception as e:  # Top-level catch — enqueues failed post for retry, then re-raises
        if not _from_retry:
            publish_queue.enqueue_failed(
                caption=full_text,
                image_path=image_url or (all_image_urls[0] if all_image_urls else None),
                content_type="tweet",
                error=str(e),
            )
        raise


# Cached username — doesn't change mid-session
_cached_username: str | None = None


async def post_thread_to_x(
    posts: list[dict],
) -> list[str]:
    """Post a thread (reply chain) to X.

    Args:
        posts: List of dicts, each with 'text' (required), optional 'image_url'
               (single image), and optional 'image_urls' (list of up to 4 images).
               'image_urls' takes precedence over 'image_url' when both are present.
               First post starts the thread, subsequent posts reply to the previous.

    Returns:
        List of tweet URLs in order.
    """
    if not posts:
        raise ValueError("Thread must have at least one post.")

    client_v2 = _get_client_v2()
    api_v1 = _get_api_v1()
    username = await _get_cached_username(client_v2)

    tweet_urls = []
    previous_tweet_id = None

    for i, post in enumerate(posts):
        text = post.get("text", "").strip()
        if not text:
            continue

        # Truncate to 280 chars
        if len(text) > 280:
            text = text[:277] + "..."

        # Consolidate image sources for this thread post
        post_image_urls: list[str] = []
        if post.get("image_urls"):
            post_image_urls = list(post["image_urls"][:MAX_IMAGES_PER_TWEET])
        elif post.get("image_url"):
            post_image_urls = [post["image_url"]]

        # Upload all images for this thread post
        media_ids: list[int] = []
        for j, img_url in enumerate(post_image_urls):
            mid = await _upload_single_image(api_v1, img_url, filename=f"thread_{i}_{j}.webp")
            if mid is not None:
                media_ids.append(mid)

        kwargs = {"text": text}
        if media_ids:
            kwargs["media_ids"] = media_ids
        if previous_tweet_id:
            kwargs["in_reply_to_tweet_id"] = previous_tweet_id

        response = await asyncio.to_thread(client_v2.create_tweet, **kwargs)
        tweet_id = response.data["id"]
        previous_tweet_id = tweet_id
        tweet_url = f"https://x.com/{username}/status/{tweet_id}"
        tweet_urls.append(tweet_url)

        logger.info("Thread post %d/%d posted: %s", i + 1, len(posts), tweet_url)

    return tweet_urls


async def publish_to_all(
    draft: dict,
    image_url: str | None = None,
    composed_path: str | None = None,
    platforms: list[str] | None = None,
) -> dict[str, str | None]:
    """Publish to all configured platforms. Returns {platform: url_or_none}.

    Uses platform_adapter to format the draft for each platform, then calls
    the appropriate publisher. Runs concurrently with asyncio.gather so a
    failure on one platform does not block the others.

    Args:
        draft: Standard draft dict with 'caption', 'hashtags', etc.
        image_url: Raw image URL (Replicate output or local path).
        composed_path: Path to the composed image (preferred over image_url).
        platforms: List of platforms to publish to. If None, uses settings.PUBLISH_PLATFORMS.

    Returns:
        Dict mapping platform name to the published URL (or None on failure).
    """
    from pathlib import Path as _Path
    from agent import platform_adapter
    from config import settings as _settings

    if platforms is None:
        platforms = _settings.PUBLISH_PLATFORMS

    # Determine the best image to use
    publish_image = image_url
    if composed_path and _Path(composed_path).exists():
        publish_image = composed_path

    # Adapt draft for each platform
    adapted = platform_adapter.adapt_for_all_platforms(
        draft, image_url=publish_image, platforms=platforms,
    )

    results: dict[str, str | None] = {}

    async def _publish_x(post: "platform_adapter.PlatformPost") -> tuple[str, str | None]:
        try:
            url = await post_to_x(
                caption=draft.get("caption", ""),
                hashtags=draft.get("hashtags", []),
                image_url=publish_image,
            )
            return "x", url
        except (tweepy.TweepyException, httpx.HTTPError, OSError) as e:
            logger.error("publish_to_all: X failed: %s", e)
            return "x", None

    async def _publish_discord(post: "platform_adapter.PlatformPost") -> tuple[str, str | None]:
        try:
            from agent import discord_bot, discord_publisher
            if not discord_bot.is_ready():
                logger.debug("publish_to_all: Discord not ready, skipping")
                return "discord", None
            url = await discord_publisher.post_to_discord(
                caption=post.text,
                hashtags=draft.get("hashtags", []),
                image_url=publish_image,
                auto_slot=draft.get("auto_slot"),
                content_type=draft.get("content_type"),
            )
            return "discord", url
        except Exception as e:  # Intentional broad catch — Discord module can raise varied errors
            logger.error("publish_to_all: Discord failed: %s", e)
            return "discord", None

    async def _publish_telegram(post: "platform_adapter.PlatformPost") -> tuple[str, str | None]:
        # Telegram publishing is handled by the bot itself (send_auto_draft).
        # This is a placeholder for future standalone Telegram channel posting.
        logger.debug("publish_to_all: Telegram channel posting not yet implemented")
        return "telegram", None

    async def _publish_linkedin(post: "platform_adapter.PlatformPost") -> tuple[str, str | None]:
        try:
            from agent.publishing.channels.registry import get_channel
            from agent.publishing.channels.base import MessageEnvelope
            channel = get_channel("linkedin")
            if channel is None or not channel.is_configured():
                logger.debug("publish_to_all: LinkedIn not configured, skipping")
                return "linkedin", None
            envelope = MessageEnvelope(
                text=post.text,
                image_url=publish_image,
                hashtags=draft.get("hashtags", []),
                content_type=draft.get("content_type", ""),
            )
            result = await channel.publish(envelope)
            return "linkedin", result.url if result.success else None
        except Exception as e:
            logger.error("publish_to_all: LinkedIn failed: %s", e)
            return "linkedin", None

    async def _publish_instagram(post: "platform_adapter.PlatformPost") -> tuple[str, str | None]:
        try:
            from agent.publishing.channels.registry import get_channel
            from agent.publishing.channels.base import MessageEnvelope
            channel = get_channel("instagram")
            if channel is None or not channel.is_configured():
                logger.debug("publish_to_all: Instagram not configured, skipping")
                return "instagram", None
            envelope = MessageEnvelope(
                text=post.text,
                image_url=publish_image,
                hashtags=draft.get("hashtags", []),
                content_type=draft.get("content_type", ""),
            )
            result = await channel.publish(envelope)
            return "instagram", result.url if result.success else None
        except Exception as e:
            logger.error("publish_to_all: Instagram failed: %s", e)
            return "instagram", None

    _PUBLISHER_MAP = {
        "x": _publish_x,
        "discord": _publish_discord,
        "telegram": _publish_telegram,
        "linkedin": _publish_linkedin,
        "instagram": _publish_instagram,
    }

    # Build tasks for all requested platforms
    tasks = []
    for platform_name, post in adapted.items():
        publisher_fn = _PUBLISHER_MAP.get(platform_name)
        if publisher_fn:
            tasks.append(publisher_fn(post))
        else:
            logger.warning("publish_to_all: No publisher for platform %s", platform_name)
            results[platform_name] = None

    # Run all publishers concurrently
    if tasks:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                logger.error("publish_to_all: Unexpected error: %s", outcome)
            elif isinstance(outcome, tuple):
                platform_name, url = outcome
                results[platform_name] = url

    return results


async def _get_cached_username(client_v2: tweepy.Client) -> str:
    """Get the authenticated user's username, cached after first call."""
    global _cached_username
    if _cached_username is None:
        me = await asyncio.to_thread(client_v2.get_me)
        _cached_username = me.data.username
    return _cached_username
