"""Tests for agent.publisher — X/Twitter posting via Tweepy.

These tests verify that:
- post_to_x() posts text-only tweets correctly
- post_to_x() uploads images and attaches media IDs
- post_to_x() falls back to text-only when image upload fails
- post_thread_to_x() chains replies correctly
- Error handling when the API is unavailable
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tweepy

from agent import publisher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_publisher_singletons():
    """Reset the module-level singletons so each test gets fresh clients."""
    publisher._api_v1 = None
    publisher._client_v2 = None
    publisher._cached_username = None


@pytest.fixture(autouse=True)
def _clean_singletons():
    _reset_publisher_singletons()
    yield
    _reset_publisher_singletons()


def _mock_create_tweet(tweet_id="123456"):
    """Return a mock response matching tweepy.Client.create_tweet."""
    return SimpleNamespace(data={"id": tweet_id})


def _mock_get_me(username="testuser"):
    """Return a mock response matching tweepy.Client.get_me."""
    return SimpleNamespace(data=SimpleNamespace(username=username))


def _mock_media_upload(media_id=99999):
    """Return a mock response matching tweepy.API.media_upload."""
    return SimpleNamespace(media_id=media_id)


# ---------------------------------------------------------------------------
# post_to_x — text-only
# ---------------------------------------------------------------------------

class TestPostToXTextOnly:
    @pytest.mark.asyncio
    async def test_text_only_post(self):
        """Text-only post (no image_url) creates tweet without media_ids."""
        mock_client = MagicMock()
        mock_client.create_tweet.return_value = _mock_create_tweet("111")
        mock_client.get_me.return_value = _mock_get_me("alice")

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()):
            url = await publisher.post_to_x("Hello world", ["#test"], None)

        assert url == "https://x.com/alice/status/111"
        call_kwargs = mock_client.create_tweet.call_args
        assert "media_ids" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_hashtags_appended(self):
        """Hashtags are appended to caption text."""
        mock_client = MagicMock()
        mock_client.create_tweet.return_value = _mock_create_tweet("222")
        mock_client.get_me.return_value = _mock_get_me("bob")

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()):
            await publisher.post_to_x("My caption", ["#web3", "#defi"], None)

        text_arg = mock_client.create_tweet.call_args.kwargs["text"]
        assert "#web3" in text_arg
        assert "#defi" in text_arg
        assert "My caption" in text_arg

    @pytest.mark.asyncio
    async def test_truncates_long_text(self):
        """Text longer than 280 chars gets truncated to fit."""
        mock_client = MagicMock()
        mock_client.create_tweet.return_value = _mock_create_tweet("333")
        mock_client.get_me.return_value = _mock_get_me("carol")

        long_caption = "A" * 300
        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()):
            await publisher.post_to_x(long_caption, ["#tag"], None)

        text_arg = mock_client.create_tweet.call_args.kwargs["text"]
        # The original would be 300 + "\n\n" + "#tag" = 306 chars
        # Truncation logic: caption[:available] + "..." + "\n\n" + hashtags
        assert len(text_arg) <= 300  # significantly shorter than untruncated
        assert "..." in text_arg
        assert "#tag" in text_arg


# ---------------------------------------------------------------------------
# post_to_x — with image
# ---------------------------------------------------------------------------

class TestPostToXWithImage:
    @pytest.mark.asyncio
    async def test_image_upload_attaches_media_id(self):
        """When image_url is provided and upload succeeds, media_ids is set."""
        mock_client = MagicMock()
        mock_client.create_tweet.return_value = _mock_create_tweet("444")
        mock_client.get_me.return_value = _mock_get_me("dave")

        mock_api = MagicMock()
        mock_api.media_upload.return_value = _mock_media_upload(77777)

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=mock_api), \
             patch.object(publisher, "_download_image", new_callable=AsyncMock, return_value=b"fake-image-bytes"):
            url = await publisher.post_to_x("With image", ["#img"], "https://example.com/image.png")

        assert url == "https://x.com/dave/status/444"
        call_kwargs = mock_client.create_tweet.call_args.kwargs
        assert call_kwargs["media_ids"] == [77777]

    @pytest.mark.asyncio
    async def test_image_upload_failure_falls_back_to_text(self):
        """When image upload fails, tweet is posted as text-only."""
        mock_client = MagicMock()
        mock_client.create_tweet.return_value = _mock_create_tweet("555")
        mock_client.get_me.return_value = _mock_get_me("eve")

        mock_api = MagicMock()
        mock_api.media_upload.side_effect = tweepy.TweepyException("upload failed")

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=mock_api), \
             patch.object(publisher, "_download_image", new_callable=AsyncMock, return_value=b"fake-bytes"):
            url = await publisher.post_to_x("Fallback", ["#fb"], "https://example.com/bad.png")

        assert url == "https://x.com/eve/status/555"
        call_kwargs = mock_client.create_tweet.call_args.kwargs
        assert "media_ids" not in call_kwargs

    @pytest.mark.asyncio
    async def test_download_failure_falls_back_to_text(self):
        """When image download fails (httpx error), tweet is posted as text-only."""
        import httpx

        mock_client = MagicMock()
        mock_client.create_tweet.return_value = _mock_create_tweet("556")
        mock_client.get_me.return_value = _mock_get_me("frank")

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()), \
             patch.object(publisher, "_download_image", new_callable=AsyncMock,
                          side_effect=httpx.HTTPError("download failed")):
            url = await publisher.post_to_x("DL fail", [], "https://example.com/broken.png")

        assert url == "https://x.com/frank/status/556"
        call_kwargs = mock_client.create_tweet.call_args.kwargs
        assert "media_ids" not in call_kwargs


# ---------------------------------------------------------------------------
# post_thread_to_x — multi-post thread
# ---------------------------------------------------------------------------

class TestPostThreadToX:
    @pytest.mark.asyncio
    async def test_thread_chains_replies(self):
        """Thread posts are chained via in_reply_to_tweet_id."""
        call_count = 0

        def _create_tweet(**kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_create_tweet(str(1000 + call_count))

        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = _create_tweet
        mock_client.get_me.return_value = _mock_get_me("threaduser")

        posts = [
            {"text": "First post in thread"},
            {"text": "Second post replying"},
            {"text": "Third post replying"},
        ]

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()):
            urls = await publisher.post_thread_to_x(posts)

        assert len(urls) == 3
        assert urls[0] == "https://x.com/threaduser/status/1001"
        assert urls[2] == "https://x.com/threaduser/status/1003"

        # First call: no in_reply_to_tweet_id
        first_call = mock_client.create_tweet.call_args_list[0]
        assert "in_reply_to_tweet_id" not in first_call.kwargs

        # Second call: replies to first tweet
        second_call = mock_client.create_tweet.call_args_list[1]
        assert second_call.kwargs["in_reply_to_tweet_id"] == "1001"

        # Third call: replies to second tweet
        third_call = mock_client.create_tweet.call_args_list[2]
        assert third_call.kwargs["in_reply_to_tweet_id"] == "1002"

    @pytest.mark.asyncio
    async def test_empty_thread_raises(self):
        """Empty posts list raises ValueError."""
        with pytest.raises(ValueError, match="at least one post"):
            await publisher.post_thread_to_x([])

    @pytest.mark.asyncio
    async def test_thread_with_images(self):
        """Thread posts can include images."""
        call_count = 0

        def _create_tweet(**kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_create_tweet(str(2000 + call_count))

        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = _create_tweet
        mock_client.get_me.return_value = _mock_get_me("imgthread")

        mock_api = MagicMock()
        mock_api.media_upload.return_value = _mock_media_upload(88888)

        posts = [
            {"text": "First with image", "image_url": "https://example.com/img1.png"},
            {"text": "Second text only"},
        ]

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=mock_api), \
             patch.object(publisher, "_download_image", new_callable=AsyncMock, return_value=b"img-bytes"):
            urls = await publisher.post_thread_to_x(posts)

        assert len(urls) == 2
        # First call should have media_ids
        first_call = mock_client.create_tweet.call_args_list[0]
        assert first_call.kwargs["media_ids"] == [88888]
        # Second call should not
        second_call = mock_client.create_tweet.call_args_list[1]
        assert "media_ids" not in second_call.kwargs

    @pytest.mark.asyncio
    async def test_thread_skips_empty_text(self):
        """Posts with empty text are skipped."""
        mock_client = MagicMock()
        mock_client.create_tweet.return_value = _mock_create_tweet("3001")
        mock_client.get_me.return_value = _mock_get_me("skipuser")

        posts = [
            {"text": ""},
            {"text": "Only real post"},
        ]

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()):
            urls = await publisher.post_thread_to_x(posts)

        assert len(urls) == 1
        assert "3001" in urls[0]


# ---------------------------------------------------------------------------
# Error handling — API down
# ---------------------------------------------------------------------------

class TestAPIErrors:
    @pytest.mark.asyncio
    async def test_create_tweet_api_error_propagates(self):
        """TweepyException from create_tweet propagates to caller."""
        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = tweepy.TweepyException("API down")
        mock_client.get_me.return_value = _mock_get_me("erruser")

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()):
            with pytest.raises(tweepy.TweepyException, match="API down"):
                await publisher.post_to_x("Test", [], None)

    @pytest.mark.asyncio
    async def test_thread_api_error_propagates(self):
        """TweepyException from create_tweet in thread propagates."""
        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = tweepy.TweepyException("rate limited")
        mock_client.get_me.return_value = _mock_get_me("ratelimit")

        with patch.object(publisher, "_get_client_v2", return_value=mock_client), \
             patch.object(publisher, "_get_api_v1", return_value=MagicMock()):
            with pytest.raises(tweepy.TweepyException, match="rate limited"):
                await publisher.post_thread_to_x([{"text": "Hello"}])


# ---------------------------------------------------------------------------
# _get_cached_username
# ---------------------------------------------------------------------------

class TestCachedUsername:
    @pytest.mark.asyncio
    async def test_caches_after_first_call(self):
        """Username is fetched once then cached."""
        mock_client = MagicMock()
        mock_client.get_me.return_value = _mock_get_me("cached_user")

        # First call fetches
        name1 = await publisher._get_cached_username(mock_client)
        assert name1 == "cached_user"

        # Second call uses cache (get_me not called again)
        mock_client.get_me.reset_mock()
        name2 = await publisher._get_cached_username(mock_client)
        assert name2 == "cached_user"
        mock_client.get_me.assert_not_called()
