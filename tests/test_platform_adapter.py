"""Tests for agent.platform_adapter — draft transformation for multi-platform publishing."""

from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

import pytest

from agent.platform_adapter import (
    PlatformPost,
    adapt_for_platform,
    adapt_for_all_platforms,
    PLATFORM_CONFIGS,
)


# ---------------------------------------------------------------------------
# adapt_for_platform — X
# ---------------------------------------------------------------------------


class TestAdaptX:
    def test_x_truncates_to_280(self):
        """Long text + hashtags should be truncated to 280 chars."""
        draft = {
            "caption": "A" * 300,
            "hashtags": ["#test"],
        }
        post = adapt_for_platform(draft, "x")
        assert len(post.text) <= 280
        assert post.text.endswith("...")
        assert post.platform == "x"

    def test_x_short_text_not_truncated(self):
        draft = {"caption": "Hello world", "hashtags": ["#brand"]}
        post = adapt_for_platform(draft, "x")
        assert "Hello world" in post.text
        assert "#brand" in post.text
        assert len(post.text) <= 280

    def test_x_no_hashtags(self):
        draft = {"caption": "Just text", "hashtags": []}
        post = adapt_for_platform(draft, "x")
        assert post.text == "Just text"

    def test_x_image_url_passthrough(self):
        draft = {"caption": "With image", "hashtags": []}
        post = adapt_for_platform(draft, "x", image_url="https://example.com/img.png")
        assert post.image_url == "https://example.com/img.png"


# ---------------------------------------------------------------------------
# adapt_for_platform — Discord
# ---------------------------------------------------------------------------


class TestAdaptDiscord:
    def test_discord_keeps_longer_text(self):
        """Discord has a 2000-char limit, so medium text should not be truncated."""
        draft = {"caption": "B" * 500, "hashtags": ["#test"]}
        post = adapt_for_platform(draft, "discord")
        assert len(post.text) == 500  # hashtags in footer, not in text
        assert post.platform == "discord"

    def test_discord_hashtags_in_metadata(self):
        draft = {"caption": "Hello", "hashtags": ["#crypto", "#defi"]}
        post = adapt_for_platform(draft, "discord")
        assert "#crypto #defi" == post.metadata["footer_hashtags"]
        assert "#crypto" not in post.text

    def test_discord_truncates_at_2000(self):
        draft = {"caption": "C" * 2500, "hashtags": []}
        post = adapt_for_platform(draft, "discord")
        assert len(post.text) <= 2000
        assert post.text.endswith("...")


# ---------------------------------------------------------------------------
# adapt_for_platform — Telegram
# ---------------------------------------------------------------------------


class TestAdaptTelegram:
    def test_telegram_keeps_full_length(self):
        draft = {"caption": "D" * 3000, "hashtags": ["#tg"]}
        post = adapt_for_platform(draft, "telegram")
        # 3000 chars + newlines + hashtag = still under 4096
        assert len(post.text) <= 4096
        assert "#tg" in post.text

    def test_telegram_truncates_at_4096(self):
        draft = {"caption": "E" * 5000, "hashtags": []}
        post = adapt_for_platform(draft, "telegram")
        assert len(post.text) <= 4096
        assert post.text.endswith("...")


# ---------------------------------------------------------------------------
# adapt_for_all_platforms
# ---------------------------------------------------------------------------


class TestAdaptAllPlatforms:
    def test_returns_all_platforms(self):
        draft = {"caption": "Multi-post", "hashtags": ["#multi"]}
        results = adapt_for_all_platforms(draft)
        assert "x" in results
        assert "discord" in results
        assert "telegram" in results
        assert all(isinstance(v, PlatformPost) for v in results.values())

    def test_specific_platforms(self):
        draft = {"caption": "Selective", "hashtags": []}
        results = adapt_for_all_platforms(draft, platforms=["x", "discord"])
        assert "x" in results
        assert "discord" in results
        assert "telegram" not in results

    def test_unsupported_platform_skipped(self):
        draft = {"caption": "Test", "hashtags": []}
        results = adapt_for_all_platforms(draft, platforms=["x", "mastodon"])
        assert "x" in results
        assert "mastodon" not in results

    def test_image_url_passed_to_all(self):
        draft = {"caption": "With image", "hashtags": []}
        results = adapt_for_all_platforms(
            draft, image_url="https://example.com/img.png", platforms=["x", "discord"],
        )
        for post in results.values():
            assert post.image_url == "https://example.com/img.png"


# ---------------------------------------------------------------------------
# publish_to_all — X failure doesn't block Discord
# ---------------------------------------------------------------------------


class TestPublishToAll:
    @pytest.mark.asyncio
    async def test_x_failure_doesnt_block_discord(self):
        """When X publishing fails, Discord should still succeed."""
        from agent import publisher

        mock_discord_url = "https://discord.com/channels/123/456/789"

        async def _mock_post_to_x(caption, hashtags, image_url, _from_retry=False):
            raise Exception("X API rate limit")

        async def _mock_post_to_discord(**kwargs):
            return mock_discord_url

        with patch.object(publisher, "post_to_x", side_effect=_mock_post_to_x), \
             patch("agent.platform_adapter.adapt_for_all_platforms") as mock_adapt, \
             patch("agent.discord_bot.is_ready", return_value=True), \
             patch("agent.discord_publisher.post_to_discord", side_effect=_mock_post_to_discord), \
             patch("config.settings.PUBLISH_PLATFORMS", ["x", "discord"]):

            # Make the adapter return proper PlatformPost objects
            mock_adapt.return_value = {
                "x": PlatformPost(platform="x", text="Test"),
                "discord": PlatformPost(platform="discord", text="Test"),
            }

            results = await publisher.publish_to_all(
                draft={"caption": "Test post", "hashtags": ["#test"]},
                image_url=None,
            )

            # X should have failed (None), Discord should have succeeded
            assert results.get("x") is None
            assert results.get("discord") == mock_discord_url

    @pytest.mark.asyncio
    async def test_publish_to_all_returns_urls(self):
        """Successful publishes return their URLs."""
        from agent import publisher

        mock_tweet_url = "https://x.com/brand/status/123"

        async def _mock_post_to_x(caption, hashtags, image_url, _from_retry=False):
            return mock_tweet_url

        with patch.object(publisher, "post_to_x", side_effect=_mock_post_to_x), \
             patch("agent.platform_adapter.adapt_for_all_platforms") as mock_adapt, \
             patch("config.settings.PUBLISH_PLATFORMS", ["x"]):

            mock_adapt.return_value = {
                "x": PlatformPost(platform="x", text="Test"),
            }

            results = await publisher.publish_to_all(
                draft={"caption": "Test", "hashtags": []},
            )

            assert results.get("x") == mock_tweet_url
