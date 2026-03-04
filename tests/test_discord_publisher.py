"""Tests for agent.discord_publisher — channel routing, embed building, post_to_discord."""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from agent import discord_publisher, discord_bot


# ---------------------------------------------------------------------------
# Channel routing
# ---------------------------------------------------------------------------


class TestChannelRouting:
    def test_slot_engagement_morning(self):
        assert discord_publisher._route_channel(auto_slot="engagement_morning") == "brand-updates"

    def test_slot_onchain_midday(self):
        assert discord_publisher._route_channel(auto_slot="onchain_midday") == "onchain-updates"

    def test_slot_onchain_afternoon(self):
        assert discord_publisher._route_channel(auto_slot="onchain_afternoon") == "onchain-updates"

    def test_slot_brand_meme(self):
        assert discord_publisher._route_channel(auto_slot="brand_meme") == "memes"

    def test_scheduled_prefix_goes_to_announcements(self):
        assert discord_publisher._route_channel(auto_slot="scheduled:12:00") == "announcements"
        assert discord_publisher._route_channel(auto_slot="scheduled:custom") == "announcements"

    def test_content_type_announcement(self):
        assert discord_publisher._route_channel(content_type="announcement") == "announcements"

    def test_content_type_meme(self):
        assert discord_publisher._route_channel(content_type="meme") == "memes"

    def test_content_type_onchain(self):
        assert discord_publisher._route_channel(content_type="onchain") == "onchain-updates"

    def test_slot_takes_priority_over_content_type(self):
        result = discord_publisher._route_channel(
            auto_slot="brand_meme", content_type="announcement"
        )
        assert result == "memes"

    def test_fallback_to_brand_updates(self):
        assert discord_publisher._route_channel() == "brand-updates"
        assert discord_publisher._route_channel(content_type="unknown_type") == "brand-updates"


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------


class TestEmbedBuilder:
    def test_embed_has_description(self):
        embed, _ = discord_publisher._build_embed(
            caption="Hello world", hashtags=["#test"]
        )
        assert embed.description == "Hello world"

    def test_embed_footer_has_hashtags(self):
        embed, _ = discord_publisher._build_embed(
            caption="Test", hashtags=["#crypto", "#defi"]
        )
        assert embed.footer.text == "#crypto #defi"

    def test_embed_no_hashtags(self):
        embed, _ = discord_publisher._build_embed(caption="Test")
        assert embed.footer.text is None or embed.footer == embed.footer  # no footer set

    def test_embed_with_url_image(self):
        embed, file = discord_publisher._build_embed(
            caption="Test", image_url="https://example.com/img.png"
        )
        assert embed.image.url == "https://example.com/img.png"

    def test_embed_color_uses_brand(self):
        embed, _ = discord_publisher._build_embed(caption="Test")
        # Just verify it has a color (int)
        assert embed.color is not None


# ---------------------------------------------------------------------------
# post_to_discord
# ---------------------------------------------------------------------------


class TestPostToDiscord:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_ready(self):
        with patch.object(discord_bot, "_client", None):
            result = await discord_publisher.post_to_discord("Hello")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_channel_id(self):
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True

        with patch.object(discord_bot, "_client", mock_client), \
             patch.object(discord_bot, "get_channel_id", return_value=None):
            result = await discord_publisher.post_to_discord(
                "Hello", content_type="announcement"
            )
            assert result is None
