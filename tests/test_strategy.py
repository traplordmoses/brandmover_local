"""Tests for agent.strategy — strategy recommendation engine."""

import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from agent.asset_audit import AssetAuditEntry, AssetInventory
from agent.strategy import (
    StrategyRecommendation,
    recommend_strategy,
    generate_config_json,
    generate_strategy_markdown,
    generate_content_calendar,
    _calendar_to_markdown,
    save_strategy,
    _ARCHETYPE_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _inventory(archetype="full_brand"):
    entries = [
        AssetAuditEntry(path="/tmp/logo.png", category="logo", quality_score=8),
        AssetAuditEntry(path="/tmp/colors.png", category="color_palette", quality_score=7),
    ]
    return AssetInventory(
        entries=entries,
        consolidated_colors=[{"hex": "#0066ff", "name": "Blue"}],
        consolidated_style=["modern", "clean"],
        missing_items=["style_guide"],
        archetype=archetype,
    )


# ---------------------------------------------------------------------------
# recommend_strategy (mocked Claude)
# ---------------------------------------------------------------------------

class TestRecommendStrategy:
    def test_returns_recommendation_from_claude(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "compositor_enabled": True,
            "badge_text": None,
            "default_mode": "image_always",
            "recommended_content_types": ["announcement", "community", "meme"],
            "visual_style_notes": "Bold and modern",
            "reasoning": "Full brand assets available",
        }))]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await recommend_strategy(
                    "TestBrand", "A test brand", ["twitter"],
                    _inventory("full_brand"), {},
                )

        rec = asyncio.run(_run())
        assert rec.archetype == "full_brand"
        assert rec.compositor_enabled is True
        assert rec.default_mode == "image_always"
        assert "announcement" in rec.recommended_content_types
        assert rec.visual_style_notes == "Bold and modern"

    def test_falls_back_to_defaults_on_bad_json(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await recommend_strategy(
                    "TestBrand", "A test brand", ["twitter"],
                    _inventory("starting_fresh"), {},
                )

        rec = asyncio.run(_run())
        assert rec.archetype == "starting_fresh"
        defaults = _ARCHETYPE_DEFAULTS["starting_fresh"]
        assert rec.compositor_enabled == defaults["compositor_enabled"]
        assert rec.recommended_content_types == defaults["recommended_content_types"]

    def test_no_inventory(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "compositor_enabled": False,
            "default_mode": "image_optional",
            "recommended_content_types": ["announcement"],
            "reasoning": "Starting from scratch",
        }))]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await recommend_strategy(
                    "NewBrand", "New", ["twitter"], None, {},
                )

        rec = asyncio.run(_run())
        assert rec.archetype == "starting_fresh"
        assert rec.compositor_enabled is False


# ---------------------------------------------------------------------------
# generate_config_json
# ---------------------------------------------------------------------------

class TestGenerateConfigJson:
    def test_schema_structure(self):
        rec = StrategyRecommendation(
            archetype="full_brand",
            compositor_enabled=True,
            default_mode="image_always",
            recommended_content_types=["announcement", "meme"],
            platforms=["x", "telegram"],
        )
        config = generate_config_json(rec, "TestBrand")

        assert config["version"] == "8.0"
        assert config["brand_name"] == "TestBrand"
        assert config["pipeline"]["compositor_enabled"] is True
        assert config["pipeline"]["default_mode"] == "image_always"
        assert config["content_types_enabled"] == ["announcement", "meme"]
        assert config["platforms"] == ["x", "telegram"]
        assert config["onboarding"]["completed"] is True
        assert config["onboarding"]["archetype"] == "full_brand"

    def test_badge_text_null(self):
        rec = StrategyRecommendation(archetype="starting_fresh", badge_text=None)
        config = generate_config_json(rec)
        assert config["pipeline"]["badge_text"] is None

    def test_platforms_default(self):
        rec = StrategyRecommendation(archetype="starting_fresh")
        config = generate_config_json(rec)
        assert config["platforms"] == ["x"]

    def test_visual_source_full_brand(self):
        rec = StrategyRecommendation(archetype="full_brand")
        config = generate_config_json(rec)
        assert config["visual_source"]["primary"] == "client_assets"
        assert config["visual_source"]["fallback"] == "ai_generated"

    def test_visual_source_has_identity(self):
        rec = StrategyRecommendation(archetype="has_identity")
        config = generate_config_json(rec)
        assert config["visual_source"]["primary"] == "hybrid"

    def test_visual_source_starting_fresh(self):
        rec = StrategyRecommendation(archetype="starting_fresh")
        config = generate_config_json(rec)
        assert config["visual_source"]["primary"] == "ai_generated"


# ---------------------------------------------------------------------------
# generate_strategy_markdown
# ---------------------------------------------------------------------------

class TestGenerateStrategyMarkdown:
    def test_contains_key_sections(self):
        rec = StrategyRecommendation(
            archetype="has_identity",
            compositor_enabled=True,
            default_mode="image_optional",
            recommended_content_types=["announcement", "community"],
            platforms=["x", "linkedin"],
            visual_style_notes="Clean and modern",
            reasoning="Has logo and some colors",
        )
        md = generate_strategy_markdown(rec, "TestBrand")
        assert "TestBrand" in md
        assert "has_identity" in md
        assert "announcement" in md
        assert "Clean and modern" in md
        assert "Has logo and some colors" in md
        assert "x, linkedin" in md


# ---------------------------------------------------------------------------
# save_strategy
# ---------------------------------------------------------------------------

class TestSaveStrategy:
    def test_creates_files(self, tmp_path):
        rec = StrategyRecommendation(
            archetype="full_brand",
            compositor_enabled=True,
            recommended_content_types=["announcement"],
        )

        with patch("agent.strategy.settings") as mock_settings:
            mock_settings.BRAND_FOLDER = str(tmp_path)
            mock_settings.BRAND_NAME = "TestBrand"
            mock_settings.AGENT_MODE = "pipeline"
            save_strategy(rec, "TestBrand")

        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "strategy.md").exists()

        config = json.loads((tmp_path / "config.json").read_text())
        assert config["brand_name"] == "TestBrand"
        assert config["onboarding"]["completed"] is True


# ---------------------------------------------------------------------------
# Content calendar
# ---------------------------------------------------------------------------

class TestCalendarToMarkdown:
    def test_generates_table(self):
        data = {
            "calendar": [
                {
                    "day": "Monday",
                    "content_type": "announcement",
                    "topic": "Product update",
                    "description": "Share the latest feature",
                    "time": "9am",
                    "platforms": ["x"],
                },
                {
                    "day": "Tuesday",
                    "content_type": "community",
                    "topic": "Community spotlight",
                    "description": "Highlight a community member",
                    "time": "12pm",
                    "platforms": ["x", "telegram"],
                },
            ],
            "weekly_theme": "Building momentum",
            "notes": "Focus on engagement",
        }
        md = _calendar_to_markdown(data, "TestBrand")
        assert "TestBrand" in md
        assert "Weekly Content Calendar" in md
        assert "Building momentum" in md
        assert "Monday" in md
        assert "announcement" in md
        assert "Tuesday" in md
        assert "community" in md
        assert "Focus on engagement" in md

    def test_empty_calendar(self):
        data = {"calendar": [], "weekly_theme": "", "notes": ""}
        md = _calendar_to_markdown(data, "Test")
        assert "Test" in md
        assert "| Day |" in md

    def test_details_section(self):
        data = {
            "calendar": [
                {
                    "day": "Wednesday",
                    "content_type": "meme",
                    "topic": "Industry humor",
                    "description": "Light-hearted meme about crypto",
                    "time": "3pm",
                    "platforms": ["x"],
                },
            ],
        }
        md = _calendar_to_markdown(data, "Test")
        assert "Daily Details" in md
        assert "Wednesday" in md
        assert "Light-hearted meme" in md


class TestGenerateContentCalendar:
    def test_generates_and_saves_calendar(self, tmp_path):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "calendar": [
                {
                    "day": "Monday",
                    "content_type": "announcement",
                    "topic": "Weekly update",
                    "description": "Share progress",
                    "time": "9am",
                    "platforms": ["x"],
                },
            ],
            "weekly_theme": "Growth week",
            "notes": "Keep it concise",
        }))]

        rec = StrategyRecommendation(
            archetype="has_identity",
            recommended_content_types=["announcement", "community"],
            platforms=["x"],
        )

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                with patch("agent.strategy.settings") as mock_settings:
                    mock_settings.BRAND_FOLDER = str(tmp_path)
                    mock_settings.BRAND_NAME = "TestBrand"
                    mock_settings.ANTHROPIC_API_KEY = "test-key"
                    return await generate_content_calendar(
                        "TestBrand", "A test brand", ["x"], rec,
                    )

        md = asyncio.run(_run())
        assert "TestBrand" in md
        assert "Monday" in md
        assert "Growth week" in md
        assert (tmp_path / "content_calendar.md").exists()
        saved = (tmp_path / "content_calendar.md").read_text()
        assert saved == md

    def test_includes_creative_data_in_prompt(self, tmp_path):
        """Calendar prompt includes creative brief and never_do when provided."""
        captured_prompt = {}
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "calendar": [], "weekly_theme": "", "notes": "",
        }))]

        rec = StrategyRecommendation(
            archetype="has_identity",
            recommended_content_types=["announcement"],
        )

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def capture_create(**kwargs):
                    captured_prompt["messages"] = kwargs.get("messages", [])
                    return mock_response
                mock_client.messages.create = capture_create
                mock_cls.return_value = mock_client
                with patch("agent.strategy.settings") as mock_settings:
                    mock_settings.BRAND_FOLDER = str(tmp_path)
                    mock_settings.BRAND_NAME = "TestBrand"
                    mock_settings.ANTHROPIC_API_KEY = "test-key"
                    return await generate_content_calendar(
                        "TestBrand", "A test brand", ["x"], rec,
                        creative_brief="garage startup confidence. hand-drawn warmth.",
                        never_do=["Never use stock photos", "Avoid corporate jargon"],
                    )

        asyncio.run(_run())
        user_msg = captured_prompt["messages"][0]["content"]
        assert "garage startup confidence" in user_msg
        assert "Never use stock photos" in user_msg
        assert "CREATIVE DIRECTION" in user_msg

    def test_omits_creative_section_when_empty(self, tmp_path):
        """Calendar prompt omits creative section when no creative data."""
        captured_prompt = {}
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "calendar": [], "weekly_theme": "", "notes": "",
        }))]

        rec = StrategyRecommendation(
            archetype="starting_fresh",
            recommended_content_types=["announcement"],
        )

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def capture_create(**kwargs):
                    captured_prompt["messages"] = kwargs.get("messages", [])
                    return mock_response
                mock_client.messages.create = capture_create
                mock_cls.return_value = mock_client
                with patch("agent.strategy.settings") as mock_settings:
                    mock_settings.BRAND_FOLDER = str(tmp_path)
                    mock_settings.BRAND_NAME = "TestBrand"
                    mock_settings.ANTHROPIC_API_KEY = "test-key"
                    return await generate_content_calendar(
                        "TestBrand", "A test brand", ["x"], rec,
                    )

        asyncio.run(_run())
        user_msg = captured_prompt["messages"][0]["content"]
        assert "CREATIVE DIRECTION" not in user_msg

    def test_handles_bad_json_response(self, tmp_path):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON at all")]

        rec = StrategyRecommendation(archetype="starting_fresh")

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                with patch("agent.strategy.settings") as mock_settings:
                    mock_settings.BRAND_FOLDER = str(tmp_path)
                    mock_settings.BRAND_NAME = "TestBrand"
                    mock_settings.ANTHROPIC_API_KEY = "test-key"
                    return await generate_content_calendar(
                        "TestBrand", "A test", ["x"], rec,
                    )

        md = asyncio.run(_run())
        # Should still produce valid markdown with empty table
        assert "TestBrand" in md
        assert "| Day |" in md
        assert (tmp_path / "content_calendar.md").exists()
