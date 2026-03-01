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
            with patch("agent.strategy.anthropic.AsyncAnthropic") as mock_cls:
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
            with patch("agent.strategy.anthropic.AsyncAnthropic") as mock_cls:
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
            with patch("agent.strategy.anthropic.AsyncAnthropic") as mock_cls:
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
        )
        config = generate_config_json(rec, "TestBrand")

        assert config["version"] == "8.0"
        assert config["brand_name"] == "TestBrand"
        assert config["pipeline"]["compositor_enabled"] is True
        assert config["pipeline"]["default_mode"] == "image_always"
        assert config["content_types_enabled"] == ["announcement", "meme"]
        assert config["onboarding"]["completed"] is True
        assert config["onboarding"]["archetype"] == "full_brand"

    def test_badge_text_null(self):
        rec = StrategyRecommendation(archetype="starting_fresh", badge_text=None)
        config = generate_config_json(rec)
        assert config["pipeline"]["badge_text"] is None


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
            visual_style_notes="Clean and modern",
            reasoning="Has logo and some colors",
        )
        md = generate_strategy_markdown(rec, "TestBrand")
        assert "TestBrand" in md
        assert "has_identity" in md
        assert "announcement" in md
        assert "Clean and modern" in md
        assert "Has logo and some colors" in md


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
