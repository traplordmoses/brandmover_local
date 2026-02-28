"""Tests for agent.scheduler — dynamic prompt builders.

Every prompt builder reads from BrandConfig, not hardcoded strings.
Tests verify that brand context flows through and that empty/missing
BrandConfig fields are handled gracefully.
"""

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch, MagicMock

from agent.compositor_config import BrandConfig, ColorEntry
from agent.scheduler import (
    _brand_context,
    _brand_name,
    _engagement_templates,
    _brand_meme_templates,
    _themes_hint,
    _visual_style,
    _voice_summary,
    _build_onchain_prompt,
    _build_engagement_prompt,
    _build_brand_meme_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> BrandConfig:
    """Build a BrandConfig with test values — NOT FOID."""
    defaults = dict(
        brand_name="ZetaCorp",
        tagline="Build something weird",
        product_description="ZetaOS — an operating system for the absurd",
        voice_traits=["playful", "irreverent", "technically precise"],
        visual_style_prompt="pastel gradients, soft blur, cotton candy clouds",
        brand_phrases=["stay weird", "code is art", "ship the strange"],
        content_themes=["absurdism", "open source", "retro computing"],
        style_keywords=["Pastel gradient", "Soft glow", "Retro terminal"],
        colors={
            "primary": ColorEntry(role="primary", name="Mint Green", hex="#00ff88", rgb=(0, 255, 136)),
        },
    )
    defaults.update(overrides)
    return BrandConfig(**defaults)


def _empty_cfg() -> BrandConfig:
    return BrandConfig()


# ---------------------------------------------------------------------------
# _visual_style
# ---------------------------------------------------------------------------

class TestVisualStyle:
    def test_uses_visual_style_prompt(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _visual_style()
            assert "pastel gradients" in result
            assert "cotton candy clouds" in result
            assert result.startswith("Image style:")

    def test_falls_back_to_style_keywords(self):
        cfg = _make_cfg(visual_style_prompt="")
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _visual_style()
            assert "Pastel gradient" in result
            assert "Soft glow" in result

    def test_falls_back_to_generic_when_empty(self):
        cfg = _empty_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _visual_style()
            assert "high quality" in result
            assert "professional" in result

    def test_no_foid_references(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _visual_style()
            assert "FOID" not in result
            assert "foid" not in result.lower()


# ---------------------------------------------------------------------------
# _voice_summary
# ---------------------------------------------------------------------------

class TestVoiceSummary:
    def test_includes_brand_name_and_traits(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _voice_summary()
            assert "ZetaCorp" in result
            assert "playful" in result
            assert "irreverent" in result

    def test_limits_to_three_traits(self):
        cfg = _make_cfg(voice_traits=["a", "b", "c", "d", "e"])
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _voice_summary()
            # Should only include first 3
            assert "a" in result
            assert "b" in result
            assert "c" in result
            assert "d" not in result

    def test_falls_back_to_settings_brand_name(self):
        cfg = _make_cfg(brand_name="")
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.settings") as mock_settings:
                mock_settings.BRAND_NAME = "FallbackBrand"
                result = _voice_summary()
                assert "FallbackBrand" in result

    def test_empty_traits_just_shows_name(self):
        cfg = _make_cfg(voice_traits=[])
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _voice_summary()
            assert "ZetaCorp voice" == result

    def test_no_foid_references(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _voice_summary()
            assert "FOID" not in result


# ---------------------------------------------------------------------------
# _brand_context
# ---------------------------------------------------------------------------

class TestBrandContext:
    def test_includes_tagline(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _brand_context()
            assert "Build something weird" in result

    def test_includes_product(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _brand_context()
            assert "ZetaOS" in result

    def test_includes_brand_phrases(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _brand_context()
            assert "stay weird" in result

    def test_empty_returns_empty_string(self):
        cfg = _empty_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _brand_context()
            assert result == ""

    def test_truncates_product_description(self):
        long_product = "x" * 500
        cfg = _make_cfg(product_description=long_product)
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _brand_context()
            # Product is truncated to 200 chars
            assert len(result) < 500


# ---------------------------------------------------------------------------
# _brand_name
# ---------------------------------------------------------------------------

class TestBrandName:
    def test_reads_from_config(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            assert _brand_name() == "ZetaCorp"

    def test_falls_back_to_settings(self):
        cfg = _make_cfg(brand_name="")
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.settings") as mock_settings:
                mock_settings.BRAND_NAME = "SettingsBrand"
                assert _brand_name() == "SettingsBrand"


# ---------------------------------------------------------------------------
# _themes_hint
# ---------------------------------------------------------------------------

class TestThemesHint:
    def test_reads_content_themes(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _themes_hint()
            assert "absurdism" in result
            assert "open source" in result

    def test_limits_to_five(self):
        cfg = _make_cfg(content_themes=["a", "b", "c", "d", "e", "f", "g"])
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _themes_hint()
            parts = result.split(", ")
            assert len(parts) == 5

    def test_empty_returns_default(self):
        cfg = _empty_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            result = _themes_hint()
            assert "culture" in result
            assert "community" in result


# ---------------------------------------------------------------------------
# _engagement_templates
# ---------------------------------------------------------------------------

class TestEngagementTemplates:
    def test_returns_four_templates(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _engagement_templates()
            assert len(templates) == 4

    def test_templates_contain_brand_name(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _engagement_templates()
            name_count = sum(1 for t in templates if "ZetaCorp" in t)
            assert name_count >= 2

    def test_templates_contain_themes(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _engagement_templates()
            # At least some templates should include themes
            themes_found = any("absurdism" in t for t in templates)
            assert themes_found

    def test_templates_contain_visual_style(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _engagement_templates()
            # Every template should end with visual style
            for t in templates:
                assert "pastel gradients" in t

    def test_templates_contain_voice(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _engagement_templates()
            voice_found = any("playful" in t for t in templates)
            assert voice_found

    def test_templates_contain_brand_context(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _engagement_templates()
            # At least some templates should include brand context
            context_found = any("Build something weird" in t for t in templates)
            assert context_found

    def test_no_foid_in_any_template(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _engagement_templates()
            for t in templates:
                assert "FOID" not in t
                assert "foid" not in t.lower()

    def test_empty_config_still_works(self):
        cfg = _empty_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.settings") as mock_settings:
                mock_settings.BRAND_NAME = "DefaultBrand"
                templates = _engagement_templates()
                assert len(templates) == 4
                for t in templates:
                    assert isinstance(t, str)
                    assert len(t) > 50


# ---------------------------------------------------------------------------
# _brand_meme_templates
# ---------------------------------------------------------------------------

class TestBrandMemeTemplates:
    def test_returns_four_templates(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _brand_meme_templates()
            assert len(templates) == 4

    def test_templates_contain_brand_name(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _brand_meme_templates()
            name_count = sum(1 for t in templates if "ZetaCorp" in t)
            assert name_count >= 2

    def test_templates_contain_product(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _brand_meme_templates()
            product_found = any("ZetaOS" in t for t in templates)
            assert product_found

    def test_templates_contain_visual_style(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _brand_meme_templates()
            for t in templates:
                assert "pastel gradients" in t

    def test_empty_product_uses_fallback(self):
        cfg = _make_cfg(product_description="")
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _brand_meme_templates()
            # Should still produce valid templates
            assert len(templates) == 4
            fallback_found = any("brand's products" in t for t in templates)
            assert fallback_found

    def test_no_foid_in_any_template(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            templates = _brand_meme_templates()
            for t in templates:
                assert "FOID" not in t
                assert "foid" not in t.lower()


# ---------------------------------------------------------------------------
# _build_onchain_prompt
# ---------------------------------------------------------------------------

class TestBuildOnchainPrompt:
    def test_quiet_board_uses_dynamic_name(self):
        cfg = _make_cfg()
        quiet_state = MagicMock()
        quiet_state.is_quiet = True
        quiet_state.events = []

        async def _run():
            return await _build_onchain_prompt()

        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.onchain.fetch_board_state", new_callable=AsyncMock, return_value=quiet_state):
                with patch("agent.scheduler.auto_state._read_state", return_value={}):
                    with patch("agent.scheduler.onchain.get_new_event_ids", return_value=[]):
                        prompt, event_ids = asyncio.run(_run())
                        assert "ZetaCorp" in prompt
                        assert "FOID" not in prompt
                        assert event_ids == []

    def test_active_board_uses_dynamic_name(self):
        cfg = _make_cfg()
        active_state = MagicMock()
        active_state.is_quiet = False
        active_state.events = [MagicMock()]

        async def _run():
            return await _build_onchain_prompt()

        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.onchain.fetch_board_state", new_callable=AsyncMock, return_value=active_state):
                with patch("agent.scheduler.auto_state._read_state", return_value={}):
                    with patch("agent.scheduler.onchain.format_onchain_summary", return_value="Summary"):
                        with patch("agent.scheduler.onchain.get_new_event_ids", return_value=["evt1"]):
                            prompt, event_ids = asyncio.run(_run())
                            assert "ZetaCorp" in prompt
                            assert "FOID" not in prompt
                            assert event_ids == ["evt1"]

    def test_quiet_board_includes_visual_style(self):
        cfg = _make_cfg()
        quiet_state = MagicMock()
        quiet_state.is_quiet = True

        async def _run():
            return await _build_onchain_prompt()

        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.onchain.fetch_board_state", new_callable=AsyncMock, return_value=quiet_state):
                with patch("agent.scheduler.auto_state._read_state", return_value={}):
                    with patch("agent.scheduler.onchain.get_new_event_ids", return_value=[]):
                        prompt, _ = asyncio.run(_run())
                        assert "pastel gradients" in prompt


# ---------------------------------------------------------------------------
# _build_engagement_prompt / _build_brand_meme_prompt
# ---------------------------------------------------------------------------

class TestBuildPromptRotation:
    def test_engagement_rotates_through_templates(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.auto_state.get_rotation_index", return_value=0):
                with patch("agent.scheduler.auto_state.advance_rotation") as mock_adv:
                    prompt = _build_engagement_prompt()
                    assert isinstance(prompt, str)
                    assert len(prompt) > 50
                    mock_adv.assert_called_once_with("engagement", 4)

    def test_brand_meme_rotates_through_templates(self):
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.auto_state.get_rotation_index", return_value=2):
                with patch("agent.scheduler.auto_state.advance_rotation") as mock_adv:
                    prompt = _build_brand_meme_prompt()
                    assert isinstance(prompt, str)
                    assert len(prompt) > 50
                    mock_adv.assert_called_once_with("brand_meme", 4)

    def test_engagement_index_wraps(self):
        """Rotation index beyond template count wraps via modulo."""
        cfg = _make_cfg()
        with patch("agent.scheduler.compositor_config.get_config", return_value=cfg):
            with patch("agent.scheduler.auto_state.get_rotation_index", return_value=100):
                with patch("agent.scheduler.auto_state.advance_rotation"):
                    prompt = _build_engagement_prompt()
                    assert isinstance(prompt, str)
