"""Tests for agent.asset_gen — asset type parsing, template loading, prompt
construction with BrandConfig injection, and model routing per asset type."""

import textwrap
from pathlib import Path
from unittest.mock import patch

from agent.asset_gen import (
    SUPPORTED_ASSET_TYPES,
    _ASSET_MODEL_MAP,
    _ASSET_QUALITY_BOOSTERS,
    _ASSET_NEGATIVE_PROMPTS,
    _DEFAULT_TEMPLATES,
    _build_asset_prompt,
    _get_brand_substitutions,
    _load_asset_template,
    parse_asset_type,
    select_asset_model,
)
from agent.compositor_config import BrandConfig, ColorEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> BrandConfig:
    """Build a BrandConfig with sensible test defaults."""
    defaults = dict(
        brand_name="TestBrand",
        tagline="Build the future",
        style_keywords=["Futuristic 3D renders", "Vibrant neon glow"],
        visual_style_prompt="dark background, neon blue glow, futuristic aesthetic",
        colors={
            "primary": ColorEntry(role="primary", name="Electric Blue", hex="#0066ff", rgb=(0, 102, 255)),
            "accent_1": ColorEntry(role="accent_1", name="Hot Pink", hex="#ff0066", rgb=(255, 0, 102)),
            "accent_2": ColorEntry(role="accent_2", name="Lavender", hex="#cdb7ff", rgb=(205, 183, 255)),
            "background": ColorEntry(role="background", name="Dark Navy", hex="#0a0a1a", rgb=(10, 10, 26)),
        },
        brand_phrases=["build the future, together", "ship it."],
        avoid_terms=["flat clip-art", "low resolution"],
    )
    defaults.update(overrides)
    return BrandConfig(**defaults)


# ---------------------------------------------------------------------------
# Asset type parsing
# ---------------------------------------------------------------------------

class TestParseAssetType:
    def test_valid_logo(self):
        asset_type, desc = parse_asset_type("logo modern crest design")
        assert asset_type == "logo"
        assert desc == "modern crest design"

    def test_valid_icon(self):
        asset_type, desc = parse_asset_type("icon a bell notification")
        assert asset_type == "icon"
        assert desc == "a bell notification"

    def test_valid_3d_asset(self):
        asset_type, desc = parse_asset_type("3d_asset floating crystal sphere")
        assert asset_type == "3d_asset"
        assert desc == "floating crystal sphere"

    def test_case_insensitive(self):
        asset_type, desc = parse_asset_type("MASCOT friendly robot helper")
        assert asset_type == "mascot"

    def test_unknown_type(self):
        asset_type, error = parse_asset_type("poster landscape scenery")
        assert asset_type is None
        assert "Unknown asset type" in error

    def test_empty_input(self):
        asset_type, error = parse_asset_type("")
        assert asset_type is None
        assert "No asset type" in error

    def test_missing_description(self):
        asset_type, error = parse_asset_type("logo")
        assert asset_type is None
        assert "description" in error.lower()

    def test_all_types_parseable(self):
        for at in SUPPORTED_ASSET_TYPES:
            asset_type, desc = parse_asset_type(f"{at} test description")
            assert asset_type == at, f"Failed to parse {at}"
            assert desc == "test description"


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

class TestTemplateLoading:
    def test_load_missing_template(self, tmp_path):
        """When no file exists, returns None."""
        with patch("agent.asset_gen._PROMPTS_DIR", tmp_path):
            result = _load_asset_template("logo")
            assert result is None

    def test_load_existing_template(self, tmp_path):
        """When file exists, returns its content."""
        template = "Logo: {description}, style: {style_keywords}"
        (tmp_path / "logo.txt").write_text(template)
        with patch("agent.asset_gen._PROMPTS_DIR", tmp_path):
            result = _load_asset_template("logo")
            assert result == template

    def test_all_types_have_default_template(self):
        """Every supported asset type has a built-in default template."""
        for at in SUPPORTED_ASSET_TYPES:
            assert at in _DEFAULT_TEMPLATES, f"No default template for {at}"

    def test_default_templates_have_placeholders(self):
        """All default templates contain the standard placeholders."""
        for at, template in _DEFAULT_TEMPLATES.items():
            for ph in ("{description}", "{style_keywords}", "{colors}", "{background}"):
                assert ph in template, f"Template {at} missing placeholder {ph}"


# ---------------------------------------------------------------------------
# Prompt construction with BrandConfig injection
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_prompt_includes_description(self):
        cfg = _make_config()
        with patch("agent.asset_gen.compositor_config.get_config", return_value=cfg):
            prompt = _build_asset_prompt("logo", "modern minimalist crest")
            assert "modern minimalist crest" in prompt

    def test_prompt_includes_style_keywords(self):
        cfg = _make_config()
        with patch("agent.asset_gen.compositor_config.get_config", return_value=cfg):
            prompt = _build_asset_prompt("banner", "launch event")
            assert "Futuristic 3D renders" in prompt

    def test_prompt_includes_colors(self):
        cfg = _make_config()
        with patch("agent.asset_gen.compositor_config.get_config", return_value=cfg):
            prompt = _build_asset_prompt("icon", "notification bell")
            assert "#0066ff" in prompt

    def test_prompt_includes_background(self):
        cfg = _make_config()
        with patch("agent.asset_gen.compositor_config.get_config", return_value=cfg):
            prompt = _build_asset_prompt("background", "starfield scene")
            assert "dark navy" in prompt.lower()

    def test_prompt_includes_visual_style_prompt(self):
        cfg = _make_config()
        with patch("agent.asset_gen.compositor_config.get_config", return_value=cfg):
            prompt = _build_asset_prompt("mascot", "friendly robot")
            assert "futuristic aesthetic" in prompt

    def test_external_template_used_when_available(self, tmp_path):
        """External template takes precedence over built-in default."""
        template = "CUSTOM: {description} | {style_keywords}"
        (tmp_path / "logo.txt").write_text(template)
        cfg = _make_config()
        with (
            patch("agent.asset_gen._PROMPTS_DIR", tmp_path),
            patch("agent.asset_gen.compositor_config.get_config", return_value=cfg),
        ):
            prompt = _build_asset_prompt("logo", "shield design")
            assert prompt.startswith("CUSTOM:")
            assert "shield design" in prompt

    def test_fallback_on_bad_external_template(self, tmp_path):
        """Falls back to default if external template has bad placeholders."""
        (tmp_path / "logo.txt").write_text("Logo: {description} {unknown_var}")
        cfg = _make_config()
        with (
            patch("agent.asset_gen._PROMPTS_DIR", tmp_path),
            patch("agent.asset_gen.compositor_config.get_config", return_value=cfg),
        ):
            prompt = _build_asset_prompt("logo", "shield design")
            # Should use the built-in template, not crash
            assert "shield design" in prompt

    def test_empty_config_still_works(self):
        """Prompt builds even with minimal BrandConfig."""
        cfg = BrandConfig()
        with patch("agent.asset_gen.compositor_config.get_config", return_value=cfg):
            prompt = _build_asset_prompt("banner", "launch event")
            assert "launch event" in prompt

    def test_brand_substitutions_content(self):
        cfg = _make_config()
        with patch("agent.asset_gen.compositor_config.get_config", return_value=cfg):
            subs = _get_brand_substitutions()
            assert "Futuristic 3D renders" in subs["style_keywords"]
            assert "futuristic aesthetic" in subs["style_keywords"]
            assert "#0066ff" in subs["colors"]
            assert "dark navy" in subs["background"]


# ---------------------------------------------------------------------------
# Model routing per asset type
# ---------------------------------------------------------------------------

class TestModelRouting:
    def test_logo_routes_to_nano_banana(self):
        model = select_asset_model("logo")
        assert "nano-banana" in model

    def test_icon_routes_to_recraft(self):
        model = select_asset_model("icon")
        assert "recraft" in model

    def test_mascot_routes_to_flux(self):
        model = select_asset_model("mascot")
        assert "flux" in model

    def test_background_routes_to_flux(self):
        model = select_asset_model("background")
        assert "flux" in model

    def test_3d_asset_routes_to_flux(self):
        model = select_asset_model("3d_asset")
        assert "flux" in model

    def test_banner_routes_to_flux(self):
        model = select_asset_model("banner")
        assert "flux" in model

    def test_social_header_routes_to_flux(self):
        model = select_asset_model("social_header")
        assert "flux" in model

    def test_all_types_have_model_mapping(self):
        """Every supported type has a model mapping."""
        for at in SUPPORTED_ASSET_TYPES:
            assert at in _ASSET_MODEL_MAP, f"No model mapping for {at}"

    def test_all_types_have_quality_boosters(self):
        for at in SUPPORTED_ASSET_TYPES:
            assert at in _ASSET_QUALITY_BOOSTERS, f"No quality boosters for {at}"

    def test_all_types_have_negative_prompts(self):
        for at in SUPPORTED_ASSET_TYPES:
            assert at in _ASSET_NEGATIVE_PROMPTS, f"No negative prompt for {at}"

    def test_unknown_type_falls_back_to_flux(self):
        model = select_asset_model("nonexistent_type")
        assert "flux" in model
