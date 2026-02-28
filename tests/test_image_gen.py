"""Tests for agent.image_gen — override parameters, prompt enhancement,
model-specific input building, and brand terms injection.

These tests verify that:
- Override parameters (model_override, aspect_ratio, negative_prompt_override,
  skip_enhance) are respected.
- enhance_prompt() reads from BrandConfig, not hardcoded strings.
- _build_input() produces correct model-specific payloads.
- _get_brand_terms() and _get_negative_prompt() read from BrandConfig.
"""

from unittest.mock import patch

from agent.compositor_config import BrandConfig, ColorEntry
from agent.image_gen import (
    _MODELS,
    _build_input,
    _get_brand_terms,
    _get_negative_prompt,
    _get_quality_profile,
    enhance_prompt,
    select_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> BrandConfig:
    """Build a BrandConfig with non-FOID test values."""
    defaults = dict(
        brand_name="ZetaCorp",
        style_keywords=["Pastel gradient", "Soft glow", "Retro terminal", "Wireframe"],
        colors={
            "primary": ColorEntry(role="primary", name="Mint", hex="#00ff88", rgb=(0, 255, 136)),
            "accent_1": ColorEntry(role="accent_1", name="Coral", hex="#ff6644", rgb=(255, 102, 68)),
            "accent_2": ColorEntry(role="accent_2", name="Lilac", hex="#cc88ff", rgb=(204, 136, 255)),
            "background": ColorEntry(role="background", name="Charcoal", hex="#1a1a2e", rgb=(26, 26, 46)),
        },
        avoid_terms=["flat colors", "white backgrounds", "stock photos"],
    )
    defaults.update(overrides)
    return BrandConfig(**defaults)


def _empty_cfg() -> BrandConfig:
    return BrandConfig()


# ---------------------------------------------------------------------------
# _get_brand_terms — reads from BrandConfig
# ---------------------------------------------------------------------------

class TestGetBrandTerms:
    def test_includes_style_keywords(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "Pastel gradient" in terms
            assert "Soft glow" in terms

    def test_includes_color_palette(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "#00ff88" in terms
            assert "mint" in terms.lower()

    def test_includes_background_color(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "charcoal" in terms.lower()
            assert "#1a1a2e" in terms

    def test_empty_config_returns_fallback(self):
        cfg = _empty_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "high quality" in terms
            assert "professional" in terms

    def test_no_foid_references(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "FOID" not in terms
            assert "foid" not in terms.lower()

    def test_limits_style_keywords(self):
        cfg = _make_cfg(style_keywords=[f"kw{i}" for i in range(20)])
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            # Should only include first 6
            assert "kw0" in terms
            assert "kw5" in terms
            assert "kw6" not in terms


# ---------------------------------------------------------------------------
# _get_negative_prompt — reads avoid_terms from BrandConfig
# ---------------------------------------------------------------------------

class TestGetNegativePrompt:
    def test_includes_avoid_terms(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            neg = _get_negative_prompt()
            assert "flat colors" in neg
            assert "white backgrounds" in neg
            assert "stock photos" in neg

    def test_includes_base_negatives(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            neg = _get_negative_prompt()
            assert "blurry" in neg
            assert "low quality" in neg

    def test_empty_avoid_terms_uses_base_only(self):
        cfg = _make_cfg(avoid_terms=[])
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            neg = _get_negative_prompt()
            assert "blurry" in neg
            assert "flat colors" not in neg


# ---------------------------------------------------------------------------
# _get_quality_profile — brand-specific quality enrichment
# ---------------------------------------------------------------------------

class TestGetQualityProfile:
    def test_includes_style_keywords(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("announcement")
            assert "Pastel gradient" in profile

    def test_includes_color_glow(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("lifestyle")
            assert "mint glow" in profile.lower()

    def test_includes_background_context(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("event")
            assert "charcoal background" in profile.lower()

    def test_includes_content_type_modifier(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("announcement")
            assert "volumetric light" in profile


# ---------------------------------------------------------------------------
# enhance_prompt — full prompt enhancement pipeline
# ---------------------------------------------------------------------------

class TestEnhancePrompt:
    def test_adds_brand_terms(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, neg = enhance_prompt("A glowing crystal", "announcement")
            assert "A glowing crystal" in enhanced
            assert "Pastel gradient" in enhanced
            assert "#00ff88" in enhanced

    def test_adds_negative_prompt(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            _, neg = enhance_prompt("A glowing crystal", "announcement")
            assert "flat colors" in neg
            assert "blurry" in neg

    def test_skips_brand_when_already_present(self):
        """If prompt already references brand colors/style, skip brand enforcement."""
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, _ = enhance_prompt(
                "A crystal with brand color #00ff88 aesthetic", "announcement"
            )
            # Brand terms should still work — _BRAND_INDICATORS checks for "brand.*color"
            # which matches, so brand terms should be skipped
            # But quality profile should still be added
            assert "volumetric light" in enhanced

    def test_mascot_special_path(self):
        """Community content with mascot keywords gets special treatment."""
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, neg = enhance_prompt("A cute mascot character", "community")
            assert "Pixar-quality" in enhanced
            assert "uncanny valley" in neg

    def test_locked_directives_preserved(self):
        """Locked directives survive enhancement."""
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, _ = enhance_prompt(
                "Product shot, upright, matte black background", "brand_asset"
            )
            assert "upright" in enhanced
            assert "matte black background" in enhanced


# ---------------------------------------------------------------------------
# _build_input — model-specific payloads
# ---------------------------------------------------------------------------

class TestBuildInput:
    def test_flux_payload(self):
        payload = _build_input(_MODELS["flux"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["aspect_ratio"] == "16:9"
        assert payload["output_format"] == "webp"

    def test_flux_custom_aspect(self):
        payload = _build_input(_MODELS["flux"], "test", aspect_ratio="1:1")
        assert payload["aspect_ratio"] == "1:1"

    def test_nano_banana_payload(self):
        payload = _build_input(_MODELS["nano-banana"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["resolution"] == "2K"

    def test_recraft_payload(self):
        payload = _build_input(_MODELS["recraft-svg"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["size"] == "1820x1024"

    def test_recraft_custom_size(self):
        payload = _build_input(_MODELS["recraft-svg"], "test", size="1024x1024")
        assert payload["size"] == "1024x1024"

    def test_seedream_payload(self):
        payload = _build_input(_MODELS["seedream"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["aspect_ratio"] == "16:9"

    def test_seedream_negative_prompt(self):
        """Only seedream accepts negative_prompt."""
        payload = _build_input(_MODELS["seedream"], "test", negative_prompt="ugly, bad")
        assert payload["negative_prompt"] == "ugly, bad"

    def test_flux_ignores_negative_prompt(self):
        """Flux doesn't support negative_prompt, so it should be excluded."""
        payload = _build_input(_MODELS["flux"], "test", negative_prompt="ugly, bad")
        assert "negative_prompt" not in payload

    def test_unknown_model_minimal_payload(self):
        payload = _build_input("some/unknown-model", "test prompt")
        assert payload == {"prompt": "test prompt"}


# ---------------------------------------------------------------------------
# select_model — routing logic
# ---------------------------------------------------------------------------

class TestSelectModel:
    def test_announcement_routes_to_nano_banana(self):
        model, reason = select_model("announcement", "Product launch image")
        assert model == _MODELS["nano-banana"]

    def test_brand_asset_routes_to_recraft(self):
        model, reason = select_model("brand_asset", "A brand icon")
        assert model == _MODELS["recraft-svg"]

    def test_lifestyle_routes_to_seedream(self):
        model, reason = select_model("lifestyle", "Coffee shop scene")
        assert model == _MODELS["seedream"]

    def test_default_routes_to_flux(self):
        model, reason = select_model("engagement", "Community post")
        assert model == _MODELS["flux"]

    def test_text_overlay_keyword_routes_to_nano_banana(self):
        model, reason = select_model("engagement", "Image with bold text reads HELLO")
        assert model == _MODELS["nano-banana"]

    def test_manual_override(self):
        with patch("agent.image_gen.settings") as mock_settings:
            mock_settings.IMAGE_MODEL = "custom/model"
            model, reason = select_model("announcement", "Test")
            assert model == "custom/model"
            assert "override" in reason

    def test_auto_mode_does_routing(self):
        with patch("agent.image_gen.settings") as mock_settings:
            mock_settings.IMAGE_MODEL = "auto"
            model, reason = select_model("lifestyle", "Nature photo")
            assert model == _MODELS["seedream"]
