"""Integration tests — cross-module brand-agnostic verification.

These tests verify that a completely non-FOID BrandConfig flows correctly
through scheduler, compositor, asset_gen, and image_gen with zero hardcoded
brand leakage. Also verifies that empty/minimal BrandConfig produces
graceful fallbacks everywhere.
"""

from unittest.mock import patch

from agent.compositor_config import BrandConfig, ColorEntry, FontEntry
from agent.scheduler import (
    _brand_context,
    _brand_name,
    _engagement_templates,
    _brand_meme_templates,
    _themes_hint,
    _visual_style,
    _voice_summary,
)
from agent.asset_gen import _build_asset_prompt, _get_brand_substitutions
from agent.image_gen import (
    _get_brand_terms,
    _get_negative_prompt,
    _get_quality_profile,
    enhance_prompt,
)
from agent import compositor


# ---------------------------------------------------------------------------
# Full alternate brand config — nothing from FOID
# ---------------------------------------------------------------------------

ZETA_CONFIG = BrandConfig(
    brand_name="ZetaCorp",
    tagline="Build something weird",
    website="https://zetacorp.dev",
    x_handle="@zetacorp",
    product_description="ZetaOS — an operating system for the absurd. Includes ZetaShell (terminal) and ZetaCanvas (whiteboard).",
    voice_traits=["playful", "irreverent", "technically precise"],
    visual_style_prompt="pastel gradients, soft blur, cotton candy clouds, dreamy aesthetic",
    brand_phrases=["stay weird", "code is art", "ship the strange"],
    content_themes=["absurdism", "open source", "retro computing", "digital gardens"],
    style_keywords=["Pastel gradient", "Soft glow", "Retro terminal", "Wireframe overlay"],
    avoid_terms=["flat colors", "corporate blue", "stock photos"],
    colors={
        "primary": ColorEntry(role="primary", name="Mint Green", hex="#00ff88", rgb=(0, 255, 136)),
        "accent_1": ColorEntry(role="accent_1", name="Coral", hex="#ff6644", rgb=(255, 102, 68)),
        "accent_2": ColorEntry(role="accent_2", name="Lilac", hex="#cc88ff", rgb=(204, 136, 255)),
        "accent_3": ColorEntry(role="accent_3", name="Sky Blue", hex="#88ccff", rgb=(136, 204, 255)),
        "background": ColorEntry(role="background", name="Charcoal", hex="#1a1a2e", rgb=(26, 26, 46)),
        "text": ColorEntry(role="text", name="Off White", hex="#f0f0f0", rgb=(240, 240, 240)),
    },
    fonts={
        "display": FontEntry(use="display", family="Poppins", weight="Black"),
        "body": FontEntry(use="body", family="Inter", weight="Regular"),
    },
    canvas_width=1920,
    canvas_height=1080,
    layout_mappings={"meme": "engagement", "lifestyle": "campaign"},
    glass_opacity=10,
    glass_blur=18,
    glass_radius=36,
    glass_inset=(50, 80, 50, 40),
    orb_alpha_base=22,
    orb_count=5,
)


# Hardcoded FOID-related strings that must never appear in dynamic output
_FOID_STRINGS = [
    "FOID", "foid", "Foid",
    "lorecraft",
    # Specific FOID product names
    "OpenClaw",
    # FOID visual style references
    "Frutiger Aero",
    "Y2K terminal",
]


def _check_no_foid(text: str, source: str = "") -> None:
    """Assert no FOID-specific strings leaked into the output."""
    for s in _FOID_STRINGS:
        assert s not in text, f"FOID leakage '{s}' in {source}: ...{text[max(0,text.find(s)-30):text.find(s)+30]}..."


# ---------------------------------------------------------------------------
# End-to-end: scheduler prompts use alternate brand
# ---------------------------------------------------------------------------

class TestSchedulerWithAlternateBrand:
    def _patch(self):
        return patch("agent.scheduler.compositor_config.get_config", return_value=ZETA_CONFIG)

    def test_visual_style_uses_zeta(self):
        with self._patch():
            result = _visual_style()
            assert "pastel gradients" in result
            _check_no_foid(result, "_visual_style")

    def test_voice_summary_uses_zeta(self):
        with self._patch():
            result = _voice_summary()
            assert "ZetaCorp" in result
            assert "playful" in result
            _check_no_foid(result, "_voice_summary")

    def test_brand_context_uses_zeta(self):
        with self._patch():
            result = _brand_context()
            assert "Build something weird" in result
            assert "ZetaOS" in result
            assert "stay weird" in result
            _check_no_foid(result, "_brand_context")

    def test_brand_name_is_zeta(self):
        with self._patch():
            assert _brand_name() == "ZetaCorp"

    def test_themes_hint_uses_zeta(self):
        with self._patch():
            result = _themes_hint()
            assert "absurdism" in result
            _check_no_foid(result, "_themes_hint")

    def test_engagement_templates_are_zeta(self):
        with self._patch():
            templates = _engagement_templates()
            for i, t in enumerate(templates):
                _check_no_foid(t, f"engagement_template[{i}]")
            # Brand name should appear in at least some templates
            assert any("ZetaCorp" in t for t in templates)
            # Visual style should appear in all
            assert all("pastel gradients" in t for t in templates)

    def test_brand_meme_templates_are_zeta(self):
        with self._patch():
            templates = _brand_meme_templates()
            for i, t in enumerate(templates):
                _check_no_foid(t, f"brand_meme_template[{i}]")
            assert any("ZetaCorp" in t for t in templates)
            assert any("ZetaOS" in t for t in templates)


# ---------------------------------------------------------------------------
# End-to-end: image_gen uses alternate brand
# ---------------------------------------------------------------------------

class TestImageGenWithAlternateBrand:
    def _patch(self):
        return patch("agent.image_gen.compositor_config.get_config", return_value=ZETA_CONFIG)

    def test_brand_terms_are_zeta(self):
        with self._patch():
            terms = _get_brand_terms()
            assert "Pastel gradient" in terms
            assert "#00ff88" in terms
            assert "charcoal" in terms.lower()
            _check_no_foid(terms, "_get_brand_terms")

    def test_negative_prompt_uses_zeta_avoid(self):
        with self._patch():
            neg = _get_negative_prompt()
            assert "flat colors" in neg
            assert "corporate blue" in neg
            _check_no_foid(neg, "_get_negative_prompt")

    def test_quality_profile_uses_zeta(self):
        with self._patch():
            profile = _get_quality_profile("announcement")
            assert "Pastel gradient" in profile
            _check_no_foid(profile, "_get_quality_profile")

    def test_enhance_prompt_injects_zeta(self):
        with self._patch():
            enhanced, neg = enhance_prompt("A floating crystal", "brand_asset")
            assert "Pastel gradient" in enhanced
            assert "#00ff88" in enhanced
            assert "flat colors" in neg
            _check_no_foid(enhanced, "enhance_prompt")


# ---------------------------------------------------------------------------
# End-to-end: asset_gen substitutions use alternate brand
# ---------------------------------------------------------------------------

class TestAssetGenWithAlternateBrand:
    def _patch(self):
        return patch("agent.asset_gen.compositor_config.get_config", return_value=ZETA_CONFIG)

    def test_brand_substitutions_are_zeta(self):
        with self._patch():
            subs = _get_brand_substitutions()
            assert "Pastel gradient" in subs["style_keywords"]
            assert "pastel gradients" in subs["style_keywords"]  # visual_style_prompt
            assert "#00ff88" in subs["colors"]
            assert "charcoal" in subs["background"]
            _check_no_foid(str(subs), "_get_brand_substitutions")

    def test_asset_prompt_uses_zeta(self):
        with self._patch():
            prompt = _build_asset_prompt("logo", "minimalist crest design")
            assert "minimalist crest design" in prompt
            assert "Pastel gradient" in prompt
            assert "#00ff88" in prompt
            _check_no_foid(prompt, "_build_asset_prompt")

    def test_banner_prompt_uses_zeta(self):
        with self._patch():
            prompt = _build_asset_prompt("banner", "launch event header")
            assert "launch event header" in prompt
            assert "charcoal" in prompt.lower()
            _check_no_foid(prompt, "_build_asset_prompt(banner)")


# ---------------------------------------------------------------------------
# End-to-end: compositor uses alternate brand
# ---------------------------------------------------------------------------

class TestCompositorWithAlternateBrand:
    def _patch(self):
        return patch("agent.compositor._brand_cfg.get_config", return_value=ZETA_CONFIG)

    def test_layout_is_zeta(self):
        with self._patch():
            layout = compositor._layout()
            assert layout.canvas_width == 1920
            assert layout.canvas_height == 1080

    def test_logo_xy_is_zeta(self):
        """ZETA_CONFIG uses default logo_padding since we didn't override."""
        with self._patch():
            lx, ly, lh = compositor._logo_xy()
            # Default padding
            assert lx == 50
            assert ly == 26

    def test_background_size_is_zeta(self):
        with self._patch():
            with patch("agent.compositor._brand_cfg.get_color_rgb") as mock_c:
                mock_c.side_effect = lambda role, fallback: (
                    ZETA_CONFIG.colors[role].rgb if role in ZETA_CONFIG.colors else fallback
                )
                profile = compositor.CompositorProfile(
                    layout="SPLIT",
                    glow_color=(0, 255, 136),
                    glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
                    glow_x_factor=1.0, glow_y_factor=1.0,
                    title_size=68, subtitle_size=21,
                    title_color=(255, 255, 255), subtitle_color=(170, 170, 170),
                    title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
                )
                bg = compositor._create_background(profile)
                assert bg.size == (1920, 1080)

    def test_layout_mappings_used(self):
        """ZETA_CONFIG has meme→engagement, lifestyle→campaign."""
        with self._patch():
            from agent.content_types import COMPOSITOR_PROFILE_MAP
            cfg = compositor._layout()
            # meme should map to "engagement" per layout_mappings
            assert cfg.layout_mappings.get("meme") == "engagement"
            # announcement has no mapping, falls through to COMPOSITOR_PROFILE_MAP
            assert cfg.layout_mappings.get("announcement") is None


# ---------------------------------------------------------------------------
# Empty BrandConfig — graceful fallbacks everywhere
# ---------------------------------------------------------------------------

EMPTY_CONFIG = BrandConfig()


class TestEmptyConfigFallbacks:
    """Verify every dynamic path degrades gracefully with an empty BrandConfig."""

    def test_scheduler_visual_style(self):
        with patch("agent.scheduler.compositor_config.get_config", return_value=EMPTY_CONFIG):
            result = _visual_style()
            assert isinstance(result, str)
            assert "Image style:" in result

    def test_scheduler_voice_summary(self):
        with patch("agent.scheduler.compositor_config.get_config", return_value=EMPTY_CONFIG):
            with patch("agent.scheduler.settings") as mock:
                mock.BRAND_NAME = "Fallback"
                result = _voice_summary()
                assert "Fallback" in result

    def test_scheduler_brand_context(self):
        with patch("agent.scheduler.compositor_config.get_config", return_value=EMPTY_CONFIG):
            result = _brand_context()
            assert result == ""

    def test_scheduler_brand_name(self):
        with patch("agent.scheduler.compositor_config.get_config", return_value=EMPTY_CONFIG):
            with patch("agent.scheduler.settings") as mock:
                mock.BRAND_NAME = "Fallback"
                assert _brand_name() == "Fallback"

    def test_scheduler_themes_hint(self):
        with patch("agent.scheduler.compositor_config.get_config", return_value=EMPTY_CONFIG):
            result = _themes_hint()
            assert "culture" in result

    def test_scheduler_engagement_templates(self):
        with patch("agent.scheduler.compositor_config.get_config", return_value=EMPTY_CONFIG):
            with patch("agent.scheduler.settings") as mock:
                mock.BRAND_NAME = "Fallback"
                templates = _engagement_templates()
                assert len(templates) == 4
                for t in templates:
                    assert isinstance(t, str)
                    assert len(t) > 30

    def test_scheduler_brand_meme_templates(self):
        with patch("agent.scheduler.compositor_config.get_config", return_value=EMPTY_CONFIG):
            with patch("agent.scheduler.settings") as mock:
                mock.BRAND_NAME = "Fallback"
                templates = _brand_meme_templates()
                assert len(templates) == 4

    def test_image_gen_brand_terms(self):
        with patch("agent.image_gen.compositor_config.get_config", return_value=EMPTY_CONFIG):
            terms = _get_brand_terms()
            assert "high quality" in terms

    def test_image_gen_negative_prompt(self):
        with patch("agent.image_gen.compositor_config.get_config", return_value=EMPTY_CONFIG):
            neg = _get_negative_prompt()
            assert "blurry" in neg

    def test_image_gen_enhance_prompt(self):
        with patch("agent.image_gen.compositor_config.get_config", return_value=EMPTY_CONFIG):
            enhanced, neg = enhance_prompt("A crystal", "announcement")
            assert "A crystal" in enhanced
            assert isinstance(neg, str)

    def test_asset_gen_substitutions(self):
        with patch("agent.asset_gen.compositor_config.get_config", return_value=EMPTY_CONFIG):
            subs = _get_brand_substitutions()
            assert "high quality" in subs["style_keywords"]
            assert subs["colors"] == ""
            assert subs["background"] == ""

    def test_asset_gen_prompt(self):
        with patch("agent.asset_gen.compositor_config.get_config", return_value=EMPTY_CONFIG):
            prompt = _build_asset_prompt("logo", "test design")
            assert "test design" in prompt

    def test_compositor_layout_defaults(self):
        with patch("agent.compositor._brand_cfg.get_config", return_value=EMPTY_CONFIG):
            lx, ly, lh = compositor._logo_xy()
            assert lx == 50
            assert ly == 26
            assert lh == 44

    def test_compositor_img_area_defaults(self):
        with patch("agent.compositor._brand_cfg.get_config", return_value=EMPTY_CONFIG):
            ix, iy, iw, ih = compositor._img_area()
            assert ix == 44
            assert iy == 90
            assert iw == 570

    def test_compositor_background_default_size(self):
        with patch("agent.compositor._brand_cfg.get_config", return_value=EMPTY_CONFIG):
            profile = compositor.CompositorProfile(
                layout="SPLIT",
                glow_color=(114, 225, 255),
                glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
                glow_x_factor=1.0, glow_y_factor=1.0,
                title_size=68, subtitle_size=21,
                title_color=(255, 255, 255), subtitle_color=(170, 170, 170),
                title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
            )
            bg = compositor._create_background(profile)
            assert bg.size == (1280, 720)
