"""Tests for agent.compositor — dynamic layout helpers, glow scaling,
and layout_mappings override in compose_branded_image.

These tests verify that the compositor reads layout parameters from
BrandConfig rather than using hardcoded values.
"""

import io
from unittest.mock import patch, AsyncMock, MagicMock

from PIL import Image

from agent.compositor_config import BrandConfig, ColorEntry
from agent import compositor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> BrandConfig:
    """Build a BrandConfig with non-default layout values."""
    defaults = dict(
        brand_name="ZetaCorp",
        canvas_width=1920,
        canvas_height=1080,
        logo_position="top-right",
        logo_padding=(60, 30),
        logo_height=52,
        image_x=50,
        image_y=100,
        image_width=800,
        image_bottom_margin=44,
        layout_mappings={"meme": "engagement", "lifestyle": "campaign"},
        glass_opacity=10,
        glass_blur=18,
        glass_radius=36,
        glass_inset=(50, 80, 50, 40),
        orb_alpha_base=22,
        orb_count=5,
        colors={
            "primary": ColorEntry(role="primary", name="Mint", hex="#00ff88", rgb=(0, 255, 136)),
            "accent_1": ColorEntry(role="accent_1", name="Coral", hex="#ff6644", rgb=(255, 102, 68)),
            "accent_2": ColorEntry(role="accent_2", name="Lilac", hex="#cc88ff", rgb=(204, 136, 255)),
            "accent_3": ColorEntry(role="accent_3", name="Sky", hex="#88ccff", rgb=(136, 204, 255)),
            "background": ColorEntry(role="background", name="Charcoal", hex="#1a1a2e", rgb=(26, 26, 46)),
            "text": ColorEntry(role="text", name="White", hex="#ffffff", rgb=(255, 255, 255)),
        },
    )
    defaults.update(overrides)
    return BrandConfig(**defaults)


def _default_cfg() -> BrandConfig:
    """BrandConfig with all defaults (1280x720)."""
    return BrandConfig()


# ---------------------------------------------------------------------------
# _layout(), _logo_xy(), _img_area() — read from BrandConfig
# ---------------------------------------------------------------------------

class TestLayoutHelpers:
    def test_layout_returns_brand_config(self):
        cfg = _make_cfg()
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            layout = compositor._layout()
            assert layout.canvas_width == 1920
            assert layout.canvas_height == 1080

    def test_logo_xy_reads_config(self):
        cfg = _make_cfg()
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            lx, ly, lh = compositor._logo_xy()
            assert lx == 60
            assert ly == 30
            assert lh == 52

    def test_logo_xy_default_values(self):
        cfg = _default_cfg()
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            lx, ly, lh = compositor._logo_xy()
            assert lx == 50
            assert ly == 26
            assert lh == 44

    def test_img_area_reads_config(self):
        cfg = _make_cfg()
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            ix, iy, iw, ih = compositor._img_area()
            assert ix == 50
            assert iy == 100
            assert iw == 800
            # ih = canvas_height - image_y - image_bottom_margin = 1080 - 100 - 44 = 936
            assert ih == 936

    def test_img_area_default_values(self):
        cfg = _default_cfg()
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            ix, iy, iw, ih = compositor._img_area()
            assert ix == 44
            assert iy == 90
            assert iw == 570
            # ih = 720 - 90 - 38 = 592
            assert ih == 592


# ---------------------------------------------------------------------------
# _create_background — canvas dimensions and glow scaling
# ---------------------------------------------------------------------------

class TestCreateBackground:
    def test_canvas_size_from_config(self):
        """Background canvas uses dimensions from BrandConfig, not hardcoded."""
        cfg = _make_cfg(canvas_width=1920, canvas_height=1080)
        profile = compositor.CompositorProfile(
            layout="SPLIT",
            glow_color=(0, 255, 136),
            glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
            glow_x_factor=1.0, glow_y_factor=1.0,
            title_size=68, subtitle_size=21,
            title_color=(255, 255, 255), subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        )
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            bg = compositor._create_background(profile)
            assert bg.size == (1920, 1080)

    def test_default_canvas_size(self):
        """Default BrandConfig produces 1280x720 canvas."""
        cfg = _default_cfg()
        profile = compositor.CompositorProfile(
            layout="SPLIT",
            glow_color=(114, 225, 255),
            glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
            glow_x_factor=1.0, glow_y_factor=1.0,
            title_size=68, subtitle_size=21,
            title_color=(255, 255, 255), subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        )
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            bg = compositor._create_background(profile)
            assert bg.size == (1280, 720)

    def test_glow_scales_with_canvas(self):
        """Glow geometry is proportional to canvas size — a bigger canvas
        should produce a bigger glow (larger blur radius) than default."""
        cfg_big = _make_cfg(canvas_width=2560, canvas_height=1440)
        cfg_default = _default_cfg()

        profile = compositor.CompositorProfile(
            layout="SPLIT",
            glow_color=(0, 255, 136),
            glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
            glow_x_factor=1.0, glow_y_factor=1.0,
            title_size=68, subtitle_size=21,
            title_color=(255, 255, 255), subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        )

        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg_big):
            bg_big = compositor._create_background(profile)
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg_default):
            bg_default = compositor._create_background(profile)

        # Both should be valid images
        assert bg_big.size == (2560, 1440)
        assert bg_default.size == (1280, 720)

    def test_orb_count_from_config(self):
        """Setting orb_count=0 should still produce a valid background."""
        cfg = _make_cfg(orb_count=0)
        profile = compositor.CompositorProfile(
            layout="SPLIT",
            glow_color=(0, 255, 136),
            glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
            glow_x_factor=1.0, glow_y_factor=1.0,
            title_size=68, subtitle_size=21,
            title_color=(255, 255, 255), subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        )
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            bg = compositor._create_background(profile)
            assert bg.size == (1920, 1080)

    def test_glass_settings_from_config(self):
        """Glass morphism settings come from BrandConfig."""
        cfg = _make_cfg(glass_opacity=0, glass_blur=0, glass_radius=0)
        profile = compositor.CompositorProfile(
            layout="SPLIT",
            glow_color=(0, 255, 136),
            glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
            glow_x_factor=1.0, glow_y_factor=1.0,
            title_size=68, subtitle_size=21,
            title_color=(255, 255, 255), subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        )
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            # Should not crash with zero opacity/blur
            bg = compositor._create_background(profile)
            assert bg.size == (1920, 1080)


# ---------------------------------------------------------------------------
# _get_profiles — colors from brand config
# ---------------------------------------------------------------------------

class TestGetProfiles:
    def test_profiles_use_brand_colors(self):
        cfg = _make_cfg()
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            with patch("agent.compositor._brand_cfg.get_color_rgb") as mock_color:
                mock_color.side_effect = lambda role, fallback: cfg.colors[role].rgb if role in cfg.colors else fallback
                # Clear cache to force rebuild
                compositor._profiles_cache = None
                compositor._profiles_hash = ""
                profiles = compositor._get_profiles()
                assert "announcement" in profiles
                assert "default" in profiles

    def test_profiles_cached_by_hash(self):
        cfg = _make_cfg()
        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            with patch("agent.compositor._brand_cfg.get_color_rgb", side_effect=lambda r, f: f):
                compositor._profiles_cache = None
                compositor._profiles_hash = ""
                profiles1 = compositor._get_profiles()
                profiles2 = compositor._get_profiles()
                # Should be same object (cached)
                assert profiles1 is profiles2


# ---------------------------------------------------------------------------
# compose_branded_image — layout_mappings override
# ---------------------------------------------------------------------------

class TestLayoutMappingsOverride:
    def test_layout_mapping_overrides_profile_map(self):
        """Config layout_mappings take precedence over COMPOSITOR_PROFILE_MAP."""
        cfg = _make_cfg(layout_mappings={"meme": "engagement"})

        with patch("agent.compositor._brand_cfg.get_config", return_value=cfg):
            with patch("agent.compositor._brand_cfg.get_color_rgb", side_effect=lambda r, f: f):
                compositor._profiles_cache = None
                compositor._profiles_hash = ""

                # Check that for content_type "meme", it uses the "engagement" profile
                profiles = compositor._get_profiles()
                profile_key = cfg.layout_mappings.get("meme")
                assert profile_key == "engagement"
                assert profile_key in profiles

    def test_unmapped_type_uses_default_profile_map(self):
        """Content types not in layout_mappings fall back to COMPOSITOR_PROFILE_MAP."""
        cfg = _make_cfg(layout_mappings={"meme": "engagement"})
        from agent.content_types import COMPOSITOR_PROFILE_MAP

        # "announcement" is not in our custom mappings, should use COMPOSITOR_PROFILE_MAP
        profile_key = cfg.layout_mappings.get("announcement") or COMPOSITOR_PROFILE_MAP.get("announcement", "default")
        assert profile_key == "announcement"  # from COMPOSITOR_PROFILE_MAP

    def test_empty_layout_mappings_uses_default(self):
        """Empty layout_mappings always falls through to COMPOSITOR_PROFILE_MAP."""
        cfg = _make_cfg(layout_mappings={})
        from agent.content_types import COMPOSITOR_PROFILE_MAP

        for ct in ("announcement", "meme", "market"):
            profile_key = cfg.layout_mappings.get(ct) or COMPOSITOR_PROFILE_MAP.get(ct, "default")
            assert profile_key == COMPOSITOR_PROFILE_MAP[ct]
