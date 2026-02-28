"""Tests for agent.compositor_config — brand guidelines parsing."""

from agent.compositor_config import (
    BrandConfig,
    _parse_colors,
    _parse_fonts,
    _parse_identity,
    _parse_style_keywords,
    _parse_avoid_terms,
    _parse_visual_effects,
)


SAMPLE_GUIDELINES = """\
# Brand Guidelines

**Brand Name:** TestBrand
**Tagline:** Build the future
**Website:** https://testbrand.io
**X Handle:** @testbrand

## COLOR PALETTE

| Role       | Name         | Hex     | RGB             |
|------------|-------------|---------|-----------------|
| Primary    | Electric Blue | #0066ff | (0, 102, 255)  |
| Accent 1   | Hot Pink     | #ff0066 | (255, 0, 102)  |
| Background | Dark Navy    | #0a0a1a | (10, 10, 26)   |

## TYPOGRAPHY

| Use              | Font       | Weight          | Style   |
|-----------------|------------|-----------------|---------|
| Display / Headlines | Orbitron | Bold           |         |
| Body            | Inter      | Regular/Medium  |         |

## ILLUSTRATION STYLE

- **Futuristic 3D renders** with metallic sheen
- **Vibrant neon glow** effects
- Avoid: flat clip-art, low resolution, stock photos

## VISUAL EFFECTS

| Effect         | Value        |
|---------------|-------------|
| Glass opacity  | 8           |
| Glass blur     | 16          |
| Glass radius   | 32          |
| Glass inset    | 50, 80, 50, 40 |
| Orb alpha      | 20          |
| Orb count      | 5           |
| Noise opacity  | 3           |
"""


def test_parse_colors():
    colors = _parse_colors(SAMPLE_GUIDELINES)
    assert "primary" in colors
    assert colors["primary"].hex == "#0066ff"
    assert colors["primary"].rgb == (0, 102, 255)
    assert "accent_1" in colors
    assert "background" in colors


def test_parse_fonts():
    fonts = _parse_fonts(SAMPLE_GUIDELINES)
    assert "display" in fonts
    assert fonts["display"].family == "Orbitron"
    assert "body" in fonts
    assert fonts["body"].family == "Inter"


def test_parse_identity():
    identity = _parse_identity(SAMPLE_GUIDELINES)
    assert identity["brand_name"] == "TestBrand"
    assert identity["tagline"] == "Build the future"
    assert identity["website"] == "https://testbrand.io"
    assert identity["x_handle"] == "@testbrand"


def test_parse_style_keywords():
    keywords = _parse_style_keywords(SAMPLE_GUIDELINES)
    assert "Futuristic 3D renders" in keywords
    assert "Vibrant neon glow" in keywords


def test_parse_avoid_terms():
    avoid = _parse_avoid_terms(SAMPLE_GUIDELINES)
    assert "flat clip-art" in avoid
    assert "low resolution" in avoid
    assert "stock photos" in avoid


def test_parse_visual_effects():
    effects = _parse_visual_effects(SAMPLE_GUIDELINES)
    assert effects["glass_opacity"] == 8
    assert effects["glass_blur"] == 16
    assert effects["glass_radius"] == 32
    assert effects["glass_inset"] == (50, 80, 50, 40)
    assert effects["orb_alpha_base"] == 20
    assert effects["orb_count"] == 5
    assert effects["noise_opacity"] == 3


def test_parse_visual_effects_missing():
    text = "# Brand\nNo effects section here."
    effects = _parse_visual_effects(text)
    assert effects == {}


def test_brand_config_defaults():
    cfg = BrandConfig()
    assert cfg.glass_opacity == 6
    assert cfg.glass_blur == 12
    assert cfg.orb_count == 7
    assert cfg.glass_inset == (40, 70, 40, 30)
