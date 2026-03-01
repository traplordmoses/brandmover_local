"""Tests for agent.compositor_config — brand guidelines parsing."""

from agent.compositor_config import (
    BrandConfig,
    _parse_colors,
    _parse_fonts,
    _parse_identity,
    _parse_style_keywords,
    _parse_avoid_terms,
    _parse_visual_effects,
    _parse_product,
    _parse_voice_traits,
    _parse_visual_style_prompt,
    _parse_brand_phrases,
    _parse_themes,
    _parse_layout_profiles,
    _parse_layout_mappings,
    _parse_compositor_section,
)


SAMPLE_GUIDELINES = """\
# Brand Guidelines

**Brand Name:** TestBrand
**Tagline:** Build the future
**Website:** https://testbrand.io
**X Handle:** @testbrand
**Product:** A platform for collaborative design — Canvas (real-time whiteboard) and Toolkit (component library).

**Key Brand Themes:** Innovation, collaboration, open source, developer experience

## VOICE & TONE

**Core personality traits:**
- Bold and direct — no fluff
- Technical but approachable — speaks developer
- Optimistic without hype — grounded enthusiasm

**Writing style:**
- Short, punchy sentences.

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

**Image generation prompt guidance:**
- "dark background, neon blue glow, glass morphism panels, futuristic aesthetic"
- "deep navy background, cyan neon highlights"
- Avoid: flat clip-art, low resolution, stock photos

## BRAND PHRASES & SLANG

**Established phrases:**
- "build the future, together" — primary tagline
- "ship it." — developer ethos
- "code is craft." — quality philosophy

**Community slang:**
- "ship" — deploy to production

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

## LAYOUT PROFILES

| Setting              | Value     |
|----------------------|-----------|
| Canvas width         | 1920      |
| Canvas height        | 1080      |
| Logo position        | top-right |
| Logo padding         | 60, 30    |
| Logo height          | 52        |
| Image x              | 50        |
| Image y              | 100       |
| Image width          | 800       |
| Image bottom margin  | 44        |

## LAYOUT MAPPINGS

| Content Type       | Profile       |
|--------------------|---------------|
| announcement       | campaign      |
| meme               | engagement    |
| lifestyle          | announcement  |

## COMPOSITOR

| Setting        | Value          |
|----------------|----------------|
| Enabled        | true           |
| Badge text     | WEB            |
| Default mode   | image_always   |
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


def test_parse_product():
    product = _parse_product(SAMPLE_GUIDELINES)
    assert "platform for collaborative design" in product
    assert "Canvas" in product
    assert "Toolkit" in product


def test_parse_voice_traits():
    traits = _parse_voice_traits(SAMPLE_GUIDELINES)
    assert len(traits) == 3
    assert "Bold and direct" in traits[0]
    assert "Technical but approachable" in traits[1]
    assert "Optimistic without hype" in traits[2]


def test_parse_visual_style_prompt():
    prompt = _parse_visual_style_prompt(SAMPLE_GUIDELINES)
    assert "dark background" in prompt
    assert "neon blue glow" in prompt
    assert "futuristic aesthetic" in prompt


def test_parse_brand_phrases():
    phrases = _parse_brand_phrases(SAMPLE_GUIDELINES)
    assert len(phrases) >= 3
    assert "build the future, together" in phrases
    assert "ship it." in phrases
    assert "code is craft." in phrases


def test_parse_themes():
    themes = _parse_themes(SAMPLE_GUIDELINES)
    assert "Innovation" in themes
    assert "collaboration" in themes
    assert "open source" in themes
    assert "developer experience" in themes


def test_parse_product_missing():
    text = "# Brand\nNo product line here."
    assert _parse_product(text) == ""


def test_parse_voice_traits_missing():
    text = "# Brand\nNo voice section."
    assert _parse_voice_traits(text) == []


def test_parse_layout_profiles():
    layout = _parse_layout_profiles(SAMPLE_GUIDELINES)
    assert layout["canvas_width"] == 1920
    assert layout["canvas_height"] == 1080
    assert layout["logo_position"] == "top-right"
    assert layout["logo_padding"] == (60, 30)
    assert layout["logo_height"] == 52
    assert layout["image_x"] == 50
    assert layout["image_y"] == 100
    assert layout["image_width"] == 800
    assert layout["image_bottom_margin"] == 44


def test_parse_layout_profiles_missing():
    text = "# Brand\nNo layout section."
    assert _parse_layout_profiles(text) == {}


def test_parse_layout_profiles_partial():
    text = """\
## LAYOUT PROFILES

| Setting        | Value |
|----------------|-------|
| Canvas width   | 1600  |
| Logo height    | 48    |
"""
    layout = _parse_layout_profiles(text)
    assert layout["canvas_width"] == 1600
    assert layout["logo_height"] == 48
    assert "canvas_height" not in layout
    assert "logo_padding" not in layout


def test_parse_layout_mappings():
    mappings = _parse_layout_mappings(SAMPLE_GUIDELINES)
    assert mappings["announcement"] == "campaign"
    assert mappings["meme"] == "engagement"
    assert mappings["lifestyle"] == "announcement"


def test_parse_layout_mappings_missing():
    text = "# Brand\nNo layout mappings."
    assert _parse_layout_mappings(text) == {}


def test_brand_config_defaults():
    cfg = BrandConfig()
    assert cfg.glass_opacity == 6
    assert cfg.glass_blur == 12
    assert cfg.orb_count == 7
    assert cfg.glass_inset == (40, 70, 40, 30)
    assert cfg.product_description == ""
    assert cfg.voice_traits == []
    assert cfg.visual_style_prompt == ""
    assert cfg.brand_phrases == []
    assert cfg.content_themes == []
    assert cfg.canvas_width == 1280
    assert cfg.canvas_height == 720
    assert cfg.logo_position == "top-left"
    assert cfg.logo_padding == (50, 26)
    assert cfg.logo_height == 44
    assert cfg.image_x == 44
    assert cfg.image_y == 90
    assert cfg.image_width == 570
    assert cfg.image_bottom_margin == 38
    assert cfg.layout_mappings == {}
    # v8 compositor defaults
    assert cfg.compositor_enabled is True
    assert cfg.badge_text is None
    assert cfg.default_mode == "image_optional"


# ---------------------------------------------------------------------------
# COMPOSITOR section parsing (v8)
# ---------------------------------------------------------------------------


def test_parse_compositor_section():
    result = _parse_compositor_section(SAMPLE_GUIDELINES)
    assert result["compositor_enabled"] is True
    assert result["badge_text"] == "WEB"
    assert result["default_mode"] == "image_always"


def test_parse_compositor_disabled():
    text = """\
## COMPOSITOR

| Setting        | Value          |
|----------------|----------------|
| Enabled        | false          |
| Badge text     | none           |
| Default mode   | text_only      |
"""
    result = _parse_compositor_section(text)
    assert result["compositor_enabled"] is False
    assert result["badge_text"] is None
    assert result["default_mode"] == "text_only"


def test_parse_compositor_missing():
    text = "# Brand\nNo compositor section here."
    result = _parse_compositor_section(text)
    assert result == {}


def test_parse_compositor_empty_badge():
    text = """\
## COMPOSITOR

| Setting        | Value          |
|----------------|----------------|
| Enabled        | true           |
| Badge text     |                |
| Default mode   | image_optional |
"""
    result = _parse_compositor_section(text)
    assert result["compositor_enabled"] is True
    assert result["badge_text"] is None
    assert result["default_mode"] == "image_optional"


def test_parse_compositor_partial():
    text = """\
## COMPOSITOR

| Setting        | Value   |
|----------------|---------|
| Enabled        | false   |
"""
    result = _parse_compositor_section(text)
    assert result["compositor_enabled"] is False
    assert "badge_text" not in result
    assert "default_mode" not in result
