"""
Template Generator — create branded templates from reference images.

Given a screenshot or reference image (e.g. a competitor post, a layout the user
likes), this module:
1. Analyzes the layout via Claude Vision to extract structural zones
2. Renders a clean template image using PIL with the client's brand colors,
   fonts, and logo — no AI image generation
3. Registers the result via TemplateMemory for future posts

The generated template is a *structural replica*, not a pixel copy. It captures
the layout intent (header bar, image area, text zone, logo placement) and
rebuilds it in the client's visual identity.
"""

import base64
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from agent import compositor_config as _cc
from agent.template_memory import (
    BrandTemplate,
    TemplateMemory,
    TemplateRegion,
    _get_text_font,
    _resize_crop,
)
from config import settings

logger = logging.getLogger(__name__)

_LOGO_PNG = Path(settings.BRAND_FOLDER) / "assets" / "logo.png"


# ---------------------------------------------------------------------------
# Claude Vision — layout analysis prompt
# ---------------------------------------------------------------------------

_LAYOUT_ANALYSIS_PROMPT = """\
Analyze this image as a social media post or design layout. Your job is to \
identify the *structural zones* — where different content types are placed.

Identify each zone with its purpose and approximate pixel coordinates.

Zone types:
- "background" — the dominant background area (always include one)
- "image" — area where a photo or illustration sits
- "header" — top bar or banner area
- "text_primary" — main headline or title text area
- "text_secondary" — subtitle, body text, or description area
- "logo" — brand logo placement
- "badge" — small label, tag, or category badge
- "footer" — bottom bar area
- "accent" — decorative element, border, or divider

For each zone provide:
- type (from the list above)
- x, y (top-left corner, in pixels from image top-left)
- width, height (in pixels)
- description (what's in this zone in the reference)
- style_notes (visual characteristics: color, opacity, rounded corners, etc.)

Also describe the overall layout pattern.

Return ONLY valid JSON:
{
  "canvas_width": 1280,
  "canvas_height": 720,
  "layout_pattern": "split-panel with image left, text right, header bar on top",
  "zones": [
    {
      "type": "background",
      "x": 0, "y": 0, "width": 1280, "height": 720,
      "description": "Dark navy background",
      "style_notes": "solid dark color, slight gradient"
    },
    {
      "type": "image",
      "x": 40, "y": 80, "width": 580, "height": 560,
      "description": "Main product image with rounded corners",
      "style_notes": "rounded corners ~20px, slight shadow"
    },
    {
      "type": "text_primary",
      "x": 660, "y": 200, "width": 560, "height": 80,
      "description": "Bold headline in large font",
      "style_notes": "white text, uppercase, large bold font"
    }
  ]
}

Return ONLY the JSON, no markdown formatting."""


# ---------------------------------------------------------------------------
# Layout analysis
# ---------------------------------------------------------------------------

@dataclass
class LayoutZone:
    type: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    description: str = ""
    style_notes: str = ""


@dataclass
class LayoutAnalysis:
    canvas_width: int = 1280
    canvas_height: int = 720
    layout_pattern: str = ""
    zones: list[LayoutZone] = field(default_factory=list)


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


async def analyze_layout(image_path: str) -> LayoutAnalysis:
    """Use Claude Vision to extract structural layout zones from a reference image."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": _LAYOUT_ANALYSIS_PROMPT},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = _parse_json_response(raw)
    except (json.JSONDecodeError, IndexError):
        logger.warning("Layout analysis returned non-JSON: %s", raw[:200])
        return LayoutAnalysis()

    zones = [
        LayoutZone(**z)
        for z in parsed.get("zones", [])
    ]

    return LayoutAnalysis(
        canvas_width=parsed.get("canvas_width", 1280),
        canvas_height=parsed.get("canvas_height", 720),
        layout_pattern=parsed.get("layout_pattern", ""),
        zones=zones,
    )


# ---------------------------------------------------------------------------
# PIL rendering — draw template from layout + brand config
# ---------------------------------------------------------------------------

def _load_logo(height: int) -> tuple[Image.Image, int] | None:
    """Load and resize brand logo to given height."""
    if not _LOGO_PNG.exists():
        return None
    try:
        logo = Image.open(str(_LOGO_PNG)).convert("RGBA")
        w = int(height * logo.width / logo.height)
        logo = logo.resize((w, height), Image.LANCZOS)
        return logo, w
    except Exception as e:
        logger.warning("Logo load failed: %s", e)
        return None


def _brand_color(role: str, fallback: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """Get brand color RGB by role."""
    return _cc.get_color_rgb(role, fallback)


def _brand_font(style: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a brand font by style name and size."""
    fonts_dir = Path(settings.BRAND_FOLDER) / "assets" / "fonts"
    font_map = _cc.get_font_map()
    info = font_map.get(style, font_map.get("regular", {}))
    if info:
        p = fonts_dir / info.get("filename", "")
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except (OSError, IOError):
                pass
    # System fallback
    for sys_font in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(sys_font, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_template(layout: LayoutAnalysis) -> Image.Image:
    """Render a branded template image from layout analysis using PIL.

    Each zone type maps to a specific drawing operation using brand colors.
    """
    cfg = _cc.get_config()
    cw, ch = layout.canvas_width, layout.canvas_height

    # Background
    bg_color = _brand_color("background", (10, 10, 26))
    canvas = Image.new("RGBA", (cw, ch), bg_color + (255,))
    draw = ImageDraw.Draw(canvas)

    # Sort zones: background first, then other zones
    bg_zones = [z for z in layout.zones if z.type == "background"]
    other_zones = [z for z in layout.zones if z.type != "background"]

    # Draw background zones (gradients, fills)
    for zone in bg_zones:
        _draw_background_zone(draw, canvas, zone, cfg)

    # Draw structural zones
    for zone in other_zones:
        _draw_zone(draw, canvas, zone, cfg)

    return canvas


def _draw_background_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw a background zone — fill with brand background color."""
    bg = _brand_color("background", (10, 10, 26))
    bg_alt = _brand_color("background_alt", (26, 26, 58))

    # If style mentions gradient, draw a vertical gradient
    if "gradient" in zone.style_notes.lower():
        for y in range(zone.y, zone.y + zone.height):
            t = (y - zone.y) / max(zone.height, 1)
            r = int(bg[0] + (bg_alt[0] - bg[0]) * t)
            g = int(bg[1] + (bg_alt[1] - bg[1]) * t)
            b = int(bg[2] + (bg_alt[2] - bg[2]) * t)
            draw.line([(zone.x, y), (zone.x + zone.width, y)], fill=(r, g, b, 255))
    else:
        draw.rectangle(
            [zone.x, zone.y, zone.x + zone.width, zone.y + zone.height],
            fill=bg + (255,),
        )


def _draw_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Dispatch zone drawing by type."""
    handlers = {
        "image": _draw_image_zone,
        "header": _draw_header_zone,
        "footer": _draw_footer_zone,
        "text_primary": _draw_text_primary_zone,
        "text_secondary": _draw_text_secondary_zone,
        "logo": _draw_logo_zone,
        "badge": _draw_badge_zone,
        "accent": _draw_accent_zone,
    }
    handler = handlers.get(zone.type)
    if handler:
        handler(draw, canvas, zone, cfg)


def _draw_image_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw an image placeholder — semi-transparent panel with border."""
    bg_alt = _brand_color("background_alt", (26, 26, 58))
    primary = _brand_color("primary", (255, 105, 180))
    # Glass-like panel
    overlay = Image.new("RGBA", (zone.width, zone.height), bg_alt + (40,))
    # Rounded corners via mask
    radius = min(20, zone.width // 10, zone.height // 10)
    mask = Image.new("L", (zone.width, zone.height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, zone.width, zone.height], radius=radius, fill=255)
    canvas.paste(overlay, (zone.x, zone.y), mask)
    # Border
    draw.rounded_rectangle(
        [zone.x, zone.y, zone.x + zone.width, zone.y + zone.height],
        radius=radius,
        outline=primary + (60,),
        width=2,
    )


def _draw_header_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw a header bar — semi-transparent panel."""
    bg_alt = _brand_color("background_alt", (26, 26, 58))
    overlay = Image.new("RGBA", (zone.width, zone.height), bg_alt + (80,))
    canvas.paste(overlay, (zone.x, zone.y), overlay)


def _draw_footer_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw a footer bar — similar to header."""
    bg_alt = _brand_color("background_alt", (26, 26, 58))
    overlay = Image.new("RGBA", (zone.width, zone.height), bg_alt + (60,))
    canvas.paste(overlay, (zone.x, zone.y), overlay)


def _draw_text_primary_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw a primary text placeholder — uppercase brand font label."""
    text_color = _brand_color("text", (255, 255, 255))
    font_size = max(16, min(zone.height // 2, 60))
    font = _brand_font("bold", font_size)
    label = cfg.brand_name.upper() if cfg.brand_name else "HEADLINE"
    draw.text(
        (zone.x + 4, zone.y + (zone.height - font_size) // 2),
        label,
        fill=text_color + (100,),
        font=font,
    )


def _draw_text_secondary_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw a secondary text placeholder — lighter, smaller font."""
    text_color = _brand_color("text", (255, 255, 255))
    font_size = max(12, min(zone.height // 2, 28))
    font = _brand_font("regular", font_size)
    label = cfg.tagline if cfg.tagline else "subtitle text"
    draw.text(
        (zone.x + 4, zone.y + (zone.height - font_size) // 2),
        label,
        fill=text_color + (70,),
        font=font,
    )


def _draw_logo_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Place the brand logo in the zone."""
    logo_height = min(zone.height, 60)
    result = _load_logo(logo_height)
    if result:
        logo_img, logo_w = result
        # Center in zone
        lx = zone.x + (zone.width - logo_w) // 2
        ly = zone.y + (zone.height - logo_height) // 2
        canvas.paste(logo_img, (lx, ly), logo_img)
    else:
        # Fallback: draw a placeholder circle
        primary = _brand_color("primary", (255, 105, 180))
        cx = zone.x + zone.width // 2
        cy = zone.y + zone.height // 2
        r = min(zone.width, zone.height) // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=primary + (80,))


def _draw_badge_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw a badge/tag placeholder — rounded pill with brand color."""
    primary = _brand_color("primary", (255, 105, 180))
    radius = min(zone.height // 2, 12)
    draw.rounded_rectangle(
        [zone.x, zone.y, zone.x + zone.width, zone.y + zone.height],
        radius=radius,
        fill=primary + (50,),
        outline=primary + (120,),
        width=1,
    )


def _draw_accent_zone(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    zone: LayoutZone,
    cfg,
) -> None:
    """Draw a decorative accent — line or glow element."""
    primary = _brand_color("primary", (255, 105, 180))
    # Thin line or bar
    if zone.width > zone.height * 3:
        # Horizontal accent line
        y_mid = zone.y + zone.height // 2
        draw.line(
            [(zone.x, y_mid), (zone.x + zone.width, y_mid)],
            fill=primary + (100,),
            width=max(2, zone.height // 3),
        )
    elif zone.height > zone.width * 3:
        # Vertical accent line
        x_mid = zone.x + zone.width // 2
        draw.line(
            [(x_mid, zone.y), (x_mid, zone.y + zone.height)],
            fill=primary + (100,),
            width=max(2, zone.width // 3),
        )
    else:
        # Square-ish accent — subtle glow
        draw.rounded_rectangle(
            [zone.x, zone.y, zone.x + zone.width, zone.y + zone.height],
            radius=8,
            fill=primary + (30,),
        )


# ---------------------------------------------------------------------------
# Layout serialization — convert to/from dicts for context storage
# ---------------------------------------------------------------------------

def layout_to_dict(layout: LayoutAnalysis) -> dict:
    """Serialize LayoutAnalysis to a JSON-safe dict for storing in user_data."""
    return {
        "canvas_width": layout.canvas_width,
        "canvas_height": layout.canvas_height,
        "layout_pattern": layout.layout_pattern,
        "zones": [
            {
                "type": z.type, "x": z.x, "y": z.y,
                "width": z.width, "height": z.height,
                "description": z.description, "style_notes": z.style_notes,
            }
            for z in layout.zones
        ],
    }


def layout_from_dict(d: dict) -> LayoutAnalysis:
    """Deserialize a dict back into LayoutAnalysis."""
    return LayoutAnalysis(
        canvas_width=d.get("canvas_width", 1280),
        canvas_height=d.get("canvas_height", 720),
        layout_pattern=d.get("layout_pattern", ""),
        zones=[LayoutZone(**z) for z in d.get("zones", [])],
    )


# ---------------------------------------------------------------------------
# Layout adjustment — Claude modifies zones based on user feedback
# ---------------------------------------------------------------------------

_ADJUST_PROMPT = """\
You are adjusting a template layout based on user feedback.

Current layout (JSON):
{layout_json}

User feedback: "{feedback}"

Apply the user's requested changes to the layout. You may:
- Move zones (change x, y)
- Resize zones (change width, height)
- Add new zones
- Remove zones
- Change zone types
- Modify canvas dimensions

Return the COMPLETE updated layout as valid JSON with the same structure.
Return ONLY the JSON, no markdown formatting."""


async def adjust_layout(layout: LayoutAnalysis, feedback: str) -> LayoutAnalysis:
    """Use Claude to adjust a layout based on user feedback text."""
    layout_json = json.dumps(layout_to_dict(layout), indent=2)
    prompt = _ADJUST_PROMPT.format(layout_json=layout_json, feedback=feedback)

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = _parse_json_response(raw)
    except (json.JSONDecodeError, IndexError):
        logger.warning("Layout adjustment returned non-JSON: %s", raw[:200])
        return layout  # Return unchanged on failure

    zones = [LayoutZone(**z) for z in parsed.get("zones", [])]
    if not zones:
        return layout  # Don't wipe zones on bad response

    return LayoutAnalysis(
        canvas_width=parsed.get("canvas_width", layout.canvas_width),
        canvas_height=parsed.get("canvas_height", layout.canvas_height),
        layout_pattern=parsed.get("layout_pattern", layout.layout_pattern),
        zones=zones,
    )


# ---------------------------------------------------------------------------
# Helpers — region building and aspect ratio
# ---------------------------------------------------------------------------

def _build_regions(layout: LayoutAnalysis) -> list[TemplateRegion]:
    """Map layout zones to TemplateMemory regions."""
    regions = []
    for z in layout.zones:
        if z.type == "image":
            regions.append(TemplateRegion(
                type="image", x=z.x, y=z.y,
                width=z.width, height=z.height,
                description=z.description,
            ))
        elif z.type in ("text_primary", "text_secondary"):
            regions.append(TemplateRegion(
                type="text", x=z.x, y=z.y,
                width=z.width, height=z.height,
                description=z.description,
            ))
        elif z.type == "logo":
            regions.append(TemplateRegion(
                type="logo", x=z.x, y=z.y,
                width=z.width, height=z.height,
                description=z.description,
            ))
    return regions


def _compute_aspect_ratio(width: int, height: int) -> str:
    """Compute a human-readable aspect ratio string."""
    if not width or not height:
        return ""
    ratio = width / height
    if abs(ratio - 16 / 9) < 0.1:
        return "16:9"
    if abs(ratio - 9 / 16) < 0.1:
        return "9:16"
    if abs(ratio - 1.0) < 0.1:
        return "1:1"
    if abs(ratio - 4 / 3) < 0.1:
        return "4:3"
    return f"{width}:{height}"


# ---------------------------------------------------------------------------
# Template generation — split into analyze+render and register phases
# ---------------------------------------------------------------------------

async def analyze_and_render(image_path: str) -> tuple[LayoutAnalysis, Image.Image]:
    """Phase 1: Analyze reference image and render a branded template preview.

    Returns (layout, rendered_image) without registering anything.
    """
    layout = await analyze_layout(image_path)
    if not layout.zones:
        raise ValueError("Could not detect any layout zones in the reference image.")

    logger.info(
        "Layout analysis: %dx%d, %d zones, pattern: %s",
        layout.canvas_width, layout.canvas_height,
        len(layout.zones), layout.layout_pattern,
    )

    template_img = render_template(layout)
    return layout, template_img


def render_and_save(layout: LayoutAnalysis) -> tuple[Image.Image, str]:
    """Re-render a layout and save to disk. Returns (image, path)."""
    template_img = render_template(layout)
    templates_dir = Path(settings.BRAND_FOLDER) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    save_path = templates_dir / f"preview_{uuid.uuid4().hex[:8]}.png"
    template_img.convert("RGB").save(str(save_path), "PNG")
    return template_img, str(save_path)


def register_layout(
    layout: LayoutAnalysis,
    image_path: str,
    name: str | None = None,
) -> BrandTemplate:
    """Phase 2: Register a finalized layout as a template in TemplateMemory."""
    tid = str(uuid.uuid4())[:8]
    cw, ch = layout.canvas_width, layout.canvas_height
    template_name = name or f"Generated {tid}"

    template = BrandTemplate(
        id=tid,
        name=template_name,
        path=image_path,
        width=cw,
        height=ch,
        regions=_build_regions(layout),
        aspect_ratio=_compute_aspect_ratio(cw, ch),
        content_types=[],  # Universal
        analysis_notes=f"Generated from reference. Layout: {layout.layout_pattern}",
    )

    memory = TemplateMemory()
    memory.add_template(template)

    logger.info(
        "Registered template '%s' (%dx%d, %d regions) from reference",
        template_name, cw, ch, len(template.regions),
    )
    return template


async def generate_template_from_reference(
    image_path: str,
    name: str | None = None,
) -> tuple[BrandTemplate, Image.Image]:
    """Full pipeline: analyze, render, register. Used by non-interactive callers."""
    layout, template_img = await analyze_and_render(image_path)

    templates_dir = Path(settings.BRAND_FOLDER) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    tid = str(uuid.uuid4())[:8]
    template_path = templates_dir / f"generated_{tid}.png"
    template_img.convert("RGB").save(str(template_path), "PNG")

    template = register_layout(layout, str(template_path), name)
    return template, template_img
