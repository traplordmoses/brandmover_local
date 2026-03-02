"""
Template Generator — create branded templates from reference images.

Given a screenshot or reference image, this module:
1. Analyzes the layout via Claude Vision to extract a precise TemplateSpec
2. Applies brand overrides (colors, fonts) to the spec
3. Renders the template frame deterministically with PIL (exact colors, shapes)
4. Supports interactive refinement via conversation (instant re-renders)
5. Registers the result via TemplateMemory for future posts

No AI image generation is used for template frames — pure PIL rendering.
"""

import base64
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx
from PIL import Image

from agent import compositor_config as _cc
from agent.template_memory import (
    BrandTemplate,
    TemplateMemory,
    TemplateRegion,
)
from agent.template_spec import (
    TemplateSpec,
    spec_to_dict,
    spec_from_dict,
)
from agent.template_renderer import (
    render_preview,
    render_template_frame,
    save_frame,
)
from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude Vision — TemplateSpec extraction
# ---------------------------------------------------------------------------

_SPEC_ANALYSIS_PROMPT = """\
Analyze this image as a social media post or design layout. Extract a precise \
visual specification that can be rendered deterministically with PIL.

You MUST return ONLY valid JSON with this exact structure:

{
  "layout_description": "A natural language description of the layout...",
  "visual_style": "glass-morphism with neon accents on dark background",
  "spec": {
    "canvas_width": 1280,
    "canvas_height": 720,
    "background": {
      "type": "solid",
      "color": "#0E0F2B"
    },
    "shapes": [
      {
        "shape": "rounded_rect",
        "x": 30, "y": 30, "width": 1220, "height": 660,
        "fill": {"type": "solid", "color": "#FFFFFF"},
        "border": {"color": "#72E1FF", "width": 1, "radius": 0},
        "corner_radius": 20,
        "opacity": 0.1,
        "z_order": 1,
        "x2": 0, "y2": 0, "line_width": 2
      }
    ],
    "text_zones": [
      {
        "x": 660, "y": 200, "width": 560, "height": 80,
        "label": "title",
        "font_family": "sans-serif",
        "font_size": 48,
        "font_weight": "bold",
        "color": "#FFFFFF",
        "alignment": "left",
        "uppercase": true,
        "outline_color": "",
        "outline_width": 0,
        "description": "Headline text area"
      }
    ],
    "image_zones": [
      {
        "x": 40, "y": 80, "width": 580, "height": 560,
        "corner_radius": 16,
        "description": "Main image area"
      }
    ],
    "logo_zones": [
      {
        "x": 50, "y": 20, "width": 100, "height": 50,
        "description": "Logo placement top-left"
      }
    ]
  }
}

Guidelines for the spec:
- **background**: Use "solid" with a hex color, or "linear_gradient"/"radial_gradient" \
with stops array [{"offset": 0.0, "color": "#hex"}, ...] and angle (degrees, 0=top-to-bottom).
- **shapes**: Decorative elements — panels, borders, accent lines, overlays. \
Use shape types: "rect", "rounded_rect", "ellipse", "line". \
For lines, set x,y as start and x2,y2 as end. \
Use opacity 0.0-1.0 for semi-transparent overlays. \
z_order controls draw order (higher = on top).
- **text_zones**: Where text will be placed at composition time. \
Detect font characteristics (serif/sans-serif, weight, size relative to canvas).
- **image_zones**: Where generated images will be composited. These become \
transparent cutouts in the final frame. Include corner_radius if rounded.
- **logo_zones**: Where the brand logo will be placed.

Measure all coordinates in pixels. Estimate the canvas dimensions from the image.

Return ONLY the JSON, no markdown formatting."""


# ---------------------------------------------------------------------------
# Legacy analysis prompt (kept for backward compat with old tests)
# ---------------------------------------------------------------------------

_LAYOUT_ANALYSIS_PROMPT = _SPEC_ANALYSIS_PROMPT


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TemplateDesign:
    layout_description: str = ""
    visual_style: str = ""
    generation_prompt: str = ""
    reference_image_path: str = ""
    generated_image_url: str = ""
    canvas_width: int = 1280
    canvas_height: int = 720
    regions: list[TemplateRegion] = field(default_factory=list)
    spec: TemplateSpec | None = None


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


# ---------------------------------------------------------------------------
# Phase 1: Analyze reference image → TemplateSpec
# ---------------------------------------------------------------------------

async def analyze_reference(image_path: str) -> TemplateDesign:
    """Use Claude Vision to extract a TemplateSpec from a reference image."""
    data, media_type = _encode_image(image_path)

    from agent._client import get_anthropic
    client = get_anthropic()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": _SPEC_ANALYSIS_PROMPT},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = _parse_json_response(raw)
    except (json.JSONDecodeError, IndexError):
        logger.warning("Spec analysis returned non-JSON: %s", raw[:200])
        return TemplateDesign(reference_image_path=image_path)

    # Build TemplateSpec from response
    spec_dict = parsed.get("spec")
    spec = spec_from_dict(spec_dict) if spec_dict else None

    # Extract regions from spec for backward compat
    regions = _spec_to_regions(spec) if spec else [
        TemplateRegion(**r) for r in parsed.get("regions", [])
    ]

    cw = spec.canvas_width if spec else parsed.get("canvas_width", 1280)
    ch = spec.canvas_height if spec else parsed.get("canvas_height", 720)

    return TemplateDesign(
        layout_description=parsed.get("layout_description", ""),
        visual_style=parsed.get("visual_style", ""),
        reference_image_path=image_path,
        canvas_width=cw,
        canvas_height=ch,
        regions=regions,
        spec=spec,
    )


def _spec_to_regions(spec: TemplateSpec) -> list[TemplateRegion]:
    """Convert TemplateSpec zones to legacy TemplateRegion list."""
    regions: list[TemplateRegion] = []
    for iz in spec.image_zones:
        regions.append(TemplateRegion(
            type="image", x=iz.x, y=iz.y, width=iz.width, height=iz.height,
            description=iz.description,
        ))
    for tz in spec.text_zones:
        regions.append(TemplateRegion(
            type="text", x=tz.x, y=tz.y, width=tz.width, height=tz.height,
            description=tz.description,
            font_family=tz.font_family, font_size=tz.font_size,
            font_weight=tz.font_weight, color=tz.color,
            alignment=tz.alignment, uppercase=tz.uppercase,
            outline_color=tz.outline_color, outline_width=tz.outline_width,
        ))
    for lz in spec.logo_zones:
        regions.append(TemplateRegion(
            type="logo", x=lz.x, y=lz.y, width=lz.width, height=lz.height,
            description=lz.description,
        ))
    return regions


# ---------------------------------------------------------------------------
# Brand overrides — replace detected colors with brand palette
# ---------------------------------------------------------------------------

def _apply_brand_overrides(spec: TemplateSpec) -> TemplateSpec:
    """Replace detected colors with brand palette colors."""
    cfg = _cc.get_config()

    # Map brand roles to colors
    bg_color = cfg.colors.get("background")
    text_color = cfg.colors.get("text")
    primary_color = cfg.colors.get("primary")
    display_font = cfg.fonts.get("display")
    body_font = cfg.fonts.get("body") or cfg.fonts.get("body_text")

    # Override background
    if bg_color and spec.background.type == "solid":
        spec.background.color = bg_color.hex

    # Override shape borders with brand primary
    if primary_color:
        for shape in spec.shapes:
            if shape.border:
                shape.border.color = primary_color.hex

    # Override text zones
    for tz in spec.text_zones:
        if text_color:
            tz.color = text_color.hex
        if tz.label in ("title", "headline") and display_font:
            tz.font_family = display_font.family
        elif body_font:
            tz.font_family = body_font.family

    return spec


# ---------------------------------------------------------------------------
# Build generation prompt (kept for backward compat)
# ---------------------------------------------------------------------------

def build_generation_prompt(design: TemplateDesign) -> str:
    """Combine layout description with brand config (legacy compat)."""
    cfg = _cc.get_config()
    color_parts = []
    for role in ("primary", "secondary", "accent", "background", "text"):
        entry = cfg.colors.get(role)
        if entry:
            color_parts.append(f"{role}: {entry.hex}")
    colors_str = ", ".join(color_parts) if color_parts else "dark modern palette"

    font_parts = []
    for use in ("display", "body"):
        entry = cfg.fonts.get(use)
        if entry:
            font_parts.append(f"{use}: {entry.family}")
    fonts_str = ", ".join(font_parts) if font_parts else "modern sans-serif"

    style_str = ", ".join(cfg.style_keywords) if cfg.style_keywords else ""
    brand_name = cfg.brand_name or "the brand"
    visual_style = design.visual_style or "modern, clean design"

    prompt = (
        f"Recreate this layout as a branded social media template for {brand_name}. "
        f"Keep the exact same spatial composition and layout structure. "
        f"\n\nLayout: {design.layout_description}"
        f"\n\nVisual style: {visual_style}"
        f"\n\nBrand colors: {colors_str}"
        f"\n\nTypography: {fonts_str}"
    )
    if style_str:
        prompt += f"\n\nStyle keywords: {style_str}"
    if cfg.tagline:
        prompt += f"\n\nTagline: {cfg.tagline}"

    prompt += (
        "\n\nIMPORTANT: This is a TEMPLATE — use placeholder text like "
        f"'{brand_name.upper()}' for headlines and 'Your message here' for body text. "
        "Leave image areas as clean placeholder zones. "
        "The template should look polished and production-ready."
    )
    design.generation_prompt = prompt
    return prompt


# ---------------------------------------------------------------------------
# Legacy: generate_template_image (deprecated, kept for compat)
# ---------------------------------------------------------------------------

async def generate_template_image(design: TemplateDesign) -> str | None:
    """DEPRECATED: Call flux-kontext-pro img2img. Use render_preview() instead."""
    from agent.image_gen import generate_img2img

    if not design.generation_prompt:
        build_generation_prompt(design)

    url = await generate_img2img(
        prompt=design.generation_prompt,
        input_image_path=design.reference_image_path,
        strength=0.75,
    )
    if url:
        design.generated_image_url = url
    return url


async def download_image(url: str) -> Image.Image | None:
    """Download an image from URL and return as PIL Image."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to download image: %s", e)
        return None


# ---------------------------------------------------------------------------
# Serialization — convert to/from dicts for user_data storage
# ---------------------------------------------------------------------------

def design_to_dict(design: TemplateDesign) -> dict:
    """Serialize TemplateDesign to a JSON-safe dict for storing in user_data."""
    d = {
        "layout_description": design.layout_description,
        "visual_style": design.visual_style,
        "generation_prompt": design.generation_prompt,
        "reference_image_path": design.reference_image_path,
        "generated_image_url": design.generated_image_url,
        "canvas_width": design.canvas_width,
        "canvas_height": design.canvas_height,
        "regions": [
            {
                "type": r.type, "x": r.x, "y": r.y,
                "width": r.width, "height": r.height,
                "description": r.description,
            }
            for r in design.regions
        ],
    }
    if design.spec:
        d["spec"] = spec_to_dict(design.spec)
    return d


def design_from_dict(d: dict) -> TemplateDesign:
    """Deserialize a dict back into TemplateDesign."""
    spec = spec_from_dict(d["spec"]) if d.get("spec") else None
    return TemplateDesign(
        layout_description=d.get("layout_description", ""),
        visual_style=d.get("visual_style", ""),
        generation_prompt=d.get("generation_prompt", ""),
        reference_image_path=d.get("reference_image_path", ""),
        generated_image_url=d.get("generated_image_url", ""),
        canvas_width=d.get("canvas_width", 1280),
        canvas_height=d.get("canvas_height", 720),
        regions=[TemplateRegion(**r) for r in d.get("regions", [])],
        spec=spec,
    )


# ---------------------------------------------------------------------------
# Spec adjustment — Claude modifies spec JSON based on user feedback
# ---------------------------------------------------------------------------

_ADJUST_SPEC_PROMPT = """\
You are adjusting a template specification based on user feedback.

Current spec:
{current_spec}

Layout description:
{layout_description}

User feedback: "{feedback}"

Modify the spec JSON to incorporate the user's feedback. You can change:
- background color/gradient
- shape positions, sizes, colors, opacity, border colors
- text zone positions, sizes, font properties, colors
- image zone positions, sizes, corner radii
- logo zone positions, sizes
- Add or remove shapes/zones

Return ONLY the updated spec as valid JSON (same structure as the input spec). \
Do NOT wrap in markdown fences. Return ONLY the JSON object."""


# Legacy prompt alias
_ADJUST_PROMPT = _ADJUST_SPEC_PROMPT


async def adjust_spec(design: TemplateDesign, feedback: str) -> TemplateDesign:
    """Use Claude to modify the spec based on user feedback, then re-render locally.

    Much faster than adjust_design() — no Replicate API call, just PIL re-render.
    """
    if not design.spec:
        # Fall back to legacy adjust if no spec available
        return await adjust_design(design, feedback)

    prompt = _ADJUST_SPEC_PROMPT.format(
        current_spec=json.dumps(spec_to_dict(design.spec), indent=2),
        layout_description=design.layout_description,
        feedback=feedback,
    )

    from agent._client import get_anthropic
    client = get_anthropic()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = _parse_json_response(raw)
    except (json.JSONDecodeError, IndexError):
        logger.warning("Spec adjustment returned non-JSON: %s", raw[:200])
        return design

    new_spec = spec_from_dict(parsed)
    design.spec = new_spec
    design.canvas_width = new_spec.canvas_width
    design.canvas_height = new_spec.canvas_height
    design.regions = _spec_to_regions(new_spec)

    return design


async def adjust_design(design: TemplateDesign, feedback: str) -> TemplateDesign:
    """Legacy: adjust generation prompt and regenerate via Replicate.

    Kept for backward compat with designs that have no spec.
    """
    _LEGACY_ADJUST = (
        "You are adjusting a template generation prompt based on user feedback.\n\n"
        "Current generation prompt:\n{current_prompt}\n\n"
        "Layout description:\n{layout_description}\n\n"
        'User feedback: "{feedback}"\n\n'
        "Rewrite the generation prompt to incorporate the user's feedback. Keep the "
        "brand identity (colors, fonts, style) but adjust the layout, composition, "
        "or visual elements as requested.\n\n"
        "Return ONLY valid JSON:\n"
        '{{\n  "generation_prompt": "the updated prompt text...",\n'
        '  "layout_description": "updated layout description if changed..."\n}}\n\n'
        "Return ONLY the JSON, no markdown formatting."
    )
    prompt = _LEGACY_ADJUST.format(
        current_prompt=design.generation_prompt,
        layout_description=design.layout_description,
        feedback=feedback,
    )

    from agent._client import get_anthropic
    client = get_anthropic()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = _parse_json_response(raw)
    except (json.JSONDecodeError, IndexError):
        logger.warning("Design adjustment returned non-JSON: %s", raw[:200])
        return design

    new_prompt = parsed.get("generation_prompt", "")
    if not new_prompt:
        return design

    design.generation_prompt = new_prompt
    if parsed.get("layout_description"):
        design.layout_description = parsed["layout_description"]

    url = await generate_template_image(design)
    if not url:
        logger.warning("Regeneration failed after adjustment")

    return design


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Main pipeline — analyze + render (preview) and register (save)
# ---------------------------------------------------------------------------

async def analyze_and_generate(image_path: str) -> tuple[TemplateDesign, Image.Image]:
    """Analyze reference image and render a branded template preview.

    Uses PIL rendering (instant, no API cost) instead of Replicate.
    Returns (design, preview_image) without registering anything.
    """
    design = await analyze_reference(image_path)
    if not design.layout_description:
        raise ValueError("Could not analyze the reference image layout.")

    logger.info(
        "Spec analysis: %dx%d, %d regions, style: %s, spec=%s",
        design.canvas_width, design.canvas_height,
        len(design.regions), design.visual_style,
        "yes" if design.spec else "no",
    )

    if design.spec:
        # Apply brand color overrides
        _apply_brand_overrides(design.spec)
        # Render preview locally with PIL — instant, no API cost
        preview = render_preview(design.spec)
    else:
        # Fallback: create a simple placeholder image
        preview = Image.new("RGB", (design.canvas_width, design.canvas_height), (14, 15, 43))

    return design, preview


def save_rendered_frame(design: TemplateDesign) -> str:
    """Render the template frame with PIL and save to disk. Returns file path."""
    templates_dir = Path(settings.BRAND_FOLDER) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    save_path = templates_dir / f"rendered_{uuid.uuid4().hex[:8]}.png"

    if design.spec:
        save_frame(design.spec, str(save_path))
    else:
        # Fallback: save a placeholder
        img = Image.new("RGBA", (design.canvas_width, design.canvas_height), (14, 15, 43, 255))
        img.save(str(save_path), "PNG")

    return str(save_path)


async def save_generated_image(design: TemplateDesign) -> str:
    """Save the template frame — renders locally if spec available, else downloads.

    Backward compat: supports both spec-based and legacy URL-based designs.
    """
    if design.spec:
        return save_rendered_frame(design)

    # Legacy path: download from URL
    templates_dir = Path(settings.BRAND_FOLDER) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    img = await download_image(design.generated_image_url)
    if not img:
        raise ValueError("Failed to download generated image for saving.")

    save_path = templates_dir / f"generated_{uuid.uuid4().hex[:8]}.png"
    img.save(str(save_path), "PNG")
    return str(save_path)


def register_design(
    design: TemplateDesign,
    image_path: str,
    name: str | None = None,
) -> BrandTemplate:
    """Register a finalized design as a template in TemplateMemory."""
    tid = str(uuid.uuid4())[:8]
    cw, ch = design.canvas_width, design.canvas_height
    template_name = name or f"Generated {tid}"

    template = BrandTemplate(
        id=tid,
        name=template_name,
        path=image_path,
        width=cw,
        height=ch,
        regions=design.regions,
        aspect_ratio=_compute_aspect_ratio(cw, ch),
        content_types=[],
        analysis_notes=(
            f"Rendered from spec. Style: {design.visual_style}. "
            f"Layout: {design.layout_description[:200]}"
        ),
        spec_json=spec_to_dict(design.spec) if design.spec else None,
        source="reference",
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
    design, preview_img = await analyze_and_generate(image_path)
    saved_path = save_rendered_frame(design)
    template = register_design(design, saved_path, name)
    return template, preview_img


# ---------------------------------------------------------------------------
# Figma import
# ---------------------------------------------------------------------------

async def import_from_figma(figma_url: str, name: str | None = None) -> tuple[TemplateDesign, Image.Image]:
    """Import a template from a Figma design URL.

    1. Parse URL → get screenshot PNG (pixel-perfect from Figma)
    2. Get child layer metadata with positions
    3. Classify layers via Claude (static vs image/text/logo placeholders)
    4. Build regions from classified layers with font info from Figma
    5. Return TemplateDesign with spec=None (Figma templates use screenshot as-is)
    """
    from agent.figma import (
        parse_figma_url,
        get_node_children_detailed,
        get_node_screenshot_with_key,
    )

    file_key, node_id = parse_figma_url(figma_url)
    if not file_key or not node_id:
        raise ValueError(f"Could not parse Figma URL: {figma_url}")

    # Get screenshot — this IS the template frame
    screenshot_result = await get_node_screenshot_with_key(file_key, node_id)
    if "error" in screenshot_result:
        raise ValueError(f"Figma screenshot failed: {screenshot_result['error']}")

    screenshot_url = screenshot_result["image_url"]
    screenshot_img = await download_image(screenshot_url)
    if not screenshot_img:
        raise ValueError("Failed to download Figma screenshot.")

    # Save screenshot locally
    templates_dir = Path(settings.BRAND_FOLDER) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    img_path = templates_dir / f"figma_{uuid.uuid4().hex[:8]}.png"
    screenshot_img.save(str(img_path), "PNG")

    width, height = screenshot_img.size

    # Get child layer metadata
    children = await get_node_children_detailed(file_key, node_id)

    # Classify layers via Claude Vision
    regions = await _classify_figma_layers(str(img_path), children, width, height)

    design = TemplateDesign(
        layout_description=f"Imported from Figma ({name or 'untitled'})",
        visual_style="Figma design",
        reference_image_path=str(img_path),
        canvas_width=width,
        canvas_height=height,
        regions=regions,
        spec=None,  # Figma templates use screenshot as-is
    )

    return design, screenshot_img


async def _classify_figma_layers(
    image_path: str,
    children: list[dict],
    canvas_width: int,
    canvas_height: int,
) -> list[TemplateRegion]:
    """Use Claude to classify Figma child layers as image/text/logo/static."""
    if not children:
        # Fall back to Claude Vision analysis if no children
        from agent.template_memory import analyze_template
        analysis = await analyze_template(image_path)
        return [TemplateRegion(**r) for r in analysis.get("regions", [])]

    children_desc = json.dumps(children[:30], indent=2)
    prompt = (
        f"Given these Figma layer definitions for a {canvas_width}x{canvas_height} canvas, "
        "classify each layer as a content placeholder or static decoration.\n\n"
        f"Layers:\n{children_desc}\n\n"
        "Return ONLY valid JSON:\n"
        '{"regions": [\n'
        '  {"type": "image"|"text"|"logo", "x": int, "y": int, "width": int, "height": int, '
        '"description": "string"}\n'
        "]}\n\n"
        "Only include layers that are content placeholders (where generated images, "
        "text, or logos should go). Skip decorative/static layers.\n"
        "Return ONLY the JSON, no markdown."
    )

    from agent._client import get_anthropic
    client = get_anthropic()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = _parse_json_response(raw)
        return [TemplateRegion(**r) for r in parsed.get("regions", [])]
    except (json.JSONDecodeError, IndexError, TypeError):
        logger.warning("Figma layer classification failed: %s", raw[:200])
        return []
