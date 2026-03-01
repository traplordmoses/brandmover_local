"""
Template Generator — create branded templates from reference images using AI.

Given a screenshot or reference image (e.g. a competitor post, a layout the user
likes), this module:
1. Analyzes the layout via Claude Vision to produce a natural language description
2. Builds a detailed generation prompt combining layout + brand identity
3. Generates a polished template image via flux-kontext-pro img2img
4. Registers the result via TemplateMemory for future posts

The reference image is used as a composition guide — flux-kontext-pro preserves
the spatial layout while applying the client's brand colors, fonts, and style.
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
from agent.image_gen import generate_img2img
from agent.template_memory import (
    BrandTemplate,
    TemplateMemory,
    TemplateRegion,
)
from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude Vision — natural language layout analysis
# ---------------------------------------------------------------------------

_LAYOUT_ANALYSIS_PROMPT = """\
Analyze this image as a social media post or design layout. Describe the \
visual structure in natural language so an AI image generator can recreate \
a similar layout with different branding.

Describe:
1. **Overall composition**: Layout pattern (split-panel, centered, grid, full-bleed, etc.), \
aspect ratio, and spatial flow.
2. **Background**: Colors, gradients, textures, or patterns.
3. **Image areas**: Where photos or illustrations sit, their shape (rounded corners, \
circular crop, full-width), and approximate position/size relative to the canvas.
4. **Text zones**: Where headlines, subtitles, and body text appear. Describe position, \
alignment, relative size, and visual weight (bold, light, uppercase).
5. **Branding elements**: Logo placement, badges, tags, watermarks — position and style.
6. **Decorative elements**: Borders, dividers, accent lines, glow effects, overlays.
7. **Visual style**: Glass-morphism, flat design, gradient mesh, neon, minimal, etc.

Also identify content regions for template registration (where generated images \
and text would be placed in future posts):

Return ONLY valid JSON:
{
  "layout_description": "A detailed natural language description of the full layout...",
  "visual_style": "glass-morphism with neon accents on dark background",
  "canvas_width": 1280,
  "canvas_height": 720,
  "regions": [
    {"type": "image", "x": 40, "y": 80, "width": 580, "height": 560, "description": "Main image area"},
    {"type": "text", "x": 660, "y": 200, "width": 560, "height": 80, "description": "Headline text"},
    {"type": "logo", "x": 50, "y": 20, "width": 100, "height": 50, "description": "Logo top-left"}
  ]
}

Return ONLY the JSON, no markdown formatting."""


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


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Phase 1a: Analyze reference image
# ---------------------------------------------------------------------------

async def analyze_reference(image_path: str) -> TemplateDesign:
    """Use Claude Vision to produce a natural language layout description."""
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
        return TemplateDesign(reference_image_path=image_path)

    regions = [
        TemplateRegion(**r) for r in parsed.get("regions", [])
    ]

    return TemplateDesign(
        layout_description=parsed.get("layout_description", ""),
        visual_style=parsed.get("visual_style", ""),
        reference_image_path=image_path,
        canvas_width=parsed.get("canvas_width", 1280),
        canvas_height=parsed.get("canvas_height", 720),
        regions=regions,
    )


# ---------------------------------------------------------------------------
# Phase 1b: Build generation prompt from layout + brand identity
# ---------------------------------------------------------------------------

def build_generation_prompt(design: TemplateDesign) -> str:
    """Combine layout description with brand config to build a flux-kontext-pro prompt."""
    cfg = _cc.get_config()

    # Collect brand colors
    color_parts = []
    for role in ("primary", "secondary", "accent", "background", "text"):
        entry = cfg.colors.get(role)
        if entry:
            color_parts.append(f"{role}: {entry.hex}")
    colors_str = ", ".join(color_parts) if color_parts else "dark modern palette"

    # Collect font info
    font_parts = []
    for use in ("display", "body"):
        entry = cfg.fonts.get(use)
        if entry:
            font_parts.append(f"{use}: {entry.family}")
    fonts_str = ", ".join(font_parts) if font_parts else "modern sans-serif"

    # Style keywords
    style_str = ", ".join(cfg.style_keywords) if cfg.style_keywords else ""

    # Brand name
    brand_name = cfg.brand_name or "the brand"

    # Visual style from analysis
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
# Phase 1c: Generate template image via flux-kontext-pro
# ---------------------------------------------------------------------------

async def generate_template_image(design: TemplateDesign) -> str | None:
    """Call flux-kontext-pro img2img to generate a branded template.

    Returns the generated image URL, or None on failure.
    """
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
    return {
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


def design_from_dict(d: dict) -> TemplateDesign:
    """Deserialize a dict back into TemplateDesign."""
    return TemplateDesign(
        layout_description=d.get("layout_description", ""),
        visual_style=d.get("visual_style", ""),
        generation_prompt=d.get("generation_prompt", ""),
        reference_image_path=d.get("reference_image_path", ""),
        generated_image_url=d.get("generated_image_url", ""),
        canvas_width=d.get("canvas_width", 1280),
        canvas_height=d.get("canvas_height", 720),
        regions=[TemplateRegion(**r) for r in d.get("regions", [])],
    )


# ---------------------------------------------------------------------------
# Design adjustment — Claude modifies prompt based on user feedback
# ---------------------------------------------------------------------------

_ADJUST_PROMPT = """\
You are adjusting a template generation prompt based on user feedback.

Current generation prompt:
{current_prompt}

Layout description:
{layout_description}

User feedback: "{feedback}"

Rewrite the generation prompt to incorporate the user's feedback. Keep the \
brand identity (colors, fonts, style) but adjust the layout, composition, \
or visual elements as requested.

Return ONLY valid JSON:
{{
  "generation_prompt": "the updated prompt text...",
  "layout_description": "updated layout description if changed..."
}}

Return ONLY the JSON, no markdown formatting."""


async def adjust_design(design: TemplateDesign, feedback: str) -> TemplateDesign:
    """Use Claude to adjust the generation prompt based on user feedback, then regenerate."""
    prompt = _ADJUST_PROMPT.format(
        current_prompt=design.generation_prompt,
        layout_description=design.layout_description,
        feedback=feedback,
    )

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
        logger.warning("Design adjustment returned non-JSON: %s", raw[:200])
        return design  # Return unchanged on failure

    new_prompt = parsed.get("generation_prompt", "")
    if not new_prompt:
        return design

    design.generation_prompt = new_prompt
    if parsed.get("layout_description"):
        design.layout_description = parsed["layout_description"]

    # Regenerate with updated prompt
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
# Main pipeline — analyze + generate (preview) and register (save)
# ---------------------------------------------------------------------------

async def analyze_and_generate(image_path: str) -> tuple[TemplateDesign, Image.Image]:
    """Phase 1: Analyze reference image and generate a branded template preview.

    Returns (design, generated_image) without registering anything.
    """
    design = await analyze_reference(image_path)
    if not design.layout_description:
        raise ValueError("Could not analyze the reference image layout.")

    logger.info(
        "Layout analysis: %dx%d, %d regions, style: %s",
        design.canvas_width, design.canvas_height,
        len(design.regions), design.visual_style,
    )

    build_generation_prompt(design)
    url = await generate_template_image(design)
    if not url:
        raise ValueError("Template image generation failed. Check REPLICATE_API_TOKEN.")

    img = await download_image(url)
    if not img:
        raise ValueError("Failed to download generated template image.")

    return design, img


async def save_generated_image(design: TemplateDesign) -> str:
    """Download the generated image and save to templates dir. Returns file path."""
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
    """Phase 2: Register a finalized design as a template in TemplateMemory."""
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
        content_types=[],  # Universal
        analysis_notes=(
            f"AI-generated from reference. Style: {design.visual_style}. "
            f"Layout: {design.layout_description[:200]}"
        ),
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
    """Full pipeline: analyze, generate, register. Used by non-interactive callers."""
    design, template_img = await analyze_and_generate(image_path)

    saved_path = await save_generated_image(design)
    template = register_design(design, saved_path, name)
    return template, template_img
