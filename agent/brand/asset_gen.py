"""
Standalone asset generation — direct /generate command without the full agent loop.

Supports: logo, icon, mascot, background, 3d_asset, banner, social_header.
Each asset type has its own model routing, quality boosters, negative prompts,
and built-in default templates with {description}, {style_keywords}, {colors},
{background} placeholders matching the externalized brand/prompts/{type}.txt format.
"""

import asyncio
import logging
from pathlib import Path

from agent import compositor_config, image_gen
from agent.tools import _staggered_generate, _handle_generate_image

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "brand" / "prompts"

# ---------------------------------------------------------------------------
# Supported asset types
# ---------------------------------------------------------------------------

SUPPORTED_ASSET_TYPES = [
    "logo", "icon", "mascot", "background", "3d_asset", "banner", "social_header",
]

# Asset type → (content_type for history logging, N options to generate)
_ASSET_ROUTING = {
    "logo":          ("brand_asset", 4),
    "icon":          ("brand_asset", 4),
    "mascot":        ("community",   4),
    "background":    ("lifestyle",   4),
    "3d_asset":      ("brand_3d",    3),
    "banner":        ("announcement", 4),
    "social_header": ("announcement", 4),
}

# ---------------------------------------------------------------------------
# Model routing per asset type
# ---------------------------------------------------------------------------

_ASSET_MODEL_MAP: dict[str, str] = {
    "logo":          "nano-banana",     # best for text-in-image
    "icon":          "recraft-svg",     # clean vector output
    "mascot":        "flux",            # character illustration
    "background":    "flux",            # atmospheric scenes
    "3d_asset":      "flux",            # 3D renders
    "banner":        "flux",            # wide format
    "social_header": "flux",            # wide format
}

# ---------------------------------------------------------------------------
# Aspect ratios and sizes per asset type
# ---------------------------------------------------------------------------

_ASSET_ASPECT_RATIO: dict[str, str] = {
    "logo":          "1:1",
    "icon":          "1:1",
    "mascot":        "3:4",
    "background":    "16:9",
    "3d_asset":      "1:1",
    "banner":        "16:9",
    "social_header": "3:1",
}

# Recraft-svg uses size strings instead of aspect ratios
_ASSET_SIZE: dict[str, str] = {
    "icon": "1024x1024",
}

# ---------------------------------------------------------------------------
# Quality boosters per asset type
# ---------------------------------------------------------------------------

_ASSET_QUALITY_BOOSTERS: dict[str, str] = {
    "logo": (
        "clean scalable vector design, centered composition, "
        "crisp edges, professional brand identity, high contrast, "
        "works at any size, simple bold shapes"
    ),
    "icon": (
        "app icon design, clean scalable vector, simple bold shapes, "
        "minimal detail, flat or subtle gradient, "
        "professional, pixel-perfect edges"
    ),
    "mascot": (
        "character mascot design, friendly and expressive, full body, "
        "3D CGI render, Pixar-quality smooth shading, "
        "soft studio lighting, high detail, polished finish, "
        "clean background, appealing personality"
    ),
    "background": (
        "wide seamless background scene, atmospheric, immersive, "
        "tileable edges, continuous pattern, "
        "high resolution, cinematic depth, volumetric lighting"
    ),
    "3d_asset": (
        "premium 3D product render, volumetric neon lighting, "
        "glass morphism, polished finish, floating object, "
        "three-quarter angle, subtle depth-of-field on background, "
        "hero product shot, 8K resolution"
    ),
    "banner": (
        "wide banner design, bold layout, eye-catching, "
        "strong visual hierarchy, professional typography area, "
        "ultra-wide 16:9, high impact, clean negative space"
    ),
    "social_header": (
        "social media header image, ultra-wide format, professional, "
        "brand-forward composition, subtle background, "
        "space for text overlay on left third, polished"
    ),
}

# ---------------------------------------------------------------------------
# Negative prompts per asset type
# ---------------------------------------------------------------------------

_BASE_NEGATIVE = (
    "blurry, low quality, low resolution, grainy, washed out, "
    "text artifacts, distorted, deformed, amateur"
)

_ASSET_NEGATIVE_PROMPTS: dict[str, str] = {
    "logo": (
        "photorealistic, complex scene, busy background, gradients, "
        "multiple objects, text, words, letters, "
        f"{_BASE_NEGATIVE}"
    ),
    "icon": (
        "photorealistic, complex scene, text, words, fine detail, "
        f"{_BASE_NEGATIVE}"
    ),
    "mascot": (
        "realistic human, photorealistic, uncanny valley, "
        "scary, horror, dark mood, extra limbs, "
        f"{_BASE_NEGATIVE}"
    ),
    "background": (
        "text, words, characters, people, logos, watermarks, "
        "visible seams, hard edges at borders, "
        f"{_BASE_NEGATIVE}"
    ),
    "3d_asset": (
        "flat lighting, gray shadows, no rim light, "
        "cheap plastic, white background, daylight, "
        f"{_BASE_NEGATIVE}"
    ),
    "banner": (
        "portrait orientation, square format, busy cluttered layout, "
        f"{_BASE_NEGATIVE}"
    ),
    "social_header": (
        "portrait orientation, square format, centered focal point, "
        f"{_BASE_NEGATIVE}"
    ),
}

# ---------------------------------------------------------------------------
# Built-in default templates (same placeholder format as external files)
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES: dict[str, str] = {
    "logo": (
        "Brand logo design: {description}. "
        "Clean scalable vector, centered composition, crisp edges, "
        "professional brand identity. "
        "Style: {style_keywords}. "
        "Color palette: {colors}. "
        "Background: {background}."
    ),
    "icon": (
        "App icon design: {description}. "
        "Clean scalable vector, simple bold shapes, minimal detail, "
        "pixel-perfect edges. "
        "Style: {style_keywords}. "
        "Color palette: {colors}. "
        "Background: {background}."
    ),
    "mascot": (
        "Character mascot design: {description}. "
        "Friendly, expressive, full body character. "
        "3D CGI render quality, appealing personality. "
        "Style: {style_keywords}. "
        "Color palette: {colors}. "
        "Background: {background}."
    ),
    "background": (
        "Wide seamless background: {description}. "
        "Atmospheric, immersive, tileable. "
        "Cinematic depth, volumetric lighting. "
        "Style: {style_keywords}. "
        "Color palette: {colors}. "
        "Background: {background}."
    ),
    "3d_asset": (
        "Premium 3D asset render: {description}. "
        "Volumetric neon lighting, glass morphism, polished finish. "
        "Floating object, three-quarter angle view. "
        "Style: {style_keywords}. "
        "Color palette: {colors}. "
        "Background: {background}."
    ),
    "banner": (
        "Wide banner design: {description}. "
        "Bold layout, eye-catching, strong visual hierarchy. "
        "Ultra-wide 16:9 format, professional. "
        "Style: {style_keywords}. "
        "Color palette: {colors}. "
        "Background: {background}."
    ),
    "social_header": (
        "Social media header image: {description}. "
        "Ultra-wide format, brand-forward composition. "
        "Space for text overlay, professional and polished. "
        "Style: {style_keywords}. "
        "Color palette: {colors}. "
        "Background: {background}."
    ),
}

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def _load_asset_template(asset_type: str) -> str | None:
    """Load an external prompt template from brand/prompts/{asset_type}.txt.

    Returns the file content if it exists, otherwise None (use built-in default).
    """
    template_path = _PROMPTS_DIR / f"{asset_type}.txt"
    if template_path.exists():
        try:
            return template_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return None


# ---------------------------------------------------------------------------
# Brand context helpers
# ---------------------------------------------------------------------------


def _get_brand_substitutions() -> dict[str, str]:
    """Build the substitution dict for template placeholders."""
    cfg = compositor_config.get_config()

    # {style_keywords}
    kw_parts: list[str] = []
    if cfg.style_keywords:
        kw_parts.extend(cfg.style_keywords[:4])
    if cfg.visual_style_prompt:
        kw_parts.append(cfg.visual_style_prompt)
    style_keywords = ", ".join(kw_parts) if kw_parts else "high quality, professional"

    # {colors}
    color_phrases: list[str] = []
    for role in ("primary", "accent_1", "accent_2"):
        entry = cfg.colors.get(role)
        if entry:
            color_phrases.append(f"{entry.name.lower()} {entry.hex}")
    colors = ", ".join(color_phrases) if color_phrases else ""

    # {background}
    bg = cfg.colors.get("background")
    background = f"{bg.name.lower()} {bg.hex}" if bg else ""

    return {
        "style_keywords": style_keywords,
        "colors": colors,
        "background": background,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_asset_prompt(asset_type: str, description: str) -> str:
    """Combine user description with asset-type template and brand context.

    Tries external template first (brand/prompts/{asset_type}.txt), then
    falls back to the built-in default template.
    """
    subs = _get_brand_substitutions()
    subs["description"] = description.strip()

    # Try external template first
    file_template = _load_asset_template(asset_type)
    if file_template:
        try:
            return file_template.format(**subs)
        except KeyError:
            # Template has unrecognized placeholders — fall through to default
            logger.warning("External template for %s has invalid placeholders, using default", asset_type)

    # Built-in default template
    template = _DEFAULT_TEMPLATES.get(asset_type)
    if template:
        return template.format(**subs)

    # Bare fallback (shouldn't happen for known types)
    parts = [description.strip()]
    if subs["style_keywords"]:
        parts.append(subs["style_keywords"])
    if subs["colors"]:
        parts.append(f"color palette: {subs['colors']}")
    if subs["background"]:
        parts.append(f"{subs['background']} background")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def select_asset_model(asset_type: str) -> str:
    """Return the Replicate model ID for an asset type."""
    model_key = _ASSET_MODEL_MAP.get(asset_type, "flux")
    return image_gen._MODELS.get(model_key, image_gen._MODELS["flux"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_asset_type(text: str) -> tuple[str | None, str]:
    """Parse asset_type and description from user input.

    Args:
        text: Raw text after /generate (e.g. "logo modern crest design")

    Returns:
        (asset_type, description) or (None, error_message) if invalid.
    """
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return None, f"No asset type provided. Available: {', '.join(SUPPORTED_ASSET_TYPES)}"

    asset_type = parts[0].lower()
    if asset_type not in _ASSET_ROUTING:
        return None, (
            f"Unknown asset type '{asset_type}'. "
            f"Available: {', '.join(sorted(SUPPORTED_ASSET_TYPES))}"
        )

    description = parts[1].strip() if len(parts) > 1 else ""
    if not description:
        return None, f"Please provide a description for the {asset_type}."

    return asset_type, description


async def generate_asset(asset_type: str, description: str) -> dict:
    """Generate asset options for a given type and description.

    Returns:
        {"asset_type": str, "urls": list[str], "prompt": str,
         "content_type": str, "model_id": str}
        or {"error": str} on failure.
    """
    asset_type = asset_type.lower().strip()
    if asset_type not in _ASSET_ROUTING:
        return {
            "error": f"Unknown asset type '{asset_type}'. "
                     f"Available: {', '.join(sorted(SUPPORTED_ASSET_TYPES))}",
        }

    content_type, n_options = _ASSET_ROUTING[asset_type]

    # --- 3d_asset delegates to existing brand_3d pipeline ---
    if asset_type == "3d_asset":
        from agent.tools import ResourceTracker
        tracker = ResourceTracker()
        result_json = await _handle_generate_image(
            {"prompt": description, "content_type": "brand_3d"},
            tracker,
        )
        import json
        try:
            data = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            data = {}
        urls = data.get("image_urls", [])
        if not urls and data.get("image_url"):
            urls = [data["image_url"]]
        return {
            "asset_type": asset_type,
            "urls": urls,
            "prompt": description,
            "content_type": content_type,
            "model_id": "flux (brand_3d pipeline)",
        }

    # --- All other types: build prompt, select model, stagger-generate ---
    prompt = _build_asset_prompt(asset_type, description)
    model_id = select_asset_model(asset_type)
    aspect_ratio = _ASSET_ASPECT_RATIO.get(asset_type)
    size = _ASSET_SIZE.get(asset_type)
    negative = _ASSET_NEGATIVE_PROMPTS.get(asset_type, _BASE_NEGATIVE)

    # Add quality boosters to the prompt
    boosters = _ASSET_QUALITY_BOOSTERS.get(asset_type, "")
    if boosters:
        prompt = f"{prompt}, {boosters}"

    logger.info("Asset gen [%s] model=%s: %s", asset_type, model_id.split("/")[-1], prompt[:150])

    callables = [
        (lambda p=prompt, ct=content_type, m=model_id, ar=aspect_ratio, sz=size, neg=negative: (
            image_gen.generate_image(
                p, ct,
                model_override=m,
                aspect_ratio=ar,
                size_override=sz,
                negative_prompt_override=neg,
            )
        ))
        for _ in range(n_options)
    ]

    results = await _staggered_generate(callables, delay=1.5)
    urls = [r for r in results if isinstance(r, str) and r]

    return {
        "asset_type": asset_type,
        "urls": urls,
        "prompt": prompt,
        "content_type": content_type,
        "model_id": model_id,
    }
