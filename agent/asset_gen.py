"""
Standalone asset generation — direct /generate command without the full agent loop.

Supports: logo, icon, mascot, background, 3d_asset, banner, social_header.
"""

import asyncio
import logging

from agent import compositor_config, image_gen
from agent.tools import _staggered_generate, _handle_generate_image

logger = logging.getLogger(__name__)

# Asset type → (content_type for model routing, N options)
_ASSET_ROUTING = {
    "logo":          ("brand_asset", 4),
    "icon":          ("brand_asset", 4),
    "mascot":        ("community", 4),
    "background":    ("lifestyle", 4),
    "3d_asset":      ("brand_3d", 3),
    "banner":        ("announcement", 4),
    "social_header": ("announcement", 4),
}

# Per-type prompt frames prepended to the user description
_ASSET_FRAMES = {
    "logo": "brand logo design, clean scalable vector, centered composition",
    "icon": "app icon design, clean scalable vector, simple bold shapes",
    "mascot": "character mascot design, friendly, expressive, full body",
    "background": "wide background scene, atmospheric, immersive",
    "banner": "wide banner design, bold layout, eye-catching",
    "social_header": "social media header image, wide format, professional",
}


def _build_asset_prompt(asset_type: str, description: str) -> str:
    """Combine user description with asset-type frame and brand context."""
    cfg = compositor_config.get_config()

    parts: list[str] = []

    # Asset-type frame
    frame = _ASSET_FRAMES.get(asset_type, "")
    if frame:
        parts.append(frame)

    # User description
    parts.append(description.strip())

    # Brand style keywords (up to 4)
    if cfg.style_keywords:
        parts.extend(cfg.style_keywords[:4])

    # Brand color palette phrase
    color_phrases = []
    for role in ("primary", "accent_1", "accent_2"):
        entry = cfg.colors.get(role)
        if entry:
            color_phrases.append(f"{entry.name.lower()} {entry.hex}")
    if color_phrases:
        parts.append("color palette: " + ", ".join(color_phrases))

    # Background color
    bg = cfg.colors.get("background")
    if bg:
        parts.append(f"{bg.name.lower()} {bg.hex} background")

    return ", ".join(parts)


async def generate_asset(asset_type: str, description: str) -> dict:
    """Generate asset options for a given type and description.

    Returns:
        {"asset_type": str, "urls": list[str], "prompt": str, "content_type": str}
    """
    asset_type = asset_type.lower().strip()
    if asset_type not in _ASSET_ROUTING:
        return {
            "error": f"Unknown asset type '{asset_type}'. "
                     f"Available: {', '.join(sorted(_ASSET_ROUTING))}",
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
        # _handle_generate_image returns JSON string with image_urls
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
        }

    # --- All other types: build prompt and stagger-generate ---
    prompt = _build_asset_prompt(asset_type, description)
    logger.info("Asset gen [%s]: %s", asset_type, prompt[:150])

    callables = [
        (lambda p=prompt, ct=content_type: image_gen.generate_image(p, ct))
        for _ in range(n_options)
    ]

    results = await _staggered_generate(callables, delay=1.5)
    urls = [r for r in results if isinstance(r, str) and r]

    return {
        "asset_type": asset_type,
        "urls": urls,
        "prompt": prompt,
        "content_type": content_type,
    }
