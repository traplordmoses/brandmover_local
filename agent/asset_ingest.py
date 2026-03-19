"""
Brand asset ingestion — analyze uploaded images and catalog them.
Uses Claude Vision to extract colors, style, tags, and suggest captions.
"""

import json
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    import base64

    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


async def analyze_for_library(image_path: str) -> dict:
    """Analyze an image with Claude Vision for brand asset cataloging.

    Returns:
    {
        "dominant_colors": [{"hex": "#...", "name": "...", "role": "primary|accent|neutral"}],
        "style_keywords": ["keyword1", "keyword2", ...],
        "content_types": ["announcement", "meme", "community", ...],
        "description": "One-line description of the image",
        "suggested_captions": ["caption1", "caption2", "caption3"],
        "brand_alignment": "high|medium|low",
        "brand_alignment_notes": "Why it does/doesn't fit the brand",
        "recommended_tags": ["tag1", "tag2", ...],
        "category": "illustration|photography|screenshot|logo|character|template|other"
    }
    """
    image_data, media_type = _encode_image(image_path)

    # Load brand context for the system prompt
    from agent import compositor_config
    cfg = compositor_config.get_config()
    brand_name = cfg.brand_name or settings.BRAND_NAME
    brand_colors = ", ".join(
        f"{c.name} ({c.hex})" for c in cfg.colors.values()
    ) if cfg.colors else "not defined"
    style_kw = ", ".join(cfg.style_keywords) if cfg.style_keywords else "not defined"

    system_prompt = (
        f"You are a brand asset analyst for {brand_name}. "
        f"The brand's visual identity uses these colors: {brand_colors}. "
        f"Style keywords: {style_kw}. "
        "Analyze images for cataloging in the brand asset library. "
        "Return ONLY valid JSON, no markdown fences."
    )

    user_prompt = (
        "Analyze this image for brand asset cataloging. "
        "Return ONLY valid JSON with this exact structure:\n"
        "{\n"
        '  "dominant_colors": [{"hex": "#rrggbb", "name": "Color Name", "role": "primary|accent|neutral"}],\n'
        '  "style_keywords": ["keyword1", "keyword2"],\n'
        '  "content_types": ["announcement", "meme", "community", "engagement", "brand_3d", "campaign", "educational", "lifestyle"],\n'
        '  "description": "One-line description of the image",\n'
        '  "suggested_captions": ["caption1", "caption2", "caption3"],\n'
        '  "brand_alignment": "high|medium|low",\n'
        '  "brand_alignment_notes": "Why it does or doesn\'t fit the brand",\n'
        '  "recommended_tags": ["tag1", "tag2"],\n'
        '  "category": "illustration|photography|screenshot|logo|character|template|other"\n'
        "}\n\n"
        "For dominant_colors, identify 3-5 colors with hex, descriptive name, and role (primary/accent/neutral).\n"
        "For content_types, list which types of social content this image could be used for.\n"
        "For suggested_captions, write 3 short captions in a lowercase, declarative, punchy voice.\n"
        "For brand_alignment, assess how well this image fits the brand's visual identity.\n"
        "For category, classify the image type.\n"
        "Return ONLY the JSON, no markdown formatting."
    )

    from agent._client import get_anthropic
    client = get_anthropic()

    response = await client.messages.create(
        model=settings.SONNET_MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude Vision returned non-JSON for asset analysis: %s", raw[:200])
        return {
            "dominant_colors": [],
            "style_keywords": [],
            "content_types": [],
            "description": "",
            "suggested_captions": [],
            "brand_alignment": "unknown",
            "brand_alignment_notes": "",
            "recommended_tags": [],
            "category": "other",
            "_raw_response": raw,
        }


async def add_to_library(
    image_path: str,
    analysis: dict,
    content_type: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Add an analyzed image to the brand asset library.

    Returns the created LibraryEntry as a dict.
    """
    from dataclasses import asdict
    from agent import asset_library

    # Determine content type: user override > first from analysis > "general"
    ct = content_type
    if not ct:
        analysis_types = analysis.get("content_types", [])
        ct = analysis_types[0] if analysis_types else "general"

    # Merge tags: analysis recommended_tags + user-provided tags
    merged_tags = list(analysis.get("recommended_tags", []))
    if tags:
        for t in tags:
            if t not in merged_tags:
                merged_tags.append(t)
    # Add category as a tag too
    category = analysis.get("category", "")
    if category and category not in merged_tags:
        merged_tags.append(category)

    entry = asset_library.add(
        image_path=image_path,
        source="uploaded",
        content_type=ct,
        prompt=analysis.get("description", ""),
        tags=merged_tags,
    )

    # Optionally update asset_inventory.json with the analysis data
    inventory_path = Path(settings.BRAND_FOLDER) / "asset_inventory.json"
    try:
        if inventory_path.exists():
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            if not isinstance(inventory, list):
                inventory = inventory.get("assets", [])
        else:
            inventory = []

        inventory.append({
            "id": entry.id,
            "analysis": analysis,
            "content_type": ct,
            "tags": merged_tags,
        })
        inventory_path.write_text(
            json.dumps(inventory, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed to update asset_inventory.json: %s", e)

    return asdict(entry)
