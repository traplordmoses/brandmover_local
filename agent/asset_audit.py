"""
Asset audit via Claude Vision — categorize uploaded assets, extract colors/style,
and determine brand archetype for onboarding.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from config import settings

logger = logging.getLogger(__name__)

_INVENTORY_PATH = Path(settings.BRAND_FOLDER) / "asset_inventory.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AssetAuditEntry:
    path: str
    category: str  # logo|icon|color_palette|font_specimen|style_guide|photography|illustration|other
    dominant_colors: list[dict] = field(default_factory=list)  # [{"hex": "#...", "name": "..."}]
    style_keywords: list[str] = field(default_factory=list)
    description: str = ""
    quality_score: int = 5  # 1-10


@dataclass
class AssetInventory:
    entries: list[AssetAuditEntry] = field(default_factory=list)
    consolidated_colors: list[dict] = field(default_factory=list)
    consolidated_style: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    archetype: str = "starting_fresh"  # full_brand|has_identity|starting_fresh


# ---------------------------------------------------------------------------
# Image encoding (reuses pattern from ingest.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single asset audit
# ---------------------------------------------------------------------------

_AUDIT_PROMPT = """\
Analyze this image as a brand asset. Return ONLY valid JSON with this structure:
{
  "category": "logo|icon|color_palette|font_specimen|style_guide|photography|illustration|other",
  "dominant_colors": [{"hex": "#rrggbb", "name": "Color Name"}],
  "style_keywords": ["keyword1", "keyword2"],
  "description": "Brief description of what this asset is",
  "quality_score": 7
}

For category:
- "logo" — brand logos, wordmarks, logomarks
- "icon" — app icons, favicons, small symbols
- "color_palette" — color swatches, palette images
- "font_specimen" — font samples, typography examples
- "style_guide" — pages from a brand style guide
- "photography" — brand photography, product shots
- "illustration" — brand illustrations, graphics
- "other" — anything else

quality_score: 1 (very low quality/unusable) to 10 (professional, print-ready)
Return ONLY the JSON, no markdown formatting."""


async def audit_single_asset(image_path: str) -> AssetAuditEntry:
    """Use Claude Vision to categorize and analyze a single brand asset."""
    image_data, media_type = _encode_image(image_path)

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
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
                    {"type": "text", "text": _AUDIT_PROMPT},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude Vision returned non-JSON for asset audit: %s", raw[:200])
        data = {}

    return AssetAuditEntry(
        path=image_path,
        category=data.get("category", "other"),
        dominant_colors=data.get("dominant_colors", []),
        style_keywords=data.get("style_keywords", []),
        description=data.get("description", ""),
        quality_score=data.get("quality_score", 5),
    )


# ---------------------------------------------------------------------------
# Batch audit + inventory
# ---------------------------------------------------------------------------

async def audit_batch(image_paths: list[str]) -> AssetInventory:
    """Audit multiple assets and consolidate into an inventory."""
    entries: list[AssetAuditEntry] = []
    for path in image_paths:
        try:
            entry = await audit_single_asset(path)
            entries.append(entry)
        except Exception as e:
            logger.warning("Failed to audit %s: %s", path, e)
            entries.append(AssetAuditEntry(
                path=path, category="other",
                description=f"Audit failed: {e}", quality_score=1,
            ))

    # Consolidate colors
    all_colors: list[dict] = []
    seen_hex: set[str] = set()
    for entry in entries:
        for color in entry.dominant_colors:
            h = color.get("hex", "").lower()
            if h and h not in seen_hex:
                seen_hex.add(h)
                all_colors.append(color)

    # Consolidate style keywords
    all_styles: list[str] = []
    seen_styles: set[str] = set()
    for entry in entries:
        for kw in entry.style_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen_styles:
                seen_styles.add(kw_lower)
                all_styles.append(kw)

    missing = detect_missing(entries)
    archetype = determine_archetype(entries)

    inventory = AssetInventory(
        entries=entries,
        consolidated_colors=all_colors,
        consolidated_style=all_styles,
        missing_items=missing,
        archetype=archetype,
    )

    return inventory


# ---------------------------------------------------------------------------
# Archetype detection
# ---------------------------------------------------------------------------

_ESSENTIAL_CATEGORIES = {"logo", "color_palette", "style_guide"}
_IDENTITY_CATEGORIES = {"logo", "icon", "color_palette"}


def detect_missing(entries: list[AssetAuditEntry]) -> list[str]:
    """Identify missing essential brand assets."""
    present = {e.category for e in entries}
    missing = []
    if "logo" not in present:
        missing.append("logo")
    if "color_palette" not in present:
        missing.append("color_palette")
    if "style_guide" not in present:
        missing.append("style_guide")
    if "font_specimen" not in present:
        missing.append("font_specimen")
    return missing


def determine_archetype(entries: list[AssetAuditEntry]) -> str:
    """Determine brand archetype based on available assets.

    - full_brand: has logo + color palette + style guide (or 5+ high-quality assets)
    - has_identity: has logo or icon with some colors
    - starting_fresh: minimal or no brand assets
    """
    if not entries:
        return "starting_fresh"

    categories = {e.category for e in entries}
    high_quality = [e for e in entries if e.quality_score >= 7]

    has_essential = _ESSENTIAL_CATEGORIES.issubset(categories)
    has_identity = bool(_IDENTITY_CATEGORIES & categories)

    if has_essential or len(high_quality) >= 5:
        return "full_brand"
    elif has_identity:
        return "has_identity"
    return "starting_fresh"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_inventory(inventory: AssetInventory) -> None:
    """Save inventory to brand/asset_inventory.json."""
    data = {
        "entries": [
            {
                "path": e.path,
                "category": e.category,
                "dominant_colors": e.dominant_colors,
                "style_keywords": e.style_keywords,
                "description": e.description,
                "quality_score": e.quality_score,
            }
            for e in inventory.entries
        ],
        "consolidated_colors": inventory.consolidated_colors,
        "consolidated_style": inventory.consolidated_style,
        "missing_items": inventory.missing_items,
        "archetype": inventory.archetype,
    }
    _INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INVENTORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Asset inventory saved: %d entries, archetype=%s", len(inventory.entries), inventory.archetype)


def load_inventory() -> AssetInventory | None:
    """Load inventory from brand/asset_inventory.json. Returns None if not found."""
    if not _INVENTORY_PATH.exists():
        return None
    try:
        data = json.loads(_INVENTORY_PATH.read_text(encoding="utf-8"))
        entries = [
            AssetAuditEntry(**e) for e in data.get("entries", [])
        ]
        return AssetInventory(
            entries=entries,
            consolidated_colors=data.get("consolidated_colors", []),
            consolidated_style=data.get("consolidated_style", []),
            missing_items=data.get("missing_items", []),
            archetype=data.get("archetype", "starting_fresh"),
        )
    except Exception as e:
        logger.warning("Failed to load asset inventory: %s", e)
        return None
