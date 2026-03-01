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
    content_potential: list[str] = field(default_factory=list)  # suggested content types
    brand_signals: list[str] = field(default_factory=list)  # detected brand traits
    recommended_formats: list[str] = field(default_factory=list)  # e.g. ["social_post", "banner"]
    # Creative understanding fields (v8.3)
    first_impression: str = ""
    what_makes_it_special: str = ""
    creative_dna: list[str] = field(default_factory=list)
    content_directions: list[str] = field(default_factory=list)
    never_do: list[str] = field(default_factory=list)
    overall_energy: str = ""
    character_system: str = ""
    presentation_formats: list[str] = field(default_factory=list)


@dataclass
class AssetInventory:
    entries: list[AssetAuditEntry] = field(default_factory=list)
    consolidated_colors: list[dict] = field(default_factory=list)
    consolidated_style: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    archetype: str = "starting_fresh"  # full_brand|has_identity|starting_fresh
    collection_analysis: dict = field(default_factory=dict)  # coherence, diversity, gaps
    brand_insights: dict = field(default_factory=dict)  # personality, audience, tone


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
You are a creative director seeing this brand asset for the first time. \
Don't just categorize it — feel it. What's the energy? What story is it telling? \
What would you NEVER pair it with?

Return ONLY valid JSON:
{
  "category": "logo|icon|color_palette|font_specimen|style_guide|photography|illustration|other",
  "dominant_colors": [{"hex": "#rrggbb", "name": "Color Name", "role": "primary|secondary|accent|neutral"}],
  "style_keywords": ["keyword1", "keyword2", "keyword3"],
  "description": "Detailed description of the asset — what it depicts, visual style, mood",
  "quality_score": 7,
  "content_potential": ["announcement", "community"],
  "brand_signals": ["professional", "tech-forward"],
  "recommended_formats": ["social_post", "banner"],
  "first_impression": "Your gut reaction in one vivid sentence — what hits you first?",
  "what_makes_it_special": "The one thing about this asset that a generic brand could never replicate",
  "creative_dna": ["hand-drawn warmth", "deliberate imperfection", "origin story energy"],
  "content_directions": ["behind-the-scenes origin stories", "raw process shots"],
  "never_do": ["Don't pair with sterile stock photography", "Never crop the hand-drawn edges"],
  "overall_energy": "One phrase capturing the vibe — e.g. 'garage startup confidence', 'quiet luxury'",
  "character_system": "If this asset were a character, who would it be? Brief description.",
  "presentation_formats": ["polaroid frame", "torn paper edge", "notebook sketch"]
}

Categories:
- "logo" — brand logos, wordmarks, logomarks
- "icon" — app icons, favicons, small symbols
- "color_palette" — color swatches, palette images
- "font_specimen" — font samples, typography examples
- "style_guide" — pages from a brand style guide
- "photography" — brand photography, product shots
- "illustration" — brand illustrations, graphics
- "other" — anything else

Classification fields:
- dominant_colors: Extract up to 5 colors. Assign each a role (primary, secondary, accent, neutral).
- style_keywords: 3-6 words describing visual style (e.g. minimalist, geometric, warm, retro, corporate).
- quality_score: 1 (unusable) to 10 (professional, print-ready).
- content_potential: Which content types could USE this asset? Options: announcement, community, meme, engagement, educational, onchain_update, brand_3d.
- brand_signals: Traits this asset communicates about the brand (e.g. "luxury", "playful", "trustworthy", "bold", "tech-forward", "community-driven").
- recommended_formats: Where this asset works best: social_post, story, banner, avatar, background, pattern, overlay.

Creative fields:
- first_impression: Your instant gut reaction — what hits you before you start analyzing?
- what_makes_it_special: The unique thing about this asset that makes it irreplaceable.
- creative_dna: 2-4 phrases capturing the creative essence — what makes this FEEL like this brand.
- content_directions: 2-3 content ideas this asset naturally suggests.
- never_do: 1-3 things that would destroy this asset's energy.
- overall_energy: One phrase capturing the vibe.
- character_system: If this asset had a personality, describe it in one sentence.
- presentation_formats: 2-3 visual formats that would showcase this asset well.

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
        content_potential=data.get("content_potential", []),
        brand_signals=data.get("brand_signals", []),
        recommended_formats=data.get("recommended_formats", []),
        first_impression=data.get("first_impression", ""),
        what_makes_it_special=data.get("what_makes_it_special", ""),
        creative_dna=data.get("creative_dna", []),
        content_directions=data.get("content_directions", []),
        never_do=data.get("never_do", []),
        overall_energy=data.get("overall_energy", ""),
        character_system=data.get("character_system", ""),
        presentation_formats=data.get("presentation_formats", []),
    )


# ---------------------------------------------------------------------------
# Batch audit + inventory
# ---------------------------------------------------------------------------

_COLLECTION_PROMPT = """\
You are analyzing a collection of brand assets. Here is a summary of each asset:

{asset_summaries}

Analyze the COLLECTION as a whole. Return ONLY valid JSON:
{{
  "collection_analysis": {{
    "visual_coherence": "high|medium|low",
    "coherence_notes": "Brief explanation of visual consistency across assets",
    "style_diversity": "high|medium|low",
    "color_harmony": "harmonious|mixed|clashing",
    "strongest_asset_types": ["logo", "photography"],
    "gaps": ["Missing consistent typography", "No pattern/texture assets"]
  }},
  "brand_insights": {{
    "personality_traits": ["professional", "innovative", "approachable"],
    "likely_audience": "Brief description of target audience based on visual cues",
    "suggested_tone": "formal|conversational|playful|authoritative|friendly",
    "visual_maturity": "polished|developing|early_stage"
  }}
}}

Return ONLY the JSON, no markdown formatting."""


async def _analyze_collection(entries: list[AssetAuditEntry]) -> tuple[dict, dict]:
    """Run Claude on asset summaries to get collection-level insights."""
    summaries = []
    for e in entries:
        colors_str = ", ".join(c.get("hex", "") for c in e.dominant_colors[:3])
        summaries.append(
            f"- {e.category}: {e.description} | "
            f"colors: {colors_str} | style: {', '.join(e.style_keywords[:4])} | "
            f"quality: {e.quality_score}/10 | signals: {', '.join(e.brand_signals[:3])}"
        )

    prompt = _COLLECTION_PROMPT.format(asset_summaries="\n".join(summaries))

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
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
        logger.warning("Collection analysis returned non-JSON: %s", raw[:200])
        data = {}

    return (
        data.get("collection_analysis", {}),
        data.get("brand_insights", {}),
    )


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

    # Collection-level analysis (only if 2+ successful entries)
    successful = [e for e in entries if e.quality_score > 1]
    collection_analysis = {}
    brand_insights = {}
    if len(successful) >= 2:
        try:
            collection_analysis, brand_insights = await _analyze_collection(successful)
        except Exception as e:
            logger.warning("Collection analysis failed: %s", e)

    inventory = AssetInventory(
        entries=entries,
        consolidated_colors=all_colors,
        consolidated_style=all_styles,
        missing_items=missing,
        archetype=archetype,
        collection_analysis=collection_analysis,
        brand_insights=brand_insights,
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
                "content_potential": e.content_potential,
                "brand_signals": e.brand_signals,
                "recommended_formats": e.recommended_formats,
                "first_impression": e.first_impression,
                "what_makes_it_special": e.what_makes_it_special,
                "creative_dna": e.creative_dna,
                "content_directions": e.content_directions,
                "never_do": e.never_do,
                "overall_energy": e.overall_energy,
                "character_system": e.character_system,
                "presentation_formats": e.presentation_formats,
            }
            for e in inventory.entries
        ],
        "consolidated_colors": inventory.consolidated_colors,
        "consolidated_style": inventory.consolidated_style,
        "missing_items": inventory.missing_items,
        "archetype": inventory.archetype,
        "collection_analysis": inventory.collection_analysis,
        "brand_insights": inventory.brand_insights,
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
        # Filter to known fields for backward compat with older inventories
        known_entry_fields = {f.name for f in AssetAuditEntry.__dataclass_fields__.values()}
        entries = [
            AssetAuditEntry(**{k: v for k, v in e.items() if k in known_entry_fields})
            for e in data.get("entries", [])
        ]
        return AssetInventory(
            entries=entries,
            consolidated_colors=data.get("consolidated_colors", []),
            consolidated_style=data.get("consolidated_style", []),
            missing_items=data.get("missing_items", []),
            archetype=data.get("archetype", "starting_fresh"),
            collection_analysis=data.get("collection_analysis", {}),
            brand_insights=data.get("brand_insights", {}),
        )
    except Exception as e:
        logger.warning("Failed to load asset inventory: %s", e)
        return None
