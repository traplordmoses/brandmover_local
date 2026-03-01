"""
Strategy recommendation engine — uses asset audit results and user inputs
to recommend a pipeline configuration for onboarding.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from agent.asset_audit import AssetInventory
from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StrategyRecommendation:
    archetype: str  # full_brand|has_identity|starting_fresh
    compositor_enabled: bool = True
    badge_text: str | None = None
    default_mode: str = "image_optional"
    recommended_content_types: list[str] = field(default_factory=list)
    visual_style_notes: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Archetype → defaults mapping
# ---------------------------------------------------------------------------

_ARCHETYPE_DEFAULTS: dict[str, dict] = {
    "full_brand": {
        "compositor_enabled": True,
        "badge_text": None,  # configurable per brand
        "default_mode": "image_always",
        "recommended_content_types": [
            "announcement", "community", "meme", "engagement",
            "educational", "brand_asset", "lifestyle", "market_commentary",
        ],
    },
    "has_identity": {
        "compositor_enabled": True,
        "badge_text": None,
        "default_mode": "image_optional",
        "recommended_content_types": [
            "announcement", "community", "educational",
        ],
    },
    "starting_fresh": {
        "compositor_enabled": False,
        "badge_text": None,
        "default_mode": "image_optional",
        "recommended_content_types": [
            "announcement", "community",
        ],
    },
}


# ---------------------------------------------------------------------------
# Claude-powered strategy recommendation
# ---------------------------------------------------------------------------

_STRATEGY_PROMPT = """\
You are a brand strategy advisor. Based on the following information about a brand,
recommend a content strategy.

Brand name: {brand_name}
Description: {description}
Target platforms: {platforms}
Brand archetype: {archetype}
Visual style preferences: {visual_preferences}

Asset inventory summary:
{inventory_summary}

Based on this, recommend:
1. Which content types this brand should focus on
2. Whether they should use image composition (compositor) or raw images
3. Visual style notes for their brand
4. Brief reasoning for your recommendations

Return ONLY valid JSON:
{{
  "compositor_enabled": true,
  "badge_text": null,
  "default_mode": "image_optional",
  "recommended_content_types": ["announcement", "community"],
  "visual_style_notes": "Clean, minimal aesthetic with bold typography",
  "reasoning": "Given the brand's early stage..."
}}"""


async def recommend_strategy(
    brand_name: str,
    description: str,
    platforms: list[str],
    inventory: AssetInventory | None,
    visual_preferences: dict,
) -> StrategyRecommendation:
    """Use Claude to generate a strategy recommendation."""
    archetype = inventory.archetype if inventory else "starting_fresh"

    # Build inventory summary
    if inventory and inventory.entries:
        inv_summary = (
            f"Assets: {len(inventory.entries)} uploaded\n"
            f"Categories: {', '.join(set(e.category for e in inventory.entries))}\n"
            f"Colors found: {len(inventory.consolidated_colors)}\n"
            f"Style keywords: {', '.join(inventory.consolidated_style[:10])}\n"
            f"Missing: {', '.join(inventory.missing_items) or 'none'}\n"
            f"Archetype: {archetype}"
        )
    else:
        inv_summary = "No assets uploaded. Archetype: starting_fresh"

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    prompt = _STRATEGY_PROMPT.format(
        brand_name=brand_name,
        description=description,
        platforms=", ".join(platforms) if platforms else "twitter",
        archetype=archetype,
        visual_preferences=json.dumps(visual_preferences) if visual_preferences else "none specified",
        inventory_summary=inv_summary,
    )

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
        logger.warning("Strategy recommendation returned non-JSON, using archetype defaults")
        data = {}

    # Merge Claude's recommendation with archetype defaults
    defaults = _ARCHETYPE_DEFAULTS.get(archetype, _ARCHETYPE_DEFAULTS["starting_fresh"])

    return StrategyRecommendation(
        archetype=archetype,
        compositor_enabled=data.get("compositor_enabled", defaults["compositor_enabled"]),
        badge_text=data.get("badge_text", defaults["badge_text"]),
        default_mode=data.get("default_mode", defaults["default_mode"]),
        recommended_content_types=data.get("recommended_content_types", defaults["recommended_content_types"]),
        visual_style_notes=data.get("visual_style_notes", ""),
        reasoning=data.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# config.json generation
# ---------------------------------------------------------------------------

def generate_config_json(rec: StrategyRecommendation, brand_name: str = "") -> dict:
    """Generate a brand/config.json from a strategy recommendation."""
    return {
        "version": "8.0",
        "brand_name": brand_name or settings.BRAND_NAME,
        "pipeline": {
            "compositor_enabled": rec.compositor_enabled,
            "badge_text": rec.badge_text,
            "default_mode": rec.default_mode,
            "agent_mode": settings.AGENT_MODE,
        },
        "content_types_enabled": rec.recommended_content_types,
        "image_generation": {
            "default_model": "auto",
            "lora_enabled": True,
        },
        "onboarding": {
            "completed": True,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "archetype": rec.archetype,
        },
    }


# ---------------------------------------------------------------------------
# Strategy markdown
# ---------------------------------------------------------------------------

def generate_strategy_markdown(rec: StrategyRecommendation, brand_name: str = "") -> str:
    """Generate a human-readable strategy summary as markdown."""
    name = brand_name or settings.BRAND_NAME
    lines = [
        f"# {name} — Content Strategy",
        "",
        f"**Archetype:** {rec.archetype}",
        f"**Compositor:** {'ON' if rec.compositor_enabled else 'OFF'}",
        f"**Badge:** {rec.badge_text or '(none)'}",
        f"**Default mode:** {rec.default_mode}",
        "",
        "## Recommended Content Types",
        "",
    ]
    for ct in rec.recommended_content_types:
        lines.append(f"- {ct}")

    if rec.visual_style_notes:
        lines.extend(["", "## Visual Style", "", rec.visual_style_notes])

    if rec.reasoning:
        lines.extend(["", "## Reasoning", "", rec.reasoning])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_strategy(rec: StrategyRecommendation, brand_name: str = "") -> None:
    """Write brand/config.json and brand/strategy.md."""
    brand_path = Path(settings.BRAND_FOLDER)
    brand_path.mkdir(parents=True, exist_ok=True)

    # config.json
    config_data = generate_config_json(rec, brand_name)
    config_path = brand_path / "config.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    logger.info("Saved config.json: %s", config_path)

    # strategy.md
    strategy_md = generate_strategy_markdown(rec, brand_name)
    strategy_path = brand_path / "strategy.md"
    strategy_path.write_text(strategy_md, encoding="utf-8")
    logger.info("Saved strategy.md: %s", strategy_path)
