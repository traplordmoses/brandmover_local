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
    platforms: list[str] = field(default_factory=lambda: ["x"])
    visual_style_notes: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Archetype → defaults mapping
# ---------------------------------------------------------------------------

_ARCHETYPE_DEFAULTS: dict[str, dict] = {
    "full_brand": {
        "compositor_enabled": True,
        "badge_text": None,
        "default_mode": "image_always",
        "visual_source": "client_assets",
        "recommended_content_types": [
            "announcement", "community", "meme", "engagement",
            "educational", "brand_asset", "lifestyle", "market_commentary",
        ],
    },
    "has_identity": {
        "compositor_enabled": True,
        "badge_text": None,
        "default_mode": "image_optional",
        "visual_source": "hybrid",
        "recommended_content_types": [
            "announcement", "community", "educational",
        ],
    },
    "starting_fresh": {
        "compositor_enabled": False,
        "badge_text": None,
        "default_mode": "image_optional",
        "visual_source": "ai_generated",
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

def _visual_source_for_archetype(archetype: str) -> str:
    """Return the default visual source for an archetype."""
    return _ARCHETYPE_DEFAULTS.get(archetype, {}).get("visual_source", "ai_generated")


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
        "platforms": rec.platforms,
        "visual_source": {
            "primary": _visual_source_for_archetype(rec.archetype),
            "fallback": "ai_generated",
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
        f"**Platforms:** {', '.join(rec.platforms)}",
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


# ---------------------------------------------------------------------------
# Content calendar generation
# ---------------------------------------------------------------------------

_CALENDAR_PROMPT = """\
You are a content strategist. Generate a weekly content calendar for the brand below.

Brand: {brand_name}
Description: {description}
Platforms: {platforms}
Archetype: {archetype}
Enabled content types: {content_types}
Visual style: {visual_style}
Posting frequency: {posting_frequency}

Create a 7-day content calendar (Monday through Sunday). Each day should have:
- Day of week
- Content type (from the enabled list)
- Topic/theme idea
- Brief description (1 sentence)
- Best posting time (general, e.g. "9am", "12pm", "6pm")
- Platform(s) to post on

Guidelines:
- Vary content types across the week
- Space out similar content (don't do 2 announcements back to back)
- Weekend content can be lighter/more engaging
- Match posting times to platform norms
- If posting frequency is low, mark some days as "rest" (no post)

Return ONLY valid JSON:
{{
  "calendar": [
    {{
      "day": "Monday",
      "content_type": "announcement",
      "topic": "Product update highlight",
      "description": "Share the latest feature or improvement",
      "time": "9am",
      "platforms": ["x"]
    }}
  ],
  "weekly_theme": "Brief theme tying the week together",
  "notes": "Any strategic notes about the cadence"
}}"""


async def generate_content_calendar(
    brand_name: str,
    description: str,
    platforms: list[str],
    rec: StrategyRecommendation,
    posting_frequency: str = "",
    creative_brief: str = "",
    never_do: list[str] | None = None,
) -> str:
    """Generate a weekly content calendar and return it as markdown.

    Also saves to brand/content_calendar.md.
    """
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    prompt = _CALENDAR_PROMPT.format(
        brand_name=brand_name,
        description=description,
        platforms=", ".join(platforms) if platforms else "x",
        archetype=rec.archetype,
        content_types=", ".join(rec.recommended_content_types),
        visual_style=rec.visual_style_notes or "modern",
        posting_frequency=posting_frequency or "daily",
    )

    # Conditionally append creative direction
    if creative_brief or never_do:
        creative_lines = ["\n\nCREATIVE DIRECTION:"]
        if creative_brief:
            creative_lines.append(f"Brief: {creative_brief}")
        if never_do:
            creative_lines.append("Never do: " + "; ".join(never_do))
        creative_lines.append(
            "Use the creative brief to inform topic choices and tone. "
            "Avoid anything in the never-do list."
        )
        prompt += "\n".join(creative_lines)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
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
        logger.warning("Calendar generation returned non-JSON, using fallback")
        data = {"calendar": [], "weekly_theme": "", "notes": ""}

    # Convert to markdown
    md = _calendar_to_markdown(data, brand_name)

    # Save to file
    cal_path = Path(settings.BRAND_FOLDER) / "content_calendar.md"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(md, encoding="utf-8")
    logger.info("Saved content_calendar.md")

    return md


def _calendar_to_markdown(data: dict, brand_name: str = "") -> str:
    """Convert calendar JSON to readable markdown."""
    name = brand_name or settings.BRAND_NAME
    lines = [
        f"# {name} — Weekly Content Calendar",
        "",
    ]

    theme = data.get("weekly_theme", "")
    if theme:
        lines.append(f"**Weekly Theme:** {theme}")
        lines.append("")

    lines.append("| Day | Type | Topic | Time | Platforms |")
    lines.append("|-----|------|-------|------|-----------|")

    for entry in data.get("calendar", []):
        day = entry.get("day", "?")
        ct = entry.get("content_type", "—")
        topic = entry.get("topic", "—")
        time_str = entry.get("time", "—")
        plats = ", ".join(entry.get("platforms", []))
        lines.append(f"| {day} | {ct} | {topic} | {time_str} | {plats} |")

    notes = data.get("notes", "")
    if notes:
        lines.extend(["", "## Notes", "", notes])

    # Details section
    details = [e for e in data.get("calendar", []) if e.get("description")]
    if details:
        lines.extend(["", "## Daily Details", ""])
        for entry in details:
            day = entry.get("day", "?")
            desc = entry.get("description", "")
            lines.append(f"- **{day}:** {desc}")

    return "\n".join(lines)
