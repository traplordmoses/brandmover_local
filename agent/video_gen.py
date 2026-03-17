"""
Remotion Video Generator — programmatic motion graphics from text briefs.

Uses Claude to generate scene JSON from a brief + BrandConfig, then renders
via Remotion CLI (npx remotion render) to produce branded promo/explainer videos.

Supports 13 scene types, stock footage asset resolution, TTS voiceover,
and ffmpeg audio mixing.

Pipeline:
  brief + BrandConfig → Claude → SceneData JSON → asset resolution →
  Remotion render → (optional) voiceover + audio mix → MP4

Public API:
    result = await generate_video("15 second promo for our launch")
"""

import asyncio
import json
import logging
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from agent.asset_pipeline import resolve_storyboard_assets
from agent.audio_pipeline import generate_voiceover, mix_audio
from agent.compositor_config import get_config as get_brand_config
from agent.paths import PROJECT_ROOT
from config import settings

logger = logging.getLogger(__name__)

REMOTION_DIR = PROJECT_ROOT / "video" / "remotion"
OUTPUT_DIR = PROJECT_ROOT / "state" / "outputs"

# Shared sync Anthropic client for scene generation (called via asyncio.to_thread)
_sync_anthropic: anthropic.Anthropic | None = None


def _get_sync_anthropic() -> anthropic.Anthropic:
    """Return a shared sync Anthropic client (lazy-initialized)."""
    global _sync_anthropic
    if _sync_anthropic is None:
        _sync_anthropic = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _sync_anthropic

# Use Sonnet for scene generation — good balance of quality and speed
_SCENE_GEN_MODEL = settings.SONNET_MODEL

# Format presets: (width, height)
_FORMAT_DIMENSIONS = {
    "square": (1080, 1080),
    "landscape": (1920, 1080),
    "portrait": (1080, 1920),
}

# ---------------------------------------------------------------------------
# Scene generation prompt — full 13-scene storyboard schema
# ---------------------------------------------------------------------------

_SCENE_GEN_SYSTEM = """\
<task>
You are a motion graphics director and video scriptwriter. Given a brief
and brand config, create a Storyboard JSON for a branded promo/explainer video.
</task>

<rules>
- Title scene first, CTA scene last — always
- Title scene MUST have a label (e.g. "INTRODUCING", "MEET", "WELCOME TO")
- CTA scene MUST have buttonText AND url fields — always include a call-to-action button
- Keep ALL text SHORT — this is video, not a document:
  * tagline lines: max 5 words each
  * text_only: max 6 words
  * feature_list items: max 4 words each (just the name, NO descriptions)
  * steps headings: max 4 words, details: max 6 words
  * CTA lines: max 4 words each
- Use the brand's colors and fonts from the config
- For short videos (15-20s): 7-8 scenes, fast pace (2-2.5s per scene). PACK IT.
- For medium videos (30-45s): 10-14 scenes, mix scene types
- For long videos (45-90s): 15-25 scenes, use varied scene types including
  stat, icon_grid, data_viz, stock_footage, icon_reveal for visual richness

CRITICAL SCENE VARIETY RULES:
- NEVER use more than 1 tagline scene per video
- NEVER use more than 1 text_only scene per video
- NEVER repeat the same scene type back-to-back
- Max 2 text-only scenes (tagline + text_only combined) per video. The rest MUST be visual.
- EVERY video MUST include at least 2 of: feature_count, stat, chat_demo, steps
  These are the high-impact visual scenes that make videos look premium.
- The ideal flow for a 20s video is:
  title → tagline → feature_count → chat_demo OR steps → feature_list → stat → cta
  Follow this pattern. Study the examples below carefully.

- stat scenes: use animate 'countUp' for large numbers
- If the brief describes a visual (bracket, grid, coins), use data_viz or icon_grid
- narration field: 1-2 sentences of what a voiceover would say (helps pacing)
- For dark-themed brands: use background 'gradient'
- For light-themed brands: use background 'clean' or 'dots'
- Total duration should match the brief. Default ~20 seconds if unspecified.
- durationFrames per scene: 50-90 frames (1.5-3 seconds). Keep it snappy.
</rules>

<scene_types>
13 scene types available:
- title: Brand logo + headline + optional subheadline/disclaimer
- tagline: Big text with accent-colored keywords. Uses lines array.
- text_only: Bold statement on clean background. Sizes: medium/large/xlarge.
- stat: Animated number/counter + label. Can count up, show raw number, handwritten suffix.
- feature_list: Staggered list. Layouts: centered-stack, left-aligned.
- chat_demo: Chat bubble mockup (user <-> bot).
- steps: Numbered instruction steps (2-3 max).
- icon_grid: Grid of repeated icons with staggered reveal + optional checkmarks.
- data_viz: bracket, dot_matrix_number, dot_grid, bar_chart.
- stock_footage: Full-bleed or inset video/image. Specify query for Pexels search.
- icon_reveal: Centered icon(s) with caption. Icon names: atom, globe, file-text, zap, shield, rocket, wallet, lock, users, chart-bar, trophy, star, coin, layers, credit-card.
- feature_count: Large number + subtitle side by side.
- cta: Closing CTA with lines array + optional url + button.
</scene_types>

<schema>
The output must be valid JSON matching this TypeScript interface:

interface SceneData {
  config: {
    width: number;       // 1080 for square/portrait, 1920 for landscape
    height: number;      // 1080 for square/landscape, 1920 for portrait
    fps: 30;
    durationInSeconds: number;
    brand: {
      name: string;
      primaryColor: string;    // hex
      accentColor: string;     // hex
      backgroundColor: string; // hex
      textColor: string;       // hex — auto from bg brightness
      fontFamily: string;
      accentFontFamily?: string;
    };
  };
  audio?: {
    voiceover?: boolean;
    musicUrl?: string;
  };
  scenes: Array<Scene>;
}

type Scene =
  | {
      type: "title";
      label?: string;          // e.g. "INTRODUCING"
      headline: string;
      subheadline?: string;
      disclaimer?: string;
      background?: "gradient" | "clean" | "dots";
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "tagline";
      supertext?: string;
      lines: Array<{ text: string; accent?: boolean }>;
      background?: "gradient" | "clean" | "dots";
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "text_only";
      text: string;
      size?: "medium" | "large" | "xlarge";
      background?: "gradient" | "clean" | "dots";
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "stat";
      value: string;            // "3" or "$2.4B"
      label: string;
      suffix?: string;          // "%" or "+"
      animate?: "countUp" | "none";
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "feature_list";
      title?: string;
      items: Array<{ icon?: string; text: string }>;
      layout?: "centered-stack" | "left-aligned";
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "chat_demo";
      messages: Array<{ text: string; isUser: boolean; label?: string }>;
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "steps";
      title?: string;
      steps: Array<{ number: string; heading: string; detail: string }>;
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "icon_grid";
      icons: Array<{ name: string; label?: string; checked?: boolean }>;
      columns?: number;
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "data_viz";
      vizType: "bracket" | "dot_matrix_number" | "dot_grid" | "bar_chart";
      data: any;               // shape depends on vizType
      caption?: string;
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "stock_footage";
      query: string;            // Pexels search query
      layout?: "full-bleed" | "inset";
      caption?: string;
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "icon_reveal";
      icons: Array<{ name: string; caption?: string }>;
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "feature_count";
      count: number | string;
      subtitle: string;
      narration?: string;
      durationFrames: number;
    }
  | {
      type: "cta";
      lines: Array<{ text: string; accent?: boolean }>;
      url?: string;
      buttonText?: string;
      background?: "gradient" | "clean" | "dots";
      narration?: string;
      durationFrames: number;
    };
</schema>

<example_1>
Brief: "15 second promo for BloFin MCP launch"
Brand: BloFin, primary=#00D26A, accent=#00D26A, bg=#0a0f0a (dark), font=Inter
Format: square (1080x1080), Theme: dark

Output:
{
  "config": {
    "width": 1080,
    "height": 1080,
    "fps": 30,
    "durationInSeconds": 16,
    "brand": {
      "name": "BloFin",
      "primaryColor": "#00D26A",
      "accentColor": "#00D26A",
      "backgroundColor": "#0a0f0a",
      "textColor": "#FFFFFF",
      "fontFamily": "Inter"
    }
  },
  "scenes": [
    {
      "type": "title",
      "label": "INTRODUCING",
      "headline": "BloFin MCP",
      "background": "gradient",
      "narration": "Introducing BloFin MCP.",
      "durationFrames": 75
    },
    {
      "type": "tagline",
      "supertext": "MODEL CONTEXT PROTOCOL",
      "lines": [
        { "text": "Your AI now speaks", "accent": false },
        { "text": "fluent trading.", "accent": true }
      ],
      "background": "gradient",
      "narration": "Your AI now speaks fluent trading.",
      "durationFrames": 90
    },
    {
      "type": "feature_count",
      "count": 3,
      "subtitle": "tools. One connection.",
      "narration": "Three powerful tools through a single connection.",
      "durationFrames": 60
    },
    {
      "type": "chat_demo",
      "messages": [
        { "text": "What's the BTC funding rate?", "isUser": true },
        { "text": "BTC funding rate: +0.0082%\\n4h interval - Next in 1h 23m", "isUser": false, "label": "BLOFIN MCP" },
        { "text": "Open a 2x long on BTC", "isUser": true },
        { "text": "Order placed\\nBTC-USDT - Long - 2x - Market", "isUser": false, "label": "BLOFIN MCP" }
      ],
      "narration": "Ask about funding rates. Place trades. All through natural conversation.",
      "durationFrames": 120
    },
    {
      "type": "steps",
      "title": "SETUP IN MINUTES",
      "steps": [
        { "number": "01", "heading": "Get your API key", "detail": "blofin.com - APIs - Create Key - Select MCP" },
        { "number": "02", "heading": "Connect to Claude", "detail": "Paste config into Claude Desktop - restart" },
        { "number": "03", "heading": "Start trading", "detail": "Ask anything. Your AI handles the rest." }
      ],
      "narration": "Set up in minutes. Get your key, connect to Claude, and start trading.",
      "durationFrames": 90
    },
    {
      "type": "cta",
      "lines": [
        { "text": "Trade smarter.", "accent": false },
        { "text": "Talk to your exchange.", "accent": true }
      ],
      "url": "blofin.com/en/blofin-mcp",
      "buttonText": "Get Started Free",
      "background": "gradient",
      "narration": "Trade smarter. Talk to your exchange.",
      "durationFrames": 60
    }
  ]
}
</example_1>

<example_2>
Brief: "60 second explainer showing Kalshi bracket trading for March Madness"
Brand: Kalshi, primary=#4F46E5, accent=#F59E0B, bg=#FFFFFF (light), font=Plus Jakarta Sans
Format: square (1080x1080), Theme: light

Output:
{
  "config": {
    "width": 1080,
    "height": 1080,
    "fps": 30,
    "durationInSeconds": 62,
    "brand": {
      "name": "Kalshi",
      "primaryColor": "#4F46E5",
      "accentColor": "#F59E0B",
      "backgroundColor": "#FFFFFF",
      "textColor": "#1A1A1A",
      "fontFamily": "Plus Jakarta Sans"
    }
  },
  "scenes": [
    {
      "type": "title",
      "label": "KALSHI PRESENTS",
      "headline": "Bracket Trading",
      "subheadline": "March Madness Edition",
      "background": "clean",
      "narration": "Kalshi presents Bracket Trading for March Madness.",
      "durationFrames": 90
    },
    {
      "type": "tagline",
      "lines": [
        { "text": "Your bracket.", "accent": false },
        { "text": "Real stakes.", "accent": true }
      ],
      "background": "dots",
      "narration": "Your bracket, with real stakes on the line.",
      "durationFrames": 75
    },
    {
      "type": "text_only",
      "text": "Predict game outcomes and earn real money.",
      "size": "large",
      "background": "clean",
      "narration": "Predict game outcomes and earn real money on every matchup.",
      "durationFrames": 90
    },
    {
      "type": "data_viz",
      "vizType": "bracket",
      "data": {
        "rounds": [
          { "label": "Sweet 16", "matchups": ["Duke vs UNC", "Kansas vs Baylor"] },
          { "label": "Elite 8", "matchups": ["Winner vs Winner"] },
          { "label": "Final Four", "matchups": ["TBD"] }
        ]
      },
      "caption": "Full tournament bracket trading",
      "narration": "Trade the full bracket from Sweet 16 through the Final Four.",
      "durationFrames": 150
    },
    {
      "type": "stat",
      "value": "67",
      "label": "games available to trade",
      "animate": "countUp",
      "narration": "Sixty-seven games available to trade this tournament.",
      "durationFrames": 90
    },
    {
      "type": "icon_grid",
      "icons": [
        { "name": "trophy", "label": "Winner", "checked": true },
        { "name": "chart-bar", "label": "Spread", "checked": true },
        { "name": "users", "label": "Matchup", "checked": true },
        { "name": "star", "label": "MVP", "checked": false },
        { "name": "layers", "label": "Parlay", "checked": false },
        { "name": "zap", "label": "Live", "checked": true }
      ],
      "columns": 3,
      "narration": "Trade winners, spreads, matchups, and live in-game events.",
      "durationFrames": 120
    },
    {
      "type": "stock_footage",
      "query": "basketball tournament crowd cheering",
      "layout": "full-bleed",
      "caption": "Feel the energy",
      "narration": "Feel the energy of every game with real skin in the bracket.",
      "durationFrames": 90
    },
    {
      "type": "feature_list",
      "title": "WHY KALSHI",
      "items": [
        { "icon": "shield", "text": "CFTC regulated exchange" },
        { "icon": "zap", "text": "Instant deposits & withdrawals" },
        { "icon": "wallet", "text": "Start with as little as $1" }
      ],
      "layout": "left-aligned",
      "narration": "Kalshi is a CFTC regulated exchange with instant deposits. Start with just one dollar.",
      "durationFrames": 120
    },
    {
      "type": "steps",
      "title": "GET STARTED",
      "steps": [
        { "number": "01", "heading": "Sign up free", "detail": "kalshi.com - 2 minute signup" },
        { "number": "02", "heading": "Fund your account", "detail": "Instant deposit via bank or card" },
        { "number": "03", "heading": "Pick your games", "detail": "Browse brackets and place trades" }
      ],
      "narration": "Sign up free, fund your account, and start picking games.",
      "durationFrames": 120
    },
    {
      "type": "icon_reveal",
      "icons": [
        { "name": "trophy", "caption": "Win big this March" }
      ],
      "narration": "Win big this March.",
      "durationFrames": 75
    },
    {
      "type": "stat",
      "value": "$2.4B",
      "label": "traded on Kalshi to date",
      "animate": "none",
      "narration": "Over 2.4 billion dollars traded on Kalshi to date.",
      "durationFrames": 90
    },
    {
      "type": "feature_count",
      "count": "1M+",
      "subtitle": "traders and counting",
      "narration": "Over one million traders and counting.",
      "durationFrames": 75
    },
    {
      "type": "cta",
      "lines": [
        { "text": "Your bracket.", "accent": false },
        { "text": "Your edge.", "accent": true }
      ],
      "url": "kalshi.com/march-madness",
      "buttonText": "Trade Now",
      "background": "dots",
      "narration": "Your bracket. Your edge. Trade now on Kalshi.",
      "durationFrames": 75
    }
  ]
}
</example_2>

Return ONLY the JSON object — no markdown, no explanation."""


# ---------------------------------------------------------------------------
# Valid scene types for post-processing validation
# ---------------------------------------------------------------------------

_VALID_SCENE_TYPES = {
    "title", "tagline", "text_only", "stat", "feature_list", "chat_demo",
    "steps", "icon_grid", "data_viz", "stock_footage", "icon_reveal",
    "feature_count", "cta",
}


@dataclass
class VideoResult:
    """Result of a video generation."""
    video_path: str = ""
    scene_data: dict | None = None
    duration_seconds: float = 0.0
    render_time_seconds: float = 0.0
    format: str = "square"
    error: str = ""


def _hex_brightness(hex_color: str) -> float:
    """Return perceived brightness of a hex color (0-255 scale)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return 128.0  # assume mid-range if unparseable
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    # Perceived brightness formula (ITU-R BT.601)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _darken_hex(hex_color: str, factor: float = 0.25) -> str:
    """Darken a hex color while preserving its hue. factor=0.25 keeps 25% brightness."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#0a0f1a"
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _brand_config_to_theme() -> dict:
    """Convert BrandConfig to the theme dict expected by Remotion.

    Auto-detects textColor from background brightness and extracts
    accentFontFamily if available.
    """
    config = get_brand_config()

    # Extract primary and accent colors
    primary_hex = "#72e1ff"  # default
    accent_hex = "#72e1ff"
    bg_hex = "#0a0f1a"

    if "primary" in config.colors:
        primary_hex = config.colors["primary"].hex
    elif config.colors:
        first = next(iter(config.colors.values()))
        primary_hex = first.hex

    if "accent_1" in config.colors:
        accent_hex = config.colors["accent_1"].hex
    elif "accent" in config.colors:
        accent_hex = config.colors["accent"].hex
    else:
        accent_hex = primary_hex

    if "background" in config.colors:
        bg_hex = config.colors["background"].hex

    # Auto-detect text color from background brightness
    brightness = _hex_brightness(bg_hex)
    text_color = "#FFFFFF" if brightness < 128 else "#1A1A1A"

    # Extract fonts
    font_family = "Inter"
    accent_font_family = None

    if "display" in config.fonts:
        font_family = config.fonts["display"].family
    elif config.fonts:
        first_font = next(iter(config.fonts.values()))
        font_family = first_font.family

    # Look for a secondary/accent font
    for key in ("accent", "body", "secondary"):
        if key in config.fonts and config.fonts[key].family != font_family:
            accent_font_family = config.fonts[key].family
            break

    theme = {
        "name": config.brand_name or settings.BRAND_NAME,
        "primaryColor": primary_hex,
        "accentColor": accent_hex,
        "backgroundColor": bg_hex,
        "textColor": text_color,
        "fontFamily": font_family,
    }

    if accent_font_family:
        theme["accentFontFamily"] = accent_font_family

    return theme


def _detect_theme(bg_hex: str) -> str:
    """Detect 'dark' or 'light' theme from background color.

    Default to dark — most video content looks better on dark backgrounds.
    Only use light if the brand bg is clearly light (brightness > 200).
    """
    return "light" if _hex_brightness(bg_hex) > 200 else "dark"


def generate_scene_json(
    brief: str,
    duration: int | None = None,
    format: str = "square",
    theme: str | None = None,
) -> dict:
    """Use Claude to generate SceneData JSON from a brief + brand config.

    Args:
        brief: The video brief/description.
        duration: Target duration in seconds. None for auto (~20s).
        format: 'square', 'landscape', or 'portrait'.
        theme: 'dark' or 'light'. Auto-detected from brand bg if None.
    """
    client = _get_sync_anthropic()
    brand_theme = _brand_config_to_theme()

    # Resolve format dimensions
    width, height = _FORMAT_DIMENSIONS.get(format, (1080, 1080))

    # Auto-detect theme if not specified
    if theme is None:
        theme = _detect_theme(brand_theme["backgroundColor"])

    # Force appropriate colors for the requested theme.
    # Instead of generic black/white, darken/lighten the BRAND's actual color
    # to preserve the brand's color identity (blue stays blue, green stays green).
    if theme == "dark":
        # Darken the brand's actual bg to ~25% brightness — cinematic but on-brand
        brand_theme["backgroundColor"] = _darken_hex(brand_theme["backgroundColor"], 0.25)
        brand_theme["textColor"] = "#ffffff"
        # If primary color has poor contrast on dark bg, force to white
        if _hex_brightness(brand_theme["primaryColor"]) < 180:
            brand_theme["primaryColor"] = "#ffffff"
    elif theme == "light":
        brand_theme["backgroundColor"] = "#f5f5f5"
        brand_theme["textColor"] = "#1a1a1a"
        # If primary color has poor contrast on light bg, force to dark
        if _hex_brightness(brand_theme["primaryColor"]) > 180:
            brand_theme["primaryColor"] = "#1a1a1a"

    # Force reliable sans-serif font for video rendering.
    # Brand fonts (Orbitron, etc.) may not be available in all render
    # environments. Inter is always loaded via Google Fonts in Remotion.
    brand_theme["fontFamily"] = "Inter"

    # Build user message with all hints
    duration_hint = f"Target duration: {duration} seconds" if duration else "Target duration: ~20 seconds"
    format_hint = f"Format: {format} ({width}x{height})"
    theme_hint = f"Theme: {theme}"

    user_message = (
        f"Brief: {brief}\n\n"
        f"Brand config:\n"
        f"- Name: {brand_theme['name']}\n"
        f"- Primary color: {brand_theme['primaryColor']}\n"
        f"- Accent color: {brand_theme['accentColor']}\n"
        f"- Background: {brand_theme['backgroundColor']}\n"
        f"- Text color: {brand_theme['textColor']}\n"
        f"- Font: {brand_theme['fontFamily']}\n"
    )
    if brand_theme.get("accentFontFamily"):
        user_message += f"- Accent font: {brand_theme['accentFontFamily']}\n"

    user_message += (
        f"\n{duration_hint}\n"
        f"{format_hint}\n"
        f"{theme_hint}\n"
    )

    response = client.messages.create(
        model=_SCENE_GEN_MODEL,
        max_tokens=4096,
        system=_SCENE_GEN_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()

    # Parse JSON — handle markdown code blocks
    if text.startswith("```"):
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()

    try:
        scene_data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            scene_data = json.loads(match.group())
        else:
            raise ValueError(
                f"Failed to parse scene JSON from LLM response: {text[:200]}"
            )

    # Validate required fields
    if "config" not in scene_data or "scenes" not in scene_data:
        raise ValueError("Scene JSON missing 'config' or 'scenes' fields")

    # --- Post-processing ---

    # Override brand theme with actual config (in case LLM deviated)
    scene_data["config"]["brand"] = brand_theme
    scene_data["config"]["width"] = width
    scene_data["config"]["height"] = height
    scene_data["config"]["fps"] = 30

    # Validate scene types — strip invalid ones
    valid_scenes = []
    for scene in scene_data["scenes"]:
        if scene.get("type") in _VALID_SCENE_TYPES:
            valid_scenes.append(scene)
        else:
            logger.warning(
                "Stripping invalid scene type: %s", scene.get("type")
            )
    scene_data["scenes"] = valid_scenes

    # Fix common LLM confusions between scene types
    for scene in scene_data["scenes"]:
        # LLM often generates icon_grid with an icons array — that's icon_reveal
        if scene.get("type") == "icon_grid" and "icons" in scene and "icon" not in scene:
            scene["type"] = "icon_reveal"
            # Ensure icons have the right shape for icon_reveal
            if isinstance(scene.get("icons"), list):
                for ic in scene["icons"]:
                    if "name" not in ic and "icon" in ic:
                        ic["name"] = ic.pop("icon")
        # icon_grid requires icon, rows, cols — add defaults if missing
        if scene.get("type") == "icon_grid":
            scene.setdefault("icon", "star")
            scene.setdefault("rows", 4)
            scene.setdefault("cols", 5)
            scene.setdefault("revealPattern", "staggered-ltr")
        # stat scene: LLM often uses "label" instead of "suffix"
        if scene.get("type") == "stat" and "label" in scene and "suffix" not in scene:
            scene["suffix"] = scene.pop("label")
        # feature_list: ensure layout defaults
        if scene.get("type") == "feature_list":
            scene.setdefault("layout", "centered-stack")

    # Truncate text that's too long (safety net for LLM verbosity)
    for scene in scene_data["scenes"]:
        stype = scene.get("type")
        if stype == "feature_list":
            for item in scene.get("items", []):
                words = item.get("text", "").split()
                if len(words) > 5:
                    item["text"] = " ".join(words[:4])
        elif stype == "text_only":
            words = scene.get("text", "").split()
            if len(words) > 8:
                scene["text"] = " ".join(words[:7])
        elif stype == "tagline":
            for line in scene.get("lines", []):
                words = line.get("text", "").split()
                if len(words) > 6:
                    line["text"] = " ".join(words[:5])
        elif stype == "steps":
            for step in scene.get("steps", []):
                words = step.get("heading", "").split()
                if len(words) > 5:
                    step["heading"] = " ".join(words[:4])
                words = step.get("detail", "").split()
                if len(words) > 8:
                    step["detail"] = " ".join(words[:7])

    # Enforce scene diversity — deduplicate consecutive same-type scenes
    deduped = []
    for scene in scene_data["scenes"]:
        if deduped and scene.get("type") == deduped[-1].get("type"):
            logger.warning("Removing consecutive duplicate scene type: %s", scene.get("type"))
            continue
        deduped.append(scene)
    scene_data["scenes"] = deduped

    # Enforce: title must have label, CTA must have button
    for scene in scene_data["scenes"]:
        if scene.get("type") == "title":
            scene.setdefault("label", "INTRODUCING")
        if scene.get("type") == "cta":
            scene.setdefault("buttonText", "Learn More")
            scene.setdefault("url", brand_theme.get("name", "").lower().replace(" ", "") + ".com")

    # Cap tagline/text_only to max 1 each — convert extras to feature_count
    tagline_count = 0
    text_only_count = 0
    for scene in scene_data["scenes"]:
        if scene.get("type") == "tagline":
            tagline_count += 1
            if tagline_count > 1:
                # Convert to feature_count as a visual alternative
                scene["type"] = "feature_count"
                lines = scene.pop("lines", [])
                scene["count"] = len(lines)
                scene["subtitle"] = lines[0].get("text", "features") if lines else "features"
                scene.pop("supertext", None)
                logger.info("Converted excess tagline to feature_count")
        elif scene.get("type") == "text_only":
            text_only_count += 1
            if text_only_count > 1:
                scene["type"] = "stat"
                scene["value"] = "100%"
                scene["suffix"] = scene.pop("text", "committed")
                scene["animate"] = "fadeIn"
                scene.pop("size", None)
                logger.info("Converted excess text_only to stat")

    # Calculate duration from scenes
    total_frames = sum(s.get("durationFrames", 90) for s in scene_data["scenes"])
    scene_data["config"]["durationInSeconds"] = round(total_frames / 30, 1)

    logger.info(
        "Generated scene JSON: %d scenes, %.1fs, %s (%dx%d)",
        len(scene_data["scenes"]),
        scene_data["config"]["durationInSeconds"],
        format,
        width,
        height,
    )
    return scene_data


async def render_video(scene_data: dict, output_path: str | None = None) -> str:
    """Render scene data to MP4 via Remotion CLI.

    Args:
        scene_data: SceneData dict matching the Remotion schema.
        output_path: Optional output path. Auto-generated if None.

    Returns:
        Path to the rendered MP4 file.
    """
    if not REMOTION_DIR.exists():
        raise RuntimeError(f"Remotion project not found at {REMOTION_DIR}")

    # Check node_modules exist
    if not (REMOTION_DIR / "node_modules").exists():
        logger.info("Installing Remotion dependencies...")
        await asyncio.to_thread(
            subprocess.run,
            ["npm", "install"],
            cwd=str(REMOTION_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not output_path:
        uid = uuid.uuid4().hex[:8]
        ts = int(time.time())
        output_path = str(OUTPUT_DIR / f"promo_{ts}_{uid}.mp4")

    # Write props to temp file
    props_path = REMOTION_DIR / "props.json"
    props_path.write_text(json.dumps(scene_data, indent=2), encoding="utf-8")

    # Extract dimensions from scene config
    width = scene_data.get("config", {}).get("width", 1080)
    height = scene_data.get("config", {}).get("height", 1080)

    # Dynamic timeout: proportional to video duration, capped at 300s
    duration_seconds = scene_data.get("config", {}).get("durationInSeconds", 20)
    render_timeout = min(max(int(duration_seconds * 10), 60), 300)

    # Render via Remotion CLI
    cmd = [
        "npx", "remotion", "render",
        "PromoVideo",
        output_path,
        f"--props={props_path}",
        f"--width={width}",
        f"--height={height}",
        "--codec=h264",
        "--image-format=jpeg",
        "--overwrite",
    ]

    logger.info("Rendering video: %s (timeout=%ds)", " ".join(cmd), render_timeout)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=str(REMOTION_DIR),
            capture_output=True,
            text=True,
            timeout=render_timeout,
        )

        if result.returncode != 0:
            error_msg = (
                result.stderr[-500:] if result.stderr else "Unknown render error"
            )
            raise RuntimeError(f"Remotion render failed: {error_msg}")

    finally:
        # Clean up props file
        props_path.unlink(missing_ok=True)

    if not Path(output_path).exists():
        raise RuntimeError(
            f"Render completed but output file not found: {output_path}"
        )

    logger.info("Video rendered: %s", output_path)
    return output_path


async def generate_video(
    brief: str,
    duration: int | None = None,
    format: str = "square",
    theme: str | None = None,
    voiceover: bool = False,
    output_path: str | None = None,
) -> VideoResult:
    """Full pipeline: brief -> scene JSON -> assets -> render -> audio -> MP4.

    Args:
        brief: The video brief/description.
        duration: Target duration in seconds. None for auto (~20s).
        format: 'square', 'landscape', or 'portrait'.
        theme: 'dark' or 'light'. Auto-detected from brand bg if None.
        voiceover: Whether to generate TTS voiceover from narrations.
        output_path: Optional output path. Auto-generated if None.

    Returns:
        VideoResult with video_path, scene_data, duration, timing, and format.
    """
    result = VideoResult(format=format)
    t0 = time.monotonic()

    try:
        # Step 1: Generate scene JSON via LLM
        scene_data = await asyncio.to_thread(
            generate_scene_json, brief, duration, format, theme
        )
        result.scene_data = scene_data
        result.duration_seconds = scene_data["config"]["durationInSeconds"]

        # Step 2: Resolve assets (download stock footage for stock_footage scenes)
        scene_data = await resolve_storyboard_assets(scene_data)

        # Step 3: Render via Remotion CLI
        video_path = await render_video(scene_data, output_path)
        result.video_path = video_path

        # Step 4: If voiceover requested, generate TTS and mix audio
        if voiceover:
            vo_path = await generate_voiceover(scene_data.get("scenes", []))
            if vo_path:
                # Check for background music URL in scene config
                music_path = None
                audio_config = scene_data.get("audio", {})
                if audio_config.get("musicUrl"):
                    # Future: download music_url to local file
                    pass

                final_path = await asyncio.to_thread(
                    mix_audio, video_path, vo_path, music_path
                )
                result.video_path = final_path
                logger.info("Voiceover applied: %s", final_path)

    except Exception as e:
        logger.error("Video generation failed: %s", e)
        result.error = str(e)

    result.render_time_seconds = round(time.monotonic() - t0, 2)
    return result
