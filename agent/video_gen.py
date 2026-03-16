"""
Remotion Video Generator — programmatic motion graphics from text briefs.

Uses Claude to generate scene JSON from a brief + BrandConfig, then renders
via Remotion CLI (npx remotion render) to produce branded promo/explainer videos.

Pipeline:
  brief + BrandConfig → Claude → SceneData JSON → Remotion → MP4

Public API:
    result = await generate_video("15 second promo for our launch")
"""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import anthropic

from agent.compositor_config import get_config as get_brand_config
from agent.paths import PROJECT_ROOT
from config import settings

logger = logging.getLogger(__name__)

REMOTION_DIR = PROJECT_ROOT / "video" / "remotion"
OUTPUT_DIR = PROJECT_ROOT / "state" / "outputs"

# Use Sonnet for scene generation — good balance of quality and speed
_SCENE_GEN_MODEL = settings.SONNET_MODEL


# ---------------------------------------------------------------------------
# Scene generation prompt
# ---------------------------------------------------------------------------

_SCENE_GEN_SYSTEM = """You are a motion graphics director. Given a brief and brand config,
output a SceneData JSON that creates a punchy branded promo video.

<rules>
- 5-7 scenes total
- Title scene first, CTA scene last
- Keep text SHORT — max 8 words per line
- Use the brand's primary and accent colors from the provided config
- Chat demo scenes should feel like a real product interaction
- Steps scenes should have 2-3 steps max
- Total duration 12-20 seconds (360-600 frames at 30fps)
- Each scene gets 60-120 frames (2-4 seconds)
</rules>

<schema>
The output must be valid JSON matching this TypeScript interface:

interface SceneData {
  config: {
    width: 1080;
    height: 1080;
    fps: 30;
    durationInSeconds: number;
    brand: {
      name: string;
      primaryColor: string;
      accentColor: string;
      backgroundColor: string;
      fontFamily: string;
    };
  };
  scenes: Array<
    | { type: "title"; label: string; headline: string; durationFrames: number }
    | { type: "tagline"; supertext: string; line1: string; line2: string; accentLine: 1 | 2; durationFrames: number }
    | { type: "feature_count"; count: number; subtitle: string; durationFrames: number }
    | { type: "chat_demo"; messages: Array<{ text: string; isUser: boolean; label?: string }>; durationFrames: number }
    | { type: "steps"; title: string; steps: Array<{ number: string; heading: string; detail: string }>; durationFrames: number }
    | { type: "cta"; line1: string; line2: string; accentLine: 1 | 2; url: string; buttonText: string; durationFrames: number }
  >;
}
</schema>

<example>
Brief: "15 second promo for BloFin MCP launch"
Brand: BloFin, primary=#00D26A, accent=#00D26A

Output:
{
  "config": {
    "width": 1080, "height": 1080, "fps": 30, "durationInSeconds": 16.5,
    "brand": {
      "name": "BloFin",
      "primaryColor": "#00D26A",
      "accentColor": "#00D26A",
      "backgroundColor": "#0a0f0a",
      "fontFamily": "Inter"
    }
  },
  "scenes": [
    { "type": "title", "label": "INTRODUCING", "headline": "BloFin", "durationFrames": 75 },
    { "type": "tagline", "supertext": "MODEL CONTEXT PROTOCOL", "line1": "Your AI now speaks", "line2": "fluent trading.", "accentLine": 2, "durationFrames": 90 },
    { "type": "feature_count", "count": 3, "subtitle": "tools. One connection.", "durationFrames": 60 },
    { "type": "chat_demo", "messages": [
      { "text": "What's the BTC funding rate?", "isUser": true },
      { "text": "BTC funding rate: +0.0082%\\n4h interval · Next in 1h 23m", "isUser": false, "label": "BLOFIN MCP" },
      { "text": "Open a 2x long on BTC", "isUser": true },
      { "text": "✓ Order placed\\nBTC-USDT · Long · 2x · Market", "isUser": false, "label": "BLOFIN MCP" }
    ], "durationFrames": 120 },
    { "type": "steps", "title": "SETUP IN MINUTES", "steps": [
      { "number": "01", "heading": "Get your API key", "detail": "blofin.com → APIs → Create Key → Select MCP" },
      { "number": "02", "heading": "Connect to Claude", "detail": "Paste config into Claude Desktop → restart" },
      { "number": "03", "heading": "Start trading", "detail": "Ask anything. Your AI handles the rest." }
    ], "durationFrames": 90 },
    { "type": "cta", "line1": "Trade smarter.", "line2": "Talk to your exchange.", "accentLine": 2, "url": "blofin.com/en/blofin-mcp", "buttonText": "Get Started Free", "durationFrames": 60 }
  ]
}
</example>

Return ONLY the JSON object — no markdown, no explanation."""


@dataclass
class VideoResult:
    """Result of a video generation."""
    video_path: str = ""
    scene_data: dict = None
    duration_seconds: float = 0.0
    render_time_seconds: float = 0.0
    error: str = ""


def _brand_config_to_theme() -> dict:
    """Convert BrandConfig to the theme dict expected by Remotion."""
    config = get_brand_config()

    # Extract primary and accent colors
    primary_hex = "#72e1ff"  # default
    accent_hex = "#72e1ff"
    bg_hex = "#0a0f1a"

    if "primary" in config.colors:
        primary_hex = config.colors["primary"].hex
    elif config.colors:
        # Use first color as primary
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

    # Extract font
    font_family = "Inter"
    if "display" in config.fonts:
        font_family = config.fonts["display"].family
    elif config.fonts:
        first_font = next(iter(config.fonts.values()))
        font_family = first_font.family

    return {
        "name": config.brand_name or settings.BRAND_NAME,
        "primaryColor": primary_hex,
        "accentColor": accent_hex,
        "backgroundColor": bg_hex,
        "fontFamily": font_family,
    }


def generate_scene_json(brief: str) -> dict:
    """Use Claude to generate SceneData JSON from a brief + brand config."""
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    brand_theme = _brand_config_to_theme()

    user_message = (
        f"Brief: {brief}\n\n"
        f"Brand config:\n"
        f"- Name: {brand_theme['name']}\n"
        f"- Primary color: {brand_theme['primaryColor']}\n"
        f"- Accent color: {brand_theme['accentColor']}\n"
        f"- Background: {brand_theme['backgroundColor']}\n"
        f"- Font: {brand_theme['fontFamily']}\n"
    )

    response = client.messages.create(
        model=_SCENE_GEN_MODEL,
        max_tokens=2048,
        system=_SCENE_GEN_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()

    # Parse JSON — handle markdown code blocks
    import re
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
            raise ValueError(f"Failed to parse scene JSON from LLM response: {text[:200]}")

    # Validate required fields
    if "config" not in scene_data or "scenes" not in scene_data:
        raise ValueError("Scene JSON missing 'config' or 'scenes' fields")

    # Ensure brand theme is set (override with actual config in case LLM deviated)
    scene_data["config"]["brand"] = brand_theme
    scene_data["config"]["width"] = 1080
    scene_data["config"]["height"] = 1080
    scene_data["config"]["fps"] = 30

    # Calculate duration from scenes
    total_frames = sum(s.get("durationFrames", 90) for s in scene_data["scenes"])
    scene_data["config"]["durationInSeconds"] = round(total_frames / 30, 1)

    logger.info(
        "Generated scene JSON: %d scenes, %.1fs",
        len(scene_data["scenes"]),
        scene_data["config"]["durationInSeconds"],
    )
    return scene_data


def render_video(scene_data: dict, output_path: str | None = None) -> str:
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
        subprocess.run(
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

    # Render via Remotion CLI
    cmd = [
        "npx", "remotion", "render",
        "PromoVideo",
        output_path,
        f"--props={props_path}",
        "--codec=h264",
        "--image-format=jpeg",
        "--overwrite",
    ]

    logger.info("Rendering video: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REMOTION_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown render error"
            raise RuntimeError(f"Remotion render failed: {error_msg}")

    finally:
        # Clean up props file
        props_path.unlink(missing_ok=True)

    if not Path(output_path).exists():
        raise RuntimeError(f"Render completed but output file not found: {output_path}")

    logger.info("Video rendered: %s", output_path)
    return output_path


async def generate_video(
    brief: str,
    output_path: str | None = None,
) -> VideoResult:
    """Full pipeline: brief → scene JSON → Remotion render → MP4.

    This is the main entry point called by the tool handler.
    """
    result = VideoResult()
    t0 = time.monotonic()

    try:
        # Step 1: Generate scene JSON
        scene_data = await asyncio.to_thread(generate_scene_json, brief)
        result.scene_data = scene_data

        # Step 2: Render via Remotion
        video_path = await asyncio.to_thread(render_video, scene_data, output_path)
        result.video_path = video_path
        result.duration_seconds = scene_data["config"]["durationInSeconds"]

    except Exception as e:
        logger.error("Video generation failed: %s", e)
        result.error = str(e)

    result.render_time_seconds = round(time.monotonic() - t0, 2)
    return result
