"""
Background generator for video promos.
Generates animated abstract backgrounds using AI video models.

Supports:
- fal.ai (Minimax Video, Kling)
- Replicate (Kling, Runway)
- Local (static image or pre-rendered loop)

The output is always a video file that can be looped/extended to match the promo duration.
"""

import os
import re
import shutil
import subprocess
import logging
from pathlib import Path
from modules.video_promo.config_schema import BackgroundConfig, BackgroundStyle

logger = logging.getLogger(__name__)

# ── Background cache ────────────────────────────────────────────────────────
# Saves raw AI-generated backgrounds to assets/video_promo_backgrounds/<brand>/
# so they can be reused without burning another fal/replicate call.

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "assets" / "video_promo_backgrounds"


def _cache_dir_for(brand_name: str) -> Path:
    """Return the cache directory for a brand, sanitizing the name."""
    safe = re.sub(r"[^\w\-]", "_", brand_name.lower().strip())
    return _CACHE_ROOT / safe


def _cache_key(style: BackgroundStyle) -> str:
    """Cache key prefix for a background style."""
    return f"bg_{style.value}"


def find_cached_background(brand_name: str, style: BackgroundStyle) -> str | None:
    """Find the most recent cached background for a brand+style combo.

    Returns the path to the cached .mp4, or None if no cache exists.
    """
    cache_dir = _cache_dir_for(brand_name)
    if not cache_dir.exists():
        return None
    prefix = _cache_key(style)
    matches = sorted(
        [f for f in cache_dir.iterdir() if f.name.startswith(prefix) and f.suffix == ".mp4"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if matches:
        logger.info("Found cached background: %s", matches[0])
        return str(matches[0])
    return None


def save_to_cache(raw_bg_path: str, brand_name: str, style: BackgroundStyle) -> str:
    """Copy a raw background into the cache. Returns the cached path."""
    cache_dir = _cache_dir_for(brand_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = _cache_key(style)
    # Number sequentially
    existing = [f for f in cache_dir.iterdir() if f.name.startswith(prefix) and f.suffix == ".mp4"]
    idx = len(existing) + 1
    dest = cache_dir / f"{prefix}_{idx:03d}.mp4"
    shutil.copy2(raw_bg_path, dest)
    logger.info("Cached background: %s → %s", raw_bg_path, dest)
    return str(dest)


# ── Prompt templates per style ──────────────────────────────────────────────

STYLE_PROMPTS: dict[BackgroundStyle, str] = {
    BackgroundStyle.LIQUID_METAL: (
        "A single floating {primary_color} liquid metal blob rotating slowly in the center "
        "of frame against a pure black background. The blob has a glossy reflective surface "
        "with {primary_color} and warm orange light reflections. Smooth organic morphing motion. "
        "No other objects. {mood}. 4K quality, shallow depth of field, studio lighting."
    ),
    BackgroundStyle.AURORA: (
        "Soft flowing {primary_color} aurora borealis ribbons gently undulating against a deep "
        "dark background. Ethereal, dreamy motion. Subtle color shifts between {primary_color} "
        "and {secondary_color}. {mood}. No text, no objects, abstract only."
    ),
    BackgroundStyle.PARTICLE_FIELD: (
        "Thousands of tiny {primary_color} glowing particles floating and drifting slowly through "
        "dark empty space. Some particles form loose clusters. Subtle depth of field blur. "
        "{mood}. No text, purely abstract."
    ),
    BackgroundStyle.SMOKE: (
        "Wisps of {primary_color} smoke or ink slowly swirling and billowing against a pure black "
        "background. Volumetric, organic flow. Occasional bright {primary_color} highlights where "
        "light catches the edges. {mood}. No text, abstract only."
    ),
}


def build_prompt(config: BackgroundConfig) -> str:
    """Build the generation prompt from config."""
    if config.style == BackgroundStyle.CUSTOM:
        raise ValueError("Cannot build prompt for CUSTOM style — provide custom_video_path instead.")

    template = STYLE_PROMPTS[config.style]
    return template.format(
        primary_color=config.primary_color,
        secondary_color=config.secondary_color or config.primary_color,
        mood=config.mood,
    )


# ── fal.ai generation ──────────────────────────────────────────────────────

async def generate_background_fal(config: BackgroundConfig, output_path: str) -> str:
    """
    Generate background video using fal.ai.

    Requires: FAL_KEY environment variable.
    Default model: fal-ai/minimax-video/video-01-live

    Returns the path to the downloaded video file.
    """
    try:
        import fal_client
    except ImportError:
        raise ImportError("fal-client not installed. Run: pip install fal-client")

    prompt = build_prompt(config)
    model = config.model or "fal-ai/minimax-video/video-01-live"

    logger.info("Generating background via fal.ai model=%s", model)
    logger.info("Prompt: %s", prompt)

    # Submit generation request
    result = await fal_client.subscribe_async(
        model,
        arguments={
            "prompt": prompt,
            "prompt_optimizer": True,
        },
    )

    # Download the video
    video_url = result["video"]["url"]
    logger.info("Background generated: %s", video_url)

    # Download with httpx
    from agent._client import get_httpx
    client = get_httpx()
    response = await client.get(video_url)
    response.raise_for_status()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info("Background saved to %s", output_path)
    return output_path


# ── Replicate generation ────────────────────────────────────────────────────

async def generate_background_replicate(config: BackgroundConfig, output_path: str) -> str:
    """
    Generate background video using Replicate.

    Requires: REPLICATE_API_TOKEN environment variable.
    Default model: kling-ai/kling-v2.0-master

    Returns the path to the downloaded video file.
    """
    try:
        import replicate
    except ImportError:
        raise ImportError("replicate not installed. Run: pip install replicate")

    prompt = build_prompt(config)
    model = config.model or "kling-ai/kling-v2.0-master"

    logger.info("Generating background via Replicate model=%s", model)
    logger.info("Prompt: %s", prompt)

    output = replicate.run(
        model,
        input={
            "prompt": prompt,
            "duration": "5",        # 5 second clip
            "aspect_ratio": "9:16",  # Vertical format
        },
    )

    # Replicate returns a URL or FileOutput
    video_url = str(output)

    from agent._client import get_httpx
    client = get_httpx()
    response = await client.get(video_url)
    response.raise_for_status()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info("Background saved to %s", output_path)
    return output_path


# ── Loop extension ──────────────────────────────────────────────────────────

def loop_background(input_path: str, output_path: str, target_duration: float, fps: int = 25) -> str:
    """
    Loop a short background video to match the target duration.
    Uses ffmpeg to seamlessly loop and trim.

    For smoother loops, we:
    1. Reverse the clip and append it (creates a ping-pong loop)
    2. Loop the ping-pong until we exceed target duration
    3. Trim to exact target duration
    """
    # Step 1: Create ping-pong (forward + reverse) for smoother looping
    pingpong_path = input_path.replace(".mp4", "_pingpong.mp4")

    cmd_pingpong = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex",
        "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[out]",
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        pingpong_path,
    ]

    logger.info("Creating ping-pong loop")
    subprocess.run(cmd_pingpong, check=True, capture_output=True)

    # Step 2: Loop the ping-pong to exceed target duration, then trim
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", pingpong_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    loop_duration = float(result.stdout.strip())
    loops_needed = int(target_duration / loop_duration) + 2

    cmd_loop = [
        "ffmpeg", "-y",
        "-stream_loop", str(loops_needed),
        "-i", pingpong_path,
        "-t", str(target_duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_path,
    ]

    logger.info("Looping to %ss", target_duration)
    subprocess.run(cmd_loop, check=True, capture_output=True)

    # Cleanup
    os.remove(pingpong_path)

    logger.info("Looped background saved to %s (%ss)", output_path, target_duration)
    return output_path


# ── Main entry point ────────────────────────────────────────────────────────

async def generate_background(
    config: BackgroundConfig,
    work_dir: str,
    target_duration: float,
    fps: int = 25,
    brand_name: str = "default",
    fresh_bg: bool = False,
) -> str:
    """
    Generate (or use custom) background video, looped to target duration.

    Checks the background cache first (assets/video_promo_backgrounds/<brand>/).
    If a cached background exists for this brand+style, uses it instead of
    calling the AI provider. Pass fresh_bg=True to force a new generation.

    After a successful AI generation, the raw clip is saved to the cache.

    Returns path to the final looped background video.
    """
    raw_bg_path = os.path.join(work_dir, "bg_raw.mp4")
    looped_bg_path = os.path.join(work_dir, "bg_looped.mp4")

    if config.style == BackgroundStyle.CUSTOM:
        if not config.custom_video_path or not os.path.exists(config.custom_video_path):
            raise FileNotFoundError(f"Custom background not found: {config.custom_video_path}")
        raw_bg_path = config.custom_video_path
    else:
        # Check cache first (unless fresh_bg requested)
        cached = None if fresh_bg else find_cached_background(brand_name, config.style)

        if cached:
            logger.info("Using cached background: %s", cached)
            raw_bg_path = cached
        elif config.provider == "fal":
            await generate_background_fal(config, raw_bg_path)
            save_to_cache(raw_bg_path, brand_name, config.style)
        elif config.provider == "replicate":
            await generate_background_replicate(config, raw_bg_path)
            save_to_cache(raw_bg_path, brand_name, config.style)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    # Loop to target duration
    loop_background(raw_bg_path, looped_bg_path, target_duration, fps)

    return looped_bg_path
