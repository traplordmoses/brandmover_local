"""
Video style reverse-engineering.
Upload a reference video -> extract frames -> Claude Vision analyzes scene-by-scene ->
maps to video_gen.py scene schema -> remaps to brand colors/fonts.
"""

import asyncio
import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from agent._client import get_anthropic
from agent.compositor_config import get_config
from config import settings

logger = logging.getLogger(__name__)

# Scene type mapping from free-form descriptions to video_gen.py types
_SCENE_TYPE_MAP = {
    "title": "title",
    "intro": "title",
    "opening": "title",
    "headline": "title",
    "tagline": "tagline",
    "slogan": "tagline",
    "feature": "feature_list",
    "features": "feature_list",
    "list": "feature_list",
    "stat": "stat",
    "statistic": "stat",
    "number": "stat",
    "counter": "stat",
    "metric": "stat",
    "cta": "cta",
    "call to action": "cta",
    "closing": "cta",
    "outro": "cta",
    "end": "cta",
    "visual": "stock_footage",
    "image": "stock_footage",
    "footage": "stock_footage",
    "photo": "stock_footage",
    "steps": "steps",
    "how to": "steps",
    "tutorial": "steps",
    "process": "steps",
    "chat": "chat_demo",
    "conversation": "chat_demo",
    "demo": "chat_demo",
    "text": "text_only",
    "statement": "text_only",
    "quote": "text_only",
    "icon": "icon_reveal",
    "icons": "icon_grid",
    "grid": "icon_grid",
    "data": "data_viz",
    "chart": "data_viz",
    "graph": "data_viz",
    "count": "feature_count",
}

# Transition style mapping
_TRANSITION_MAP = {
    "cut": "cut",
    "hard cut": "cut",
    "fade": "fade",
    "dissolve": "fade",
    "crossfade": "fade",
    "slide": "slide",
    "wipe": "slide",
    "push": "slide",
    "zoom": "zoom",
    "scale": "zoom",
    "ken burns": "zoom",
}

# Motion mapping to video_gen animation styles
_MOTION_MAP = {
    "fade in": "fade",
    "fade": "fade",
    "slide left": "slide",
    "slide right": "slide",
    "slide up": "slide",
    "slide down": "slide",
    "zoom": "zoom",
    "zoom in": "zoom",
    "zoom out": "zoom",
    "static": "none",
    "none": "none",
    "pop": "scale",
    "bounce": "scale",
    "scale up": "scale",
}


async def extract_keyframes(video_path: str, max_frames: int = 12) -> list[str]:
    """Extract keyframes from a video at regular intervals.

    Tries keyframe extraction first, falls back to fps=1 if too few frames.
    Returns list of JPEG temp file paths.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    tmp_dir = tempfile.mkdtemp(prefix="brandmover_vr_")
    output_pattern = str(Path(tmp_dir) / "frame_%03d.jpg")

    # Get video duration for logging
    duration = await _get_video_duration(video_path)
    logger.info("Video duration: %.1fs, extracting up to %d keyframes", duration, max_frames)

    # Try keyframe extraction first
    cmd_keyframes = [
        "ffmpeg", "-i", video_path,
        "-vf", "select=eq(pict_type\\,I),scale=640:-1",
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        output_pattern,
    ]

    frames = await _run_ffmpeg(cmd_keyframes, tmp_dir, output_pattern)

    # If keyframe extraction yields too few frames, fall back to fps-based
    if len(frames) < 3:
        logger.info("Keyframe extraction got %d frames, falling back to fps=1", len(frames))
        # Clean up sparse results
        for f in frames:
            try:
                Path(f).unlink(missing_ok=True)
            except OSError:
                pass

        output_pattern_fps = str(Path(tmp_dir) / "fps_%03d.jpg")
        # Calculate fps to get roughly max_frames across the video
        target_fps = max(0.5, min(2.0, max_frames / max(duration, 1.0)))
        cmd_fps = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={target_fps:.2f},scale=640:-1",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            output_pattern_fps,
        ]
        frames = await _run_ffmpeg(cmd_fps, tmp_dir, output_pattern_fps)

    logger.info("Extracted %d frames from video", len(frames))
    return frames


async def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path,
    ]
    try:
        result = await asyncio.to_thread(
            subprocess.run, cmd,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception as e:
        logger.warning("ffprobe failed: %s", e)
    return 0.0


async def _run_ffmpeg(cmd: list[str], tmp_dir: str, pattern: str) -> list[str]:
    """Run an ffmpeg command and return sorted list of output frame paths."""
    try:
        result = await asyncio.to_thread(
            subprocess.run, cmd,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg failed (rc=%d): %s", result.returncode, result.stderr[:500])
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out")
    except FileNotFoundError:
        logger.error("ffmpeg not found -- install ffmpeg to use video analysis")
        raise RuntimeError("ffmpeg is not installed. Install it with: brew install ffmpeg")

    # Collect output frames
    frames = sorted(Path(tmp_dir).glob("*.jpg"))
    return [str(f) for f in frames]


async def analyze_video_style(video_path: str, frames: list[str]) -> dict:
    """Analyze video style by sending frames to Claude Vision.

    Sends all frames in a single call for holistic analysis.
    Returns structured analysis dict.
    """
    if not frames:
        raise ValueError("No frames to analyze")

    # Build the message content with all frames
    content: list[dict] = []

    # Get video duration for context
    duration = await _get_video_duration(video_path)

    content.append({
        "type": "text",
        "text": (
            f"I'm showing you {len(frames)} keyframes extracted from a {duration:.0f}-second video. "
            f"The frames are in chronological order. Analyze the video's visual style."
        ),
    })

    # Add each frame as an image
    for i, frame_path in enumerate(frames):
        frame_data = await asyncio.to_thread(_read_frame_b64, frame_path)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_data,
            },
        })
        content.append({
            "type": "text",
            "text": f"[Frame {i + 1} of {len(frames)}]",
        })

    system_prompt = """\
You are a motion graphics analyst. Analyze the video frames shown and provide a detailed style breakdown.

Return a JSON object with this exact structure (no markdown, just raw JSON):
{
    "style_summary": "Brief 1-2 sentence description of the video's overall visual style and aesthetic",
    "color_palette": [
        {"hex": "#XXXXXX", "name": "descriptive name", "role": "primary|accent|background|text|highlight"}
    ],
    "typography": {
        "font_style": "sans-serif|serif|monospace|display|handwritten",
        "weight": "light|regular|medium|bold|black",
        "case": "uppercase|lowercase|mixed|title"
    },
    "pacing": "fast|medium|slow",
    "transition_style": "cut|fade|slide|zoom|mixed",
    "scenes": [
        {
            "frame_index": 0,
            "description": "What is visually shown in this frame",
            "scene_type": "title|tagline|feature|stat|cta|visual|text|steps|list|data|icon",
            "text_content": "Any visible text in the frame (empty string if none)",
            "duration_estimate": "2s",
            "motion": "fade in|slide left|slide right|zoom|static|scale up"
        }
    ],
    "what_makes_it_work": "2-3 sentences on why this video is visually effective",
    "recreation_notes": "2-3 sentences on the key techniques needed to recreate this style"
}

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation
- Include one scene entry per frame shown
- Be specific about colors -- extract actual hex values from the frames
- Identify the dominant visual language (minimal, bold, corporate, playful, etc.)
- Note any recurring visual motifs or patterns
- duration_estimate should reflect typical timing for that type of scene"""

    client = get_anthropic()
    response = await client.messages.create(
        model=settings.SONNET_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )

    # Parse the response
    raw_text = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()
    if raw_text.startswith("json"):
        raw_text = raw_text[4:].strip()

    try:
        analysis = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude analysis: %s\nRaw: %s", e, raw_text[:500])
        raise ValueError(f"Claude returned invalid JSON: {e}")

    # Validate required fields
    required = ["style_summary", "color_palette", "typography", "scenes"]
    for key in required:
        if key not in analysis:
            analysis[key] = [] if key in ("color_palette", "scenes") else {}

    # Set defaults
    analysis.setdefault("pacing", "medium")
    analysis.setdefault("transition_style", "cut")
    analysis.setdefault("what_makes_it_work", "")
    analysis.setdefault("recreation_notes", "")

    logger.info(
        "Video analysis complete: %d scenes, %d colors, style=%s",
        len(analysis.get("scenes", [])),
        len(analysis.get("color_palette", [])),
        analysis.get("style_summary", "")[:60],
    )

    return analysis


def _read_frame_b64(frame_path: str) -> str:
    """Read a frame file and return base64-encoded data."""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


async def remap_to_brand(analysis: dict) -> dict:
    """Remap analyzed video style to FOID brand identity.

    Takes the raw analysis and produces a video_gen.py compatible scene JSON,
    replacing colors, fonts, and scene types with brand equivalents.
    """
    cfg = get_config()

    # Extract brand colors with fallbacks
    primary_color = _get_brand_color(cfg, "primary", "#6B9FD4")
    accent_color = _get_brand_color(cfg, "accent_1", "#FFD700")
    bg_color = _get_brand_color(cfg, "background", "#5B8FC4")
    text_color = "#FFFFFF"

    # Determine text color based on background brightness
    bg_rgb = _hex_to_rgb(bg_color)
    if bg_rgb:
        brightness = (bg_rgb[0] * 299 + bg_rgb[1] * 587 + bg_rgb[2] * 114) / 1000
        text_color = "#FFFFFF" if brightness < 128 else "#1A1A2E"

    # Get brand font
    display_font = "Orbitron"
    if cfg.fonts:
        display_entry = cfg.fonts.get("display")
        if display_entry:
            display_font = display_entry.family

    brand_name = cfg.brand_name or "FOID Foundation"

    # Calculate total duration from scene estimates
    scenes = analysis.get("scenes", [])
    total_duration = 0.0
    for scene in scenes:
        est = scene.get("duration_estimate", "2s")
        try:
            total_duration += float(est.replace("s", "").strip())
        except (ValueError, AttributeError):
            total_duration += 2.0

    total_duration = max(10.0, min(90.0, total_duration))

    # Map pacing to fps-adjusted frame counts
    pacing = analysis.get("pacing", "medium")
    base_frames = {"fast": 55, "medium": 70, "slow": 90}.get(pacing, 70)

    # Build remapped scenes
    remapped_scenes = []
    for i, scene in enumerate(scenes):
        mapped = _map_scene_to_brand(
            scene, i, len(scenes), base_frames,
            primary_color, accent_color, bg_color, text_color, brand_name,
        )
        if mapped:
            remapped_scenes.append(mapped)

    # Ensure we have at least title and cta
    if not remapped_scenes:
        remapped_scenes = [
            {
                "type": "title",
                "label": "INTRODUCING",
                "headline": brand_name,
                "background": "gradient",
                "durationFrames": 75,
            },
            {
                "type": "cta",
                "lines": [{"text": "Learn More", "accent": True}],
                "buttonText": "Get Started",
                "url": cfg.website or "https://example.com",
                "background": "gradient",
                "durationFrames": 75,
            },
        ]
    else:
        # Ensure first scene is title
        if remapped_scenes[0].get("type") != "title":
            remapped_scenes.insert(0, {
                "type": "title",
                "label": "INTRODUCING",
                "headline": brand_name,
                "background": "gradient",
                "durationFrames": 75,
            })
        # Ensure last scene is cta
        if remapped_scenes[-1].get("type") != "cta":
            remapped_scenes.append({
                "type": "cta",
                "lines": [{"text": "Join Us", "accent": True}],
                "buttonText": "Get Started",
                "url": cfg.website or "https://example.com",
                "background": "gradient",
                "durationFrames": 75,
            })

    # Recalculate total duration from actual frames
    total_frames = sum(s.get("durationFrames", 70) for s in remapped_scenes)
    actual_duration = total_frames / 30.0

    result = {
        "config": {
            "width": 1080,
            "height": 1080,
            "fps": 30,
            "durationInSeconds": round(actual_duration, 1),
            "brand": {
                "name": brand_name,
                "primaryColor": primary_color,
                "accentColor": accent_color,
                "backgroundColor": bg_color,
                "textColor": text_color,
                "fontFamily": display_font,
            },
        },
        "scenes": remapped_scenes,
    }

    logger.info(
        "Remapped %d scenes to brand style (%.1fs, %s)",
        len(remapped_scenes), actual_duration, brand_name,
    )
    return result


def _get_brand_color(cfg, role: str, fallback: str) -> str:
    """Get a brand color hex by role, with fallback."""
    entry = cfg.colors.get(role)
    return entry.hex if entry else fallback


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return None
    try:
        return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
    except ValueError:
        return None


def _map_scene_to_brand(
    scene: dict, index: int, total: int, base_frames: int,
    primary: str, accent: str, bg: str, text_color: str, brand_name: str,
) -> dict | None:
    """Map a single analyzed scene to a video_gen.py compatible scene dict."""
    raw_type = scene.get("scene_type", "visual").lower().strip()
    text_content = scene.get("text_content", "").strip()
    description = scene.get("description", "").strip()

    # Map to video_gen scene type
    mapped_type = _SCENE_TYPE_MAP.get(raw_type, "text_only")

    # Calculate duration frames from estimate
    est = scene.get("duration_estimate", "2s")
    try:
        seconds = float(est.replace("s", "").strip())
    except (ValueError, AttributeError):
        seconds = 2.0
    duration_frames = max(45, min(120, int(seconds * 30)))

    # Pick background based on position
    if index == 0 or index == total - 1:
        background = "gradient"
    elif mapped_type in ("feature_list", "steps", "icon_grid"):
        background = "dots"
    else:
        background = "clean"

    narration = scene.get("narration", description)

    # Build scene based on type
    if mapped_type == "title":
        return {
            "type": "title",
            "label": "INTRODUCING" if index == 0 else "",
            "headline": text_content or brand_name,
            "background": background,
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "tagline":
        lines = []
        if text_content:
            words = text_content.split()
            mid = len(words) // 2
            lines = [
                {"text": " ".join(words[:mid]), "accent": False},
                {"text": " ".join(words[mid:]), "accent": True},
            ]
        else:
            lines = [
                {"text": description[:30] if description else "Built Different", "accent": True},
            ]
        return {
            "type": "tagline",
            "lines": lines,
            "background": background,
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "text_only":
        return {
            "type": "text_only",
            "text": text_content or description[:50] or "Innovation Starts Here",
            "size": "large",
            "background": background,
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "stat":
        # Try to extract a number from text content
        import re
        numbers = re.findall(r"[\d,.]+[%+KMB]*", text_content) if text_content else []
        value = numbers[0] if numbers else "100"
        suffix = ""
        if value.endswith(("%", "+", "K", "M", "B")):
            suffix = value[-1]
            value = value[:-1]
        label = text_content.replace(value, "").strip() if text_content else description[:30]
        return {
            "type": "stat",
            "value": value,
            "label": label or "Growth",
            "suffix": suffix or "+",
            "animate": "countUp",
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "feature_list":
        items = []
        if text_content:
            parts = [p.strip() for p in text_content.replace("\n", ",").split(",") if p.strip()]
            items = [{"text": p[:30]} for p in parts[:6]]
        if not items:
            items = [{"text": description[:30] or "Key Feature"}]
        return {
            "type": "feature_list",
            "title": "Features",
            "items": items,
            "layout": "centered-stack",
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "steps":
        items = []
        if text_content:
            parts = [p.strip() for p in text_content.replace("\n", ",").split(",") if p.strip()]
            items = [{"heading": p[:20], "detail": ""} for p in parts[:3]]
        if not items:
            items = [{"heading": "Step 1", "detail": description[:30] or "Get Started"}]
        return {
            "type": "steps",
            "items": items,
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "chat_demo":
        messages = [
            {"text": text_content or "How does it work?", "isUser": True},
            {"text": description[:60] or "Let me show you!", "isUser": False, "label": brand_name},
        ]
        return {
            "type": "chat_demo",
            "messages": messages,
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "feature_count":
        import re
        numbers = re.findall(r"\d+", text_content) if text_content else []
        return {
            "type": "feature_count",
            "number": numbers[0] if numbers else "10",
            "subtitle": text_content or description[:30] or "Key Metrics",
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "icon_reveal":
        return {
            "type": "icon_reveal",
            "icons": ["rocket"],
            "caption": text_content or description[:40] or "Discover More",
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "icon_grid":
        return {
            "type": "icon_grid",
            "icons": ["star", "globe", "zap", "shield"],
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "data_viz":
        return {
            "type": "data_viz",
            "vizType": "bar_chart",
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "stock_footage":
        return {
            "type": "stock_footage",
            "query": description[:40] or "technology innovation",
            "layout": "full-bleed",
            "narration": narration,
            "durationFrames": duration_frames,
        }

    if mapped_type == "cta":
        return {
            "type": "cta",
            "lines": [
                {"text": text_content or "Get Started", "accent": True},
            ],
            "buttonText": "Learn More",
            "url": "https://example.com",
            "background": "gradient",
            "narration": narration,
            "durationFrames": duration_frames,
        }

    # Fallback: text_only
    return {
        "type": "text_only",
        "text": text_content or description[:50] or "Innovation",
        "size": "large",
        "background": background,
        "narration": narration,
        "durationFrames": duration_frames,
    }


async def format_breakdown(analysis: dict) -> str:
    """Format video analysis as a readable Telegram message (HTML).

    Returns a string safe for parse_mode='HTML'.
    """
    from xml.sax.saxutils import escape

    lines: list[str] = []

    # Header
    lines.append("<b>Video Style Breakdown</b>")
    lines.append("")

    # Style summary
    summary = analysis.get("style_summary", "")
    if summary:
        lines.append(f"<b>Style:</b> {escape(summary)}")
        lines.append("")

    # Color palette
    palette = analysis.get("color_palette", [])
    if palette:
        lines.append("<b>Color Palette:</b>")
        for color in palette[:8]:
            hex_val = color.get("hex", "")
            name = color.get("name", "")
            role = color.get("role", "")
            role_str = f" ({escape(role)})" if role else ""
            lines.append(f"  {escape(hex_val)} - {escape(name)}{role_str}")
        lines.append("")

    # Typography
    typo = analysis.get("typography", {})
    if typo:
        parts = []
        if typo.get("font_style"):
            parts.append(typo["font_style"])
        if typo.get("weight"):
            parts.append(typo["weight"])
        if typo.get("case"):
            parts.append(typo["case"])
        if parts:
            lines.append(f"<b>Typography:</b> {escape(', '.join(parts))}")

    # Pacing and transitions
    pacing = analysis.get("pacing", "")
    transition = analysis.get("transition_style", "")
    if pacing or transition:
        meta_parts = []
        if pacing:
            meta_parts.append(f"pacing: {pacing}")
        if transition:
            meta_parts.append(f"transitions: {transition}")
        lines.append(f"<b>Rhythm:</b> {escape(', '.join(meta_parts))}")
    lines.append("")

    # Scene-by-scene breakdown
    scenes = analysis.get("scenes", [])
    if scenes:
        lines.append(f"<b>Scene Breakdown ({len(scenes)} scenes):</b>")
        for i, scene in enumerate(scenes):
            desc = scene.get("description", "")
            stype = scene.get("scene_type", "")
            text = scene.get("text_content", "")
            dur = scene.get("duration_estimate", "")
            motion = scene.get("motion", "")

            scene_label = f"<b>{i + 1}.</b> [{escape(stype)}]" if stype else f"<b>{i + 1}.</b>"
            scene_parts = [scene_label]
            if desc:
                scene_parts.append(escape(desc[:80]))
            if text:
                scene_parts.append(f'<i>"{escape(text[:60])}"</i>')
            meta = []
            if dur:
                meta.append(dur)
            if motion:
                meta.append(motion)
            if meta:
                scene_parts.append(f"({escape(', '.join(meta))})")

            lines.append(" ".join(scene_parts))
        lines.append("")

    # What makes it work
    effective = analysis.get("what_makes_it_work", "")
    if effective:
        lines.append(f"<b>Why it works:</b> {escape(effective)}")
        lines.append("")

    # Recreation notes
    notes = analysis.get("recreation_notes", "")
    if notes:
        lines.append(f"<b>Recreation notes:</b> {escape(notes)}")

    return "\n".join(lines)
