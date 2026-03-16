"""
Video Styler — intelligent post-processing for social media videos.

Analyzes input video dimensions and applies the best treatment:
- Mobile recording (9:16) → phone mockup with gradient background
- Desktop recording (16:9) → clean crop, no fake phone frame
- Any aspect → proper framing for X/Twitter engagement

Pipeline:
1. Probe input video (dimensions, duration, aspect ratio)
2. Choose framing strategy based on content
3. Generate gradient background (PIL, fast)
4. Build ffmpeg filter graph: scale → overlay → text → encode
5. Output H.264 MP4 optimized for X/Twitter

Best practices baked in:
- 9:16 vertical gets highest engagement on X mobile feeds
- Content fills 75-85% of frame (no tiny content in huge borders)
- Clean font rendering using project fonts (Poppins, Orbitron)
- Narration text as styled pills with background, not raw drawtext
- Subtle zoom (1.0→1.05 max) — too much zoom = jitter
- CRF 18-20 for quality, H.264 High profile
"""

import base64
import io
import json
import logging
import math
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from agent.paths import PROJECT_ROOT
from config import settings

logger = logging.getLogger(__name__)

_OUTPUTS_DIR = PROJECT_ROOT / "state" / "outputs"
_FONTS_DIR = PROJECT_ROOT / "brand" / "assets" / "fonts"

# Best font for video narration text
_NARRATION_FONT = str(_FONTS_DIR / "Poppins-SemiBold.ttf")
_NARRATION_FONT_FALLBACK = str(_FONTS_DIR / "Poppins-Bold.ttf")


@dataclass
class VideoStyle:
    """Configuration for video styling."""
    # Framing
    frame_mode: str = "auto"       # "auto" | "phone" | "desktop" | "none"
    # auto: picks phone for 9:16 input, desktop for 16:9

    # Background — holographic pastels matching Moonshot reference
    # Saturated cyan + pink/lavender flowing blobs
    bg_colors: list[str] = field(default_factory=lambda: [
        "#4de8e8", "#e07ff0", "#a88cff", "#ffacc8",
    ])

    # Ken Burns zoom (subtle)
    zoom_enabled: bool = True
    zoom_factor: float = 1.04      # max 4% zoom — subtle, no jitter

    # Output — 9:16 vertical is best for X engagement
    aspect: str = "9:16"           # "9:16" | "1:1" | "16:9"
    output_fps: int = 30
    output_crf: int = 18           # high quality
    max_dimension: int = 1080      # width for 9:16, or larger side

    # Text
    font_path: str = ""            # auto-detected if empty
    font_size: int = 0             # auto-scaled if 0
    text_position: str = "bottom"  # "bottom" | "top"


def _get_font_path(style: VideoStyle) -> str:
    """Get the best available font file path."""
    if style.font_path and Path(style.font_path).exists():
        return style.font_path
    if Path(_NARRATION_FONT).exists():
        return _NARRATION_FONT
    if Path(_NARRATION_FONT_FALLBACK).exists():
        return _NARRATION_FONT_FALLBACK
    return ""  # ffmpeg will use its default


def _output_dimensions(style: VideoStyle) -> tuple[int, int]:
    """Calculate output width x height from aspect ratio and max dimension."""
    if style.aspect == "9:16":
        w = style.max_dimension
        h = int(w * 16 / 9)
        # Round to even numbers (required for H.264)
        return w, h - (h % 2)
    elif style.aspect == "1:1":
        return style.max_dimension, style.max_dimension
    else:  # 16:9
        h = style.max_dimension
        w = int(h * 16 / 9)
        return w - (w % 2), h


def _probe_video(path: str) -> dict:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout:
        return {}
    data = json.loads(result.stdout)

    # Extract key info
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "duration": float(fmt.get("duration", 10)),
        "codec": video_stream.get("codec_name", ""),
        "fps": _parse_fps(video_stream.get("r_frame_rate", "30/1")),
    }


def _parse_fps(fps_str: str) -> float:
    """Parse ffprobe frame rate string like '30/1' or '29.97'."""
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den)
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        return 30.0


def _choose_frame_mode(input_w: int, input_h: int, style: VideoStyle) -> str:
    """Intelligently choose framing based on input aspect ratio."""
    if style.frame_mode != "auto":
        return style.frame_mode

    if input_w == 0 or input_h == 0:
        return "desktop"

    ratio = input_w / input_h

    if ratio < 0.7:
        # Portrait / mobile recording → phone mockup makes sense
        return "phone"
    elif ratio > 1.4:
        # Widescreen desktop → clean desktop framing, no fake phone
        return "desktop"
    else:
        # Square-ish → desktop treatment
        return "desktop"


def _escape_ffmpeg(text: str) -> str:
    """Escape text for ffmpeg drawtext filter."""
    # Order matters: backslash first, then special chars
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace('"', '\\"')
    text = text.replace("%", "%%")
    return text


def _build_gradient_image(width: int, height: int, colors: list[str], output_path: str) -> str:
    """Generate a holographic gradient background with organic frosted-glass shapes.

    Matches the Moonshot/PENGU reference: bright white base with distinct translucent
    organic shapes (like frosted glass ribbons / aurora waves) floating over soft
    pastel color fields. NOT a flat gradient — visible forms with soft edges.
    """
    from PIL import Image, ImageDraw, ImageFilter

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    parsed = [hex_to_rgb(c) for c in colors]
    if len(parsed) < 2:
        parsed = [(77, 232, 232), (224, 127, 240), (168, 140, 255), (255, 172, 200)]

    # Layer 1: Bright base
    base = Image.new("RGB", (width, height), (238, 244, 250))

    # Layer 2: Broad color fields — subtle hue map underneath the shapes
    wash_configs = [
        (0.20, 0.40, 0.60, 0.80, 0, 90),   # cyan tint, left
        (0.80, 0.30, 0.50, 0.60, 1, 80),   # pink tint, top right
        (0.70, 0.80, 0.50, 0.50, 3, 75),   # warm pink, bottom right
    ]

    for cx_pct, cy_pct, rw_pct, rh_pct, ci, alpha in wash_configs:
        color = parsed[ci % len(parsed)]
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = int(width * cx_pct), int(height * cy_pct)
        rw, rh = int(width * rw_pct), int(height * rh_pct)
        draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(*color, alpha))
        blur_r = int(min(rw, rh) * 0.5)
        layer = layer.filter(ImageFilter.GaussianBlur(radius=max(blur_r, 25)))
        base.paste(
            Image.composite(Image.new("RGB", (width, height), color), base, layer.split()[3]),
        )

    # Layer 3: Frosted-glass organic shapes — THE hero visual
    # High alpha, LOW blur = visible translucent forms with soft but defined edges
    glass_shapes = [
        # Large cyan wave — left side, flowing diagonally
        ([(0.0, 0.42), (0.08, 0.30), (0.35, 0.25), (0.55, 0.38),
          (0.48, 0.58), (0.22, 0.65), (0.0, 0.58)], 0, 200, 0.035),
        # Pink flowing panel — upper right
        ([(0.42, 0.02), (0.70, 0.06), (1.0, 0.15), (1.0, 0.40),
          (0.75, 0.35), (0.48, 0.18)], 1, 190, 0.035),
        # Warm pink blob — bottom right
        ([(0.48, 0.65), (0.75, 0.55), (1.0, 0.68), (1.0, 0.92),
          (0.65, 0.92), (0.42, 0.78)], 3, 180, 0.03),
        # Lavender accent — top left
        ([(0.0, 0.06), (0.25, 0.02), (0.40, 0.12), (0.30, 0.28),
          (0.08, 0.24), (0.0, 0.15)], 2, 170, 0.03),
        # Cyan accent — bottom left
        ([(0.0, 0.75), (0.20, 0.68), (0.35, 0.78), (0.25, 0.95),
          (0.0, 0.95)], 0, 175, 0.03),
        # Small pink highlight — center
        ([(0.30, 0.45), (0.50, 0.40), (0.55, 0.50), (0.40, 0.55)], 1, 120, 0.025),
    ]

    for points_pct, ci, alpha, blur_frac in glass_shapes:
        color = parsed[ci % len(parsed)]
        # Lighten slightly for frosted glass
        glass_color = tuple(min(255, c + 15) for c in color)

        shape = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        points = [(int(x * width), int(y * height)) for x, y in points_pct]
        draw.polygon(points, fill=(*glass_color, alpha))

        # Low blur — soft edges but shape is CLEARLY visible
        blur_r = int(min(width, height) * blur_frac)
        shape = shape.filter(ImageFilter.GaussianBlur(radius=max(blur_r, 10)))

        base.paste(
            Image.composite(
                Image.new("RGB", (width, height), glass_color),
                base,
                shape.split()[3],
            ),
        )

    # Layer 4: White highlight streaks — light refracting through glass
    for pts_pct, alpha in [
        ([(0.06, 0.38), (0.35, 0.32), (0.42, 0.37), (0.28, 0.44), (0.04, 0.44)], 70),
        ([(0.55, 0.10), (0.80, 0.13), (0.85, 0.22), (0.70, 0.24), (0.50, 0.16)], 65),
        ([(0.40, 0.70), (0.65, 0.64), (0.75, 0.73), (0.55, 0.78)], 50),
    ]:
        streak = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(streak)
        pts = [(int(x * width), int(y * height)) for x, y in pts_pct]
        draw.polygon(pts, fill=(255, 255, 255, alpha))
        blur_r = int(min(width, height) * 0.025)
        streak = streak.filter(ImageFilter.GaussianBlur(radius=max(blur_r, 6)))
        base.paste(
            Image.composite(
                Image.new("RGB", (width, height), (255, 255, 255)),
                base,
                streak.split()[3],
            ),
        )

    # Minimal final smoothing
    base = base.filter(ImageFilter.GaussianBlur(radius=2))

    base.save(output_path, quality=95)
    logger.info("Generated holographic background: %dx%d → %s", width, height, output_path)
    return output_path


def _build_phone_frame_png(width: int, height: int, screen_rect: tuple, output_path: str) -> str:
    """Generate a photorealistic iPhone frame overlay PNG.

    Matches the Moonshot reference video: thick dark bezel, prominent rounded
    corners, side buttons, speaker slit, Dynamic Island. The inner screen area
    is transparent so the video content shows through.

    screen_rect: (x, y, w, h) where the video content shows through.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    sx, sy, sw, sh = screen_rect

    # Thick bezel like a real iPhone — reference shows ~3-4% of phone width
    bezel = max(int(sw * 0.035), 10)
    # Prominent corner radius — iPhones have very rounded corners
    corner_r = int(sw * 0.12)

    phone_left = sx - bezel
    phone_top = sy - bezel
    phone_right = sx + sw + bezel
    phone_bottom = sy + sh + bezel

    # ── Phone body (outer shell) ──
    # Slightly lighter edge for depth (like a real phone edge catch light)
    draw.rounded_rectangle(
        [phone_left - 2, phone_top - 2, phone_right + 2, phone_bottom + 2],
        radius=corner_r + bezel + 2,
        fill=(50, 50, 50, 255),  # subtle edge highlight
    )

    # Main phone body — deep black
    draw.rounded_rectangle(
        [phone_left, phone_top, phone_right, phone_bottom],
        radius=corner_r + bezel,
        fill=(15, 15, 15, 255),
    )

    # ── Inner screen cutout (transparent) ──
    draw.rounded_rectangle(
        [sx, sy, sx + sw, sy + sh],
        radius=corner_r,
        fill=(0, 0, 0, 0),
    )

    # ── Dynamic Island ──
    di_w = int(sw * 0.28)
    di_h = max(int(sh * 0.022), 10)
    di_x = sx + (sw - di_w) // 2
    di_y = sy + int(sh * 0.012)
    draw.rounded_rectangle(
        [di_x, di_y, di_x + di_w, di_y + di_h],
        radius=di_h // 2,
        fill=(15, 15, 15, 255),
    )

    # ── Side buttons (left side) ──
    btn_x = phone_left - 3
    btn_w = 3

    # Silent switch (small, near top)
    sw_y = phone_top + int(sh * 0.12)
    sw_h = int(sh * 0.03)
    draw.rounded_rectangle(
        [btn_x, sw_y, btn_x + btn_w, sw_y + sw_h],
        radius=1,
        fill=(40, 40, 40, 255),
    )

    # Volume up
    vu_y = phone_top + int(sh * 0.20)
    vu_h = int(sh * 0.055)
    draw.rounded_rectangle(
        [btn_x, vu_y, btn_x + btn_w, vu_y + vu_h],
        radius=1,
        fill=(40, 40, 40, 255),
    )

    # Volume down
    vd_y = vu_y + vu_h + int(sh * 0.015)
    vd_h = int(sh * 0.055)
    draw.rounded_rectangle(
        [btn_x, vd_y, btn_x + btn_w, vd_y + vd_h],
        radius=1,
        fill=(40, 40, 40, 255),
    )

    # ── Power button (right side) ──
    pwr_x = phone_right
    pwr_y = phone_top + int(sh * 0.22)
    pwr_h = int(sh * 0.07)
    draw.rounded_rectangle(
        [pwr_x, pwr_y, pwr_x + btn_w, pwr_y + pwr_h],
        radius=1,
        fill=(40, 40, 40, 255),
    )

    # ── Bottom speaker grille (subtle dots) ──
    speaker_y = phone_bottom - int(bezel * 0.55)
    speaker_cx = sx + sw // 2
    dot_r = max(int(bezel * 0.06), 1)
    dot_spacing = max(int(bezel * 0.25), 4)
    for i in range(-3, 4):
        dot_x = speaker_cx + i * dot_spacing
        draw.ellipse(
            [dot_x - dot_r, speaker_y - dot_r, dot_x + dot_r, speaker_y + dot_r],
            fill=(40, 40, 40, 255),
        )

    img.save(output_path)
    return output_path


def _build_rounded_mask_png(
    content_w: int, content_h: int, corner_radius: int, output_path: str,
) -> str:
    """Generate a rounded-rectangle alpha mask sized to the video content.

    RGBA image: fully opaque white inside the rounded rect, fully transparent outside.
    Used in ffmpeg with alphamerge to clip video to rounded corners.
    """
    from PIL import Image, ImageDraw

    mask = Image.new("RGBA", (content_w, content_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [0, 0, content_w - 1, content_h - 1],
        radius=corner_radius,
        fill=(255, 255, 255, 255),
    )
    mask.save(output_path)
    return output_path


def apply_style(
    input_video: str,
    style: VideoStyle | None = None,
    narration_texts: list[dict] | None = None,
    output_path: str | None = None,
) -> str:
    """Apply intelligent video styling to a raw recording.

    Probes the input video, chooses the best framing strategy,
    and produces a polished output optimized for X/Twitter.

    Args:
        input_video: Path to raw video (WebM/MP4).
        style: VideoStyle configuration. Defaults to smart auto mode.
        narration_texts: Optional list of {"text": str, "start": float, "end": float}.
        output_path: Where to write output. Auto-generated if None.

    Returns:
        Path to the styled output MP4.
    """
    if style is None:
        style = VideoStyle()

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not output_path:
        ts = int(time.time())
        uid = uuid.uuid4().hex[:6]
        output_path = str(_OUTPUTS_DIR / f"styled_{ts}_{uid}.mp4")

    # ── Step 1: Probe input ──────────────────────────────────────────
    info = _probe_video(input_video)
    in_w = info.get("width", 1280)
    in_h = info.get("height", 720)
    duration = info.get("duration", 10)
    logger.info("Input: %dx%d, %.1fs, %s", in_w, in_h, duration, info.get("codec", "?"))

    # ── Step 2: Choose framing strategy ──────────────────────────────
    frame_mode = _choose_frame_mode(in_w, in_h, style)
    ow, oh = _output_dimensions(style)
    font_path = _get_font_path(style)
    logger.info("Strategy: %s framing, output %dx%d (%s)", frame_mode, ow, oh, style.aspect)

    # ── Step 3: Generate background ──────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        bg_path = f.name
    _build_gradient_image(ow, oh, style.bg_colors, bg_path)

    # ── Step 4: Calculate content placement ──────────────────────────
    if frame_mode == "phone":
        # Phone mockup — content fills 86% of width for maximum visual impact
        content_w = int(ow * 0.86)
        # Maintain input aspect ratio
        content_h = int(content_w * in_h / in_w) if in_w > 0 else int(content_w * 16 / 9)
        # Cap height to 88% of output (leave room for narration text below)
        if content_h > int(oh * 0.88):
            content_h = int(oh * 0.88)
            content_w = int(content_h * in_w / in_h) if in_h > 0 else content_h

        # Ensure even dimensions
        content_w -= content_w % 2
        content_h -= content_h % 2

        content_x = (ow - content_w) // 2
        content_y = (oh - content_h) // 2 - int(oh * 0.03)  # above center, leave room for text

        # Generate phone frame overlay
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            frame_path = f.name
        _build_phone_frame_png(ow, oh, (content_x, content_y, content_w, content_h), frame_path)

        # Generate rounded-corner mask to clip video content
        screen_corner_r = int(content_w * 0.12)  # match phone frame's screen cutout radius
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_path = f.name
        _build_rounded_mask_png(content_w, content_h, screen_corner_r, mask_path)

    elif frame_mode == "desktop":
        # Desktop — content fills ~88% of frame, clean rounded corners, no fake phone
        content_w = int(ow * 0.88)
        content_h = int(content_w * in_h / in_w) if in_w > 0 else int(content_w * 9 / 16)
        if content_h > int(oh * 0.75):
            content_h = int(oh * 0.75)
            content_w = int(content_h * in_w / in_h) if in_h > 0 else content_h

        content_w -= content_w % 2
        content_h -= content_h % 2

        content_x = (ow - content_w) // 2
        content_y = (oh - content_h) // 2 - int(oh * 0.04)  # above center, leave room for text
        frame_path = None
        mask_path = None

    else:  # "none"
        content_w = ow
        content_h = oh
        content_x = 0
        content_y = 0
        frame_path = None
        mask_path = None

    logger.info("Content placement: %dx%d at (%d,%d)", content_w, content_h, content_x, content_y)

    # ── Step 5: Build ffmpeg filter graph ────────────────────────────
    total_frames = int(duration * style.output_fps)
    corner_r = int(min(content_w, content_h) * 0.03)  # subtle rounded corners

    # Background with subtle Ken Burns zoom
    if style.zoom_enabled:
        zf = style.zoom_factor
        # Pre-scale background larger, then zoompan crops into it
        pre_w = int(ow * zf) + (int(ow * zf) % 2)
        pre_h = int(oh * zf) + (int(oh * zf) % 2)
        zoom_rate = (zf - 1.0) / max(total_frames, 1)
        bg_filter = (
            f"[0:v]scale={pre_w}:{pre_h},"
            f"loop=loop=-1:size=1:start=0,"
            f"setpts=N/{style.output_fps}/TB,"
            f"zoompan=z='1+{zoom_rate:.8f}*on':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={ow}x{oh}:fps={style.output_fps}[bg]"
        )
    else:
        bg_filter = (
            f"[0:v]scale={ow}:{oh},"
            f"loop=loop=-1:size=1:start=0,"
            f"setpts=N/{style.output_fps}/TB[bg]"
        )

    # Scale video content to fill the allocated area (crop to fit, no black bars)
    video_filter = (
        f"[1:v]scale={content_w}:{content_h}:force_original_aspect_ratio=increase,"
        f"crop={content_w}:{content_h}"
    )

    if mask_path:
        # Clip video to rounded corners using alpha mask, then overlay on background
        # Mask is content-sized (content_w x content_h) — same size as scaled video
        mask_idx = 3 if frame_path else 2
        full_filter = (
            f"{bg_filter};"
            f"{video_filter},format=rgba[content_raw];"
            f"[{mask_idx}:v]alphaextract[amask];"
            f"[content_raw][amask]alphamerge[content];"
            f"[bg][content]overlay={content_x}:{content_y}:format=auto"
        )
    else:
        # No mask — overlay content directly
        full_filter = (
            f"{bg_filter};"
            f"{video_filter}[content];"
            f"[bg][content]overlay={content_x}:{content_y}"
        )

    # Overlay phone frame if applicable
    if frame_path:
        full_filter += f"[pre_frame];[pre_frame][2:v]overlay=0:0"

    # ── Step 6: Add narration text overlays ──────────────────────────
    narration_text_files = []  # track temp files for cleanup
    if narration_texts:
        # Auto-scale font size — larger for readability on mobile feeds
        font_size = style.font_size if style.font_size > 0 else max(int(ow * 0.038), 28)
        box_pad = max(int(font_size * 0.45), 10)

        font_opt = f"fontfile='{font_path}':" if font_path else ""

        for idx, nt in enumerate(narration_texts):
            raw_text = nt.get("text", "").strip()
            if not raw_text:
                continue
            start = nt.get("start", 0)
            end = nt.get("end", start + 3)

            # Write text to a temp file to avoid all escaping issues
            text_file = str(_OUTPUTS_DIR / f"_narr_{idx}.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(raw_text)
            narration_text_files.append(text_file)

            if style.text_position == "top":
                y_expr = f"{int(oh * 0.06)}"
            else:
                y_expr = f"h-{int(oh * 0.08)}"

            full_filter += (
                f",drawtext={font_opt}"
                f"textfile='{text_file}':"
                f"fontsize={font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.55:boxborderw={box_pad}:"
                f"x=(w-text_w)/2:y={y_expr}:"
                f"enable='between(t\\,{start}\\,{end})'"
            )

    full_filter += "[out]"

    # ── Step 7: Encode ───────────────────────────────────────────────
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", bg_path,       # input 0: background
        "-i", input_video,                   # input 1: video
    ]

    if frame_path:
        cmd += ["-i", frame_path]            # input 2: phone frame overlay

    if mask_path:
        cmd += ["-loop", "1", "-i", mask_path]  # input 3 (or 2): rounded corner mask

    cmd += [
        "-filter_complex", full_filter,
        "-map", "[out]",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-preset", "medium",
        "-crf", str(style.output_crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",          # web-optimized
        "-t", str(duration),
        "-an",                               # no audio
        output_path,
    ]

    logger.info("Styling video: %s → %s (%.1fs, %s mode)", input_video, output_path, duration, frame_mode)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr[-800:] if result.stderr else "no output")
        # Fallback: simple re-encode with scaling to output dimensions
        fallback_cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                   f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v", "libx264", "-preset", "fast", "-crf", str(style.output_crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", output_path,
        ]
        subprocess.run(fallback_cmd, capture_output=True, timeout=120)
        logger.info("Fallback encode complete: %s", output_path)
    else:
        logger.info("Styled video complete: %s (%s, %dx%d)", output_path, frame_mode, ow, oh)

    # Clean up temp files
    Path(bg_path).unlink(missing_ok=True)
    if frame_path:
        Path(frame_path).unlink(missing_ok=True)
    if mask_path:
        Path(mask_path).unlink(missing_ok=True)
    for tf in narration_text_files:
        Path(tf).unlink(missing_ok=True)

    return output_path


def cut_and_stitch(
    input_video: str,
    segments: list[dict],
    output_path: str | None = None,
    crossfade_duration: float = 0.3,
) -> str:
    """Cut segments from a video and stitch them together with crossfade transitions.

    This is the video editor — takes a raw recording and produces a tight highlight reel.

    Args:
        input_video: Path to source video.
        segments: List of {"start": float, "end": float, "label": str}.
                  Times in seconds. label is optional, for logging.
        output_path: Where to write output. Auto-generated if None.
        crossfade_duration: Crossfade between segments in seconds. 0 = hard cut.

    Returns:
        Path to the stitched output MP4.
    """
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not output_path:
        ts = int(time.time())
        uid = uuid.uuid4().hex[:6]
        output_path = str(_OUTPUTS_DIR / f"edited_{ts}_{uid}.mp4")

    if not segments:
        logger.warning("No segments provided, returning input unchanged")
        return input_video

    # Single segment — simple trim, no stitching needed
    if len(segments) == 1:
        seg = segments[0]
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seg["start"]),
            "-i", input_video,
            "-t", str(seg["end"] - seg["start"]),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("Trim failed: %s", result.stderr[-400:])
        return output_path

    # Multiple segments — extract each, then concat with optional crossfade
    tmp_clips = []
    try:
        for i, seg in enumerate(segments):
            clip_path = str(_OUTPUTS_DIR / f"_clip_{i:03d}.mp4")
            duration = seg["end"] - seg["start"]
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg["start"]),
                "-i", input_video,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an",
                "-movflags", "+faststart",
                clip_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                tmp_clips.append(clip_path)
                logger.info("Cut segment %d: %.1f-%.1fs (%s)",
                           i, seg["start"], seg["end"], seg.get("label", ""))
            else:
                logger.error("Failed to cut segment %d: %s", i, result.stderr[-200:])

        if not tmp_clips:
            logger.error("No clips extracted")
            return input_video

        if len(tmp_clips) == 1:
            Path(tmp_clips[0]).rename(output_path)
            return output_path

        # Build crossfade filter graph if requested
        if crossfade_duration > 0 and len(tmp_clips) > 1:
            inputs = []
            for clip in tmp_clips:
                inputs += ["-i", clip]

            # Get durations for offset calculation
            clip_durations = []
            for clip in tmp_clips:
                info = _probe_video(clip)
                clip_durations.append(info.get("duration", 3.0))

            # Build xfade chain with simple sequential labels:
            # [0:v][1:v]xfade=...[x1]; [x1][2:v]xfade=...[x2]; etc.
            filter_parts = []
            cumulative = clip_durations[0] - crossfade_duration

            filter_parts.append(
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:"
                f"offset={cumulative:.3f}[x1]"
            )
            cumulative += clip_durations[1] - crossfade_duration

            for i in range(2, len(tmp_clips)):
                filter_parts.append(
                    f"[x{i-1}][{i}:v]xfade=transition=fade:"
                    f"duration={crossfade_duration}:offset={cumulative:.3f}[x{i}]"
                )
                cumulative += clip_durations[i] - crossfade_duration

            final_label = f"x{len(tmp_clips) - 1}"
            filter_complex = ";".join(filter_parts)

            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex,
                "-map", f"[{final_label}]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                output_path,
            ]
        else:
            # Simple concat demuxer (no crossfade)
            concat_file = str(_OUTPUTS_DIR / "_concat_list.txt")
            with open(concat_file, "w") as f:
                for clip in tmp_clips:
                    f.write(f"file '{clip}'\n")

            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                output_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            logger.error("Stitch failed: %s", result.stderr[-400:])
            # Fallback: just use concat without crossfade
            concat_file = str(_OUTPUTS_DIR / "_concat_list.txt")
            with open(concat_file, "w") as f:
                for clip in tmp_clips:
                    f.write(f"file '{clip}'\n")
            fallback_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file, "-c", "copy", output_path,
            ]
            subprocess.run(fallback_cmd, capture_output=True, timeout=60)

        logger.info("Stitched %d segments → %s", len(tmp_clips), output_path)

    finally:
        # Clean up temp clips
        for clip in tmp_clips:
            Path(clip).unlink(missing_ok=True)
        Path(_OUTPUTS_DIR / "_concat_list.txt").unlink(missing_ok=True)

    return output_path


def edit_and_style(
    input_video: str,
    segments: list[dict],
    style: VideoStyle | None = None,
    narration_texts: list[dict] | None = None,
    crossfade_duration: float = 0.3,
    output_path: str | None = None,
) -> str:
    """Full pipeline: cut segments → stitch → style with phone mockup + gradient.

    This is the one-call editor+styler for the bot's video production pipeline.
    """
    # Step 1: Cut and stitch
    edited = cut_and_stitch(
        input_video, segments,
        crossfade_duration=crossfade_duration,
    )

    # Step 2: Adjust narration timestamps to match edited timeline
    if narration_texts:
        # Remap narration times based on segment cuts
        adjusted = _remap_narration(narration_texts, segments)
    else:
        adjusted = None

    # Step 3: Apply style
    return apply_style(edited, style, adjusted, output_path)


def _remap_narration(
    narrations: list[dict],
    segments: list[dict],
) -> list[dict]:
    """Remap narration timestamps from raw video time to edited video time."""
    remapped = []
    edit_offset = 0.0

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_duration = seg_end - seg_start

        for nar in narrations:
            nar_start = nar.get("start", 0)
            nar_end = nar.get("end", nar_start + 3)

            # Check if narration overlaps with this segment
            if nar_start < seg_end and nar_end > seg_start:
                # Clamp to segment bounds
                new_start = max(nar_start - seg_start, 0) + edit_offset
                new_end = min(nar_end - seg_start, seg_duration) + edit_offset
                if new_end > new_start:
                    remapped.append({
                        "text": nar["text"],
                        "start": round(new_start, 2),
                        "end": round(new_end, 2),
                    })

        edit_offset += seg_duration

    return remapped


async def async_apply_style(
    input_video: str,
    style: VideoStyle | None = None,
    narration_texts: list[dict] | None = None,
    output_path: str | None = None,
) -> str:
    """Async wrapper for apply_style."""
    import asyncio
    return await asyncio.to_thread(
        apply_style, input_video, style, narration_texts, output_path,
    )


async def async_edit_and_style(
    input_video: str,
    segments: list[dict],
    style: VideoStyle | None = None,
    narration_texts: list[dict] | None = None,
    crossfade_duration: float = 0.3,
    output_path: str | None = None,
) -> str:
    """Async wrapper for edit_and_style."""
    import asyncio
    return await asyncio.to_thread(
        edit_and_style, input_video, segments, style,
        narration_texts, crossfade_duration, output_path,
    )


# ---------------------------------------------------------------------------
# Self-review quality gate
# ---------------------------------------------------------------------------

_REVIEW_MODEL = "claude-haiku-4-5-20251001"

_REVIEW_SYSTEM_PROMPT = """You are a video quality reviewer for a brand's product demo.
You're reviewing a styled demo video that shows off the product's features.

Analyze the extracted frames and give an honest quality assessment. Check for:

1. **Blank/black screens** — any frames that are entirely black, white, or solid color with no content
2. **Stuck content** — same exact screen visible for too many consecutive frames (suggests dead time)
3. **Missing key moments** — wallet connection, card swiping, voting should be visible
4. **Loading/error states** — spinners, error messages, "connecting" screens that shouldn't be in final cut
5. **Visual quality** — phone mockup looks clean, gradient background visible, no rendering glitches
6. **Pacing** — does the video feel like it moves through the feature demo at a good pace?
7. **Onboarding/skip screens** — these should be cut out, not in the final video

Respond with a JSON object:
{
  "pass": true/false,
  "score": 1-10,
  "duration_looks_correct": true/false,
  "issues": ["list of specific problems found"],
  "stuck_frames": [{"frame_index": N, "description": "what's stuck"}],
  "good_moments": ["list of things that look good"],
  "suggestions": ["specific actionable suggestions to improve"],
  "summary": "one-line verdict"
}

Be critical. The user has high quality standards and wants to catch problems before the video is sent.
Only return the JSON object, nothing else."""


def _extract_review_frames(video_path: str, count: int = 8) -> list[dict]:
    """Extract evenly-spaced frames from a video for review.

    Returns list of Claude Vision image content blocks (base64 JPEG, 400px wide).
    """
    from PIL import Image

    info = _probe_video(video_path)
    duration = info.get("duration", 10)

    # Extract frames at even intervals
    interval = duration / (count + 1)
    frames = []

    for i in range(1, count + 1):
        timestamp = interval * i
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            frame_path = f.name

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.2f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            Path(frame_path).unlink(missing_ok=True)
            continue

        try:
            img = Image.open(frame_path)
            # Downscale to 400px wide for token efficiency
            max_w = 400
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode()

            frames.append({
                "timestamp": round(timestamp, 1),
                "image_block": {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
            })
        finally:
            Path(frame_path).unlink(missing_ok=True)

    return frames


def review_video(video_path: str, expected_duration: float | None = None) -> dict:
    """Self-review a video by extracting frames and analyzing with Claude Vision.

    Returns a quality report dict with pass/fail, score, issues, and suggestions.
    The bot should call this after styling and BEFORE sending to the user.
    """
    if not Path(video_path).exists():
        return {"pass": False, "score": 0, "error": f"Video not found: {video_path}"}

    info = _probe_video(video_path)
    actual_duration = info.get("duration", 0)
    width = info.get("width", 0)
    height = info.get("height", 0)

    logger.info(
        "Self-reviewing video: %s (%.1fs, %dx%d)",
        video_path, actual_duration, width, height,
    )

    # Extract frames — more for longer videos
    frame_count = min(max(int(actual_duration / 3), 6), 12)
    frames = _extract_review_frames(video_path, count=frame_count)

    if not frames:
        return {
            "pass": False,
            "score": 0,
            "error": "Could not extract any frames from video",
            "issues": ["Frame extraction failed — video may be corrupt"],
        }

    # Build the review prompt with all frames
    content = []
    content.append({
        "type": "text",
        "text": (
            f"Review this demo video of {settings.BRAND_NAME}.\n"
            f"Video info: {actual_duration:.1f}s duration, {width}x{height}, "
            f"{len(frames)} frames extracted at even intervals.\n"
        ),
    })

    if expected_duration:
        content.append({
            "type": "text",
            "text": (
                f"Expected duration: {expected_duration:.1f}s. "
                f"Actual duration: {actual_duration:.1f}s. "
                f"{'MISMATCH — flag this.' if abs(actual_duration - expected_duration) > 3 else 'Duration looks correct.'}"
            ),
        })

    for i, frame in enumerate(frames):
        content.append({
            "type": "text",
            "text": f"\nFrame {i+1}/{len(frames)} at t={frame['timestamp']}s:",
        })
        content.append(frame["image_block"])

    # Call Claude Vision for review
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model=_REVIEW_MODEL,
            max_tokens=1024,
            system=_REVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        review_text = response.content[0].text.strip()
        # Parse JSON from response
        try:
            report = json.loads(review_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            match = re.search(r'\{[\s\S]*\}', review_text)
            if match:
                report = json.loads(match.group())
            else:
                report = {
                    "pass": False,
                    "score": 5,
                    "raw_response": review_text,
                    "issues": ["Could not parse structured review"],
                }

        # Add metadata
        report["video_path"] = video_path
        report["actual_duration"] = round(actual_duration, 1)
        report["resolution"] = f"{width}x{height}"
        report["frames_analyzed"] = len(frames)
        report["tokens_used"] = response.usage.input_tokens + response.usage.output_tokens

        if expected_duration and abs(actual_duration - expected_duration) > 3:
            report.setdefault("issues", []).append(
                f"Duration mismatch: expected {expected_duration:.1f}s, got {actual_duration:.1f}s"
            )
            report["duration_looks_correct"] = False

        logger.info(
            "Self-review complete: %s (score=%s, pass=%s)",
            video_path, report.get("score"), report.get("pass"),
        )
        return report

    except Exception as e:
        logger.error("Self-review API call failed: %s", e)
        return {
            "pass": False,
            "score": 0,
            "error": f"Review API call failed: {str(e)[:200]}",
            "video_path": video_path,
            "actual_duration": round(actual_duration, 1),
        }


async def async_review_video(
    video_path: str, expected_duration: float | None = None,
) -> dict:
    """Async wrapper for review_video."""
    import asyncio
    return await asyncio.to_thread(review_video, video_path, expected_duration)
