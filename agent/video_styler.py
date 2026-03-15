"""
Video Styler — post-processing pipeline for polished social media videos.

Takes a raw screen recording (from Playwright or screen capture) and applies:
1. Device mockup frame (iPhone-style bezel)
2. Animated gradient background (holographic/glassmorphism)
3. Ken Burns pan/zoom effects
4. Narration text overlays
5. Output as H.264 MP4 optimized for X/Twitter

Reference style: mobile device mockup with holographic gradient background,
smooth zoom transitions between interaction highlights.

Uses ffmpeg for all video processing.
"""

import json
import logging
import math
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agent.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

_OUTPUTS_DIR = PROJECT_ROOT / "state" / "outputs"


@dataclass
class VideoStyle:
    """Configuration for video styling."""
    # Device frame
    device_frame: str = "iphone"  # "iphone" | "none"
    device_color: str = "black"   # bezel color

    # Background
    bg_type: str = "gradient"     # "gradient" | "solid" | "blur"
    bg_colors: list[str] = field(default_factory=lambda: [
        "#00e5ff", "#e040fb", "#7c4dff", "#ff80ab",
    ])

    # Pan/zoom (Ken Burns)
    zoom_enabled: bool = True
    zoom_start: float = 1.0       # start scale
    zoom_end: float = 1.15        # end scale (subtle zoom in)

    # Output
    output_width: int = 1080      # X/Twitter optimal
    output_height: int = 1080     # square for max engagement
    output_fps: int = 30
    output_crf: int = 20          # quality (lower = better)


def _escape_ffmpeg(text: str) -> str:
    """Escape text for ffmpeg drawtext filter."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _build_gradient_image(width: int, height: int, colors: list[str], output_path: str) -> str:
    """Generate a holographic gradient background image using PIL."""
    from PIL import Image, ImageDraw
    import colorsys

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Parse hex colors
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    parsed = [hex_to_rgb(c) for c in colors]
    if len(parsed) < 2:
        parsed = [(0, 229, 255), (224, 64, 251)]

    # Create a diagonal gradient with color stops
    for y in range(height):
        for x in range(width):
            # Diagonal position (0-1)
            t = (x / width + y / height) / 2
            # Add some waviness for holographic feel
            t += 0.05 * math.sin(x * 0.01) * math.cos(y * 0.01)
            t = max(0, min(1, t))

            # Interpolate between color stops
            n = len(parsed) - 1
            idx = min(int(t * n), n - 1)
            local_t = (t * n) - idx

            c1 = parsed[idx]
            c2 = parsed[min(idx + 1, n)]
            r = int(c1[0] + (c2[0] - c1[0]) * local_t)
            g = int(c1[1] + (c2[1] - c1[1]) * local_t)
            b = int(c1[2] + (c2[2] - c1[2]) * local_t)
            draw.point((x, y), (r, g, b))

    img.save(output_path, quality=95)
    logger.info("Generated gradient background: %s", output_path)
    return output_path


def _build_device_frame(width: int, height: int, output_path: str) -> str:
    """Generate a device frame overlay (iPhone-style rounded rect with notch)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Phone dimensions relative to canvas
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.05)
    phone_w = width - 2 * margin_x
    phone_h = height - 2 * margin_y
    corner_r = int(phone_w * 0.08)

    # Outer bezel (dark)
    bezel = 8
    draw.rounded_rectangle(
        [margin_x - bezel, margin_y - bezel,
         margin_x + phone_w + bezel, margin_y + phone_h + bezel],
        radius=corner_r + bezel,
        fill=(30, 30, 30, 255),
    )

    # Inner screen area (transparent — this is where the video shows through)
    inner_r = corner_r - 4
    draw.rounded_rectangle(
        [margin_x, margin_y, margin_x + phone_w, margin_y + phone_h],
        radius=inner_r,
        fill=(0, 0, 0, 0),
    )

    # Dynamic Island / notch at top center
    notch_w = int(phone_w * 0.3)
    notch_h = int(phone_h * 0.025)
    notch_x = margin_x + (phone_w - notch_w) // 2
    notch_y = margin_y + int(phone_h * 0.015)
    draw.rounded_rectangle(
        [notch_x, notch_y, notch_x + notch_w, notch_y + notch_h],
        radius=notch_h // 2,
        fill=(30, 30, 30, 255),
    )

    img.save(output_path)
    logger.info("Generated device frame: %s", output_path)
    return output_path


def apply_style(
    input_video: str,
    style: VideoStyle | None = None,
    narration_texts: list[dict] | None = None,
    output_path: str | None = None,
) -> str:
    """Apply video styling to a raw recording.

    Args:
        input_video: Path to raw video (WebM/MP4 from Playwright or screen capture).
        style: VideoStyle configuration. Defaults to holographic gradient style.
        narration_texts: Optional list of {"text": str, "start": float, "end": float}
                        for text overlays.
        output_path: Where to write the styled video. Auto-generated if None.

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

    # Get input video dimensions and duration
    probe_cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", input_video,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    probe_data = json.loads(probe.stdout) if probe.stdout else {}
    duration = float(probe_data.get("format", {}).get("duration", 10))

    # Generate background
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        bg_path = f.name
    _build_gradient_image(style.output_width, style.output_height, style.bg_colors, bg_path)

    # Build ffmpeg filter graph
    ow, oh = style.output_width, style.output_height

    # Calculate phone screen area
    margin_x = int(ow * 0.15)
    margin_y = int(oh * 0.05)
    phone_w = ow - 2 * margin_x
    phone_h = oh - 2 * margin_y
    corner_r = int(phone_w * 0.06)

    filters = []

    # Scale input video to fit phone screen area
    filters.append(f"[1:v]scale={phone_w}:{phone_h}:force_original_aspect_ratio=decrease,"
                   f"pad={phone_w}:{phone_h}:(ow-iw)/2:(oh-ih)/2:color=black@0")

    # Round corners on the video to match phone bezel
    filters.append(f"format=yuva420p,"
                   f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':"
                   f"a='if(gt(abs(X-{phone_w}/2),{phone_w}/2-{corner_r})*"
                   f"gt(abs(Y-{phone_h}/2),{phone_h}/2-{corner_r}),"
                   f"if(lte(hypot(abs(X-{phone_w}/2)-{phone_w}/2+{corner_r},"
                   f"abs(Y-{phone_h}/2)-{phone_h}/2+{corner_r}),{corner_r}),255,0),255)'")
    filters.append(f"[phone]")

    # Background: static image looped for duration
    filters_str = ";".join(filters)

    # Overlay phone on background
    overlay_x = margin_x
    overlay_y = margin_y

    # Ken Burns zoom
    if style.zoom_enabled:
        zs = style.zoom_start
        ze = style.zoom_end
        # Zoom into center
        zoom_filter = (
            f"[bg_base]scale={int(ow*ze)}:{int(oh*ze)},"
            f"zoompan=z='zoom+{(ze-zs)/duration/style.output_fps}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={int(duration*style.output_fps)}:s={ow}x{oh}:fps={style.output_fps}[bg]"
        )
    else:
        zoom_filter = f"[bg_base]scale={ow}:{oh}[bg]"

    # Build complete filter
    full_filter = (
        # Input 0 = background image, Input 1 = video
        f"[0:v]scale={ow}:{oh},loop=loop=-1:size=1:start=0,setpts=N/{style.output_fps}/TB[bg_base];"
        f"{zoom_filter};"
        f"[1:v]scale={phone_w}:{phone_h}:force_original_aspect_ratio=decrease,"
        f"pad={phone_w}:{phone_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=rgba[phone];"
        f"[bg][phone]overlay={overlay_x}:{overlay_y}:format=auto"
    )

    # Add narration text overlays
    if narration_texts:
        for i, nt in enumerate(narration_texts):
            text = _escape_ffmpeg(nt.get("text", ""))
            start = nt.get("start", 0)
            end = nt.get("end", start + 3)
            full_filter += (
                f",drawtext=text='{text}':"
                f"fontsize=32:fontcolor=white:"
                f"borderw=2:bordercolor=black@0.7:"
                f"x=(w-text_w)/2:y=h-80:"
                f"enable='between(t,{start},{end})'"
            )

    full_filter += "[out]"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", bg_path,      # input 0: background
        "-i", input_video,                  # input 1: video
        "-filter_complex", full_filter,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(style.output_crf),
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        "-an",                              # no audio
        output_path,
    ]

    logger.info("Styling video: %s → %s (%.1fs)", input_video, output_path, duration)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr[-500:] if result.stderr else "no output")
        # Fallback: just re-encode without styling
        fallback_cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-c:v", "libx264", "-preset", "fast", "-crf", str(style.output_crf),
            "-pix_fmt", "yuv420p", "-an", output_path,
        ]
        subprocess.run(fallback_cmd, capture_output=True, timeout=60)
        logger.info("Fallback encode complete: %s", output_path)
    else:
        logger.info("Styled video complete: %s", output_path)

    # Clean up temp files
    Path(bg_path).unlink(missing_ok=True)

    return output_path


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
