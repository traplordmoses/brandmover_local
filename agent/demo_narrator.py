"""
Demo Narrator — ffmpeg post-processing for demo recordings.

Adds text overlays (narration captions) to recorded WebM videos and
converts to H.264 MP4. Also provides clip stitching.

Public API:
    timeline = build_narration_timeline(steps)
    mp4 = add_narration_overlay(input_webm, output_mp4, timeline)
    mp4 = convert_webm_to_mp4(input_webm, output_mp4)
    mp4 = stitch_clips(clip_paths, output_path)
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------

@dataclass
class NarrationEntry:
    text: str
    start: float
    end: float


def build_narration_timeline(steps) -> list[dict]:
    """Compute narration timeline from demo steps.

    Each step's narration is displayed starting when the step begins
    (cumulative wait of previous steps) and ending when the next step begins.

    Args:
        steps: List of DemoStep objects (or dicts with 'narration' and 'wait').

    Returns:
        List of {text, start, end} dicts for non-empty narrations.
    """
    timeline = []
    cursor = 0.0

    for step in steps:
        wait = getattr(step, "wait", None)
        if wait is None:
            wait = step.get("wait", 2.0) if isinstance(step, dict) else 2.0
        wait = float(wait)

        narration = getattr(step, "narration", None)
        if narration is None:
            narration = step.get("narration", "") if isinstance(step, dict) else ""

        start = cursor
        end = cursor + wait

        if narration:
            timeline.append({
                "text": narration,
                "start": round(start, 3),
                "end": round(end, 3),
            })

        cursor = end

    return timeline


# ---------------------------------------------------------------------------
# ffmpeg text escaping
# ---------------------------------------------------------------------------

def _escape_drawtext(text: str) -> str:
    """Escape text for ffmpeg drawtext filter.

    ffmpeg drawtext requires escaping of: ' : \\ and other special chars.
    """
    # Escape backslash first, then other special chars
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\u2019")  # Replace apostrophe with unicode right single quote
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


# ---------------------------------------------------------------------------
# ffmpeg commands
# ---------------------------------------------------------------------------

def _build_drawtext_filters(timeline: list[dict]) -> str:
    """Build ffmpeg drawtext filter chain from narration timeline."""
    filters = []
    for entry in timeline:
        text = _escape_drawtext(entry["text"])
        start = entry["start"]
        end = entry["end"]
        f = (
            f"drawtext=text='{text}'"
            f":fontsize=28"
            f":fontcolor=white"
            f":borderw=2"
            f":bordercolor=black"
            f":x=(w-text_w)/2"
            f":y=h-60"
            f":enable='between(t,{start},{end})'"
        )
        filters.append(f)
    return ",".join(filters)


def add_narration_overlay(
    input_webm: str,
    output_mp4: str,
    timeline: list[dict],
) -> str:
    """Add narration text overlays to a video and convert to MP4.

    Args:
        input_webm: Path to input WebM file from Playwright.
        output_mp4: Desired output MP4 path.
        timeline: List of {text, start, end} dicts.

    Returns:
        Path to the output MP4 file.
    """
    if not Path(input_webm).exists():
        raise FileNotFoundError(f"Input video not found: {input_webm}")

    drawtext = _build_drawtext_filters(timeline)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_webm,
        "-vf", drawtext,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        output_mp4,
    ]

    logger.info("ffmpeg narration: %s → %s (%d captions)", input_webm, output_mp4, len(timeline))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr[-500:]}")

    logger.info("Narrated video: %s", output_mp4)
    return output_mp4


def convert_webm_to_mp4(input_webm: str, output_mp4: str) -> str:
    """Convert WebM to MP4 without narration overlays.

    Args:
        input_webm: Path to input WebM.
        output_mp4: Desired output path.

    Returns:
        Path to the output MP4.
    """
    if not Path(input_webm).exists():
        raise FileNotFoundError(f"Input video not found: {input_webm}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_webm,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        output_mp4,
    ]

    logger.info("ffmpeg convert: %s → %s", input_webm, output_mp4)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr[-500:]}")

    return output_mp4


def stitch_clips(clip_paths: list[str], output_path: str) -> str:
    """Stitch multiple video clips using ffmpeg concat demuxer.

    Args:
        clip_paths: Ordered list of video file paths to concatenate.
        output_path: Desired output file path.

    Returns:
        Path to the stitched output file.
    """
    if not clip_paths:
        raise ValueError("No clips to stitch")

    for cp in clip_paths:
        if not Path(cp).exists():
            raise FileNotFoundError(f"Clip not found: {cp}")

    # Write concat list file
    list_path = Path(output_path).with_suffix(".txt")
    lines = [f"file '{p}'" for p in clip_paths]
    list_path.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]

    logger.info("ffmpeg stitch: %d clips → %s", len(clip_paths), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed (exit {result.returncode}): {result.stderr[-500:]}")

    return output_path
