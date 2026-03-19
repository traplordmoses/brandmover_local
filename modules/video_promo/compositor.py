"""
Compositor for video promos.

Takes:
- A looped background video
- A sequence of transparent PNG frames (text card overlay)
- An optional audio track

And composites them into the final output .mp4 using ffmpeg.
"""

import os
import subprocess
import logging
from modules.video_promo.config_schema import VideoPromoConfig

logger = logging.getLogger(__name__)


def composite_video(
    config: VideoPromoConfig,
    background_video: str,
    overlay_frame_pattern: str,
    total_frames: int,
    output_path: str,
) -> str:
    """
    Composite background video with text card overlay frames and optional audio.

    Uses ffmpeg's overlay filter with PNG sequence input for alpha compositing.

    Returns:
        Path to the final composited video.
    """
    # ── Build ffmpeg filter graph ──
    inputs = [
        "-i", background_video,
        "-framerate", str(config.fps),
        "-i", overlay_frame_pattern,
    ]

    # Scale background to FILL the frame (crop, don't letterbox)
    filter_parts = [
        f"[0:v]scale={config.width}:{config.height}:force_original_aspect_ratio=increase,"
        f"crop={config.width}:{config.height},setsar=1[bg]",
        "[bg][1:v]overlay=0:0:format=auto[out]",
    ]
    filter_graph = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_graph,
        "-map", "[out]",
    ]

    # Add audio if configured
    if config.audio.audio_path and os.path.exists(config.audio.audio_path):
        cmd.extend(["-i", config.audio.audio_path])

        audio_filters = []
        if config.audio.fade_in_seconds > 0:
            audio_filters.append(f"afade=t=in:st=0:d={config.audio.fade_in_seconds}")
        if config.audio.fade_out_seconds > 0:
            fade_out_start = config.total_duration_seconds - config.audio.fade_out_seconds
            audio_filters.append(f"afade=t=out:st={fade_out_start}:d={config.audio.fade_out_seconds}")
        if config.audio.volume != 1.0:
            audio_filters.append(f"volume={config.audio.volume}")

        if audio_filters:
            cmd.extend(["-af", ",".join(audio_filters)])

        cmd.extend(["-map", "2:a"])
        cmd.extend(["-shortest"])
    else:
        cmd.extend(["-an"])

    # Output encoding settings
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(config.fps),
        "-t", str(config.total_duration_seconds),
        "-movflags", "+faststart",
        output_path,
    ])

    logger.info("Compositing video...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg error:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg compositing failed:\n{result.stderr}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Expected output not found: {output_path}")

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Final video saved: %s (%.1f MB)", output_path, file_size)

    return output_path


def add_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    volume: float = 0.8,
    fade_in: float = 0.5,
    fade_out: float = 1.0,
) -> str:
    """Utility: Add audio track to an existing video."""
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())

    audio_filters = []
    if fade_in > 0:
        audio_filters.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        fade_out_start = duration - fade_out
        audio_filters.append(f"afade=t=out:st={fade_out_start}:d={fade_out}")
    if volume != 1.0:
        audio_filters.append(f"volume={volume}")

    af_str = ",".join(audio_filters) if audio_filters else None

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
    ]
    if af_str:
        cmd.extend(["-af", af_str])
    cmd.extend([
        "-map", "0:v",
        "-map", "1:a",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ])

    subprocess.run(cmd, check=True, capture_output=True)
    logger.info("Audio added: %s", output_path)
    return output_path
