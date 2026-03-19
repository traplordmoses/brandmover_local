"""
Video Promo Module — Main Orchestrator

Orchestrates the full pipeline:
  1. Generate AI background (or use custom)
  2. Render text card frames
  3. Composite everything into final video

Usage:
    from modules.video_promo import generate_promo_video

    config = VideoPromoConfig(...)
    output_path = await generate_promo_video(config)
"""

import os
import shutil
import asyncio
import logging
import tempfile
from pathlib import Path

from modules.video_promo.config_schema import VideoPromoConfig, BackgroundStyle
from modules.video_promo.background_gen import generate_background
from modules.video_promo.text_card_renderer import render_text_card_frames
from modules.video_promo.compositor import composite_video

logger = logging.getLogger(__name__)


async def generate_promo_video(
    config: VideoPromoConfig,
    work_dir: str | None = None,
    keep_intermediates: bool = False,
    fresh_bg: bool = False,
) -> str:
    """
    Generate a complete branded promo video.

    Args:
        config: Full VideoPromoConfig with all settings
        work_dir: Working directory for intermediate files.
                  If None, uses a temp directory.
        keep_intermediates: If True, don't clean up intermediate files.

    Returns:
        Path to the final .mp4 file.
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="video_promo_")
    else:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    frames_dir = os.path.join(work_dir, "text_frames")

    logger.info("=" * 60)
    logger.info("VIDEO PROMO GENERATION STARTING")
    logger.info("  Output: %s", config.output_path)
    logger.info("  Duration: %ss @ %sfps", config.total_duration_seconds, config.fps)
    logger.info("  Resolution: %sx%s", config.width, config.height)
    logger.info("  Background: %s", config.background.style.value)
    logger.info("=" * 60)

    try:
        # ── Step 1: Generate background ──
        logger.info("[1/3] GENERATING BACKGROUND...")

        bg_video_path = await generate_background(
            config=config.background,
            work_dir=work_dir,
            target_duration=config.total_duration_seconds,
            fps=config.fps,
            brand_name=config.brand_name,
            fresh_bg=fresh_bg,
        )
        logger.info("  Background ready: %s", bg_video_path)

        # ── Step 2: Render text card frames ──
        logger.info("[2/3] RENDERING TEXT CARD FRAMES...")

        frame_pattern, total_frames = await asyncio.to_thread(
            render_text_card_frames, config, frames_dir,
        )
        logger.info("  Text frames ready: %d frames", total_frames)

        # ── Step 3: Composite ──
        logger.info("[3/3] COMPOSITING FINAL VIDEO...")

        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)

        final_path = await asyncio.to_thread(
            composite_video, config, bg_video_path, frame_pattern, total_frames, config.output_path,
        )

        logger.info("=" * 60)
        logger.info("VIDEO COMPLETE: %s", final_path)
        file_size = os.path.getsize(final_path) / (1024 * 1024)
        logger.info("  Size: %.1f MB", file_size)
        logger.info("=" * 60)

        return final_path

    finally:
        if not keep_intermediates:
            try:
                if os.path.exists(frames_dir):
                    shutil.rmtree(frames_dir)
                logger.info("Cleaned up intermediate frames.")
            except Exception as e:
                logger.warning("Cleanup error (non-fatal): %s", e)


# ── Sync wrapper for non-async contexts ─────────────────────────────────────

def generate_promo_video_sync(
    config: VideoPromoConfig, fresh_bg: bool = False, **kwargs
) -> str:
    """Synchronous wrapper around generate_promo_video()."""
    return asyncio.run(generate_promo_video(config, fresh_bg=fresh_bg, **kwargs))


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m modules.video_promo.video_promo_module <config.json> [--keep]")
        sys.exit(1)

    config_path = sys.argv[1]
    keep = "--keep" in sys.argv
    fresh = "--fresh-bg" in sys.argv

    with open(config_path) as f:
        config_data = json.load(f)

    config = VideoPromoConfig(**config_data)

    output = generate_promo_video_sync(config, keep_intermediates=keep, fresh_bg=fresh)
    print(f"\nDone! Video saved to: {output}")
