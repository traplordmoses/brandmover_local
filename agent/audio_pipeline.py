"""
Audio Pipeline — TTS voiceover and audio mixing for video generation.

Supports ElevenLabs TTS (primary) and OpenAI TTS (fallback).
FFmpeg for mixing voiceover + background music onto rendered video.
"""
import asyncio
import logging
import os
import subprocess
import uuid
from pathlib import Path

from agent._client import get_httpx
from agent.paths import PROJECT_ROOT
from config import settings

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "state" / "outputs"

ELEVENLABS_API_KEY = settings.ELEVENLABS_API_KEY
OPENAI_API_KEY = settings.OPENAI_API_KEY


def _collect_narrations(scenes: list[dict]) -> list[dict]:
    """Extract narration text and timing from scenes."""
    narrations = []
    cumulative_frames = 0
    fps = 30

    for scene in scenes:
        narration = scene.get("narration", "")
        duration_frames = scene.get("durationFrames", 90)

        if narration:
            narrations.append({
                "text": narration,
                "start_seconds": cumulative_frames / fps,
                "duration_seconds": duration_frames / fps,
            })

        cumulative_frames += duration_frames

    return narrations


async def generate_voiceover_elevenlabs(
    text: str,
    voice_id: str = "",
    output_path: str | None = None,
) -> str:
    """Generate TTS audio via ElevenLabs API."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not output_path:
        output_path = str(OUTPUT_DIR / f"vo_{uuid.uuid4().hex[:8]}.mp3")

    client = get_httpx()
    resp = await client.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    Path(output_path).write_bytes(resp.content)

    logger.info("ElevenLabs TTS: %d chars -> %s", len(text), output_path)
    return output_path


async def generate_voiceover_openai(
    text: str,
    voice: str = "alloy",
    output_path: str | None = None,
) -> str:
    """Generate TTS audio via OpenAI TTS API (fallback)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not output_path:
        output_path = str(OUTPUT_DIR / f"vo_{uuid.uuid4().hex[:8]}.mp3")

    client = get_httpx()
    resp = await client.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": voice,
        },
        timeout=60,
    )
    resp.raise_for_status()
    Path(output_path).write_bytes(resp.content)

    logger.info("OpenAI TTS: %d chars -> %s", len(text), output_path)
    return output_path


async def generate_voiceover(
    scenes: list[dict],
    voice_id: str | None = None,
) -> str | None:
    """Generate voiceover from scene narrations. Returns path to audio file."""
    narrations = _collect_narrations(scenes)
    if not narrations:
        return None

    # Combine all narration text with pauses
    full_text = " ... ".join(n["text"] for n in narrations)

    if not full_text.strip():
        return None

    # Try ElevenLabs first, fall back to OpenAI
    try:
        if ELEVENLABS_API_KEY:
            return await generate_voiceover_elevenlabs(
                full_text, voice_id or settings.ELEVENLABS_VOICE_ID
            )
    except Exception as e:
        logger.warning("ElevenLabs TTS failed, trying OpenAI: %s", e)

    try:
        if OPENAI_API_KEY:
            return await generate_voiceover_openai(full_text)
    except Exception as e:
        logger.warning("OpenAI TTS also failed: %s", e)

    logger.warning("No TTS available -- skipping voiceover")
    return None


def mix_audio(
    video_path: str,
    voiceover_path: str | None = None,
    music_path: str | None = None,
    music_volume: float = 0.15,
) -> str:
    """Mix voiceover and/or background music onto a rendered video via ffmpeg.

    Returns path to the final video with audio.
    """
    if not voiceover_path and not music_path:
        return video_path

    output_path = video_path.replace(".mp4", "_audio.mp4")

    cmd = ["ffmpeg", "-y", "-i", video_path]
    filter_parts = []
    input_idx = 1

    if voiceover_path:
        cmd.extend(["-i", voiceover_path])
        filter_parts.append(f"[{input_idx}:a]volume=1.0[vo]")
        input_idx += 1

    if music_path:
        cmd.extend(["-i", music_path])
        filter_parts.append(f"[{input_idx}:a]volume={music_volume}[bg]")
        input_idx += 1

    if voiceover_path and music_path:
        filter_complex = (
            ";".join(filter_parts)
            + ";[vo][bg]amix=inputs=2:duration=shortest[aout]"
        )
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
        ])
    elif voiceover_path:
        cmd.extend([
            "-filter_complex",
            filter_parts[0].replace("[vo]", "[aout]"),
            "-map", "0:v", "-map", "[aout]",
        ])
    elif music_path:
        cmd.extend([
            "-filter_complex",
            filter_parts[0].replace("[bg]", "[aout]"),
            "-map", "0:v", "-map", "[aout]",
        ])

    cmd.extend(["-c:v", "copy", "-c:a", "aac", "-shortest", output_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error("Audio mixing failed: %s", result.stderr[-300:])
            return video_path  # fall back to silent video
    except Exception as e:
        logger.error("Audio mixing error: %s", e)
        return video_path

    logger.info("Audio mixed: %s", output_path)
    return output_path
