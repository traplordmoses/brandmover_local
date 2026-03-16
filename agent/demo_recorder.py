"""
Demo Recorder — screen-record feature walkthroughs of web apps.

Uses Playwright for browser automation with video recording (WebM) or
screenshot fallback. Pairs with demo_narrator.py for ffmpeg post-processing.

Public API:
    result = await record_demo("demos/scripts/swipe-feature-demo.json")
"""

import asyncio
import ipaddress
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

from agent.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEMOS_DIR = PROJECT_ROOT / "demos"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DemoStep:
    action: str          # "goto" | "click" | "fill" | "wait" | "screenshot" | "scroll"
    target: str = ""     # URL path or CSS selector
    value: str = ""      # Text for fill action
    narration: str = ""  # Caption text for this step
    wait: float = 2.0    # Seconds to hold after action


@dataclass
class DemoScript:
    name: str
    url: str             # Base URL for the site to record
    steps: list[DemoStep]
    viewport_width: int = 390       # mobile default — best for social video
    viewport_height: int = 844      # iPhone 14/15 viewport
    mode: str = "video"  # "video" | "screenshot"


@dataclass
class DemoResult:
    script_name: str
    mode: str
    video_path: str = ""
    screenshot_paths: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# SSRF protection (mirrors agent/web_fetch.py)
# ---------------------------------------------------------------------------

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def validate_url(url: str) -> None:
    """Validate URL scheme and block private IPs. Raises ValueError on failure."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs allowed, got: {parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Could not parse hostname from URL: {url}")
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname {hostname!r}: {e}")
    for info in addrinfos:
        addr = ipaddress.ip_address(info[4][0])
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(f"Access to private/internal address {addr} is blocked")


# ---------------------------------------------------------------------------
# Script loading
# ---------------------------------------------------------------------------

def load_demo_script(path: str) -> DemoScript:
    """Load a demo script from a JSON file, applying defaults."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Demo script not found: {path}")
    raw = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Demo script must be a JSON object")
    if "name" not in raw or "url" not in raw:
        raise ValueError("Demo script must have 'name' and 'url' fields")
    if "steps" not in raw or not isinstance(raw["steps"], list):
        raise ValueError("Demo script must have a 'steps' array")

    steps = []
    for i, s in enumerate(raw["steps"]):
        if not isinstance(s, dict) or "action" not in s:
            raise ValueError(f"Step {i} must be an object with an 'action' field")
        steps.append(DemoStep(
            action=s["action"],
            target=s.get("target", ""),
            value=s.get("value", ""),
            narration=s.get("narration", ""),
            wait=float(s.get("wait", 2.0)),
        ))

    return DemoScript(
        name=raw["name"],
        url=raw["url"],
        steps=steps,
        viewport_width=int(raw.get("viewport_width", 1280)),
        viewport_height=int(raw.get("viewport_height", 720)),
        mode=raw.get("mode", "video"),
    )


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def _execute_step(page, step: DemoStep, base_url: str) -> None:
    """Execute a single demo step on the Playwright page."""
    action = step.action

    if action == "goto":
        target = step.target
        if target.startswith("/"):
            target = base_url.rstrip("/") + target
        page.goto(target, wait_until="networkidle", timeout=30000)

    elif action == "click":
        page.locator(step.target).first.click(timeout=10000)

    elif action == "fill":
        page.locator(step.target).first.fill(step.value, timeout=10000)

    elif action == "scroll":
        amount = int(step.value) if step.value else 400
        page.mouse.wheel(0, amount)

    elif action == "wait":
        pass  # wait is handled after dispatch

    elif action == "screenshot":
        pass  # screenshot is handled by the caller

    else:
        logger.warning("Unknown demo action: %s", action)

    # Hold for the configured wait duration
    if step.wait > 0:
        import time as _time
        _time.sleep(step.wait)


# ---------------------------------------------------------------------------
# Screenshot mode (PIL captions)
# ---------------------------------------------------------------------------

def _add_caption(img: Image.Image, text: str) -> Image.Image:
    """Add a semi-transparent caption bar at the bottom of an image."""
    if not text:
        return img
    img = img.convert("RGBA")
    w, h = img.size

    # Semi-transparent overlay bar
    bar_height = 60
    overlay = Image.new("RGBA", (w, bar_height), (0, 0, 0, 180))
    img.paste(overlay, (0, h - bar_height), overlay)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default(size=24)
        except TypeError:
            font = ImageFont.load_default()

    bb = font.getbbox(text)
    text_w = bb[2] - bb[0]
    text_x = (w - text_w) // 2
    text_y = h - bar_height + (bar_height - (bb[3] - bb[1])) // 2 - bb[1]
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

    return img


def _run_screenshot_sync(script: DemoScript, output_dir: Path) -> DemoResult:
    """Run demo in screenshot mode — one PNG per step with PIL captions."""
    from playwright.sync_api import sync_playwright

    validate_url(script.url)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = DemoResult(script_name=script.name, mode="screenshot")
    t0 = time.monotonic()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(viewport={
                    "width": script.viewport_width,
                    "height": script.viewport_height,
                })

                for i, step in enumerate(script.steps):
                    _execute_step(page, step, script.url)

                    # Take screenshot for every step
                    filename = f"step_{i:03d}_{step.action}.png"
                    raw_path = output_dir / f"_raw_{filename}"
                    page.screenshot(path=str(raw_path))

                    # Add caption
                    img = Image.open(raw_path)
                    img = _add_caption(img, step.narration)
                    final_path = output_dir / filename
                    img.save(str(final_path), format="PNG")
                    raw_path.unlink(missing_ok=True)

                    result.screenshot_paths.append(str(final_path))
                    logger.info("Screenshot: %s", final_path)

            finally:
                browser.close()

    except Exception as e:
        logger.error("Screenshot recording failed: %s", e)
        result.error = str(e)

    result.duration_seconds = round(time.monotonic() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Video mode (Playwright record_video_dir)
# ---------------------------------------------------------------------------

def _run_recording_sync(script: DemoScript, output_dir: Path) -> DemoResult:
    """Run demo with Playwright video recording + per-step screenshots."""
    from playwright.sync_api import sync_playwright

    validate_url(script.url)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_tmp = output_dir / "_video_tmp"
    video_tmp.mkdir(parents=True, exist_ok=True)
    result = DemoResult(script_name=script.name, mode="video")
    t0 = time.monotonic()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(
                    viewport={
                        "width": script.viewport_width,
                        "height": script.viewport_height,
                    },
                    record_video_dir=str(video_tmp),
                    record_video_size={
                        "width": script.viewport_width,
                        "height": script.viewport_height,
                    },
                )
                page = context.new_page()

                for i, step in enumerate(script.steps):
                    _execute_step(page, step, script.url)

                    # Take thumbnail screenshot for each step
                    filename = f"step_{i:03d}_{step.action}.png"
                    shot_path = output_dir / filename
                    page.screenshot(path=str(shot_path))
                    result.screenshot_paths.append(str(shot_path))

                # Close context to finalize video
                video_path_obj = page.video.path()
                context.close()

                # Move the WebM to output dir
                webm_path = output_dir / f"{script.name}_raw.webm"
                if video_path_obj and Path(video_path_obj).exists():
                    Path(video_path_obj).rename(webm_path)
                    result.video_path = str(webm_path)
                    logger.info("Video recorded: %s", webm_path)

            finally:
                browser.close()

        # Clean up temp video dir
        import shutil
        shutil.rmtree(video_tmp, ignore_errors=True)

    except Exception as e:
        logger.error("Video recording failed: %s", e)
        result.error = str(e)

    result.duration_seconds = round(time.monotonic() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def record_demo(
    script_path: str,
    mode_override: str | None = None,
) -> DemoResult:
    """Record a demo walkthrough. Main entry point.

    Args:
        script_path: Path to a demo script JSON file.
        mode_override: "video" or "screenshot" to override script default.

    Returns:
        DemoResult with paths to outputs.
    """
    script = load_demo_script(script_path)
    if mode_override:
        script.mode = mode_override

    # Build output directory
    ts = int(time.time())
    uid = uuid.uuid4().hex[:6]
    run_name = f"{script.name}_{ts}_{uid}"
    output_dir = DEMOS_DIR / run_name

    if script.mode == "video":
        result = await asyncio.to_thread(_run_recording_sync, script, output_dir)

        # Convert WebM → MP4 (no narration text — let video_styler handle overlays
        # to avoid double captions when both narrator and styler add text)
        if result.video_path and not result.error:
            try:
                from agent.demo_narrator import convert_webm_to_mp4
                mp4_path = str(DEMOS_DIR / f"{run_name}.mp4")
                mp4_path = await asyncio.to_thread(
                    convert_webm_to_mp4, result.video_path, mp4_path
                )
                result.video_path = mp4_path
                logger.info("Converted to MP4: %s", mp4_path)
            except Exception as e:
                logger.warning("ffmpeg conversion failed, keeping raw WebM: %s", e)

        # Fallback to screenshot mode if video failed
        if result.error:
            logger.warning("Video failed, falling back to screenshot mode: %s", result.error)
            result = await asyncio.to_thread(_run_screenshot_sync, script, output_dir)

    else:
        result = await asyncio.to_thread(_run_screenshot_sync, script, output_dir)

    return result
