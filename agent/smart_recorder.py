"""
Smart Recorder — vision-in-the-loop browser agent for autonomous demo videos.

Instead of blindly executing scripted steps, this agent:
1. Takes a high-level goal ("show the swipe feature, upload a meme, vote")
2. Navigates the site using Playwright
3. After each action, screenshots the page and sends it to Claude Vision
4. Claude decides the next action based on what it actually sees
5. Adapts to loading states, modals, errors, and unexpected UI
6. Records the entire session as video via Playwright record_video_dir
7. Builds a narration timeline from what actually happened

Uses Claude Sonnet for vision reasoning (fast + cheap per screenshot).
Max 20 steps per recording to control API costs (~$0.50 worst case).
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from agent.demo_recorder import validate_url, DemoResult
from agent.paths import PROJECT_ROOT
from config import settings

logger = logging.getLogger(__name__)

DEMOS_DIR = PROJECT_ROOT / "demos"
_SITE_GUIDE_PATH = PROJECT_ROOT / "brand" / "references" / "foidfun-site-guide.md"

# Vision model — Haiku for high rate limits and low cost
# Sonnet hits 30k input tokens/min on free tier with screenshots
_VISION_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RecordingStep:
    """One step in the recorded session."""
    action: str
    target: str = ""
    value: str = ""
    narration: str = ""
    timestamp: float = 0.0
    screenshot_path: str = ""
    error: str = ""


@dataclass
class SmartDemoResult:
    """Extended DemoResult with reasoning metadata."""
    script_name: str
    mode: str = "video"
    video_path: str = ""
    screenshot_paths: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str = ""
    steps_taken: list[RecordingStep] = field(default_factory=list)
    narration_timeline: list[dict] = field(default_factory=list)
    reasoning_steps: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Browser tools for Claude
# ---------------------------------------------------------------------------

BROWSER_TOOLS = [
    {
        "name": "click",
        "description": (
            "Click an element on the page. Use CSS selectors or text-based selectors. "
            "Examples: 'button:has-text(\"Connect\")', '.submit-btn', 'text=PROPOSE A MEME'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS or text selector for the element."},
                "narration": {"type": "string", "description": "Brief narration (5-15 words) of what this click does for the viewer."},
            },
            "required": ["selector", "narration"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input field.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector for the input."},
                "text": {"type": "string", "description": "Text to type."},
                "narration": {"type": "string", "description": "Brief narration of this action."},
            },
            "required": ["selector", "text", "narration"],
        },
    },
    {
        "name": "goto",
        "description": "Navigate to a URL path on the site. Use relative paths like '/swipe', '/gallery', '/pray'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "URL path like '/swipe' or '/gallery'."},
                "narration": {"type": "string", "description": "Brief narration of where we're going."},
            },
            "required": ["path", "narration"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["down", "up"]},
                "amount": {"type": "integer", "description": "Pixels to scroll. Default 400."},
                "narration": {"type": "string", "description": "Brief narration."},
            },
            "required": ["direction", "narration"],
        },
    },
    {
        "name": "swipe",
        "description": (
            "Perform a swipe/drag gesture on an element. Use for swiping meme cards "
            "left (reject) or right (approve). The gesture starts at the element center "
            "and drags in the specified direction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector for the element to swipe."},
                "direction": {"type": "string", "enum": ["left", "right", "up", "down"], "description": "Direction to swipe."},
                "distance": {"type": "integer", "description": "Pixels to drag. Default 200. Use 250+ for a decisive swipe."},
                "narration": {"type": "string", "description": "Brief narration of this swipe action."},
            },
            "required": ["selector", "direction", "narration"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for loading, animations, or transitions. Use when you see a spinner or loading state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "How long to wait (1-5 seconds)."},
                "reason": {"type": "string", "description": "Why we're waiting."},
            },
            "required": ["seconds", "reason"],
        },
    },
    {
        "name": "upload_file",
        "description": "Upload a file to a file input element. Use for meme submission or image uploads.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector for the file input or upload zone."},
                "asset_key": {"type": "string", "description": "Key from prepare_assets dict, e.g. 'meme'."},
                "narration": {"type": "string", "description": "Brief narration."},
            },
            "required": ["selector", "asset_key", "narration"],
        },
    },
    {
        "name": "finish",
        "description": "Signal that the demo recording is complete. Call this when all goals are achieved or when you've shown enough.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Final narration line for the video ending."},
            },
            "required": ["summary"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    goals: list[str],
    base_url: str,
    max_steps: int,
    asset_keys: list[str],
) -> str:
    """Build the system prompt with site knowledge and goals."""

    # Load site guide if available
    site_guide = ""
    if _SITE_GUIDE_PATH.exists():
        try:
            site_guide = _SITE_GUIDE_PATH.read_text(encoding="utf-8")
        except Exception:
            pass

    goals_text = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(goals))

    assets_text = ""
    if asset_keys:
        assets_text = (
            "\n\nYou have these prepared assets available for upload_file:\n"
            + "\n".join(f"  - {k}" for k in asset_keys)
        )

    return f"""You are a browser automation agent recording a product demo video of {base_url}.

You are controlling a mobile browser (iPhone viewport). After each action you take, you will see a screenshot of the current page state. Based on what you see, decide the next action.

## Your Goals
Record a compelling demo that shows:
{goals_text}

## Rules
- You have a maximum of {max_steps} actions. Be efficient but thorough.
- After each action, you'll see the resulting screenshot. Adapt to what you see.
- If the page is loading (spinner, skeleton, loading text), call `wait` for 2-3 seconds.
- If a modal or dialog appears, interact with it.
- If something fails (element not found), try a different approach or a different selector.
- Every action MUST include a `narration` field — this becomes the text overlay in the final video.
- Keep narration short and punchy (5-15 words). Write it for social media viewers.
- When you've achieved all goals or shown enough, call `finish`.
- Use text-based selectors when possible: 'text=Connect', 'button:has-text("Submit")'
- The site has a bottom navigation bar with: Home, Pray, Board, Swipe, Gallery, MiFOID, Faucet
- IMPORTANT: The gallery route is /gallery, NOT /loreboard. The loreboard/collage is at /board.

## Critical Interaction Rules

### Wallet Connection (DO THIS FIRST)
The site requires a connected wallet for most features. Here's exactly how:
1. Click the "CONNECT" button (usually in the window title bar area)
2. A RainbowKit modal will appear with wallet options — click "FOID Wallet"
3. A setup dialog appears explaining FOID Wallet — click "Continue"
4. A "CHOOSE A PIN" form appears with TWO fields:
   - First field: "Enter PIN" — type "foid123456" (must be 6+ characters)
   - Second field: "Confirm PIN" — type "foid123456" (must match exactly)
5. Click "Create Wallet" button
6. The biometric step will auto-complete (virtual authenticator handles it)
7. Wait 3-5 seconds for wallet creation to finish
8. You'll know it worked when: "DISCONNECTED" changes to a wallet address, or the CONNECT button disappears
9. If you see an error, try the whole flow again from step 1
10. DO NOT proceed to other features until the wallet is connected

### Swiping Meme Cards (THE STAR OF THE VIDEO)
- On `/swipe`, meme cards appear as images you can vote on
- Use the `swipe` tool with direction "right" to approve, "left" to reject
- Target the meme card element (look for the card/image in the viewport)
- SWIPE AT LEAST 5 CARDS — this is the hero moment of the demo
- Alternate between swipe right (approve) and swipe left (reject) to show both
- PACE IS EVERYTHING: swipe the card IMMEDIATELY after it appears — do NOT analyze it, do NOT describe it, do NOT think about it. Just swipe.
- Call swipe tools BACK TO BACK with zero delay. No wait calls between swipes.
- The "X remaining" counter should visibly decrease — this proves the votes are real
- If you see "No live proposals", click the "CLOSED" tab to show past votes
- SPEED RULE: each card gets ONE swipe tool call, then immediately swipe the next card. No screenshots between swipes — just swipe, swipe, swipe.

### Onboarding / Tutorial
- If you see an onboarding carousel with slides (LOREBOARD, PRAY, SWIPE, GALLERY, etc.) and a "SKIP" button in the top-right corner, click "SKIP" immediately. Don't spend time going through tutorial slides.
- If you see "ENTER" or "NEXT >" buttons on a tutorial, click SKIP to get to the actual app faster.

### General Rules
- Do NOT just visit pages and move on if they show empty/disconnected states. Interact!
- If you see actual content (meme cards, proposals, gallery entries), spend time showing them — scroll, click, explore.
- The video should show the app being USED, not just visited. Click buttons, interact with UI elements.
- If a modal or popup appears, interact with it — don't ignore it.
- After clicking any button that triggers a transaction or modal, wait 2-3 seconds for the UI to respond.
- Be efficient — don't waste steps on menus, dropdowns, or settings unless it's part of the demo goal.
{assets_text}

## Site Guide
{site_guide if site_guide else "No site guide available. Navigate based on what you see in the screenshots."}"""


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------

def _take_screenshot(page, path: str) -> dict:
    """Take screenshot, downscale for token efficiency, return Claude Vision image block.

    Full-res PNG saved to disk for reference. A 50%-scaled JPEG is sent to Claude
    to stay well under rate limits (~4x fewer image tokens than full PNG).
    """
    page.screenshot(path=path)

    # Downscale for Claude — reduces image tokens dramatically
    try:
        from PIL import Image
        import io
        img = Image.open(path)
        # Scale to max 400px wide (enough for Claude to read UI elements)
        max_w = 400
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        media_type = "image/jpeg"
    except ImportError:
        # Fallback: send full PNG
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        media_type = "image/png"

    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64},
    }


# ---------------------------------------------------------------------------
# Browser action executor
# ---------------------------------------------------------------------------

def _execute_action(
    page,
    tool_name: str,
    tool_input: dict,
    base_url: str,
    prepare_assets: dict,
) -> dict:
    """Execute a browser action via Playwright. Returns status dict."""

    try:
        if tool_name == "click":
            selector = tool_input["selector"]
            page.locator(selector).first.click(timeout=5000)
            time.sleep(1.5)  # wait for UI response
            return {"status": "ok", "action": "click", "target": selector}

        elif tool_name == "type_text":
            selector = tool_input["selector"]
            text = tool_input["text"]
            page.locator(selector).first.fill(text, timeout=5000)
            time.sleep(0.5)
            return {"status": "ok", "action": "type", "target": selector, "text": text}

        elif tool_name == "goto":
            path = tool_input["path"]
            url = base_url.rstrip("/") + path if path.startswith("/") else path
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            time.sleep(2.0)  # wait for hydration
            return {"status": "ok", "action": "goto", "url": url}

        elif tool_name == "scroll":
            direction = tool_input.get("direction", "down")
            amount = tool_input.get("amount", 400)
            if direction == "up":
                amount = -amount
            page.mouse.wheel(0, amount)
            time.sleep(0.8)
            return {"status": "ok", "action": "scroll", "direction": direction, "amount": amount}

        elif tool_name == "swipe":
            selector = tool_input["selector"]
            direction = tool_input.get("direction", "right")
            distance = tool_input.get("distance", 200)
            # Get element center position
            box = page.locator(selector).first.bounding_box(timeout=5000)
            if not box:
                return {"error": f"Could not find element for swipe: {selector}"}
            cx = box["x"] + box["width"] / 2
            cy = box["y"] + box["height"] / 2
            # Calculate end position
            dx, dy = 0, 0
            if direction == "right":
                dx = distance
            elif direction == "left":
                dx = -distance
            elif direction == "down":
                dy = distance
            elif direction == "up":
                dy = -distance
            # Perform drag gesture (slow for visual effect in recording)
            page.mouse.move(cx, cy)
            page.mouse.down()
            # Move in small increments for smooth visual in recording
            steps = 15
            for i in range(1, steps + 1):
                page.mouse.move(
                    cx + dx * i / steps,
                    cy + dy * i / steps,
                )
                time.sleep(0.03)
            page.mouse.up()
            time.sleep(0.6)  # brief pause for card transition
            return {"status": "ok", "action": "swipe", "direction": direction, "distance": distance}

        elif tool_name == "wait":
            seconds = min(float(tool_input.get("seconds", 2)), 5.0)
            time.sleep(seconds)
            return {"status": "ok", "action": "wait", "seconds": seconds}

        elif tool_name == "upload_file":
            selector = tool_input["selector"]
            asset_key = tool_input.get("asset_key", "")
            asset_path = prepare_assets.get(asset_key, "")
            if not asset_path or not Path(asset_path).exists():
                return {"error": f"Asset '{asset_key}' not found. Available: {list(prepare_assets.keys())}"}
            # Use set_input_files for file inputs
            page.locator(selector).set_input_files(str(asset_path))
            time.sleep(1.0)
            return {"status": "ok", "action": "upload", "asset": asset_key}

        elif tool_name == "finish":
            return {"status": "finished", "summary": tool_input.get("summary", "")}

        else:
            return {"error": f"Unknown action: {tool_name}"}

    except PlaywrightTimeoutError:
        return {"error": f"Element not found or timed out: {tool_input.get('selector', 'N/A')}. Try a different selector or wait for loading."}
    except Exception as e:
        return {"error": f"Action failed: {str(e)[:200]}"}


# ---------------------------------------------------------------------------
# Main agent loop (synchronous — runs inside asyncio.to_thread)
# ---------------------------------------------------------------------------

def _run_smart_recording_sync(
    name: str,
    url: str,
    goals: list[str],
    viewport_w: int,
    viewport_h: int,
    max_steps: int,
    prepare_assets: dict,
    output_dir: Path,
) -> SmartDemoResult:
    """The synchronous vision-in-the-loop agent loop."""

    validate_url(url)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_tmp = output_dir / "_video_tmp"
    video_tmp.mkdir(parents=True, exist_ok=True)

    result = SmartDemoResult(script_name=name)
    t0 = time.monotonic()
    recording_start = 0.0
    total_tokens = 0

    # Init Claude client (sync — we're in a thread)
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    system_prompt = _build_system_prompt(
        goals=goals,
        base_url=url,
        max_steps=max_steps,
        asset_keys=list(prepare_assets.keys()),
    )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(
                    viewport={"width": viewport_w, "height": viewport_h},
                    record_video_dir=str(video_tmp),
                    record_video_size={"width": viewport_w, "height": viewport_h},
                )
                page = context.new_page()

                # Enable virtual WebAuthn authenticator via CDP
                # This allows FOID Wallet's passkey/biometric flow to work
                # in headless Chromium (no real fingerprint sensor needed)
                try:
                    cdp = context.new_cdp_session(page)
                    cdp.send("WebAuthn.enable")
                    cdp.send("WebAuthn.addVirtualAuthenticator", {
                        "options": {
                            "protocol": "ctap2",
                            "transport": "internal",
                            "hasResidentKey": True,
                            "hasUserVerification": True,
                            "isUserVerified": True,
                            "automaticPresenceSimulation": True,
                        }
                    })
                    logger.info("Virtual WebAuthn authenticator enabled")
                except Exception as e:
                    logger.warning("Could not enable virtual authenticator: %s", e)

                # Navigate to starting URL
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                time.sleep(3.0)  # let the page fully hydrate
                recording_start = time.monotonic()

                # Take initial screenshot
                init_screenshot_path = str(output_dir / "step_000_start.png")
                init_image = _take_screenshot(page, init_screenshot_path)
                result.screenshot_paths.append(init_screenshot_path)

                # Build initial messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is the starting state of the page. Begin recording the demo. Achieve the goals efficiently."},
                            init_image,
                        ],
                    },
                ]

                # Track conversation screenshots for context window management
                screenshot_count = 1

                # ── Agent loop ───────────────────────────────────────
                for step_num in range(1, max_steps + 1):
                    logger.info("Smart recorder step %d/%d", step_num, max_steps)

                    # Call Claude
                    try:
                        response = client.messages.create(
                            model=_VISION_MODEL,
                            max_tokens=1024,
                            system=system_prompt,
                            tools=BROWSER_TOOLS,
                            messages=messages,
                        )
                    except Exception as e:
                        logger.error("Claude API error: %s", e)
                        result.error = f"Claude API error: {str(e)[:200]}"
                        break

                    total_tokens += response.usage.input_tokens + response.usage.output_tokens

                    # Check stop reason
                    if response.stop_reason == "end_turn":
                        logger.info("Claude ended turn (no more actions)")
                        break

                    # Find tool_use block
                    tool_use = None
                    text_content = ""
                    for block in response.content:
                        if block.type == "tool_use":
                            tool_use = block
                        elif block.type == "text":
                            text_content = block.text

                    if not tool_use:
                        logger.info("No tool call from Claude, ending recording")
                        break

                    tool_name = tool_use.name
                    tool_input = tool_use.input

                    # Check for finish
                    if tool_name == "finish":
                        narration = tool_input.get("summary", "")
                        timestamp = time.monotonic() - recording_start
                        result.steps_taken.append(RecordingStep(
                            action="finish", narration=narration, timestamp=timestamp,
                        ))
                        # Append assistant + tool result to close the loop cleanly
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": tool_use.id,
                                         "content": json.dumps({"status": "finished"})}],
                        })
                        logger.info("Recording finished: %s", narration)
                        break

                    # Execute browser action
                    action_result = _execute_action(
                        page, tool_name, tool_input, url, prepare_assets,
                    )

                    # Take screenshot after action — skip for consecutive swipes
                    # to maintain fast pacing in the recording
                    timestamp = time.monotonic() - recording_start
                    screenshot_path = str(output_dir / f"step_{step_num:03d}_{tool_name}.png")
                    prev_was_swipe = (step_num > 1 and len(result.steps_taken) > 0
                                      and result.steps_taken[-1].action == "swipe")
                    if tool_name == "swipe" and prev_was_swipe:
                        # Skip screenshot between consecutive swipes — just report result
                        screenshot_image = None
                    else:
                        screenshot_image = _take_screenshot(page, screenshot_path)
                        result.screenshot_paths.append(screenshot_path)
                        screenshot_count += 1

                    # Record step
                    narration = tool_input.get("narration", tool_input.get("reason", ""))
                    result.steps_taken.append(RecordingStep(
                        action=tool_name,
                        target=tool_input.get("selector", tool_input.get("path", "")),
                        value=tool_input.get("text", ""),
                        narration=narration,
                        timestamp=timestamp,
                        screenshot_path=screenshot_path,
                        error=action_result.get("error", ""),
                    ))

                    # Build tool result — include screenshot unless skipped for pacing
                    tool_result_content = [
                        {"type": "text", "text": json.dumps(action_result)},
                    ]
                    if screenshot_image:
                        tool_result_content.append(screenshot_image)

                    # Append assistant response + tool result
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tool_use.id,
                             "content": tool_result_content},
                        ],
                    })

                    # Context window management: if we have too many screenshots,
                    # summarize older turns to reduce image tokens
                    if screenshot_count > 8:
                        messages = _compact_messages(messages)
                        screenshot_count = 4  # approximate after compaction

                    logger.info(
                        "Step %d: %s(%s) → %s | narration: %s",
                        step_num, tool_name,
                        tool_input.get("selector", tool_input.get("path", ""))[:40],
                        "ok" if "error" not in action_result else action_result["error"][:50],
                        narration[:50],
                    )

                # ── Finalize video ───────────────────────────────────
                video_path_obj = page.video.path()
                context.close()

                webm_path = output_dir / f"{name}_raw.webm"
                if video_path_obj and Path(video_path_obj).exists():
                    Path(video_path_obj).rename(webm_path)
                    result.video_path = str(webm_path)

            finally:
                browser.close()

        # Clean up temp video dir
        import shutil
        shutil.rmtree(video_tmp, ignore_errors=True)

    except Exception as e:
        logger.error("Smart recording failed: %s", e)
        result.error = str(e)

    # Build narration timeline from steps
    result.narration_timeline = _build_timeline(result.steps_taken)
    result.reasoning_steps = len(result.steps_taken)
    result.total_tokens = total_tokens
    result.duration_seconds = round(time.monotonic() - t0, 2)

    logger.info(
        "Smart recording '%s': %d steps, %d tokens, %.1fs, video=%s",
        name, result.reasoning_steps, total_tokens,
        result.duration_seconds, bool(result.video_path),
    )
    return result


# ---------------------------------------------------------------------------
# Context compaction — remove old screenshots to manage token budget
# ---------------------------------------------------------------------------

def _compact_messages(messages: list[dict]) -> list[dict]:
    """Remove screenshots from older messages, keeping only recent ones.

    Keeps the first message (initial screenshot) and last 3 tool results with images.
    Older tool results have their images replaced with text descriptions.
    """
    compacted = []
    # Find indices of messages with images (tool results)
    image_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            has_image = any(
                isinstance(c, dict) and (
                    c.get("type") == "image" or
                    (c.get("type") == "tool_result" and isinstance(c.get("content"), list) and
                     any(isinstance(cc, dict) and cc.get("type") == "image" for cc in c["content"]))
                )
                for c in msg["content"]
            )
            if has_image:
                image_indices.append(i)

    # Keep first and last 3 image messages
    keep_indices = set()
    if image_indices:
        keep_indices.add(image_indices[0])  # initial screenshot
        for idx in image_indices[-3:]:       # last 3
            keep_indices.add(idx)

    for i, msg in enumerate(messages):
        if i in image_indices and i not in keep_indices:
            # Strip images from this message, replace with text summary
            new_content = []
            for c in msg.get("content", []):
                if isinstance(c, dict):
                    if c.get("type") == "image":
                        new_content.append({"type": "text", "text": "[screenshot removed for context efficiency]"})
                    elif c.get("type") == "tool_result" and isinstance(c.get("content"), list):
                        # Keep text parts, remove images
                        new_inner = []
                        for cc in c["content"]:
                            if isinstance(cc, dict) and cc.get("type") == "image":
                                new_inner.append({"type": "text", "text": "[screenshot removed]"})
                            else:
                                new_inner.append(cc)
                        new_content.append({**c, "content": new_inner})
                    else:
                        new_content.append(c)
                else:
                    new_content.append(c)
            compacted.append({**msg, "content": new_content})
        else:
            compacted.append(msg)

    return compacted


# ---------------------------------------------------------------------------
# Narration timeline builder
# ---------------------------------------------------------------------------

def _build_timeline(steps: list[RecordingStep]) -> list[dict]:
    """Build narration timeline from recorded steps."""
    timeline = []
    for i, step in enumerate(steps):
        if not step.narration:
            continue
        start = step.timestamp
        # End at next step's timestamp, or 3 seconds after
        if i + 1 < len(steps):
            end = steps[i + 1].timestamp
        else:
            end = start + 3.0
        # Ensure minimum display time
        if end - start < 1.5:
            end = start + 1.5
        timeline.append({
            "text": step.narration,
            "start": round(start, 2),
            "end": round(end, 2),
        })
    return timeline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def smart_record_demo(
    goals: list[str],
    url: str = "https://foidfun.vercel.app",
    name: str = "smart-demo",
    viewport: str = "mobile",
    max_steps: int = 20,
    prepare_assets: dict | None = None,
) -> SmartDemoResult:
    """Record an autonomous demo walkthrough using vision-guided browser agent.

    Args:
        goals: High-level goals like ["show the swipe feature", "upload a meme"].
        url: Base URL to record.
        name: Short name for the recording.
        viewport: "mobile" (390x844) or "desktop" (1280x720).
        max_steps: Maximum reasoning steps (controls cost). Default 20.
        prepare_assets: Dict of asset_key → file_path for uploads.
                       e.g. {"meme": "/path/to/meme.jpg"}

    Returns:
        SmartDemoResult with video path, screenshots, narration timeline.
    """
    if viewport == "desktop":
        vp_w, vp_h = 1280, 720
    else:
        vp_w, vp_h = 390, 844

    if prepare_assets is None:
        prepare_assets = {}

    # Build output directory
    ts = int(time.time())
    uid = uuid.uuid4().hex[:6]
    run_name = f"{name}_{ts}_{uid}"
    output_dir = DEMOS_DIR / run_name

    result = await asyncio.to_thread(
        _run_smart_recording_sync,
        name=name,
        url=url,
        goals=goals,
        viewport_w=vp_w,
        viewport_h=vp_h,
        max_steps=max_steps,
        prepare_assets=prepare_assets,
        output_dir=output_dir,
    )

    # Convert WebM → MP4 if video was recorded
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
            logger.warning("MP4 conversion failed: %s", e)

    return result
