"""
Tool registry for agent mode.
Defines the 7 tools available to Claude and their async handler functions.
Each handler receives (input_dict, ResourceTracker) and returns a string.
"""

import asyncio
import json
import logging
import re
import shlex
import subprocess
import time as _time
from pathlib import Path

from PIL import Image as _PILImage

from agent import feedback, figma, guidelines, image_gen, state as _state
from agent.resource_log import ResourceTracker
from config import settings

logger = logging.getLogger(__name__)

# Allowlist of OpenClaw scripts that can be executed
_OPENCLAW_ALLOWLIST = {
    "read_vault.js",
    "create_campaign.js",
    "schedule_content.js",
    "log_activity.js",
    "browse_tasks.js",
    "claim_task.js",
    "submit_task.js",
    "check_balance.js",
    "list_campaigns.js",
    "list_activities.js",
    "verify_contract.js",
    "get_task_details.js",
}

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic ToolParam format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "read_brand_guidelines",
        "description": (
            "Load the full brand context: guidelines, example posts, and reference materials "
            "(PDFs, docs). Use this as your first step to understand the brand voice, tone, "
            "colors, hashtags, dos/don'ts, and visual style before generating any content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_references",
        "description": (
            "Get a quick inventory of available reference files (PDFs, brand assets, campaign briefs) "
            "without loading their full content. Useful to see what's available before deciding "
            "what to consult."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check_figma_design",
        "description": (
            "Fetch design information from the brand's Figma file. Can retrieve styles, "
            "design tokens (colors, typography), node metadata, or screenshots. "
            "Use this to check official brand colors, typography, and visual references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["styles", "tokens", "metadata", "screenshot"],
                    "description": "What to fetch: 'styles' for published styles, 'tokens' for design tokens from a node, 'metadata' for node structure, 'screenshot' for a rendered image.",
                },
                "node_id": {
                    "type": "string",
                    "description": "Figma node ID (e.g. '0:5'). Optional — defaults to the configured page node.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "generate_image",
        "description": (
            "Generate an image using Replicate with smart model routing. The model is auto-selected "
            "based on content_type: announcements → Nano Banana (text overlays), brand assets → "
            "Recraft SVG, lifestyle/events → Seedream, general → Flux 1.1 Pro. "
            "For BloFin: use black/orange (#FF8800) color scheme, bold futuristic crypto aesthetic, "
            "dark backgrounds, 3D matte black metallic objects with orange glow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed image generation prompt following brand visual guidelines.",
                },
                "content_type": {
                    "type": "string",
                    "enum": ["announcement", "lifestyle", "event", "educational", "brand_asset", "community", "market_commentary", "brand_3d"],
                    "description": "Content type for smart model routing. Determines which image model is used.",
                    "default": "announcement",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "img2img",
        "description": (
            "Generate an image based on an existing reference image and a text prompt using "
            "flux-kontext-pro (img2img). Use this when the user has uploaded a reference photo, "
            "or when generating Finny the mascot (blue round head, orange spacesuit, white helmet "
            "ring, gold B logo, deadpan expression). For Finny requests, reference images are "
            "auto-loaded from brand assets if reference_image_path is not provided."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the desired image output.",
                },
                "reference_image_path": {
                    "type": "string",
                    "description": "Absolute path to the reference image on disk. Leave empty to auto-detect Finny references.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "read_feedback_history",
        "description": (
            "Read the history of approved/rejected drafts and any learned brand preferences. "
            "Use this to understand what the user likes and dislikes before generating content, "
            "so you can avoid repeating past mistakes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "log_resource_usage",
        "description": (
            "Record what resources you consulted during this generation. "
            "Call this near the end to log which files, Figma nodes, scripts, and APIs you used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of resources consulted (e.g. 'guidelines.md, brand PDF, feedback history').",
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "execute_openclaw_script",
        "description": (
            "Execute an OpenClaw onchain script (Node.js). Available scripts: "
            "read_vault.js, create_campaign.js, schedule_content.js, log_activity.js, "
            "browse_tasks.js, claim_task.js, submit_task.js, check_balance.js, "
            "list_campaigns.js, list_activities.js, verify_contract.js, get_task_details.js. "
            "Use these for blockchain operations like logging campaigns, reading the vault, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script_name": {
                    "type": "string",
                    "description": "Name of the script file (e.g. 'browse_tasks.js').",
                },
                "args": {
                    "type": "string",
                    "description": "Space-separated arguments to pass to the script.",
                    "default": "",
                },
            },
            "required": ["script_name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _handle_read_brand_guidelines(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    context = guidelines.get_brand_context()
    tracker.log_file("guidelines.md")
    tracker.log_file("references")
    tracker.log_api("brand_context")
    return context


async def _handle_read_references(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    summary = guidelines.get_reference_summary()
    tracker.log_file("reference_inventory")
    return summary


async def _handle_check_figma_design(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    action = input_dict.get("action", "styles")
    node_id = input_dict.get("node_id")

    if action == "styles":
        result = await figma.get_file_styles()
    elif action == "tokens":
        result = await figma.get_design_tokens(node_id)
    elif action == "metadata":
        result = await figma.get_node_metadata(node_id)
    elif action == "screenshot":
        result = await figma.get_node_screenshot(node_id)
    else:
        result = {"error": f"Unknown action: {action}"}

    tracker.log_figma(node_id or settings.FIGMA_NODE_ID)
    tracker.log_api("figma")
    return json.dumps(result, indent=2)


_REFS_DIR = Path(settings.BRAND_FOLDER) / "references"


async def _handle_generate_image(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    prompt = input_dict.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "No prompt provided"})

    content_type = input_dict.get("content_type", "announcement")

    # 0. brand_3d — dedicated 3D asset pipeline
    # Always: master prompt splice + category refs + optional logo refs
    # LoRA trigger (BLOFIN3D) appended as suffix when available (never prepended)
    if content_type == "brand_3d":
        master_prompt = _state.get_3d_master_prompt()
        lora_path = Path(settings.BRAND_FOLDER) / "loras" / "blofin3d.safetensors"
        lora_ready = lora_path.exists()

        # --- Step 1: Build final prompt (master prompt splice always) ---
        if master_prompt:
            marker = "GENERATION REQUEST"
            idx = master_prompt.find(marker)
            if idx != -1:
                final_prompt = master_prompt[:idx] + f"GENERATION REQUEST\n\n{prompt}"
            else:
                final_prompt = f"{master_prompt}\n\nGENERATION REQUEST\n\n{prompt}"
            logger.info("brand_3d: master prompt spliced")
        else:
            final_prompt = prompt
            logger.warning("brand_3d: no master prompt found — using raw prompt")

        # Append LoRA trigger as SUFFIX (not prefix) to avoid rendering as text
        if lora_ready:
            final_prompt = f"{final_prompt}\n\nBLOFIN3D"
            logger.info("brand_3d: LoRA ready — appended BLOFIN3D trigger as suffix")

        # Enforce pure black background (overrides any ambient light leakage)
        final_prompt += "\n\nCRITICAL: background must be pure #000000 black, no gradients, no warm tones, no brown, no ambient light, no floor reflection, no vignette."

        # --- Step 2: Collect reference images (category + logo) ---
        training_dir = Path(settings.BRAND_FOLDER) / "assets" / "blofin3d_training"
        ref_images = _select_3d_refs(training_dir, prompt)

        # Inject logo refs when prompt mentions logo/B logo/BloFin logo
        if _LOGO_PATTERN.search(prompt):
            logo_dir = training_dir / "logos"
            if logo_dir.is_dir():
                logo_refs = sorted(logo_dir.glob("*.png"))
                if logo_refs:
                    # Add logo refs to end of stack, avoid duplicates
                    existing_paths = {str(p) for p in ref_images}
                    for lr in logo_refs:
                        if str(lr) not in existing_paths:
                            ref_images.append(lr)
                    logger.info("brand_3d: injected %d logo refs (prompt mentions logo)", len(logo_refs))

        # --- Step 3: Generate N=3 options in parallel ---
        _N_OPTIONS = 3

        if ref_images:
            ref_grid = _stitch_grid([str(p) for p in ref_images], max_images=4, label="3d_ref")
            logger.info("brand_3d: stitched %d refs into grid (lora=%s), generating %d options", len(ref_images), lora_ready, _N_OPTIONS)
            tracker.log_api("replicate:flux-kontext-pro (brand_3d + refs x%d)" % _N_OPTIONS)
            results = await asyncio.gather(
                *[image_gen.generate_img2img(final_prompt, ref_grid, strength=0.15) for _ in range(_N_OPTIONS)],
                return_exceptions=True,
            )
            # Clean up stitched temp file
            try:
                Path(ref_grid).unlink(missing_ok=True)
            except Exception:
                pass
            urls = [r for r in results if isinstance(r, str) and r]
            if urls:
                _state.save_last_generated(urls[0], "brand_3d")
                return json.dumps({
                    "image_url": urls[0],
                    "image_urls": urls,
                    "model": "flux-kontext-pro",
                    "reason": "brand_3d with master prompt + refs" + (" + LoRA" if lora_ready else ""),
                    "prompt_used": final_prompt[:500],
                    "lora_ready": lora_ready,
                    "options_generated": len(urls),
                })
            logger.warning("brand_3d img2img failed (all %d options) — falling back to text-to-image", _N_OPTIONS)

        # Text-to-image fallback (no refs available or img2img failed)
        # Use flux-1.1-pro (never nano-banana-pro for brand_3d)
        model_id = "black-forest-labs/flux-1.1-pro"
        tracker.log_api("replicate:flux-1.1-pro (brand_3d fallback x%d)" % _N_OPTIONS)
        results = await asyncio.gather(
            *[image_gen.generate_image(final_prompt, content_type="community") for _ in range(_N_OPTIONS)],
            return_exceptions=True,
        )
        urls = [r for r in results if isinstance(r, str) and r]
        if urls:
            _state.save_last_generated(urls[0], "brand_3d")
            return json.dumps({
                "image_url": urls[0],
                "image_urls": urls,
                "model": model_id,
                "reason": "brand_3d fallback (flux-1.1-pro)" + (" + LoRA" if lora_ready else ""),
                "prompt_used": final_prompt[:500],
                "lora_ready": lora_ready,
                "options_generated": len(urls),
            })
        return json.dumps({"error": "brand_3d image generation failed", "model": model_id, "prompt_used": final_prompt[:500]})

    # 1. Check for active style profile for this content_type
    active_profile = _state.get_active_profile(content_type)
    if active_profile:
        profile_refs = _state.get_profile_refs(active_profile)
        if profile_refs:
            # Stitch up to 3 refs into a grid
            if len(profile_refs) >= 3:
                input_ref = _stitch_grid(profile_refs[:3], label="style")
            else:
                input_ref = profile_refs[-1]  # most recent single ref

            # Get profile-specific settings
            profiles = _state.get_style_profiles()
            profile_data = profiles.get(active_profile, {})
            strength = profile_data.get("strength", 0.3)
            prefix = profile_data.get("prompt_prefix", "")

            if prefix:
                prompt = f"{prefix}, {prompt}"
            prompt += ", visual reference: maintain same composition style, lighting, and layout"

            logger.info(
                "Using style profile '%s' for %s: %d refs, strength=%.2f",
                active_profile, content_type, len(profile_refs), strength,
            )
            tracker.log_api(f"replicate:flux-kontext-pro (style profile: {active_profile})")
            url = await image_gen.generate_img2img(prompt, input_ref, strength=strength)

            # Clean up stitched temp file
            if input_ref.startswith("/tmp/style_stitched_"):
                try:
                    Path(input_ref).unlink(missing_ok=True)
                except Exception:
                    pass

            if url:
                return json.dumps({
                    "image_url": url,
                    "model": "flux-kontext-pro",
                    "reason": f"style profile: {active_profile}",
                    "prompt_used": prompt,
                })
            logger.warning("Style profile img2img failed — falling back")

    # 2. Fallback: check for approved references matching this content_type
    approved_refs = sorted(_REFS_DIR.glob(f"approved_{content_type}_*.png"))
    if approved_refs:
        latest_ref = str(approved_refs[-1])
        prompt = (
            f"{prompt}, visual reference: maintain same composition style, "
            f"lighting, and layout as previous approved posts of this type"
        )
        logger.info("Using approved style ref for %s: %s (strength=0.3)", content_type, latest_ref)
        tracker.log_api("replicate:flux-kontext-pro (style ref)")
        url = await image_gen.generate_img2img(prompt, latest_ref, strength=0.3)
        if url:
            return json.dumps({"image_url": url, "model": "flux-kontext-pro", "reason": "style reference from approved", "prompt_used": prompt})
        logger.warning("img2img style ref failed — falling back to text-to-image")

    # 3. Pure text-to-image generation
    model_id, reason = image_gen.select_model(content_type, prompt)
    tracker.log_api(f"replicate:{model_id.split('/')[-1]}")

    url = await image_gen.generate_image(prompt, content_type=content_type)

    if url:
        return json.dumps({"image_url": url, "model": model_id, "reason": reason, "prompt_used": prompt})
    else:
        return json.dumps({"error": "Image generation failed or REPLICATE_API_TOKEN not set", "model": model_id, "prompt_used": prompt})


_FINNY_ASSETS_DIR = Path(settings.BRAND_FOLDER) / "assets"
_FINNY_IDENTITY = (
    "round blue alien fish head, big black eyes, blue fin antenna, "
    "orange spacesuit with white helmet ring and gold B logo, deadpan grumpy expression"
)


def _stitch_grid(image_paths: list[str], max_images: int = 3, label: str = "ref") -> str:
    """Stitch up to max_images reference images into a horizontal grid.
    Returns path to the stitched image in /tmp."""
    paths = image_paths[:max_images]
    images = [_PILImage.open(p).convert("RGB") for p in paths]

    # Normalize to same height (use the smallest)
    min_h = min(img.height for img in images)
    resized = []
    for img in images:
        w = int(min_h * img.width / img.height)
        resized.append(img.resize((w, min_h), _PILImage.LANCZOS))

    total_w = sum(img.width for img in resized)
    grid = _PILImage.new("RGB", (total_w, min_h))
    x = 0
    for img in resized:
        grid.paste(img, (x, 0))
        x += img.width

    out_path = f"/tmp/{label}_stitched_{int(_time.time())}.jpg"
    grid.save(out_path, "JPEG", quality=95)
    logger.info("Stitched %d %s refs into grid: %s (%dx%d)", len(resized), label, out_path, total_w, min_h)
    return out_path


# ---------------------------------------------------------------------------
# brand_3d smart category routing for reference image selection
# ---------------------------------------------------------------------------

# Keyword → category folder(s) mapping. First match wins.
_3D_CATEGORY_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"server|cube|MCP|circuit|protocol|node|network|tech", re.IGNORECASE), ["platforms_and_bases"]),
    (re.compile(r"coin|USDT|token", re.IGNORECASE), ["coins_and_tokens", "usdt_coin_refs"]),
    (re.compile(r"safe|vault|lock", re.IGNORECASE), ["safes_and_security"]),
    (re.compile(r"gift|box|reward", re.IGNORECASE), ["gift_boxes"]),
    (re.compile(r"container|jar|glass", re.IGNORECASE), ["containers_and_vessels"]),
    (re.compile(r"trophy|scene|rocket", re.IGNORECASE), ["scenes_and_compositions"]),
    (re.compile(r"icon|feature", re.IGNORECASE), ["feature_icons"]),
]

# Logo keyword pattern — checked separately so logo refs are ADDED to the stack
_LOGO_PATTERN = re.compile(r"\blogo\b|B\s*logo|BloFin\s*logo", re.IGNORECASE)


def _select_3d_refs(training_dir: Path, prompt: str, max_refs: int = 3) -> list[Path]:
    """Select up to max_refs reference images from blofin3d_training/ subdirectories.

    Uses keyword matching to route to the best category folders.
    Falls back to pulling 1 image from each of the top 3 most populated categories.
    """
    if not training_dir.is_dir():
        return []

    # Try keyword-based category routing
    for pattern, folders in _3D_CATEGORY_RULES:
        if pattern.search(prompt):
            pool: list[Path] = []
            for folder in folders:
                cat_dir = training_dir / folder
                if cat_dir.is_dir():
                    pool.extend(sorted(cat_dir.glob("*.png")))
            if pool:
                selected = pool[:max_refs]
                logger.info(
                    "brand_3d refs: keyword '%s' → %s (%d refs)",
                    pattern.pattern, folders, len(selected),
                )
                return selected

    # No keyword match — return empty (no random refs)
    logger.info("brand_3d refs: no keyword match — skipping refs")
    return []


def _build_finny_prompt(user_prompt: str) -> str:
    """Rewrite a Finny prompt into the BFL-recommended structure for character consistency."""
    return (
        f"This character — {_FINNY_IDENTITY} — is now {user_prompt}. "
        f"Keep exact character design, same face, same suit colors, same proportions. "
        f"Change the background and scene while keeping the character in the exact same "
        f"position, scale, and pose."
    )


async def _handle_img2img(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    prompt = input_dict.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "No prompt provided"})

    reference_image_path = input_dict.get("reference_image_path") or None

    # Auto-detect Finny references when no explicit path and prompt mentions finny/mascot
    is_finny = re.search(r"finny|mascot", prompt, re.IGNORECASE)
    if reference_image_path is None and is_finny:
        found = []
        for i in range(1, 10):
            p = _FINNY_ASSETS_DIR / f"finny_reference_{i}.png"
            if p.exists():
                found.append(str(p))

        if found:
            # Stitch multiple refs into a grid for Kontext (multiple angles in one image)
            if len(found) >= 3:
                reference_image_path = _stitch_grid(found, max_images=3, label="finny")
            else:
                reference_image_path = found[0]

            prompt = _build_finny_prompt(prompt)
            logger.info("Auto-selected %d Finny reference(s): input=%s", len(found), reference_image_path)
        else:
            logger.warning("Finny prompt but no finny_reference_*.png found - falling back to text-to-image")
            url = await image_gen.generate_image(prompt, content_type="community")
            tracker.log_api("replicate:flux-1.1-pro (finny fallback)")
            if url:
                return json.dumps({"image_url": url, "note": "Finny references not found, used text-to-image fallback"})
            return json.dumps({"error": "Image generation failed"})

    if reference_image_path is None:
        logger.info("img2img called with no reference image and no Finny keyword - falling back to generate_image")
        url = await image_gen.generate_image(prompt, content_type="announcement")
        tracker.log_api("replicate:flux-1.1-pro (no-ref fallback)")
        if url:
            return json.dumps({"image_url": url, "note": "No reference image provided, used text-to-image"})
        return json.dumps({"error": "Image generation failed"})

    tracker.log_api("replicate:flux-kontext-pro")
    url = await image_gen.generate_img2img(prompt, reference_image_path)

    # Clean up stitched temp file
    if reference_image_path.startswith("/tmp/finny_stitched_"):
        try:
            Path(reference_image_path).unlink(missing_ok=True)
        except Exception:
            pass

    if url:
        return json.dumps({"image_url": url, "model": "flux-kontext-pro", "reference": reference_image_path, "prompt_used": prompt})
    return json.dumps({"error": "img2img generation failed", "reference": reference_image_path, "prompt_used": prompt})


async def _handle_read_feedback_history(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    context = feedback.get_feedback_context()
    tracker.log_file("feedback.json")
    return context


async def _handle_log_resource_usage(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    summary = input_dict.get("summary", "")
    logger.info("Agent logged resource usage: %s", summary)
    return f"Resource usage logged: {summary}\nCurrent tracker: {tracker.to_summary()}"


async def _handle_execute_openclaw_script(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    script_name = input_dict.get("script_name", "")
    args = input_dict.get("args", "")

    # Validate against allowlist
    if script_name not in _OPENCLAW_ALLOWLIST:
        return json.dumps({"error": f"Script '{script_name}' not in allowlist. Available: {sorted(_OPENCLAW_ALLOWLIST)}"})

    script_path = Path(settings.OPENCLAW_SCRIPTS_DIR) / script_name
    if not script_path.exists():
        return json.dumps({"error": f"Script not found at {script_path}. Install OpenClaw skills first."})

    # Sanitize args — reject shell metacharacters, use shlex for safe splitting
    _UNSAFE_CHARS = re.compile(r"[;&|`$(){}!<>\\]")
    if args and _UNSAFE_CHARS.search(args):
        return json.dumps({"error": "Arguments contain unsafe characters. Only alphanumeric, hyphens, underscores, dots, and spaces are allowed."})

    cmd = ["node", str(script_path)]
    if args:
        try:
            cmd.extend(shlex.split(args))
        except ValueError as e:
            return json.dumps({"error": f"Invalid arguments: {e}"})

    tracker.log_script(script_name)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(script_path.parent),
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            return json.dumps({"exit_code": result.returncode, "stdout": output, "stderr": error})
        return output if output else "(no output)"

    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Script {script_name} timed out after 60 seconds"})
    except Exception as e:
        return json.dumps({"error": f"Failed to execute {script_name}: {e}"})


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------

_HANDLERS = {
    "read_brand_guidelines": _handle_read_brand_guidelines,
    "read_references": _handle_read_references,
    "check_figma_design": _handle_check_figma_design,
    "generate_image": _handle_generate_image,
    "img2img": _handle_img2img,
    "read_feedback_history": _handle_read_feedback_history,
    "log_resource_usage": _handle_log_resource_usage,
    "execute_openclaw_script": _handle_execute_openclaw_script,
}


async def execute_tool(
    tool_name: str, input_dict: dict, tracker: ResourceTracker
) -> str:
    """
    Execute a tool by name. Returns the tool's string result.
    Raises KeyError if tool_name is not registered.
    """
    handler = _HANDLERS.get(tool_name)
    if not handler:
        raise KeyError(f"Unknown tool: {tool_name}")
    return await handler(input_dict, tracker)
