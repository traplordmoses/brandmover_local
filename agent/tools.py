"""
Tool registry for agent mode.
Defines the 7 tools available to Claude and their async handler functions.
Each handler receives (input_dict, ResourceTracker) and returns a string.
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path

from agent import feedback, figma, guidelines, image_gen
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
                    "enum": ["announcement", "lifestyle", "event", "educational", "brand_asset", "community", "market_commentary"],
                    "description": "Content type for smart model routing. Determines which image model is used.",
                    "default": "announcement",
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


async def _handle_generate_image(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    prompt = input_dict.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "No prompt provided"})

    content_type = input_dict.get("content_type", "announcement")
    model_id, reason = image_gen.select_model(content_type, prompt)
    tracker.log_api(f"replicate:{model_id.split('/')[-1]}")

    url = await image_gen.generate_image(prompt, content_type=content_type)

    if url:
        return json.dumps({"image_url": url, "model": model_id, "reason": reason, "prompt_used": prompt})
    else:
        return json.dumps({"error": "Image generation failed or REPLICATE_API_TOKEN not set", "model": model_id, "prompt_used": prompt})


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

    cmd = ["node", str(script_path)]
    if args:
        cmd.extend(args.split())

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
