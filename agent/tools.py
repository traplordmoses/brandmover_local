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
import subprocess  # nosec B404 — mitigated by _OPENCLAW_ALLOWLIST + shlex + _UNSAFE_CHARS
import tempfile
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from agent import asset_library, content_types, feedback, figma, guidelines, image_gen, lora_pipeline, state as _state
from agent.resource_log import ResourceTracker
from config import settings

logger = logging.getLogger(__name__)

# Regex for detecting unsafe shell metacharacters in user-supplied arguments.
# Shared with agent.onchain to avoid duplication.
UNSAFE_SHELL_CHARS = re.compile(r"[;&|`$(){}!<>\\\n\r\t]")


def _str_param(d: dict, key: str, default: str = "") -> str:
    """Extract a string parameter, coercing non-strings."""
    v = d.get(key, default)
    return str(v) if v is not None else default


# Allowlist of OpenClaw scripts that can be executed.
# Loaded from brand/openclaw_allowlist.txt if present, else uses this default set.
_DEFAULT_OPENCLAW_ALLOWLIST = {
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


def _load_openclaw_allowlist() -> set[str]:
    """Load script allowlist from brand/openclaw_allowlist.txt or use defaults.

    The config file is one script name per line. Blank lines and #comments are ignored.
    """
    config_path = Path(settings.BRAND_FOLDER) / "openclaw_allowlist.txt"
    if not config_path.exists():
        return set(_DEFAULT_OPENCLAW_ALLOWLIST)
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
        names = {
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")
        }
        if names:
            logger.info("Loaded %d OpenClaw scripts from %s", len(names), config_path)
            return names
    except OSError as e:
        logger.warning("Failed to read openclaw_allowlist.txt: %s — using defaults", e)
    return set(_DEFAULT_OPENCLAW_ALLOWLIST)


_OPENCLAW_ALLOWLIST = _load_openclaw_allowlist()

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
            "Use the brand's color scheme and visual style as defined in the guidelines."
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
                    "enum": list(content_types.AGENT_SELECTABLE_TYPES),
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
            "or when generating a brand mascot. For mascot requests, reference images are "
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
                    "description": "Absolute path to the reference image on disk. Leave empty to auto-detect mascot references.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "read_feedback_history",
        "description": (
            "Read learned content preferences distilled from past approvals and rejections. "
            "Returns actionable patterns about what works and what doesn't."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "verify_draft",
        "description": (
            "Check your draft against quality rules BEFORE calling finish. "
            "Returns a 0-100 score with per-dimension feedback. "
            "If score < 75, revise and verify again. If >= 75, proceed to finish. "
            "Always call this before finish to ensure quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "caption": {"type": "string", "description": "Draft caption text."},
                "title": {"type": "string", "description": "Draft title (overlay text)."},
                "subtitle": {"type": "string", "description": "Draft subtitle."},
                "image_prompt": {"type": "string", "description": "Draft image prompt."},
                "content_type": {"type": "string", "description": "Content type."},
            },
            "required": ["caption"],
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
        "name": "think",
        "description": (
            "Use this to think step-by-step, plan your approach, or reason about the request "
            "before taking action. Returns 'ok'. Use this before your first real tool call and "
            "whenever you need to reason."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning, planning, or analysis",
                },
            },
            "required": ["thought"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Call this when your draft is complete. Submit the final content. "
            "Do not output raw JSON in your text response — always submit your final draft through this tool. "
            "For threads, set format to 'thread' and provide thread_posts array."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "caption": {"type": "string", "description": "Post text (or first post for threads)."},
                "hashtags": {"type": "array", "items": {"type": "string"}},
                "alt_text": {"type": "string"},
                "image_prompt": {"type": "string"},
                "content_type": {"type": "string"},
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
                "platform": {"type": "string"},
                "format": {
                    "type": "string",
                    "enum": ["single", "thread", "calendar", "report"],
                    "description": "Output format. Defaults to 'single'. Use 'thread' for multi-post threads.",
                },
                "thread_posts": {
                    "type": "array",
                    "description": "For threads: array of post objects. Each has 'text' (required) and optional 'image_prompt'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "image_prompt": {"type": "string"},
                        },
                        "required": ["text"],
                    },
                },
                "calendar_entries": {
                    "type": "array",
                    "description": "For calendars: array of {date, time, theme, type, topic, status} entries.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "time": {"type": "string"},
                            "theme": {"type": "string"},
                            "type": {"type": "string"},
                            "topic": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                },
                "report_type": {
                    "type": "string",
                    "description": "For reports: 'performance', 'campaign', 'feedback', or 'custom'.",
                },
                "report_sections": {
                    "type": "array",
                    "description": (
                        "For custom reports: sections array. Each: {heading, content, type}. "
                        "type is 'text', 'table', 'stats', or 'rich'. "
                        "For rich: use semantic HTML with content-block, callout, layer-card, "
                        "diagram-container, separator classes. Prefer rich for polished reports."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"},
                            "type": {"type": "string", "enum": ["text", "table", "stats", "rich"]},
                        },
                    },
                },
            },
            "required": ["caption"],
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
    # --- Creative variations ---
    {
        "name": "suggest_variations",
        "description": (
            "After producing a draft, suggest 1-2 alternative creative directions. "
            "Each variation should take a meaningfully different approach — different tone, "
            "hook, or visual concept. Use when the request is ambiguous or when multiple "
            "strong approaches exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "variations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "approach": {"type": "string", "description": "Brief description of the creative direction"},
                            "caption": {"type": "string"},
                            "image_prompt": {"type": "string"},
                            "content_type": {"type": "string"},
                            "title": {"type": "string"},
                            "subtitle": {"type": "string"},
                        },
                        "required": ["approach", "caption", "image_prompt"],
                    },
                },
            },
            "required": ["variations"],
        },
    },
    # --- Skills system --- agent-created persistent capabilities
    {
        "name": "use_skill",
        "description": (
            "Load a saved skill by name. Returns the skill's full instructions (SKILL.md) "
            "and any bundled scripts. Follow the instructions to execute the skill. "
            "Call list_skills first if you're not sure which skill to use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name (e.g. 'meme-generator', 'trending-topics').",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "create_skill",
        "description": (
            "Save a reusable skill for future sessions. Use this after you've solved a novel "
            "problem and want to remember HOW you did it. A skill contains: a SKILL.md with "
            "instructions, and optional scripts the agent can run later. Skills compound — "
            "each one makes you more capable in future sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill identifier (lowercase, hyphens allowed, e.g. 'meme-generator').",
                },
                "description": {
                    "type": "string",
                    "description": "One-line description of what this skill does (shown in skill registry).",
                },
                "skill_md": {
                    "type": "string",
                    "description": (
                        "Full SKILL.md content in markdown. Should include: "
                        "YAML frontmatter (name, description), When to Use, How to Use, "
                        "and any code/commands. Follow the SKILL.md format."
                    ),
                },
                "scripts": {
                    "type": "string",
                    "description": (
                        "Optional JSON object of {filename: code} for reusable scripts. "
                        "E.g. '{\"make_meme.py\": \"from PIL import Image...\"}'. "
                        "Scripts are saved to the skill's scripts/ directory."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "If true, replace an existing skill with the same name.",
                    "default": False,
                },
            },
            "required": ["name", "description", "skill_md"],
        },
    },
    {
        "name": "list_skills",
        "description": (
            "List all available skills with their descriptions. Use this to discover "
            "what capabilities have been saved from previous sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # --- Subagent delegation ---
    {
        "name": "delegate_task",
        "description": (
            "Delegate a research or analysis sub-task to a lightweight sub-agent. "
            "The sub-agent runs with Haiku for speed/cost and has access to brand "
            "guidelines, feedback, and skills. Use for: competitor research, data "
            "gathering, analysis tasks, or any work that can run independently."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What the sub-agent should do. Be specific.",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context to provide (optional).",
                    "default": "",
                },
            },
            "required": ["task"],
        },
    },
    # --- Semantic memory ---
    {
        "name": "search_memory",
        "description": (
            "Search past generations by relevance to find what worked before. "
            "Returns similar past content with approval status. Use this to learn "
            "from past successes and avoid repeating rejected approaches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g., 'partnership announcement', 'meme about staking').",
                },
                "status_filter": {
                    "type": "string",
                    "enum": ["approved", "rejected", "draft"],
                    "description": "Only show entries with this status. Omit for all.",
                },
            },
            "required": ["query"],
        },
    },
    # ── Content repurposing ──
    {
        "name": "repurpose_content",
        "description": (
            "Take a high-performing past post and repurpose it into a different format. "
            "E.g., turn a popular tweet into a thread, an image post into a carousel concept, "
            "or a thread into a summary image post. Use when the user asks to 'repurpose', "
            "'remix', or 'turn X into Y', or when you notice a past post had high engagement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_caption": {
                    "type": "string",
                    "description": "The original post caption to repurpose",
                },
                "source_format": {
                    "type": "string",
                    "enum": ["single", "thread", "carousel"],
                    "description": "Original format",
                },
                "target_format": {
                    "type": "string",
                    "enum": ["single", "thread", "carousel", "quote_card", "infographic_concept"],
                    "description": "Desired output format",
                },
                "angle": {
                    "type": "string",
                    "description": "Optional: specific angle or twist for the repurposed version",
                },
            },
            "required": ["source_caption", "target_format"],
        },
    },
    # ── Trend research ──
    {
        "name": "research_trends",
        "description": (
            "Research trending topics and competitor activity in your brand's niche. "
            "Returns 3-5 actionable content angles based on what's currently resonating. "
            "Use when the user asks about trends, what to post next, or when planning "
            "content strategy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "niche": {
                    "type": "string",
                    "description": "The brand's niche or industry (e.g., 'AI marketing', 'DeFi', 'fitness')",
                },
                "focus": {
                    "type": "string",
                    "enum": ["trending_topics", "competitor_angles", "content_gaps", "all"],
                    "description": "What to research",
                },
            },
            "required": ["niche"],
        },
    },
    # ── Growth threads ──
    {
        "name": "plan_growth_thread",
        "description": (
            "Plan a growth-optimized Twitter thread. Threads are the #1 organic growth "
            "tool on X. This tool structures threads with proven growth hooks: bold opening "
            "claim, evidence/story, twist/insight, and a 'follow for more' CTA. Use when "
            "the user wants to create a thread or when the content would benefit from "
            "multi-post depth."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The thread topic",
                },
                "angle": {
                    "type": "string",
                    "description": "The unique angle or hot take",
                },
                "target_length": {
                    "type": "integer",
                    "description": "Number of posts (5-12 recommended for growth)",
                    "default": 7,
                },
                "include_follow_cta": {
                    "type": "boolean",
                    "description": "Add 'follow for more' in final post",
                    "default": True,
                },
            },
            "required": ["topic"],
        },
    },
    # ── Video promo generation ──
    {
        "name": "generate_promo_video",
        "description": (
            "Generate a short-form branded promo video (Reels/TikTok/Shorts). "
            "Composites an AI-generated background with a glassmorphism text card "
            "featuring typewriter-animated conversation. Outputs a 1080x1920 .mp4."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Bold title text for the card. Use \\n for line breaks, e.g. 'FOID\\nMCP'.",
                },
                "subtitle": {
                    "type": "string",
                    "description": "Smaller subtitle below the title, e.g. '// Example conversation with AI'.",
                },
                "conversation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "description": "Speaker label: 'You', 'AI', etc."},
                            "text": {"type": "string", "description": "Message text. Use \\n for line breaks."},
                        },
                        "required": ["role", "text"],
                    },
                    "description": "The demo conversation to animate with typewriter effect.",
                },
                "background_style": {
                    "type": "string",
                    "enum": ["liquid_metal", "aurora", "particle_field", "smoke", "custom"],
                    "description": "Background animation style. Default: liquid_metal.",
                },
                "background_color": {
                    "type": "string",
                    "description": "Primary color for the background, e.g. 'amber gold', 'electric blue'.",
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "Total video duration in seconds. Default: 15.",
                },
                "output_filename": {
                    "type": "string",
                    "description": "Output filename (saved to state/outputs/), e.g. 'promo_video.mp4'.",
                },
                "fresh_bg": {
                    "type": "boolean",
                    "description": "Force new AI background generation even if a cached background exists for this brand+style.",
                },
            },
            "required": ["title", "conversation"],
        },
    },
    # ── Campaign creation ──
    {
        "name": "create_campaign",
        "description": (
            "Create a multi-day content campaign with individual posts that will be auto-scheduled. "
            "Use this when the user wants a content plan, campaign, or series of posts spread across days."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short campaign name (kebab-case)"},
                "brief": {"type": "string", "description": "Campaign brief/strategy"},
                "posts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "integer", "description": "Day number (1-based)"},
                            "time": {"type": "string", "description": "Time to post, e.g. '11:11am', '3:33pm'"},
                            "caption": {"type": "string", "description": "The exact post copy"},
                            "content_type": {"type": "string"},
                            "image_prompt": {"type": "string", "description": "Image generation prompt, if needed"},
                            "narrative_role": {
                                "type": "string",
                                "enum": ["hook", "buildup", "climax", "resolution", "cta"],
                            },
                            "emotional_tone": {
                                "type": "string",
                                "enum": ["curiosity", "excitement", "urgency", "trust", "celebration"],
                            },
                        },
                        "required": ["day", "time", "caption"],
                    },
                },
            },
            "required": ["name", "posts"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _handle_read_brand_guidelines(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Brand guidelines are pre-loaded in the system prompt — return a pointer."""
    tracker.log_file("brand/guidelines.md (pre-loaded)")
    return json.dumps({
        "status": "Brand guidelines are already loaded in your system prompt above. "
                  "You have full access to the brand voice, visual style, and content rules. "
                  "Proceed with generation using the context you already have."
    })


async def _handle_read_references(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    summary = await asyncio.to_thread(guidelines.get_reference_summary)
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
    prompt = _str_param(input_dict, "prompt")
    if not prompt:
        return json.dumps({"error": "No prompt provided"})

    content_type = input_dict.get("content_type", "announcement")

    # Check visual source preference from config.json
    from agent.compositor_config import _load_config_json
    _vs_config = _load_config_json()
    _visual_source = (_vs_config.get("visual_source", {}).get("primary", "ai_generated")
                      if _vs_config else "ai_generated")

    # Check asset library for a reusable match before generating
    # (skip for ai_generated mode — always generate fresh)
    if _visual_source in ("client_assets", "hybrid"):
        existing = asset_library.suggest(prompt, content_type)
        if existing:
            lib_path = asset_library.get_library_path(existing)
            if lib_path:
                asset_library.mark_used(existing.id)
                logger.info("Reusing library asset %s (%s mode)", existing.id, _visual_source)
                return json.dumps({
                    "image_url": str(lib_path),
                    "model": "library",
                    "reason": f"reused library asset {existing.id} ({_visual_source} mode)",
                    "prompt_used": prompt,
                    "library_entry_id": existing.id,
                })

    # 0. brand_3d — dedicated 3D asset pipeline
    # Always: master prompt splice + category refs + optional logo refs
    # LoRA trigger (BRAND3D) appended as suffix when available (never prepended)
    if content_type == "brand_3d":
        master_prompt = _state.get_3d_master_prompt()
        active_lora = lora_pipeline.get_active_lora()
        lora_ready = active_lora is not None

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
            trigger = active_lora.get("trigger_word", "BRAND3D")
            final_prompt = f"{final_prompt}\n\n{trigger}"
            logger.info("brand_3d: LoRA ready — appended %s trigger as suffix", trigger)

        # Enforce pure black background (overrides any ambient light leakage)
        final_prompt += "\n\nCRITICAL: background must be pure #000000 black, no gradients, no warm tones, no brown, no ambient light, no floor reflection, no vignette."

        # --- Step 2: Collect reference images (category + logo) ---
        training_dir = Path(settings.BRAND_FOLDER) / "assets" / "brand3d_training"
        ref_images = await asyncio.to_thread(_select_3d_refs, training_dir, prompt)

        # Inject logo refs when prompt mentions logo
        _logo_contrast_temps: list[str] = []  # track contrast files for cleanup
        if _LOGO_PATTERN.search(prompt):
            logo_dir = training_dir / "logos"
            if logo_dir.is_dir():
                logo_refs = sorted(logo_dir.glob("*.png"))
                if logo_refs:
                    existing_paths = {str(p) for p in ref_images}
                    for lr in logo_refs:
                        usable, tmp = _prepare_logo_ref(lr)
                        if tmp:
                            _logo_contrast_temps.append(tmp)
                        if str(usable) not in existing_paths:
                            ref_images.append(usable)
                            existing_paths.add(str(usable))
                    logger.info("brand_3d: injected %d logo refs (prompt mentions logo, %d contrast-boosted)", len(logo_refs), len(_logo_contrast_temps))

        # Inject MCP logo refs as safety net (always when folder has files)
        mcp_refs_dir = training_dir / "mcp_refs"
        if mcp_refs_dir.is_dir():
            mcp_ref_files = sorted(mcp_refs_dir.glob("*.png"))
            if mcp_ref_files:
                existing_paths = {str(p) for p in ref_images}
                for mr in mcp_ref_files:
                    if str(mr) not in existing_paths:
                        ref_images.append(mr)
                logger.info("brand_3d: injected %d MCP refs", len(mcp_ref_files))

        # --- Step 3: Generate N=3 options with staggered calls ---
        _N_OPTIONS = 3
        _STAGGER_DELAY = 1.5  # seconds between calls to avoid 429s

        ref_grid = None
        try:
            if ref_images:
                ref_grid = await asyncio.to_thread(_stitch_grid, [str(p) for p in ref_images], 4, "3d_ref")
                logger.info("brand_3d: stitched %d refs into grid (lora=%s), generating %d options (staggered %.1fs)", len(ref_images), lora_ready, _N_OPTIONS, _STAGGER_DELAY)
                tracker.log_api("replicate:flux-kontext-pro (brand_3d + refs x%d)" % _N_OPTIONS)
                results = await _staggered_generate(
                    [lambda p=final_prompt, g=ref_grid: image_gen.generate_img2img(p, g, strength=0.15) for _ in range(_N_OPTIONS)],
                    delay=_STAGGER_DELAY,
                )
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
            results = await _staggered_generate(
                [lambda p=final_prompt: image_gen.generate_image(p, content_type="community") for _ in range(_N_OPTIONS)],
                delay=_STAGGER_DELAY,
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
        finally:
            # Clean up stitched temp file + contrast logo temps even on exception
            if ref_grid:
                try:
                    Path(ref_grid).unlink(missing_ok=True)
                except OSError as e:
                    logger.debug("Temp cleanup failed for %s: %s", ref_grid, e)
            for _tmp in _logo_contrast_temps:
                try:
                    Path(_tmp).unlink(missing_ok=True)
                except OSError as e:
                    logger.debug("Logo temp cleanup failed for %s: %s", _tmp, e)

    # 1. Check for active style profile for this content_type
    active_profile = _state.get_active_profile(content_type)
    if active_profile:
        profile_refs = _state.get_profile_refs(active_profile)
        if profile_refs:
            # Stitch up to 3 refs into a grid
            if len(profile_refs) >= 3:
                input_ref = await asyncio.to_thread(_stitch_grid, profile_refs[:3], 3, "style")
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
            if input_ref.startswith(tempfile.gettempdir()):
                try:
                    Path(input_ref).unlink(missing_ok=True)
                except OSError as e:
                    logger.debug("Style ref cleanup failed for %s: %s", input_ref, e)

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

    # Query template image region for optimal aspect ratio
    from agent import template_memory as _tm
    region_aspect = _tm.get_image_region_aspect_ratio(content_type)

    url = await image_gen.generate_image(prompt, content_type=content_type, aspect_ratio=region_aspect)

    if url:
        try:
            asset_library.add(url, "generated", content_type, prompt=prompt)
        except OSError as e:
            logger.debug("Asset library add failed: %s", e)
        return json.dumps({"image_url": url, "model": model_id, "reason": reason, "prompt_used": prompt})
    else:
        return json.dumps({"error": "Image generation failed or REPLICATE_API_TOKEN not set", "model": model_id, "prompt_used": prompt})


_MASCOT_ASSETS_DIR = Path(settings.BRAND_FOLDER) / "assets"


def _prepare_logo_ref(logo_path: Path) -> tuple[Path, str | None]:
    """Check if a logo image is too dark for the model to see and create a
    contrast-boosted version in /tmp/ if needed.

    Returns (usable_path, tmp_path_or_None).
    tmp_path is set only when a contrast file was created (caller must clean up).
    """
    try:
        from PIL import Image as _PILImage, ImageOps as _PILImageOps

        img = _PILImage.open(logo_path)

        # If the image has transparency, flatten onto a white background
        if img.mode in ("RGBA", "LA", "PA"):
            bg = _PILImage.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])  # use alpha as mask
            tmp_fd = tempfile.NamedTemporaryFile(suffix=".png", prefix="logo_contrast_", delete=False)
            tmp_name = tmp_fd.name
            tmp_fd.close()
            bg.save(tmp_name, "PNG")
            logger.info("brand_3d logo: flattened %s onto white background", logo_path.name)
            return Path(tmp_name), tmp_name

        # Check mean brightness — if predominantly dark, invert
        rgb = img.convert("RGB")
        import numpy as _np
        mean_val = _np.array(rgb).mean()
        if mean_val < 30:
            inverted = _PILImageOps.invert(rgb)
            tmp_fd = tempfile.NamedTemporaryFile(suffix=".png", prefix="logo_contrast_", delete=False)
            tmp_name = tmp_fd.name
            tmp_fd.close()
            inverted.save(tmp_name, "PNG")
            logger.info("brand_3d logo: inverted %s (mean=%.1f → too dark)", logo_path.name, mean_val)
            return Path(tmp_name), tmp_name

        # Logo is fine as-is
        return logo_path, None
    except (OSError, ValueError, ImportError) as e:
        logger.warning("brand_3d logo: failed to preprocess %s: %s", logo_path.name, e)
        return logo_path, None


async def _staggered_generate(
    callables: list,
    delay: float = 1.5,
    max_retries: int = 2,
) -> list:
    """Run generation callables with staggered start times and retry on 429.

    Each callable is fired with `delay` seconds between launches, then all
    are awaited concurrently.  If a call fails with a 429, it retries up to
    `max_retries` times with exponential backoff (delay * 2^attempt).
    Returns a list of results (str URLs or Exceptions).
    """

    async def _run_with_retry(fn, index: int):
        # Stagger: wait index * delay before starting
        if index > 0:
            await asyncio.sleep(index * delay)
        for attempt in range(max_retries + 1):
            try:
                result = await fn()
                if result:
                    return result
                return None
            except (httpx.HTTPStatusError, httpx.TimeoutException, OSError, ValueError, RuntimeError) as e:
                is_429 = "429" in str(e)
                if is_429 and attempt < max_retries:
                    backoff = delay * (2 ** (attempt + 1))
                    logger.info("brand_3d option %d: 429 rate-limited, retrying in %.1fs (attempt %d/%d)", index + 1, backoff, attempt + 1, max_retries)
                    await asyncio.sleep(backoff)
                    continue
                return e

    tasks = [_run_with_retry(fn, i) for i, fn in enumerate(callables)]
    return await asyncio.gather(*tasks)


def _stitch_grid(image_paths: list[str], max_images: int = 3, label: str = "ref") -> str:
    """Stitch up to max_images reference images into a horizontal grid.
    Returns path to the stitched image in /tmp."""
    from PIL import Image as _PILImage

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

    tmp_fd = tempfile.NamedTemporaryFile(suffix=".jpg", prefix=f"{label}_stitched_", delete=False)
    out_path = tmp_fd.name
    tmp_fd.close()
    grid.save(out_path, "JPEG", quality=95)
    logger.info("Stitched %d %s refs into grid: %s (%dx%d)", len(resized), label, out_path, total_w, min_h)
    return out_path


# ---------------------------------------------------------------------------
# brand_3d smart category routing for reference image selection
# ---------------------------------------------------------------------------

# Auto-discovered at runtime from subdirectory names in brand3d_training/.
# Each subdirectory name is matched against the prompt as a keyword.
# e.g. a folder named "coins_and_tokens" matches prompts containing "coin" or "token".

# Logo keyword pattern — checked separately so logo refs are ADDED to the stack
_LOGO_PATTERN = re.compile(r"\blogo\b|brand\s*logo", re.IGNORECASE)


def _select_3d_refs(training_dir: Path, prompt: str, max_refs: int = 3) -> list[Path]:
    """Select up to max_refs reference images from brand3d_training/ subdirectories.

    Auto-discovers category folders and matches subdirectory name keywords against
    the prompt. e.g. a folder named "coins_and_tokens" matches "coin" or "token".
    """
    if not training_dir.is_dir():
        return []

    resolved_root = training_dir.resolve()
    prompt_lower = prompt.lower()

    # Auto-discover subdirectories and match keywords from folder names
    for cat_dir in sorted(training_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        # Path containment check — prevent symlink escapes
        if not cat_dir.resolve().is_relative_to(resolved_root):
            logger.warning("Skipping symlink outside training_dir: %s", cat_dir)
            continue
        # Split folder name into keywords (e.g. "coins_and_tokens" → ["coins", "tokens"])
        keywords = [w for w in cat_dir.name.lower().replace("_", " ").split() if len(w) > 2 and w != "and"]
        if any(kw in prompt_lower for kw in keywords):
            pool = [
                p for p in sorted(cat_dir.glob("*.png"))[:50]
                if p.resolve().is_relative_to(resolved_root)
            ]
            if pool:
                selected = pool[:max_refs]
                logger.info(
                    "brand_3d refs: folder '%s' matched (%d refs)",
                    cat_dir.name, len(selected),
                )
                return selected

    # No keyword match — return empty (no random refs)
    logger.info("brand_3d refs: no keyword match — skipping refs")
    return []


def _build_mascot_prompt(user_prompt: str) -> str:
    """Rewrite a mascot prompt into the BFL-recommended structure for character consistency."""
    return (
        f"This character is now {user_prompt}. "
        f"Keep exact character design, same face, same colors, same proportions. "
        f"Change the background and scene while keeping the character in the exact same "
        f"position, scale, and pose."
    )


async def _handle_img2img(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    prompt = _str_param(input_dict, "prompt")
    if not prompt:
        return json.dumps({"error": "No prompt provided"})

    reference_image_path = input_dict.get("reference_image_path") or None

    # Auto-detect mascot references when no explicit path and prompt mentions mascot
    is_mascot = re.search(r"mascot|character", prompt, re.IGNORECASE)
    if reference_image_path is None and is_mascot:
        found = []
        for i in range(1, 10):
            p = _MASCOT_ASSETS_DIR / f"mascot_reference_{i}.png"
            if p.exists():
                found.append(str(p))

        if found:
            # Stitch multiple refs into a grid for Kontext (multiple angles in one image)
            if len(found) >= 3:
                reference_image_path = await asyncio.to_thread(_stitch_grid, found, 3, "mascot")
            else:
                reference_image_path = found[0]

            prompt = _build_mascot_prompt(prompt)
            logger.info("Auto-selected %d mascot reference(s): input=%s", len(found), reference_image_path)
        else:
            logger.warning("Mascot prompt but no mascot_reference_*.png found - falling back to text-to-image")
            url = await image_gen.generate_image(prompt, content_type="community")
            tracker.log_api("replicate:flux-1.1-pro (mascot fallback)")
            if url:
                return json.dumps({"image_url": url, "note": "Mascot references not found, used text-to-image fallback"})
            return json.dumps({"error": "Image generation failed"})

    if reference_image_path is None:
        logger.info("img2img called with no reference image and no mascot keyword - falling back to generate_image")
        url = await image_gen.generate_image(prompt, content_type="announcement")
        tracker.log_api("replicate:flux-1.1-pro (no-ref fallback)")
        if url:
            return json.dumps({"image_url": url, "note": "No reference image provided, used text-to-image"})
        return json.dumps({"error": "Image generation failed"})

    tracker.log_api("replicate:flux-kontext-pro")
    url = await image_gen.generate_img2img(prompt, reference_image_path)

    # Clean up stitched temp file
    if reference_image_path.startswith(tempfile.gettempdir()):
        try:
            Path(reference_image_path).unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Mascot ref cleanup failed for %s: %s", reference_image_path, e)

    if url:
        return json.dumps({"image_url": url, "model": "flux-kontext-pro", "reference": reference_image_path, "prompt_used": prompt})
    return json.dumps({"error": "img2img generation failed", "reference": reference_image_path, "prompt_used": prompt})


async def _handle_read_feedback_history(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    from agent.session import load_session
    session = load_session()
    tracker.log_file("agent_session.json")

    if not session.learned_preferences:
        return json.dumps({
            "preferences": [],
            "message": "No learned preferences yet. Preferences are auto-extracted from approval/rejection patterns.",
        })

    return json.dumps({
        "preferences": session.learned_preferences,
        "count": len(session.learned_preferences),
        "message": "These are distilled preferences learned from past approvals and rejections.",
    })


async def _handle_log_resource_usage(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    summary = input_dict.get("summary", "")
    logger.info("Agent logged resource usage: %s", summary)
    return f"Resource usage logged: {summary}\nCurrent tracker: {tracker.to_summary()}"


async def _handle_think(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    logger.info("Agent thinking: %s", _str_param(input_dict, "thought")[:200])
    return "ok"


async def _handle_finish(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    # Coerce caption to string to guard against non-string tool_input values
    if "caption" in input_dict:
        input_dict["caption"] = _str_param(input_dict, "caption")
    return json.dumps({"status": "complete", "draft": input_dict})


async def _handle_execute_openclaw_script(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    from agent.audit_log import audit
    script_name = input_dict.get("script_name", "")
    args = input_dict.get("args", "")
    audit("execute_openclaw_script", script=script_name, args=args[:200])

    # Validate against allowlist
    if script_name not in _OPENCLAW_ALLOWLIST:
        return json.dumps({"error": f"Script '{script_name}' not in allowlist. Available: {sorted(_OPENCLAW_ALLOWLIST)}"})

    script_path = Path(settings.OPENCLAW_SCRIPTS_DIR) / script_name
    if not script_path.exists():
        return json.dumps({"error": f"Script '{script_name}' not found. Install OpenClaw skills first."})

    # Sanitize args — allowlist approach: only permit safe characters
    if args and not re.match(r'^[a-zA-Z0-9\-_.,:\s/=@"\']+$', args):
        return json.dumps({"error": "Arguments contain unsafe characters. Only alphanumeric, hyphens, underscores, dots, colons, commas, and spaces are allowed."})

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
    except (subprocess.SubprocessError, OSError) as e:
        return json.dumps({"error": f"Failed to execute {script_name}: {e}"})


# ---------------------------------------------------------------------------
# Skill handlers
# ---------------------------------------------------------------------------

async def _handle_use_skill(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Load a skill's full content and return it as context for the agent."""
    from agent.skills import load_skill
    name = input_dict.get("name", "")
    if not name:
        return json.dumps({"error": "No skill name provided."})

    skill = load_skill(name)
    if not skill:
        return json.dumps({"error": f"Skill '{name}' not found. Use list_skills to see available skills."})

    tracker.log_skill(name)

    result = {"name": skill["name"], "instructions": skill["content"]}
    if skill["scripts"]:
        result["scripts"] = skill["scripts"]
    if skill["references"]:
        result["references"] = skill["references"]
    return json.dumps(result, indent=2)


async def _handle_create_skill(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Create a new persistent skill."""
    from agent.skills import create_skill
    name = input_dict.get("name", "")
    description = input_dict.get("description", "")
    skill_md = input_dict.get("skill_md", "")
    overwrite = input_dict.get("overwrite", False)

    if not name or not description or not skill_md:
        return json.dumps({"error": "name, description, and skill_md are all required."})

    # Parse scripts JSON string if provided
    scripts = None
    scripts_raw = input_dict.get("scripts")
    if scripts_raw:
        try:
            scripts = json.loads(scripts_raw) if isinstance(scripts_raw, str) else scripts_raw
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid scripts JSON: {e}"})

    result = create_skill(
        name=name,
        description=description,
        skill_md=skill_md,
        scripts=scripts,
        overwrite=overwrite,
    )
    tracker.log_api(f"create_skill:{name}")
    return json.dumps(result)


async def _handle_list_skills(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """List all registered skills."""
    from agent.skills import load_registry
    skills = load_registry()
    if not skills:
        return json.dumps({
            "skills": [],
            "message": "No skills created yet. Use create_skill to save reusable capabilities.",
        })
    return json.dumps({"skills": skills, "count": len(skills)}, indent=2)


# ---------------------------------------------------------------------------
# Subagent + memory handlers
# ---------------------------------------------------------------------------

async def _handle_delegate_task(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Spawn a sub-agent for a focused research/analysis task."""
    from agent.subagent import delegate_task
    task = input_dict.get("task", "")
    context = input_dict.get("context", "")
    if not task:
        return json.dumps({"error": "No task provided."})

    tracker.log_api("subagent:delegate")
    result = await delegate_task(task=task, context=context, tracker=tracker)
    return json.dumps(result, indent=2)


async def _handle_research_trends(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Research trending topics via a sub-agent."""
    from agent.subagent import delegate_task

    niche = input_dict.get("niche", "")
    focus = input_dict.get("focus", "all")
    if not niche:
        return json.dumps({"error": "No niche provided."})

    focus_instructions = {
        "trending_topics": "Focus on what topics are trending right now in this space. What are people talking about?",
        "competitor_angles": "Focus on what competitors are doing well. What content angles are getting engagement?",
        "content_gaps": "Focus on content gaps -- what topics are underserved? Where is there opportunity?",
        "all": "Cover trending topics, competitor angles, AND content gaps.",
    }

    task = (
        f"Research trending topics and content opportunities in the '{niche}' niche.\n\n"
        f"{focus_instructions.get(focus, focus_instructions['all'])}\n\n"
        f"Return exactly 3-5 actionable content angles. For each angle provide:\n"
        f"1. TOPIC: A specific topic or hook\n"
        f"2. WHY NOW: Why this is timely or resonating\n"
        f"3. CONTENT IDEA: A concrete post idea the brand could create\n"
        f"4. FORMAT: Best format (single post, thread, image, video)\n\n"
        f"Be specific and actionable. No generic advice."
    )

    tracker.log_api("subagent:research_trends")
    result = await delegate_task(task=task, context=f"Brand niche: {niche}", tracker=tracker)
    return json.dumps(result, indent=2)


async def _handle_search_memory(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Search past generations by relevance."""
    from agent.memory import search_past_generations
    query = input_dict.get("query", "")
    if not query:
        return json.dumps({"error": "No query provided."})

    status_filter = input_dict.get("status_filter")
    results = search_past_generations(query, top_k=5, status_filter=status_filter)
    tracker.log_api("memory:search")

    if not results:
        return json.dumps({"results": [], "message": "No relevant past generations found."})
    return json.dumps({"results": results, "count": len(results)}, indent=2)


async def _handle_generate_promo_video(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Generate a short-form branded promo video with AI background + text card."""
    from modules.video_promo import (
        generate_promo_video,
        VideoPromoConfig,
        TextCardConfig,
        BackgroundConfig,
        BrandOverlay,
        ConversationLine,
    )
    from agent import compositor_config

    title = _str_param(input_dict, "title")
    if not title:
        return json.dumps({"error": "title is required"})

    conversation_raw = input_dict.get("conversation", [])
    if not conversation_raw:
        return json.dumps({"error": "conversation is required (list of {role, text})"})

    # Build conversation lines
    conversation = []
    for line in conversation_raw:
        if isinstance(line, dict) and "role" in line and "text" in line:
            conversation.append(ConversationLine(role=line["role"], text=line["text"]))
        else:
            return json.dumps({"error": f"Invalid conversation line: {line}"})

    # Pull brand config for defaults
    try:
        cfg = compositor_config.get_config()
    except (OSError, KeyError, ValueError):
        cfg = None

    # Resolve paths
    brand_folder = Path(settings.BRAND_FOLDER)
    output_dir = Path(settings.STATE_FOLDER) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = _str_param(input_dict, "output_filename", "promo_video.mp4")
    if not output_filename.endswith(".mp4"):
        output_filename += ".mp4"
    output_path = str(output_dir / output_filename)

    # Resolve fonts
    font_bold = str(brand_folder / "assets" / "fonts" / "Inter-Bold.ttf")
    font_regular = str(brand_folder / "assets" / "fonts" / "Inter-Regular.ttf")

    # Build brand overlay if logo exists
    logo_path = brand_folder / "assets" / "logo.png"
    brand_overlay = None
    if logo_path.exists():
        brand_overlay = BrandOverlay(logo_path=str(logo_path))

    # Build config
    bg_style = _str_param(input_dict, "background_style", "liquid_metal")
    bg_color = _str_param(input_dict, "background_color", "amber gold")
    duration = input_dict.get("duration_seconds", 15.0)
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 15.0

    brand_name = settings.BRAND_NAME
    fresh_bg = bool(input_dict.get("fresh_bg", False))

    config = VideoPromoConfig(
        output_path=output_path,
        total_duration_seconds=duration,
        brand_name=brand_name,
        text_card=TextCardConfig(
            title=title,
            subtitle=input_dict.get("subtitle"),
            conversation=conversation,
        ),
        background=BackgroundConfig(
            style=bg_style,
            primary_color=bg_color,
            mood=getattr(cfg, "mood_keywords", None) or "cinematic, dark, luxurious",
        ),
        brand=brand_overlay,
        font_bold=font_bold,
        font_regular=font_regular,
    )

    tracker.log_api("generate_promo_video")

    # Check if using cached background
    from modules.video_promo.background_gen import find_cached_background, BackgroundStyle as _BgStyle
    try:
        style_enum = _BgStyle(bg_style)
        cached = None if fresh_bg else find_cached_background(brand_name, style_enum)
    except ValueError:
        cached = None
    if cached:
        logger.info("Generating promo video (cached bg): %s", output_path)
    else:
        logger.info("Generating promo video (fresh bg via %s): %s",
                     config.background.provider, output_path)

    try:
        final_path = await generate_promo_video(config, fresh_bg=fresh_bg)
        result = {
            "status": "complete",
            "video_path": final_path,
            "duration": duration,
            "resolution": f"{config.width}x{config.height}",
            "background_cached": cached is not None,
        }
        if fresh_bg:
            result["fresh_bg"] = True
        return json.dumps(result)
    except Exception as e:  # Intentional broad catch — video pipeline spans FFmpeg, PIL, and external APIs
        logger.exception("Promo video generation failed")
        return json.dumps({"error": f"Video generation failed: {type(e).__name__}: {str(e)[:300]}"})


async def _handle_create_campaign(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Create a multi-day campaign and immediately schedule all posts."""
    from agent import campaigns
    from agent.scheduling import schedule_queue

    name = _str_param(input_dict, "name")
    if not name:
        return json.dumps({"error": "Campaign name is required."})

    brief = _str_param(input_dict, "brief", "")
    posts = input_dict.get("posts", [])
    if not posts:
        return json.dumps({"error": "At least one post is required."})

    # Determine local timezone for converting day/time to UTC timestamps
    local_tz = schedule_queue._get_local_tz()
    today = datetime.now(local_tz).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today.strftime("%Y-%m-%d")

    # Build campaign slots and schedule each post immediately
    slots: list[dict] = []
    schedule_results: list[dict] = []
    errors: list[str] = []

    for i, post in enumerate(posts):
        day = post.get("day", i + 1)
        time_str = post.get("time", "9:00am")
        caption = post.get("caption", "")
        content_type = post.get("content_type", "")
        image_prompt = post.get("image_prompt", "")
        narrative_role = post.get("narrative_role", "")
        emotional_tone = post.get("emotional_tone", "")

        # Parse the time string to get a UTC timestamp
        # Calculate the target date first
        target_date = today + timedelta(days=day - 1)
        # Combine date with the parsed time
        ts, label_str = schedule_queue.parse_time(time_str, now=target_date)
        if ts is None:
            # Fallback: try parsing as HH:MMam/pm manually
            try:
                time_str_clean = time_str.strip().lower()
                parsed_dt = None
                for fmt in ("%I:%M%p", "%I:%M %p", "%H:%M"):
                    try:
                        parsed_dt = datetime.strptime(time_str_clean, fmt)
                        break
                    except ValueError:
                        continue
                if parsed_dt:
                    target_dt = target_date.replace(
                        hour=parsed_dt.hour,
                        minute=parsed_dt.minute,
                        second=0,
                        microsecond=0,
                    )
                    ts = target_dt.timestamp()
                else:
                    errors.append(f"Post {i+1} (day {day}): could not parse time '{time_str}', defaulting to 9:00am")
                    target_dt = target_date.replace(hour=9, minute=0, second=0, microsecond=0)
                    ts = target_dt.timestamp()
            except Exception:
                errors.append(f"Post {i+1} (day {day}): could not parse time '{time_str}', defaulting to 9:00am")
                target_dt = target_date.replace(hour=9, minute=0, second=0, microsecond=0)
                ts = target_dt.timestamp()

        # Build the prompt for the scheduler
        narrative_ctx = ""
        if narrative_role or emotional_tone:
            parts = []
            if narrative_role:
                parts.append(f"Role: {narrative_role}")
            if emotional_tone:
                parts.append(f"Tone: {emotional_tone}")
            narrative_ctx = f"\nNarrative: {' | '.join(parts)}"

        full_prompt = f"[CAMPAIGN: {name}]\n"
        if brief:
            full_prompt += f"Campaign brief: {brief}\n"
        full_prompt += f"Post this exact copy (do not change the wording):\n\n{caption}"
        if image_prompt:
            full_prompt += f"\n\nImage prompt: {image_prompt}"
        if narrative_ctx:
            full_prompt += narrative_ctx

        queue_label = f"{name} D{day}/{time_str}"
        # Build pre-approved draft so scheduler posts directly
        draft_dict = {"caption": caption, "_campaign_scheduled": True}
        if content_type:
            draft_dict["content_type"] = content_type
        if image_prompt:
            draft_dict["image_prompt"] = image_prompt

        item = schedule_queue.add_scheduled(
            prompt=full_prompt,
            scheduled_utc=ts,
            label=queue_label,
            draft=draft_dict,
        )

        slot = {
            "day": day,
            "slot_label": time_str,
            "copy": caption,
            "content_type": content_type,
            "narrative_role": narrative_role,
            "emotional_tone": emotional_tone,
            "media_note": image_prompt,
            "status": "scheduled" if item else "pending",
            "schedule_queue_id": item["id"] if item else "",
        }
        slots.append(slot)

        if item:
            target_dt_display = datetime.fromtimestamp(ts, tz=local_tz)
            schedule_results.append({
                "day": day,
                "time": time_str,
                "queue_id": item["id"],
                "scheduled_for": target_dt_display.strftime("%Y-%m-%d %I:%M %p %Z"),
                "caption_preview": caption[:60],
            })
        else:
            errors.append(f"Post {i+1} (day {day}): duplicate detected, skipped")

    # Create the campaign record
    result = campaigns.create_campaign(
        name=name,
        brief=brief,
        slots=slots,
        start_date=start_date,
    )

    if not result.get("success"):
        return json.dumps(result)

    tracker.log_api(f"create_campaign:{name}")

    # Build summary table
    table_lines = ["| Day | Time | Caption | Status |", "|-----|------|---------|--------|"]
    for sr in schedule_results:
        cap = sr["caption_preview"]
        table_lines.append(f"| {sr['day']} | {sr['time']} | {cap}... | Queued ({sr['queue_id']}) |")

    summary = {
        "status": "campaign_created",
        "campaign_name": name,
        "posts_scheduled": len(schedule_results),
        "posts_total": len(posts),
        "schedule": schedule_results,
        "schedule_table": "\n".join(table_lines),
        "errors": errors if errors else [],
        "message": (
            f"Campaign '{name}' created with {len(schedule_results)}/{len(posts)} posts "
            f"immediately queued for auto-posting. No /approve needed. "
            f"The scheduler will fire each post at the scheduled time."
        ),
    }
    return json.dumps(summary, indent=2)


async def _handle_suggest_variations(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Store creative variations suggested by the agent."""
    variations = input_dict.get("variations", [])
    if not variations:
        return json.dumps({"error": "No variations provided."})

    stored = []
    for v in variations:
        approach = v.get("approach", "")
        caption = v.get("caption", "")
        if not approach or not caption:
            continue
        stored.append({
            "approach": approach,
            "caption": caption,
            "image_prompt": v.get("image_prompt", ""),
            "content_type": v.get("content_type", ""),
            "title": v.get("title", ""),
            "subtitle": v.get("subtitle", ""),
        })

    if not stored:
        return json.dumps({"error": "No valid variations (each needs approach + caption)."})

    tracker.log_api("suggest_variations")
    return json.dumps({
        "status": "variations_stored",
        "count": len(stored),
        "variations": stored,
    })


async def _handle_repurpose_content(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Repurpose an existing post into a different format using Claude Haiku."""
    from agent.model_fallback import call_with_fallback

    source_caption = _str_param(input_dict, "source_caption")
    if not source_caption:
        return json.dumps({"error": "No source_caption provided"})

    target_format = _str_param(input_dict, "target_format")
    if not target_format:
        return json.dumps({"error": "No target_format provided"})

    source_format = _str_param(input_dict, "source_format", "single")
    angle = _str_param(input_dict, "angle")

    format_instructions = {
        "single": "a single concise post (280 chars max for Twitter)",
        "thread": "a multi-post thread (3-7 posts, each under 280 chars). Return as a JSON array of strings.",
        "carousel": "a carousel concept with 3-8 slides. Return as a JSON array of {slide_number, heading, body, visual_note} objects.",
        "quote_card": "a quote card with a punchy pull-quote (under 100 chars) and optional attribution. Return as {quote, attribution, context}.",
        "infographic_concept": "an infographic concept with a title, 3-5 key data points or steps, and a visual description. Return as {title, points: [{label, detail}], visual_description}.",
    }

    format_desc = format_instructions.get(target_format, f"a {target_format} format post")

    system_prompt = (
        "You are a content repurposing specialist. Take an existing post and "
        "transform it into a different format while preserving the core message "
        "and brand voice. Return ONLY valid JSON, no markdown fences, no explanation."
    )

    user_msg = (
        f"Original post ({source_format} format):\n"
        f"---\n{source_caption}\n---\n\n"
        f"Repurpose this into: {format_desc}\n"
    )
    if angle:
        user_msg += f"Angle/twist: {angle}\n"
    user_msg += (
        "\nReturn a JSON object with:\n"
        '- "format": the target format name\n'
        '- "content": the repurposed content (structure depends on format)\n'
        '- "rationale": one sentence explaining what you changed and why\n'
    )

    try:
        response = await call_with_fallback(
            messages=[{"role": "user", "content": user_msg}],
            system=system_prompt,
            max_tokens=2048,
            primary_model="claude-haiku-4-5-20251001",
        )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        result = json.loads(text)
        tracker.log_api("repurpose_content")
        return json.dumps(result)

    except json.JSONDecodeError:
        # Return the raw text if it's not valid JSON
        tracker.log_api("repurpose_content")
        return json.dumps({
            "format": target_format,
            "content": text,
            "rationale": "Raw response (could not parse as JSON)",
        })
    except Exception as e:
        logger.warning("Repurpose content failed: %s", e)
        return json.dumps({"error": f"Repurpose failed: {e}"})


async def _handle_verify_draft(input_dict: dict, tracker: ResourceTracker) -> str:
    """Run quality scoring + brand alignment on a draft before submission."""
    from agent.scoring import score_draft
    from agent.brand_alignment import score_brand_alignment
    from agent.self_review import draft_quality_gate

    draft = {
        "caption": input_dict.get("caption", ""),
        "title": input_dict.get("title", ""),
        "subtitle": input_dict.get("subtitle", ""),
        "image_prompt": input_dict.get("image_prompt", ""),
        "content_type": input_dict.get("content_type", ""),
    }

    # Run all three quality systems
    quality_score = score_draft(draft)
    gate = draft_quality_gate(draft)

    try:
        brand_score = score_brand_alignment(draft)
    except (OSError, KeyError, TypeError, ValueError):
        brand_score = {"alignment_score": -1, "drift_flags": [], "checks": []}

    # Build actionable feedback
    issues = []
    for r in quality_score.get("results", []):
        if r["score"] < 0.7:
            issues.append(f"- {r['name']}: {r['detail']} (score: {r['score']:.0%})")

    for check in gate.get("checks", []):
        if not check.get("passed"):
            issues.append(f"- HARD RULE FAIL: {check['rule']} — {check.get('detail', '')}")

    if brand_score.get("alignment_score", 100) >= 0:
        for flag in brand_score.get("drift_flags", []):
            issues.append(f"- Brand drift: {flag}")

    total_score = quality_score["total_score"]
    grade = quality_score["grade"]
    passed = total_score >= 75 and gate.get("passed", False)

    result = {
        "score": round(total_score),
        "grade": grade,
        "passed": passed,
        "verdict": "READY — proceed to finish" if passed else "NEEDS WORK — revise and verify again",
        "issues": issues if issues else ["No issues found."],
        "auto_fixed": gate.get("auto_fixed", []),
    }
    return json.dumps(result)


async def _handle_plan_growth_thread(
    input_dict: dict, tracker: ResourceTracker
) -> str:
    """Plan a growth-optimized Twitter thread using Haiku for speed."""
    from agent._client import get_anthropic

    topic = _str_param(input_dict, "topic")
    if not topic:
        return json.dumps({"error": "No topic provided."})

    angle = _str_param(input_dict, "angle", "")
    target_length = input_dict.get("target_length", 7)
    include_follow_cta = input_dict.get("include_follow_cta", True)

    # Clamp target_length to 3-15
    target_length = max(3, min(15, target_length))

    # Load brand name from guidelines for the CTA
    brand_name = settings.BRAND_NAME if hasattr(settings, "BRAND_NAME") else ""
    brand_handle = f"@{brand_name}" if brand_name else "us"

    angle_line = f"Angle/hot take: {angle}\n" if angle else ""
    if include_follow_cta:
        cta_instruction = (
            f'"Follow {brand_handle} for more on {topic}" + optional link.'
        )
    else:
        cta_instruction = "Strong closing statement summarizing the thread's value."

    planning_prompt = (
        f"Plan a growth-optimized Twitter/X thread on the topic: {topic}\n"
        f"{angle_line}"
        f"Target length: {target_length} posts\n\n"
        f"Follow the proven growth thread formula:\n"
        f"- Post 1 (HOOK): Bold claim, surprising stat, or provocative question. "
        f"Must stop the scroll. Under 280 chars.\n"
        f"- Post 2 (BRIDGE): 'Here's why...' or 'Let me explain...' — transition "
        f"that promises value.\n"
        f"- Posts 3 to {target_length - 2} (BODY): Evidence, stories, data points. "
        f"Each post must be standalone-valuable. Each under 280 chars.\n"
        f"- Post {target_length - 1} (PAYOFF): The twist or unexpected insight that "
        f"rewards reading the whole thread.\n"
        f"- Post {target_length} (CTA): {cta_instruction}\n\n"
        f"Return valid JSON with this exact structure:\n"
        f'{{"thread_plan": {{"topic": str, "hook_type": str, "estimated_reach_multiplier": str, '
        f'"posts": [{{"post_number": int, "role": str, "text": str, "tip": str}}]}}}}\n\n'
        f"Roles are: hook, bridge, body, payoff, cta\n"
        f"Tips are brief notes on WHY this post works for growth."
    )

    client = get_anthropic()
    tracker.log_api("anthropic:haiku (growth thread planner)")

    try:
        response = await client.messages.create(
            model=settings.HAIKU_MODEL,
            max_tokens=2000,
            system="You are a Twitter growth strategist. Plan viral threads that maximize follower growth. Always return valid JSON.",
            messages=[{"role": "user", "content": planning_prompt}],
        )

        text = response.content[0].text if response.content else ""

        # Try to parse JSON from the response
        try:
            # Strip markdown fences if present
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            # Return the raw text if JSON parsing fails
            plan = {"thread_plan": {"raw_text": text, "topic": topic}}

        return json.dumps(plan, indent=2)
    except Exception as e:
        logger.error("Growth thread planning failed: %s", e)
        return json.dumps({"error": f"Thread planning failed: {str(e)[:200]}"})


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
    "think": _handle_think,
    "finish": _handle_finish,
    "execute_openclaw_script": _handle_execute_openclaw_script,
    "use_skill": _handle_use_skill,
    "create_skill": _handle_create_skill,
    "list_skills": _handle_list_skills,
    "delegate_task": _handle_delegate_task,
    "research_trends": _handle_research_trends,
    "search_memory": _handle_search_memory,
    "generate_promo_video": _handle_generate_promo_video,
    "repurpose_content": _handle_repurpose_content,
    "verify_draft": _handle_verify_draft,
    "suggest_variations": _handle_suggest_variations,
    "plan_growth_thread": _handle_plan_growth_thread,
    "create_campaign": _handle_create_campaign,
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def strip_json_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON content.

    Handles patterns like ```json\\n{...}\\n``` and ```\\n{...}\\n```.
    Returns the unwrapped text, or the original text if no fences found.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def tool_description(tool_name: str, tool_input: dict) -> str:
    """Brief human-readable description of a tool call.

    Used for the progress callback (on_tool_call) so the Telegram UI can show
    the user what's happening: "Generating brand image...", "Fetching URL...", etc.
    Each description is a short present-tense phrase with the most relevant parameter.
    """
    descs = {
        # Base tools (from agent/tools.py)
        "read_brand_guidelines": "Loading brand guidelines and references...",
        "read_references": "Checking available reference materials...",
        "check_figma_design": f"Checking Figma design ({tool_input.get('action', 'styles')})...",
        "generate_image": "Generating brand image...",
        "read_feedback_history": "Reviewing feedback history...",
        "log_resource_usage": "Logging resources used...",
        "img2img": f"Generating image from reference: {tool_input.get('reference_image_path', 'auto')}...",
        "execute_openclaw_script": f"Running {tool_input.get('script_name', 'script')}...",
        # Unified-only tools (from agent/unified_tools.py)
        "get_pending_draft": "Checking pending draft...",
        "revise_draft": f"Revising draft: {tool_input.get('feedback', '?')[:60]}...",
        "check_auto_post_status": "Checking auto-post schedule...",
        "web_fetch": f"Fetching {tool_input.get('url', 'URL')[:60]}...",
        "save_session_plan": "Saving content plan...",
        "get_session_plan": "Checking session plan...",
        "update_plan_item": f"Updating plan item #{tool_input.get('item_id', '?')}...",
        "execute_code": f"Running script: {tool_input.get('description', 'computation')}...",
        "register_draft": f"Registering {Path(tool_input.get('image_path', '')).name if tool_input.get('image_path') else '?'} as draft...",
        "send_file": f"Sending file: {Path(tool_input.get('file_path', '')).name if tool_input.get('file_path') else '?'}...",
        "read_state_file": f"Reading {tool_input.get('file_path', 'file')}...",
        "run_self_review": "Analyzing performance and updating preferences...",
        "start_autonomous_plan": "Working through plan autonomously...",
        "show_queued_draft": f"Loading draft #{tool_input.get('item_id', '?')} for review...",
        "approve_draft": "Approving draft...",
        "post_approved": "Posting approved draft to X...",
        "schedule_post": f"Scheduling post for {tool_input.get('time_description', '?')}...",
        "list_scheduled_posts": "Checking scheduled posts...",
        "cancel_scheduled_post": f"Cancelling scheduled post {tool_input.get('item_id', '?')}...",
        # New tools (screenshot, image editing, notes, git, channel, snippets)
        "take_screenshot": f"Capturing screenshot of {tool_input.get('url', 'page')[:50]}...",
        "edit_image": f"Editing image: {len(tool_input.get('operations', []))} operation(s)...",
        "save_note": f"Saving note: {tool_input.get('key', '?')}...",
        "get_notes": f"Retrieving note(s){': ' + tool_input.get('key', '') if tool_input.get('key') else ''}...",
        "git_info": f"Git {tool_input.get('action', 'info')}...",
        "read_telegram_channel": "Reading channel messages...",
        "save_snippet": f"Saving snippet: {tool_input.get('label', '?')[:40]}...",
        "list_snippets": "Listing saved snippets...",
        "use_snippet": f"Loading snippet {tool_input.get('id', '?')}...",
        # Generic
        "think": "Reasoning...",
        "finish": "Submitting final draft...",
        "research_trends": f"Researching trends in {tool_input.get('niche', 'your niche')}...",
        "generate_promo_video": f"Generating promo video: {tool_input.get('title', '?')[:40]}...",
        "suggest_variations": "Suggesting creative variations...",
        "plan_growth_thread": f"Planning growth thread on {tool_input.get('topic', 'topic')}...",
        "create_campaign": f"Creating campaign '{tool_input.get('name', '?')}' with {len(tool_input.get('posts', []))} posts...",
    }
    return descs.get(tool_name, f"Executing {tool_name}...")


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
