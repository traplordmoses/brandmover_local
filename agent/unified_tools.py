"""
Extended tool registry for unified brain.

ARCHITECTURE:
This is the TOOL LAYER — it defines everything the agent can DO. Each tool has:
1. A definition dict (JSON schema for Claude's tool-use API)
2. An async handler function that actually executes the action

The module follows a TWO-TIER DISPATCH pattern:
- _UNIFIED_HANDLERS: New tools added specifically for the unified brain
- _base_execute_tool: Falls through to the original 8 tools from tools.py

This layered approach means we can add new tools without touching the original
tool registry, and the unified brain gets ALL tools (old + new).

TOOL CATEGORIES:
- Draft management: get_pending_draft, revise_draft, approve_draft
- Publishing: post_approved, schedule_post, list_scheduled, cancel_scheduled
- Content creation: execute_code, register_draft
- Research: web_fetch, read_state_file, take_screenshot
- Image editing: edit_image (Pillow-based operations)
- Memory: save_note, get_notes, save_snippet, list_snippets, use_snippet
- Dev tools: git_info, read_telegram_channel
- Planning: save_session_plan, get_session_plan, update_plan_item

SECURITY NOTES:
- take_screenshot: URL scheme validation (http/https only) prevents SSRF
- git_info: Args sanitized with strict regex to prevent command injection
- edit_image: Output paths contained to state/outputs/ directory
- execute_code: Full system access BY DESIGN (the agent needs it to be powerful)

INTERVIEW TALKING POINT:
"Tools are declarative JSON schemas that Claude discovers through its tool-use API.
Adding a new capability is: define the schema, write an async handler, register it.
The agent automatically learns to use new tools from their descriptions — no
prompt engineering needed for individual tools. We use a two-tier dispatch so
new tools don't affect the battle-tested original tool registry."
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from agent import auto_state, publisher, schedule_queue, session_plan, state, web_fetch
from agent.resource_log import ResourceTracker
from agent.tools import TOOL_DEFINITIONS as _BASE_TOOL_DEFINITIONS
from agent.tools import execute_tool as _base_execute_tool

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS_DIR = _PROJECT_ROOT / "state" / "outputs"


# New tool definitions

_get_pending_draft_def = {
    "name": "get_pending_draft",
    "description": (
        "Check if there's a pending draft awaiting approval. Returns the draft details "
        "(caption, image_prompt, content_type) or indicates no draft is pending. "
        "Use this to make informed revision decisions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_revise_draft_def = {
    "name": "revise_draft",
    "description": (
        "Clear the current pending draft and return its details so you can generate "
        "a revised version. Use when the user gives conversational feedback on a pending "
        "draft (e.g. 'change the image', 'make it shorter', 'use my photo'). After calling "
        "this, generate a revised version and output a new JSON draft block."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "Brief summary of what the user wants changed.",
            },
        },
        "required": ["feedback"],
    },
}

_check_auto_post_status_def = {
    "name": "check_auto_post_status",
    "description": (
        "Check the auto-posting schedule status: how many posts today, when the last "
        "post was, whether auto-posting is paused. Use this to answer questions about "
        "the posting schedule."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


_web_fetch_def = {
    "name": "web_fetch",
    "description": (
        "Fetch and read the content of a web page, tweet, or article. Use when the "
        "user shares a URL or asks you to look at something online. Returns the page "
        "title, metadata, and text content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters of page content to return. Default 15000.",
            },
        },
        "required": ["url"],
    },
}


_save_session_plan_def = {
    "name": "save_session_plan",
    "description": (
        "Save a content plan for the current session. Use when you and the operator "
        "have discussed a multi-post strategy and agreed on a plan."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_name": {
                "type": "string",
                "description": "Short name for this plan (e.g. 'Product launch campaign').",
            },
            "items": {
                "type": "array",
                "description": "List of planned content pieces.",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "What this content piece is about.",
                        },
                        "tone": {
                            "type": "string",
                            "description": "Desired tone/angle (e.g. 'understated, deadpan').",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes or constraints.",
                        },
                    },
                    "required": ["description"],
                },
            },
        },
        "required": ["plan_name", "items"],
    },
}

_get_session_plan_def = {
    "name": "get_session_plan",
    "description": (
        "Get the current session plan status. Shows what content is planned, "
        "completed, and next."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_update_plan_item_def = {
    "name": "update_plan_item",
    "description": (
        "Update a plan item's status or notes. Use after a draft is approved/rejected "
        "to track progress, or to skip an item."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "integer",
                "description": "The plan item ID to update.",
            },
            "status": {
                "type": "string",
                "description": "New status: pending, generating, review, approved, rejected, skipped.",
            },
            "notes": {
                "type": "string",
                "description": "Optional notes to attach to this item.",
            },
        },
        "required": ["item_id"],
    },
}


_execute_code_def = {
    "name": "execute_code",
    "description": (
        "Execute a Python script for data processing, image creation, web scraping, "
        "or any computation. Full access to Pillow (PIL), httpx, numpy, and all "
        "installed packages. Can download images from URLs, create graphics with "
        "brand fonts, and generate sophisticated visual content. "
        "Timeout: 60 seconds. Write outputs to state/outputs/."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what the script does (for logging).",
            },
        },
        "required": ["code", "description"],
    },
}

_send_file_def = {
    "name": "send_file",
    "description": (
        "Send a file to the user in Telegram. Use after generating a report, "
        "document, image, or any file the user should receive."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to send (usually in state/outputs/).",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption to send with the file.",
            },
        },
        "required": ["file_path"],
    },
}

_read_state_file_def = {
    "name": "read_state_file",
    "description": (
        "Read any file from the state/ or brand/ directory. Use when you need "
        "raw data for analysis — feedback history, generation logs, schedule "
        "state, brand guidelines, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path relative to project root (e.g. 'state/feedback.json', 'brand/guidelines.md').",
            },
        },
        "required": ["file_path"],
    },
}


_run_self_review_def = {
    "name": "run_self_review",
    "description": (
        "Analyze your own recent performance — approval rates, rejection patterns, "
        "friction points — and update your learned preferences. Use when the operator "
        "asks how you've been performing or asks you to improve."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


_start_autonomous_plan_def = {
    "name": "start_autonomous_plan",
    "description": (
        "Start working through the session plan autonomously. You'll generate "
        "all remaining plan items in sequence, queuing each draft for later review. "
        "Use when the operator says to cook everything, work through the plan, "
        "or that they'll review later."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_show_queued_draft_def = {
    "name": "show_queued_draft",
    "description": (
        "Show a specific queued draft from an autonomous plan run. "
        "Use when the operator asks to see draft #N or review a specific item."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "integer",
                "description": "The plan item ID to show the draft for.",
            },
        },
        "required": ["item_id"],
    },
}


_approve_draft_def = {
    "name": "approve_draft",
    "description": (
        "Approve the current pending draft. Moves it from 'pending' to 'approved' state. "
        "Use this when the user expresses approval of a draft (e.g. 'approve', 'i approve', "
        "'looks good', 'love it', 'perfect'). After approving, ask if they want to post now "
        "or schedule for later."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_post_approved_def = {
    "name": "post_approved",
    "description": (
        "Post the currently approved draft to X/Twitter immediately. "
        "Only works if there is an approved draft (user said 'yes'/'approve' first). "
        "Use this when the user says 'post it', 'send it', 'publish', etc. after approving."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_schedule_post_def = {
    "name": "schedule_post",
    "description": (
        "Schedule a post for a future time. If an approved draft exists, schedules that "
        "draft for direct posting (no regeneration). Otherwise, takes a prompt to generate "
        "content at the scheduled time. Supports natural language times: '3pm', 'tomorrow 9am', "
        "'in 2 hours', 'friday 3:30pm'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "time_description": {
                "type": "string",
                "description": "Natural language time expression, e.g. '3pm', 'tomorrow 9am', 'in 2 hours'.",
            },
            "prompt": {
                "type": "string",
                "description": "Content generation prompt (only needed if no approved draft exists).",
            },
        },
        "required": ["time_description"],
    },
}

_list_scheduled_posts_def = {
    "name": "list_scheduled_posts",
    "description": (
        "List all pending scheduled posts with their IDs, times, and labels. "
        "Use when the user asks what's scheduled or wants to check the queue."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_cancel_scheduled_post_def = {
    "name": "cancel_scheduled_post",
    "description": (
        "Cancel a scheduled post by its ID. Use when the user wants to remove "
        "a scheduled post from the queue."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "The scheduled post ID to cancel.",
            },
        },
        "required": ["item_id"],
    },
}

_create_campaign_def = {
    "name": "create_campaign",
    "description": (
        "Create a multi-day content campaign and schedule all posts. "
        "Use when the user describes a campaign plan with multiple posts across days. "
        "Each slot can have pre-written copy (posted as-is) or a generation prompt. "
        "After creation, all posts are automatically scheduled into the queue."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short campaign name, e.g. 'welcome-to-foid'.",
            },
            "brief": {
                "type": "string",
                "description": "Campaign theme and objective.",
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format. Defaults to today.",
            },
            "post_times": {
                "type": "object",
                "description": (
                    "Mapping of slot labels to local times in HH:MM format. "
                    "e.g. {\"morning\": \"09:00\", \"evening\": \"18:00\"}. "
                    "Each slot's slot_label determines which time it uses."
                ),
            },
            "slots": {
                "type": "array",
                "description": "List of post slots.",
                "items": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "integer", "description": "Day number (1-indexed)."},
                        "slot_label": {
                            "type": "string",
                            "description": "Time slot label: 'morning', 'evening', etc.",
                        },
                        "copy": {
                            "type": "string",
                            "description": "Pre-written post copy. If provided, posted exactly as-is.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Generation prompt (used if no copy provided).",
                        },
                        "angle": {
                            "type": "string",
                            "description": "Brief description of this post's angle/topic.",
                        },
                        "content_type": {
                            "type": "string",
                            "description": "Content type: engagement, announcement, educational, etc.",
                        },
                        "media_note": {
                            "type": "string",
                            "description": "Note about required media, e.g. '[screenshot of swipe UI]'.",
                        },
                    },
                    "required": ["day"],
                },
            },
        },
        "required": ["name", "brief", "slots"],
    },
}

_campaign_status_def = {
    "name": "campaign_status",
    "description": (
        "Get the status of a campaign or list all campaigns. "
        "Use when the user asks about campaign progress."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Campaign name. If omitted, lists all campaigns.",
            },
        },
        "required": [],
    },
}

_record_walkthrough_def = {
    "name": "record_walkthrough",
    "description": (
        "Record a video walkthrough or screenshot sequence of a website. "
        "Uses Playwright to automate browser actions (goto, click, scroll, wait) "
        "and records the session as MP4 video with text overlays, or as a series "
        "of screenshots. Use for: product demos, feature walkthroughs, onboarding "
        "recordings, capturing UI states. Returns path to the output file(s)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL to record, e.g. 'https://foidfun.vercel.app'.",
            },
            "steps": {
                "type": "array",
                "description": "Sequence of browser actions to perform.",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["goto", "click", "fill", "wait", "screenshot", "scroll"],
                            "description": "Action type.",
                        },
                        "target": {
                            "type": "string",
                            "description": "URL path (for goto) or CSS selector (for click/fill).",
                        },
                        "value": {
                            "type": "string",
                            "description": "Text to type (for fill action).",
                        },
                        "narration": {
                            "type": "string",
                            "description": "Caption text overlay for this step.",
                        },
                        "wait": {
                            "type": "number",
                            "description": "Seconds to wait after action (default 2.0).",
                        },
                    },
                    "required": ["action"],
                },
            },
            "mode": {
                "type": "string",
                "enum": ["video", "screenshot"],
                "description": "Recording mode. 'video' produces MP4, 'screenshot' produces PNGs. Default: video.",
            },
            "name": {
                "type": "string",
                "description": "Short name for the recording (used in filenames).",
            },
        },
        "required": ["url", "steps"],
    },
}

_style_video_def = {
    "name": "style_video",
    "description": (
        "Apply polished social media styling to a raw screen recording or video. "
        "Adds: iPhone device mockup frame, holographic gradient background, "
        "Ken Burns zoom animation, and optional text overlays. "
        "Outputs an H.264 MP4 optimized for X/Twitter. "
        "Use after record_walkthrough to make the output look professional, "
        "or on any raw video/recording that needs a polished look."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "input_video": {
                "type": "string",
                "description": "Path to the raw video file (WebM or MP4).",
            },
            "device_frame": {
                "type": "string",
                "enum": ["iphone", "none"],
                "description": "Device frame to wrap the video in. Default: iphone.",
            },
            "bg_colors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hex colors for gradient background. Default: holographic cyan/pink/purple.",
            },
            "zoom_enabled": {
                "type": "boolean",
                "description": "Enable Ken Burns zoom effect. Default: true.",
            },
            "square_output": {
                "type": "boolean",
                "description": "Output as 1080x1080 square (best for X engagement). Default: true.",
            },
            "narration_texts": {
                "type": "array",
                "description": "Text overlays with timing.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "start": {"type": "number", "description": "Start time in seconds."},
                        "end": {"type": "number", "description": "End time in seconds."},
                    },
                    "required": ["text", "start", "end"],
                },
            },
        },
        "required": ["input_video"],
    },
}

_register_draft_def = {
    "name": "register_draft",
    "description": (
        "Register a file produced by execute_code (or any local file in state/outputs/) "
        "as a pending draft so it enters the normal approve/schedule/post pipeline. "
        "Use after execute_code creates a meme, graphic, or any visual content that "
        "the user should be able to approve and post."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "caption": {
                "type": "string",
                "description": "Post caption text (can be empty for image-only posts like memes).",
            },
            "image_path": {
                "type": "string",
                "description": "Path to the image file, usually in state/outputs/.",
            },
            "content_type": {
                "type": "string",
                "description": "Content type: meme, announcement, brand_3d, etc. Defaults to 'meme'.",
            },
            "alt_text": {
                "type": "string",
                "description": "Accessible image description.",
            },
            "title": {
                "type": "string",
                "description": "Optional title for compositor overlay.",
            },
            "subtitle": {
                "type": "string",
                "description": "Optional subtitle for compositor overlay.",
            },
        },
        "required": ["image_path"],
    },
}

_take_screenshot_def = {
    "name": "take_screenshot",
    "description": (
        "Take a screenshot of a web page. Use for checking websites, capturing tweets, "
        "visual reference, content creation, verifying posts went live."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to screenshot.",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture the full scrollable page (default: false).",
            },
            "width": {
                "type": "integer",
                "description": "Viewport width in pixels (default: 1280).",
            },
            "height": {
                "type": "integer",
                "description": "Viewport height in pixels (default: 720).",
            },
        },
        "required": ["url"],
    },
}

_edit_image_def = {
    "name": "edit_image",
    "description": (
        "Perform common image operations using Pillow. Use for: adding text overlay to photos, "
        "resizing, cropping, compositing images, adding borders, watermarks, or brand elements. "
        "Faster than writing a full execute_code script for simple image edits."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_path": {
                "type": "string",
                "description": "Path to the source image file.",
            },
            "operations": {
                "type": "array",
                "description": (
                    "Array of operations to apply in order. Each has a 'type' field: "
                    "'text_overlay' (text, position top/bottom/center, font_size, color, stroke_color), "
                    "'resize' (width, height), 'crop' (left, top, right, bottom), "
                    "'composite' (overlay_path, x, y, opacity), 'border' (width, color)."
                ),
                "items": {"type": "object"},
            },
            "output_path": {
                "type": "string",
                "description": "Output file path (default: state/outputs/edited_{timestamp}.png).",
            },
        },
        "required": ["source_path", "operations"],
    },
}

_save_note_def = {
    "name": "save_note",
    "description": (
        "Save a persistent note that survives across conversations. Use for: remembering "
        "operator preferences, key dates (launch date, deadlines), recurring instructions, "
        "brand-specific context that isn't in guidelines, or anything the operator says "
        "'remember this'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short identifier like 'launch_date' or 'annio_preferences'.",
            },
            "content": {
                "type": "string",
                "description": "The note content to save.",
            },
        },
        "required": ["key", "content"],
    },
}

_get_notes_def = {
    "name": "get_notes",
    "description": (
        "Retrieve saved notes. Call with no key to get all notes, or with a specific key."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Optional specific note key to retrieve.",
            },
        },
        "required": [],
    },
}

_git_info_def = {
    "name": "git_info",
    "description": (
        "Read git repository information. Use for: checking recent changes, understanding "
        "what was modified, reading diffs, seeing commit history. Helpful for self-awareness "
        "of your own codebase evolution."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["log", "diff", "show", "status"],
                "description": "Git action: log, diff, show, or status.",
            },
            "args": {
                "type": "string",
                "description": "Optional args: commit count for log, ref for diff/show.",
            },
        },
        "required": ["action"],
    },
}

_read_telegram_channel_def = {
    "name": "read_telegram_channel",
    "description": (
        "Read recent messages from a Telegram channel or group. Use for: checking community "
        "sentiment, seeing what people are talking about, finding content ideas from "
        "community discussions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "Channel/group ID (defaults to configured channel).",
            },
            "limit": {
                "type": "integer",
                "description": "Number of messages to retrieve (default: 20, max: 50).",
            },
        },
        "required": [],
    },
}

_save_snippet_def = {
    "name": "save_snippet",
    "description": (
        "Save a piece of content (caption, idea, analysis, draft text) for later use. "
        "Use when the conversation produces something worth keeping — a good caption, "
        "a content angle, research findings, or anything the operator might want to "
        "reference later."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Short description of the snippet.",
            },
            "content": {
                "type": "string",
                "description": "The content to save.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for filtering (e.g. ['meme', 'caption']).",
            },
        },
        "required": ["label", "content"],
    },
}

_list_snippets_def = {
    "name": "list_snippets",
    "description": "List saved snippets, optionally filtered by tag.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tag": {
                "type": "string",
                "description": "Optional tag to filter by.",
            },
            "limit": {
                "type": "integer",
                "description": "Max snippets to return (default: 10).",
            },
        },
        "required": [],
    },
}

_use_snippet_def = {
    "name": "use_snippet",
    "description": "Retrieve a specific snippet by ID for use in content generation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The snippet ID to retrieve.",
            },
        },
        "required": ["id"],
    },
}

UNIFIED_TOOL_DEFINITIONS = _BASE_TOOL_DEFINITIONS + [
    _get_pending_draft_def,
    _revise_draft_def,
    _approve_draft_def,
    _check_auto_post_status_def,
    _web_fetch_def,
    _save_session_plan_def,
    _get_session_plan_def,
    _update_plan_item_def,
    _execute_code_def,
    _register_draft_def,
    _send_file_def,
    _read_state_file_def,
    _run_self_review_def,
    _start_autonomous_plan_def,
    _show_queued_draft_def,
    _post_approved_def,
    _schedule_post_def,
    _list_scheduled_posts_def,
    _cancel_scheduled_post_def,
    _take_screenshot_def,
    _edit_image_def,
    _save_note_def,
    _get_notes_def,
    _git_info_def,
    _read_telegram_channel_def,
    _save_snippet_def,
    _list_snippets_def,
    _use_snippet_def,
    _create_campaign_def,
    _campaign_status_def,
    _record_walkthrough_def,
    _style_video_def,
]


# New tool handlers

async def _handle_get_pending_draft(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    pending = state.get_pending(user_id=user_id)
    if not pending:
        return json.dumps({"status": "no_pending_draft"})
    return json.dumps({
        "status": "pending",
        "caption": pending.get("caption", ""),
        "content_type": pending.get("content_type", "unknown"),
        "image_prompt": pending.get("image_prompt", ""),
        "original_request": pending.get("original_request", ""),
        "revision": state.get_draft_revision_count(user_id=user_id),
        "has_image": bool(pending.get("image_url")),
    })


async def _handle_revise_draft(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    pending = state.get_pending(user_id=user_id)
    if not pending:
        return json.dumps({"error": "No pending draft to revise"})

    old_draft = {
        "caption": pending.get("caption", ""),
        "image_prompt": pending.get("image_prompt", ""),
        "content_type": pending.get("content_type", "unknown"),
        "original_request": pending.get("original_request", ""),
        "alt_text": pending.get("alt_text", ""),
        "has_image": bool(pending.get("image_url")),
        "revision": state.get_draft_revision_count(user_id=user_id),
    }

    feedback_text = input_dict.get("feedback", "")

    # Log rejection to feedback history
    from agent import feedback as _fb
    try:
        await _fb.async_log_feedback(
            request=pending.get("original_request", ""),
            draft=pending, accepted=False,
            feedback_text=feedback_text,
            resources_used=pending.get("resources_used", []),
        )
    except Exception as e:
        logger.warning("Failed to log revision feedback: %s", e)

    state.clear_pending(user_id=user_id)

    return json.dumps({
        "status": "draft_cleared",
        "previous_draft": old_draft,
        "feedback": feedback_text,
        "message": "Previous draft cleared. Generate a revised version addressing the feedback.",
    })


async def _handle_check_auto_post_status(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    summary = auto_state.get_status_summary()
    return json.dumps(summary)


async def _handle_web_fetch(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    url = input_dict.get("url", "")
    if not url:
        return json.dumps({"error": "No URL provided"})
    max_chars = input_dict.get("max_chars", 15000)
    tracker.log_api(f"web_fetch:{url[:60]}")
    return await web_fetch.fetch_url(url, max_chars=max_chars)


async def _handle_save_session_plan(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    plan_name = input_dict.get("plan_name", "Untitled plan")
    items = input_dict.get("items", [])
    if not items:
        return json.dumps({"error": "No items provided"})
    plan = session_plan.save_plan(plan_name, items)
    return json.dumps(plan)


async def _handle_get_session_plan(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    plan = session_plan.get_plan()
    if not plan:
        return json.dumps({"status": "no_active_plan"})
    return json.dumps(plan)


async def _handle_update_plan_item(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    item_id = input_dict.get("item_id")
    if item_id is None:
        return json.dumps({"error": "item_id is required"})
    status = input_dict.get("status")
    notes = input_dict.get("notes")
    item = session_plan.update_item(int(item_id), status=status, notes=notes)
    if item is None:
        return json.dumps({"error": f"Item #{item_id} not found or plan expired"})
    plan = session_plan.get_plan()
    return json.dumps({"updated_item": item, "current_item": plan.get("current_item") if plan else None})


async def _handle_execute_code(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    code = input_dict.get("code", "")
    description = input_dict.get("description", "script")
    if not code.strip():
        return json.dumps({"error": "No code provided"})

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot output files before execution
    before = set(_OUTPUTS_DIR.iterdir()) if _OUTPUTS_DIR.exists() else set()

    # Write code to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(_PROJECT_ROOT), delete=False,
    ) as f:
        f.write(code)
        script_path = f.name

    logger.info("execute_code: running '%s' (%d chars)", description, len(code))
    tracker.log_api(f"execute_code:{description[:40]}")

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(_PROJECT_ROOT),
            env={
                **{k: v for k, v in os.environ.items()
                   if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))},
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        stdout = proc.stdout[:10000] if proc.stdout else ""
        stderr = proc.stderr[:5000] if proc.stderr else ""
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "Script timed out after 60 seconds"
    except Exception as e:
        stdout = ""
        stderr = f"Execution failed: {e}"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    # Detect new files in state/outputs/
    after = set(_OUTPUTS_DIR.iterdir()) if _OUTPUTS_DIR.exists() else set()
    new_files = sorted(str(p) for p in (after - before))

    result = {"stdout": stdout, "stderr": stderr, "output_files": new_files}
    if stderr and not stdout:
        logger.warning("execute_code '%s' had errors: %s", description, stderr[:200])
    else:
        logger.info("execute_code '%s' completed, %d output files", description, len(new_files))

    return json.dumps(result)


async def _handle_register_draft(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    image_path = input_dict.get("image_path", "")
    if not image_path:
        return json.dumps({"error": "image_path is required"})

    # Resolve relative paths against project root
    path = Path(image_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    if not path.exists():
        return json.dumps({"error": f"File not found: {image_path}"})

    caption = input_dict.get("caption", "")
    content_type = input_dict.get("content_type", "meme")
    alt_text = input_dict.get("alt_text", "")
    title = input_dict.get("title", "")
    subtitle = input_dict.get("subtitle", "")

    # Use alt_text as fallback caption if caption is empty
    if not caption and alt_text:
        caption = alt_text

    # Save as pending draft with local file path as image_url
    state.save_pending(
        caption=caption,
        hashtags=[],
        image_url=str(path),
        alt_text=alt_text,
        image_prompt="",
        original_request=caption or f"[registered from {path.name}]",
        content_type=content_type,
        user_id=user_id,
    )

    # Also save as last_composed so the publish flow uses the local file
    state.set_last_composed(str(path), content_type, user_id=user_id)

    logger.info("register_draft: %s registered as pending draft (content_type=%s)", path.name, content_type)

    msg = f"Draft registered with {path.name}. Ready for approve → post/schedule."
    if not caption:
        msg += " WARNING: Caption is empty — add a caption before posting."

    return json.dumps({
        "status": "registered",
        "image_path": str(path),
        "content_type": content_type,
        "caption": caption,
        "message": msg,
    })


async def _handle_send_file(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    file_path = input_dict.get("file_path", "")
    caption = input_dict.get("caption")

    if not file_path:
        return json.dumps({"error": "No file_path provided"})

    # Resolve relative paths against project root
    path = Path(file_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    if not path.exists():
        return json.dumps({"error": f"File not found: {path}"})

    if not tool_context:
        return json.dumps({"error": "No Telegram context available — cannot send file"})

    bot = tool_context.get("bot")
    chat_id = tool_context.get("chat_id")
    if not bot or not chat_id:
        return json.dumps({"error": "Missing bot or chat_id in tool context"})

    tracker.log_api(f"send_file:{path.name}")

    try:
        with open(path, "rb") as f:
            await bot.send_document(chat_id=chat_id, document=f, caption=caption)
        logger.info("send_file: sent %s to chat %s", path.name, chat_id)
        return json.dumps({"status": "sent", "file": path.name})
    except Exception as e:
        logger.error("send_file failed: %s", e)
        return json.dumps({"error": f"Failed to send file: {e}"})


async def _handle_read_state_file(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    file_path = input_dict.get("file_path", "")
    if not file_path:
        return json.dumps({"error": "No file_path provided"})

    # Resolve and validate path is under state/ or brand/
    path = Path(file_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    try:
        resolved = path.resolve()
        state_dir = (_PROJECT_ROOT / "state").resolve()
        brand_dir = (_PROJECT_ROOT / "brand").resolve()
        if not (str(resolved).startswith(str(state_dir)) or str(resolved).startswith(str(brand_dir))):
            return json.dumps({"error": "Access denied — only state/ and brand/ files are readable"})
    except (OSError, ValueError):
        return json.dumps({"error": "Invalid path"})

    if not resolved.exists():
        return json.dumps({"error": f"File not found: {file_path}"})
    if not resolved.is_file():
        return json.dumps({"error": f"Not a file: {file_path}"})

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return json.dumps({"error": "Binary file — cannot read as text"})
    except OSError as e:
        return json.dumps({"error": f"Read failed: {e}"})

    # Pretty-print JSON files
    if resolved.suffix == ".json":
        try:
            data = json.loads(content)
            content = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass  # return raw content

    # Cap output size
    if len(content) > 30000:
        content = content[:30000] + f"\n\n[... truncated, {len(content)} total chars ...]"

    return json.dumps({"file": file_path, "content": content})


async def _handle_run_self_review(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    from agent.self_review import run_self_review
    from agent.self_review_scheduler import mark_review_complete

    tracker.log_api("run_self_review")
    result = await run_self_review()
    if not result.get("error"):
        mark_review_complete()
    return json.dumps(result)


async def _handle_start_autonomous_plan(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    plan = session_plan.get_plan()
    if not plan:
        return json.dumps({"error": "No active session plan"})

    pending_items = [it for it in plan["items"] if it["status"] == "pending"]
    if not pending_items:
        return json.dumps({"error": "No pending items in the plan"})

    session_plan.set_autonomous(True)

    # Send initial status to Telegram
    bot = tool_context.get("bot") if tool_context else None
    chat_id = tool_context.get("chat_id") if tool_context else None

    total = len(pending_items)
    completed = 0
    errors = []

    for item in pending_items:
        item_id = item["id"]
        desc = item["description"]
        tone = item.get("tone", "")

        session_plan.update_item(item_id, status="generating")

        # Send progress to Telegram
        if bot and chat_id:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"working on #{item_id} of {len(plan['items'])}: {desc[:80]}...",
                )
            except Exception:
                pass

        # Generate content for this item using the existing tool chain
        prompt = f"Generate content: {desc}"
        if tone:
            prompt += f" (tone: {tone})"
        if item.get("notes"):
            prompt += f" — {item['notes']}"

        try:
            from agent.unified_brain import run_unified
            from agent.conversation_context import ConversationContext

            ctx = ConversationContext()
            result = await run_unified(
                message=prompt,
                context=ctx,
                user_id=user_id,
                tool_context=tool_context,
            )
            if result.draft:
                session_plan.save_queued_draft(
                    item_id, result.draft, image_url=result.image_url,
                )
                session_plan.update_item(item_id, status="review")
                completed += 1
            else:
                session_plan.update_item(
                    item_id, status="rejected",
                    notes="Autonomous generation produced no draft",
                )
                errors.append(f"#{item_id}: no draft generated")
        except Exception as e:
            logger.error("Autonomous plan: item #%d failed: %s", item_id, e)
            session_plan.update_item(item_id, status="rejected", notes=str(e)[:200])
            errors.append(f"#{item_id}: {e}")

    # Summary
    queued = session_plan.get_queued_drafts()
    summary = {
        "status": "complete",
        "total_items": total,
        "drafts_generated": completed,
        "errors": errors,
        "queued_drafts": len(queued),
        "message": (
            f"Finished all {total} items. {completed} drafts queued for review. "
            f"Say 'show me #1' to start reviewing."
            if not errors else
            f"Finished {completed}/{total} items ({len(errors)} failed). "
            f"Say 'show me #1' to review completed drafts."
        ),
    }

    # Notify Telegram
    if bot and chat_id:
        try:
            await bot.send_message(chat_id=chat_id, text=summary["message"])
        except Exception:
            pass

    return json.dumps(summary)


async def _handle_show_queued_draft(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    item_id = input_dict.get("item_id")
    if item_id is None:
        return json.dumps({"error": "item_id is required"})

    entry = session_plan.get_queued_draft_by_item(int(item_id))
    if not entry:
        return json.dumps({"error": f"No queued draft for item #{item_id}"})

    # Mark as reviewed
    session_plan.mark_draft_reviewed(int(item_id))

    # Load into pending state so the normal approve/reject flow works
    draft = entry.get("draft", {})
    image_url = entry.get("image_url")

    state.save_pending(
        caption=draft.get("caption", ""),
        hashtags=draft.get("hashtags", []),
        image_url=image_url,
        alt_text=draft.get("alt_text", ""),
        image_prompt=draft.get("image_prompt", ""),
        original_request=draft.get("original_request", f"Plan item #{item_id}"),
        content_type=draft.get("content_type"),
        user_id=user_id,
    )

    # Set current_item to this one for plan tracking
    plan = session_plan.get_plan()
    if plan:
        plan["current_item"] = int(item_id)
        plan["updated_at"] = time.time()
        session_plan._write_plan(plan)

    return json.dumps({
        "status": "loaded",
        "item_id": item_id,
        "draft": draft,
        "image_url": image_url,
        "message": (
            f"Draft for item #{item_id} loaded as pending. "
            "The operator can now approve, reject, or edit it."
        ),
    })


async def _handle_approve_draft(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    pending = state.get_pending(user_id=user_id)
    if not pending:
        return json.dumps({"error": "No pending draft to approve."})

    approved = state.approve_pending(user_id=user_id)
    state.clear_draft_history(user_id=user_id)

    # Update session plan if active
    try:
        plan = session_plan.get_plan()
        if plan:
            current_id = plan.get("current_item")
            if current_id:
                session_plan.update_item(current_id, status="approved")
    except Exception:
        pass

    caption = approved.get("caption", "")[:100] if approved else ""
    return json.dumps({
        "status": "approved",
        "caption_preview": caption,
        "message": "Draft approved. Ask: post now or schedule for later?",
    })


async def _handle_post_approved(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    approved = state.get_approved(user_id=user_id)
    if not approved:
        return json.dumps({"error": "No approved draft to post. The user must approve a draft first."})

    caption = approved.get("caption", "")
    hashtags = approved.get("hashtags", [])
    image_url = approved.get("image_url")

    # Prefer composed image if available
    composed_path, _ = state.get_last_composed(user_id=user_id)
    publish_image = image_url
    if composed_path and Path(composed_path).exists():
        publish_image = composed_path

    tracker.log_api("post_to_x")
    try:
        tweet_url = await publisher.post_to_x(caption, hashtags, publish_image)
    except Exception as e:
        logger.error("post_approved failed: %s", e)
        return json.dumps({"error": f"X posting failed: {e}"})

    # Post to Discord (fire-and-forget)
    discord_url = None
    try:
        from agent import discord_bot, discord_publisher
        if discord_bot.is_ready():
            discord_url = await discord_publisher.post_to_discord(
                caption=caption, hashtags=hashtags, image_url=publish_image,
                auto_slot=approved.get("auto_slot"),
                content_type=approved.get("content_type"),
            )
    except Exception as e:
        logger.warning("Discord posting failed (non-fatal): %s", e)

    # Record auto_slot if applicable
    auto_slot = approved.get("auto_slot")
    if auto_slot:
        from agent import auto_state as _as
        _as.record_post(
            slot_name=auto_slot, caption=caption,
            tweet_url=tweet_url, event_ids=approved.get("auto_event_ids"),
        )

    state.clear_approved(user_id=user_id)

    # Clean up composed file
    if composed_path and Path(composed_path).exists():
        try:
            Path(composed_path).unlink(missing_ok=True)
        except Exception:
            pass
        state.clear_last_composed(user_id=user_id)

    result = {"status": "posted", "tweet_url": tweet_url}
    if discord_url:
        result["discord_url"] = discord_url
    return json.dumps(result)


async def _handle_schedule_post(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    time_desc = input_dict.get("time_description", "")
    if not time_desc:
        return json.dumps({"error": "time_description is required"})

    ts, display = schedule_queue.parse_time(time_desc)
    if ts is None:
        return json.dumps({"error": display})

    approved = state.get_approved(user_id=user_id)
    prompt = input_dict.get("prompt", "")

    if approved:
        # Schedule the pre-approved draft for direct posting
        draft_data = {
            "caption": approved.get("caption", ""),
            "hashtags": approved.get("hashtags", []),
            "image_url": approved.get("image_url"),
            "alt_text": approved.get("alt_text", ""),
            "image_prompt": approved.get("image_prompt", ""),
            "content_type": approved.get("content_type"),
        }
        # Include composed image path if available
        composed_path, _ = state.get_last_composed(user_id=user_id)
        if composed_path and Path(composed_path).exists():
            draft_data["composed_path"] = composed_path

        label = approved.get("caption", "")[:40]
        item = schedule_queue.add_scheduled(
            prompt=approved.get("original_request", label),
            scheduled_utc=ts,
            label=label,
            draft=draft_data,
        )
        if item is None:
            return json.dumps({
                "status": "duplicate",
                "message": "This post is already scheduled around that time.",
            })
        state.clear_approved(user_id=user_id)
        return json.dumps({
            "status": "scheduled",
            "item_id": item["id"],
            "scheduled_for": display,
            "type": "pre_approved_draft",
            "message": f"Approved draft scheduled for {display}. It will be posted directly.",
        })
    elif prompt:
        # Schedule a generate-at-time item
        item = schedule_queue.add_scheduled(prompt=prompt, scheduled_utc=ts, label=prompt[:40])
        if item is None:
            return json.dumps({
                "status": "duplicate",
                "message": "This post is already scheduled around that time.",
            })
        return json.dumps({
            "status": "scheduled",
            "item_id": item["id"],
            "scheduled_for": display,
            "type": "generate_at_time",
            "message": f"Scheduled for {display}. Content will be generated and queued for review.",
        })
    else:
        return json.dumps({"error": "No approved draft and no prompt provided. Approve a draft first or provide a prompt."})


async def _handle_list_scheduled_posts(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    from datetime import datetime, timezone
    items = schedule_queue.list_scheduled()
    if not items:
        return json.dumps({"status": "empty", "message": "No scheduled posts."})
    result = []
    for item in items:
        scheduled_time = datetime.fromtimestamp(
            item["scheduled_utc"], tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        result.append({
            "id": item["id"],
            "label": item.get("label", ""),
            "scheduled_for": scheduled_time,
            "status": item["status"],
            "has_draft": bool(item.get("draft")),
        })
    return json.dumps({"scheduled_posts": result, "count": len(result)})


async def _handle_cancel_scheduled_post(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    item_id = input_dict.get("item_id", "")
    if not item_id:
        return json.dumps({"error": "item_id is required"})
    success = schedule_queue.cancel_scheduled(item_id)
    if success:
        return json.dumps({"status": "cancelled", "item_id": item_id})
    return json.dumps({"error": f"Item {item_id} not found or already completed"})


# ---------------------------------------------------------------------------
# take_screenshot handler
# Uses Playwright (headless Chromium) to capture web page screenshots.
# Useful for: verifying posts went live, capturing tweets, visual reference.
#
# SECURITY: URL scheme validation prevents SSRF attacks (file://, ftp://, etc.)
# RESOURCE SAFETY: Browser is wrapped in try/finally to prevent leaked processes.
# ---------------------------------------------------------------------------

async def _handle_take_screenshot(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    url = input_dict.get("url", "")
    if not url:
        return json.dumps({"error": "url is required"})

    # Validate URL scheme (prevent file://, ftp://, internal IPs)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return json.dumps({"error": "Only http/https URLs are supported"})

    full_page = input_dict.get("full_page", False)
    width = input_dict.get("width", 1280)
    height = input_dict.get("height", 720)

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(_OUTPUTS_DIR / f"screenshot_{int(time.time())}_{uuid.uuid4().hex[:6]}.png")

    tracker.log_api(f"take_screenshot:{url[:40]}")
    logger.info("take_screenshot: %s (full_page=%s)", url, full_page)

    try:
        def _run_screenshot():
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch()
                try:
                    page = browser.new_page(viewport={"width": width, "height": height})
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    page.screenshot(path=out_path, full_page=full_page)
                finally:
                    browser.close()

        await asyncio.to_thread(_run_screenshot)
        logger.info("Screenshot saved: %s", out_path)
        return json.dumps({"path": out_path, "url": url})
    except Exception as e:
        logger.error("take_screenshot failed: %s", e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# edit_image handler
# Uses Pillow for common image operations: text overlay (meme text), resize,
# crop, composite (overlay images), and border. This is a convenience tool —
# faster than writing a full execute_code script for simple edits.
#
# SECURITY: Relative paths resolved against project root. Output paths
# contained to state/outputs/ to prevent arbitrary file writes.
# ---------------------------------------------------------------------------

def _find_font(preferred: str = "Impact") -> str | None:
    """Find a font file, preferring brand fonts then system fonts."""
    from config import settings as _s
    brand_fonts = Path(_s.BRAND_FOLDER) / "assets" / "fonts"
    if brand_fonts.exists():
        for f in brand_fonts.iterdir():
            if f.suffix in (".ttf", ".otf"):
                return str(f)
    # Common system paths
    for candidate in [
        f"/usr/share/fonts/truetype/msttcorefonts/{preferred.lower()}.ttf",
        f"/System/Library/Fonts/{preferred}.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


async def _handle_edit_image(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    source_path = input_dict.get("source_path", "")
    operations = input_dict.get("operations", [])
    if not source_path:
        return json.dumps({"error": "source_path is required"})
    # Resolve relative paths against project root
    src = Path(source_path)
    if not src.is_absolute():
        src = _PROJECT_ROOT / src
    if not src.exists():
        return json.dumps({"error": f"Source image not found: {source_path}"})
    source_path = str(src)
    if not operations:
        return json.dumps({"error": "No operations provided"})

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = input_dict.get("output_path") or str(
        _OUTPUTS_DIR / f"edited_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    )
    # Contain output to outputs dir if user-provided
    if input_dict.get("output_path"):
        out_resolved = Path(output_path).resolve()
        if not str(out_resolved).startswith(str(_OUTPUTS_DIR.resolve()) + "/"):
            output_path = str(_OUTPUTS_DIR / out_resolved.name)

    tracker.log_api("edit_image")
    logger.info("edit_image: %s → %d operations", source_path, len(operations))

    try:
        def _run_edit():
            from PIL import Image, ImageDraw, ImageFont, ImageFilter

            img = Image.open(source_path).convert("RGBA")

            for op in operations:
                op_type = op.get("type", "")

                if op_type == "text_overlay":
                    text = op.get("text", "")
                    position = op.get("position", "bottom")
                    font_size = op.get("font_size", 48)
                    color = op.get("color", "white")
                    stroke_color = op.get("stroke_color", "black")

                    font_path = _find_font()
                    font = (
                        ImageFont.truetype(font_path, font_size)
                        if font_path
                        else ImageFont.load_default()
                    )
                    draw = ImageDraw.Draw(img)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x = (img.width - tw) // 2
                    if position == "top":
                        y = int(img.height * 0.05)
                    elif position == "center":
                        y = (img.height - th) // 2
                    else:  # bottom
                        y = int(img.height * 0.85) - th
                    draw.text((x, y), text, font=font, fill=color,
                              stroke_width=max(1, font_size // 16),
                              stroke_fill=stroke_color)

                elif op_type == "resize":
                    w = op.get("width", img.width)
                    h = op.get("height", img.height)
                    img = img.resize((w, h), Image.LANCZOS)

                elif op_type == "crop":
                    left = op.get("left", 0)
                    top = op.get("top", 0)
                    right = op.get("right", img.width)
                    bottom = op.get("bottom", img.height)
                    img = img.crop((left, top, right, bottom))

                elif op_type == "composite":
                    overlay_path = op.get("overlay_path", "")
                    if not Path(overlay_path).exists():
                        continue
                    overlay = Image.open(overlay_path).convert("RGBA")
                    opacity = op.get("opacity", 1.0)
                    if opacity < 1.0:
                        alpha = overlay.split()[3]
                        alpha = alpha.point(lambda p: int(p * opacity))
                        overlay.putalpha(alpha)
                    x = op.get("x", 0)
                    y = op.get("y", 0)
                    img.paste(overlay, (x, y), overlay)

                elif op_type == "border":
                    bw = op.get("width", 5)
                    bcolor = op.get("color", "white")
                    from PIL import ImageOps
                    img = ImageOps.expand(img, border=bw, fill=bcolor)

            # Save as RGB PNG
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[3])
                bg.save(output_path)
            else:
                img.save(output_path)

        await asyncio.to_thread(_run_edit)
        logger.info("edit_image saved: %s", output_path)
        return json.dumps({"path": output_path})
    except Exception as e:
        logger.error("edit_image failed: %s", e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# save_note / get_notes handlers
# Persistent key-value notes that survive across conversations.
# Stored in state/agent_notes.json as a simple {key: value} dict.
# Notes are also injected into the system prompt via unified_prompt.py
# so the agent always has them in context without calling get_notes.
# ---------------------------------------------------------------------------

_NOTES_FILE = _PROJECT_ROOT / "state" / "agent_notes.json"


def _read_notes() -> dict:
    if not _NOTES_FILE.exists():
        return {}
    try:
        return json.loads(_NOTES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_notes(notes: dict) -> None:
    _NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _NOTES_FILE.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")


async def _handle_save_note(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    key = input_dict.get("key", "").strip()
    content = input_dict.get("content", "").strip()
    if not key or not content:
        return json.dumps({"error": "Both key and content are required"})

    notes = _read_notes()
    existed = key in notes
    notes[key] = content
    _write_notes(notes)
    tracker.log_api("save_note")
    action = "updated" if existed else "saved"
    logger.info("save_note: %s '%s'", action, key)
    return json.dumps({"status": action, "key": key})


async def _handle_get_notes(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    key = input_dict.get("key")
    notes = _read_notes()
    if key:
        content = notes.get(key)
        if content is None:
            return json.dumps({"error": f"No note found with key '{key}'"})
        return json.dumps({"key": key, "content": content})
    return json.dumps({"notes": notes, "count": len(notes)})


# ---------------------------------------------------------------------------
# git_info handler
# Reads git repository information: log, diff, show, status.
# Gives the agent self-awareness of its own codebase evolution.
#
# SECURITY: Args are sanitized with a strict regex that only allows
# alphanumeric chars, ~, ^, ., -, / — prevents command injection and
# prevents reading sensitive files via git show ":path/to/.env".
# ---------------------------------------------------------------------------

async def _handle_git_info(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    action = input_dict.get("action", "log")
    args = input_dict.get("args", "")

    # Sanitize args: allow only safe git ref patterns (alphanumeric, ~, ^, ., -, /)
    if args and not re.match(r'^[a-zA-Z0-9_.~^/\-]+$', args):
        return json.dumps({"error": "Invalid args: only alphanumeric, ~, ^, ., -, / allowed"})

    tracker.log_api(f"git_info:{action}")

    cmd_map = {
        "log": ["git", "log", "--oneline", "-n", args or "10"],
        "diff": ["git", "diff", args or "HEAD~1"],
        "show": ["git", "show", args or "HEAD", "--stat"],
        "status": ["git", "status", "--short"],
    }
    cmd = cmd_map.get(action)
    if not cmd:
        return json.dumps({"error": f"Unknown action: {action}"})

    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd,
            capture_output=True, text=True, timeout=10,
            cwd=str(_PROJECT_ROOT),
        )
        output = proc.stdout or ""
        stderr = proc.stderr or ""
        # Truncate large outputs
        max_len = 3000 if action == "show" else 5000
        if len(output) > max_len:
            output = output[:max_len] + "\n... (truncated)"
        return json.dumps({"output": output, "stderr": stderr[:500] if stderr else ""})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Git command timed out (10s)"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# read_telegram_channel handler
# Reads messages logged by the channel message logger (see telegram_bot.py).
# Bots can't read channel history directly via Telegram API, so we use a
# passive approach: a MessageHandler in telegram_bot.py silently logs messages
# from configured channels to state/channel_messages.json (rolling 100 msgs).
# This tool reads from that file.
# ---------------------------------------------------------------------------

_CHANNEL_MESSAGES_FILE = _PROJECT_ROOT / "state" / "channel_messages.json"


def _read_channel_messages() -> list[dict]:
    if not _CHANNEL_MESSAGES_FILE.exists():
        return []
    try:
        return json.loads(_CHANNEL_MESSAGES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def log_channel_message(chat_id: int, author: str, text: str, timestamp: float) -> None:
    """Called from the Telegram message handler to log channel messages."""
    messages = _read_channel_messages()
    messages.append({
        "chat_id": chat_id,
        "author": author,
        "text": text[:500],
        "timestamp": timestamp,
    })
    # Keep last 100 messages
    messages = messages[-100:]
    _CHANNEL_MESSAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CHANNEL_MESSAGES_FILE.write_text(
        json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def _handle_read_telegram_channel(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    channel_id = input_dict.get("channel_id", "")
    limit = min(input_dict.get("limit", 20), 50)

    tracker.log_api("read_telegram_channel")

    messages = _read_channel_messages()
    if channel_id:
        try:
            cid = int(channel_id)
            messages = [m for m in messages if m.get("chat_id") == cid]
        except ValueError:
            pass

    messages = messages[-limit:]
    return json.dumps({"messages": messages, "count": len(messages)})


# ---------------------------------------------------------------------------
# save_snippet / list_snippets / use_snippet handlers
# Content snippet library — saves captions, ideas, research findings for reuse.
# Stored in state/snippets.json as an array of tagged entries (max 100).
# Each snippet has: id, label, content, tags[], saved_at timestamp.
# Useful for: building a content bank, saving good captions for reuse,
# storing research findings for later content generation.
# ---------------------------------------------------------------------------

_SNIPPETS_FILE = _PROJECT_ROOT / "state" / "snippets.json"


def _read_snippets() -> list[dict]:
    if not _SNIPPETS_FILE.exists():
        return []
    try:
        return json.loads(_SNIPPETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_snippets(snippets: list[dict]) -> None:
    _SNIPPETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SNIPPETS_FILE.write_text(
        json.dumps(snippets, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def _handle_save_snippet(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    label = input_dict.get("label", "").strip()
    content = input_dict.get("content", "").strip()
    tags = input_dict.get("tags", [])
    if not label or not content:
        return json.dumps({"error": "Both label and content are required"})

    snippet = {
        "id": uuid.uuid4().hex[:8],
        "label": label,
        "content": content,
        "tags": tags,
        "saved_at": time.time(),
    }
    snippets = _read_snippets()
    snippets.append(snippet)
    # Keep max 100
    if len(snippets) > 100:
        snippets = snippets[-100:]
    _write_snippets(snippets)
    tracker.log_api("save_snippet")
    logger.info("save_snippet: '%s' (id=%s)", label, snippet["id"])
    return json.dumps({"status": "saved", "id": snippet["id"], "label": label})


async def _handle_list_snippets(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    tag = input_dict.get("tag", "")
    limit = input_dict.get("limit", 10)

    snippets = _read_snippets()
    if tag:
        snippets = [s for s in snippets if tag in s.get("tags", [])]
    # Most recent first
    snippets = list(reversed(snippets))[:limit]
    # Return compact summaries
    results = [
        {"id": s["id"], "label": s["label"], "tags": s.get("tags", []),
         "preview": s["content"][:100]}
        for s in snippets
    ]
    return json.dumps({"snippets": results, "count": len(results)})


async def _handle_use_snippet(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    snippet_id = input_dict.get("id", "")
    if not snippet_id:
        return json.dumps({"error": "id is required"})

    snippets = _read_snippets()
    for s in snippets:
        if s["id"] == snippet_id:
            return json.dumps(s)
    return json.dumps({"error": f"Snippet '{snippet_id}' not found"})


async def _handle_create_campaign(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    from agent import campaigns

    name = input_dict.get("name", "").strip()
    brief = input_dict.get("brief", "").strip()
    slots = input_dict.get("slots", [])
    start_date = input_dict.get("start_date", "")
    post_times = input_dict.get("post_times")
    if not name or not brief or not slots:
        return json.dumps({"error": "name, brief, and slots are required"})

    result = campaigns.create_campaign(
        name=name,
        brief=brief,
        slots=slots,
        start_date=start_date,
        post_times=post_times,
    )
    if not result["success"]:
        return json.dumps(result)

    # Auto-schedule all posts
    sched_result = campaigns.schedule_campaign_posts(name)
    result["scheduling"] = sched_result
    return json.dumps(result)


async def _handle_campaign_status(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    from agent import campaigns

    name = input_dict.get("name", "").strip()
    if name:
        progress = campaigns.get_campaign_progress(name)
        campaign = campaigns.get_campaign(name)
        if campaign:
            progress["slots"] = campaign.get("slots", [])
        return json.dumps(progress)

    all_campaigns = campaigns.list_campaigns()
    summaries = []
    for c in all_campaigns:
        slots = c.get("slots", [])
        posted = sum(1 for s in slots if s.get("status") == "posted")
        summaries.append({
            "name": c["name"],
            "status": c.get("status"),
            "progress": f"{posted}/{len(slots)}",
            "brief": c.get("brief", "")[:100],
        })
    return json.dumps({"campaigns": summaries})


async def _handle_record_walkthrough(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    """Record a video walkthrough or screenshot sequence using the demo recorder."""
    url = input_dict.get("url", "")
    steps = input_dict.get("steps", [])
    mode = input_dict.get("mode", "video")
    name = input_dict.get("name", "walkthrough")

    if not url or not steps:
        return json.dumps({"error": "url and steps are required"})

    # Write a temporary demo script JSON and call the recorder
    import tempfile
    script_data = {
        "name": name,
        "url": url,
        "mode": mode,
        "viewport_width": 1280,
        "viewport_height": 720,
        "steps": steps,
    }

    tracker.log_api(f"record_walkthrough:{url[:40]}")

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="demo_"
        ) as f:
            json.dump(script_data, f)
            script_path = f.name

        from agent.demo_recorder import record_demo
        result = await record_demo(script_path, mode_override=mode)

        # Clean up temp script
        Path(script_path).unlink(missing_ok=True)

        if result.error:
            return json.dumps({"error": result.error})

        output = {
            "mode": result.mode,
            "duration_seconds": result.duration_seconds,
        }
        if result.video_path:
            output["video_path"] = result.video_path
        if result.screenshot_paths:
            output["screenshot_paths"] = result.screenshot_paths

        return json.dumps(output)
    except Exception as e:
        logger.error("record_walkthrough failed: %s", e)
        return json.dumps({"error": str(e)})


async def _handle_style_video(
    input_dict: dict, tracker: ResourceTracker, user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    """Apply polished styling to a raw video."""
    input_video = input_dict.get("input_video", "")
    if not input_video or not Path(input_video).exists():
        return json.dumps({"error": f"Video not found: {input_video}"})

    from agent.video_styler import VideoStyle, async_apply_style

    style = VideoStyle()
    if input_dict.get("device_frame") == "none":
        style.device_frame = "none"
    if input_dict.get("bg_colors"):
        style.bg_colors = input_dict["bg_colors"]
    if input_dict.get("zoom_enabled") is False:
        style.zoom_enabled = False
    if input_dict.get("square_output") is False:
        style.output_width = 1280
        style.output_height = 720

    narration_texts = input_dict.get("narration_texts")

    tracker.log_api("style_video")

    try:
        output = await async_apply_style(
            input_video, style=style, narration_texts=narration_texts,
        )
        return json.dumps({"path": output, "style": style.device_frame})
    except Exception as e:
        logger.error("style_video failed: %s", e)
        return json.dumps({"error": str(e)})


_UNIFIED_HANDLERS = {
    "get_pending_draft": _handle_get_pending_draft,
    "revise_draft": _handle_revise_draft,
    "check_auto_post_status": _handle_check_auto_post_status,
    "web_fetch": _handle_web_fetch,
    "save_session_plan": _handle_save_session_plan,
    "get_session_plan": _handle_get_session_plan,
    "update_plan_item": _handle_update_plan_item,
    "execute_code": _handle_execute_code,
    "register_draft": _handle_register_draft,
    "send_file": _handle_send_file,
    "read_state_file": _handle_read_state_file,
    "run_self_review": _handle_run_self_review,
    "start_autonomous_plan": _handle_start_autonomous_plan,
    "show_queued_draft": _handle_show_queued_draft,
    "approve_draft": _handle_approve_draft,
    "post_approved": _handle_post_approved,
    "schedule_post": _handle_schedule_post,
    "list_scheduled_posts": _handle_list_scheduled_posts,
    "cancel_scheduled_post": _handle_cancel_scheduled_post,
    "take_screenshot": _handle_take_screenshot,
    "edit_image": _handle_edit_image,
    "save_note": _handle_save_note,
    "get_notes": _handle_get_notes,
    "git_info": _handle_git_info,
    "read_telegram_channel": _handle_read_telegram_channel,
    "save_snippet": _handle_save_snippet,
    "list_snippets": _handle_list_snippets,
    "use_snippet": _handle_use_snippet,
    "create_campaign": _handle_create_campaign,
    "campaign_status": _handle_campaign_status,
    "record_walkthrough": _handle_record_walkthrough,
    "style_video": _handle_style_video,
}


async def execute_tool(
    tool_name: str,
    input_dict: dict,
    tracker: ResourceTracker,
    user_id: int | None = None,
    tool_context: dict | None = None,
) -> str:
    """Execute a tool by name. Dispatches to base tools or unified-only tools."""
    handler = _UNIFIED_HANDLERS.get(tool_name)
    if handler:
        return await handler(
            input_dict, tracker, user_id=user_id, tool_context=tool_context,
        )
    # Delegate to existing tool registry
    return await _base_execute_tool(tool_name, input_dict, tracker)
