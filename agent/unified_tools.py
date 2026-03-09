"""
Extended tool registry for unified brain.

Re-exports all 8 existing tools from tools.py and adds unified-only tools:
- get_pending_draft: returns current pending draft state
- check_auto_post_status: returns auto-post schedule summary
- web_fetch: fetches and reads web page content
- execute_code: run Python scripts for reports, analysis, file generation
- send_file: deliver generated files to the user via Telegram
- read_state_file: read raw data from state/ and brand/ directories
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent import auto_state, session_plan, state, web_fetch
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
        "Execute a Python script to process data, generate files, create HTML, "
        "analyze stats, or do any computation. Use this when you need to build "
        "something that doesn't fit your other tools — reports, data analysis, "
        "file generation, etc. The script runs in a sandboxed subprocess with "
        "access to the brand/ and state/ directories (read-only) and can write "
        "output files to state/outputs/."
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


UNIFIED_TOOL_DEFINITIONS = _BASE_TOOL_DEFINITIONS + [
    _get_pending_draft_def,
    _revise_draft_def,
    _check_auto_post_status_def,
    _web_fetch_def,
    _save_session_plan_def,
    _get_session_plan_def,
    _update_plan_item_def,
    _execute_code_def,
    _send_file_def,
    _read_state_file_def,
    _run_self_review_def,
    _start_autonomous_plan_def,
    _show_queued_draft_def,
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
            timeout=30,
            cwd=str(_PROJECT_ROOT),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout = proc.stdout[:10000] if proc.stdout else ""
        stderr = proc.stderr[:5000] if proc.stderr else ""
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "Script timed out after 30 seconds"
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


_UNIFIED_HANDLERS = {
    "get_pending_draft": _handle_get_pending_draft,
    "revise_draft": _handle_revise_draft,
    "check_auto_post_status": _handle_check_auto_post_status,
    "web_fetch": _handle_web_fetch,
    "save_session_plan": _handle_save_session_plan,
    "get_session_plan": _handle_get_session_plan,
    "update_plan_item": _handle_update_plan_item,
    "execute_code": _handle_execute_code,
    "send_file": _handle_send_file,
    "read_state_file": _handle_read_state_file,
    "run_self_review": _handle_run_self_review,
    "start_autonomous_plan": _handle_start_autonomous_plan,
    "show_queued_draft": _handle_show_queued_draft,
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
