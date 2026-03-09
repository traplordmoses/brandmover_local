"""
Unified agent brain — single LLM loop with personality + tools.

Replaces the two-brain architecture (chat.handle_casual_chat + engine.run_agent)
with one entry point that handles both chat and content generation.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

import anthropic

from agent.conversation_context import ConversationContext
from agent.engine import (
    _try_parse_draft,
    _sanitize_draft,
    _extract_image_url,
    _extract_image_urls,
    OnToolCall,
)
from agent.resource_log import ResourceTracker
from agent.unified_prompt import build_unified_system_prompt
from agent.unified_tools import UNIFIED_TOOL_DEFINITIONS, execute_tool
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResult:
    """Result of a unified brain run."""
    response_text: str = ""
    draft: dict | None = None
    image_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    resources: ResourceTracker = field(default_factory=ResourceTracker)
    tool_calls_made: list[str] = field(default_factory=list)
    turns_used: int = 0
    total_time: float = 0.0
    is_generation: bool = False


async def run_unified(
    message: str,
    context: ConversationContext,
    on_tool_call: OnToolCall | None = None,
    user_id: int | None = None,
    tool_context: dict | None = None,
) -> UnifiedResult:
    """Run the unified brain for any message.

    The LLM decides whether to chat, generate content, or both.
    Uses personality + memory + tools in every call.

    Args:
        message: User's message text.
        context: Conversation context with history.
        on_tool_call: Optional progress callback(tool_name, description).
        user_id: Telegram user ID for per-user state.
        tool_context: Optional dict with bot/chat_id for tools that need Telegram access.

    Returns:
        UnifiedResult with response text and optional draft.
    """
    t_start = time.time()
    result = UnifiedResult()
    tracker = ResourceTracker()
    result.resources = tracker

    from agent._client import get_anthropic
    client = get_anthropic()

    system_prompt = build_unified_system_prompt(context, user_id=user_id)

    # Build messages from conversation history + current message
    messages = []
    for turn in context.conversation_history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
        })
    messages.append({"role": "user", "content": message})

    max_turns = settings.AGENT_MAX_TURNS
    tool_call_log = []

    for turn in range(max_turns):
        result.turns_used = turn + 1

        # Turn 0: auto (LLM can chat without tools)
        # Last turn: no tools (force final answer)
        # Middle: auto
        tool_choice = (
            {"type": "none"} if turn >= max_turns - 1
            else {"type": "auto"}
        )

        try:
            response = await client.messages.create(
                model=settings.SONNET_MODEL,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=UNIFIED_TOOL_DEFINITIONS,
                tool_choice=tool_choice,
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error("Unified brain API error on turn %d: %s", turn + 1, e)
            result.response_text = f"API error: {e}"
            break

        # Track token usage
        if hasattr(response, "usage") and response.usage:
            tracker.add_tokens(response.usage.input_tokens, response.usage.output_tokens)

        # Process response blocks
        assistant_content = response.content
        tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
        text_blocks = [b for b in assistant_content if b.type == "text"]

        # Collect text output
        for tb in text_blocks:
            result.response_text += tb.text + "\n"

        # If no tool calls, we're done
        if not tool_use_blocks or response.stop_reason == "end_turn":
            logger.info(
                "Unified brain finished after %d turns (stop_reason=%s)",
                turn + 1, response.stop_reason,
            )
            break

        # Append assistant message
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute each tool call
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block.name
            tool_input = tool_block.input
            result.tool_calls_made.append(tool_name)

            logger.info("Unified brain calling tool: %s (input: %s)", tool_name, str(tool_input)[:200])

            if on_tool_call:
                brief = _tool_description(tool_name, tool_input)
                await on_tool_call(tool_name, brief)

            try:
                tool_result = await execute_tool(
                    tool_name, tool_input, tracker,
                    user_id=user_id, tool_context=tool_context,
                )
                if len(tool_result) > 15000:
                    tool_result = tool_result[:15000] + "\n\n[... truncated to 15000 chars ...]"
            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e)
                tool_result = json.dumps({"error": str(e)})

            log_entry = {
                "name": tool_name,
                "input": tool_input,
                "result": tool_result if tool_name in ("generate_image", "img2img") else tool_result[:500],
            }

            # Pre-extract image URL
            if tool_name in ("generate_image", "img2img"):
                try:
                    parsed = json.loads(tool_result)
                    if "image_url" in parsed:
                        log_entry["image_url"] = parsed["image_url"]
                except (json.JSONDecodeError, TypeError):
                    pass

            tool_call_log.append(log_entry)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": tool_result,
            })

        messages.append({"role": "user", "content": tool_results})

    # Post-processing
    result.response_text = result.response_text.strip()
    result.total_time = round(time.time() - t_start, 1)

    # Try to parse a draft from the response
    draft = _try_parse_draft(result.response_text)
    if draft:
        result.draft = _sanitize_draft(draft)
        result.is_generation = True

    # Backfill content_type from generate_image tool call
    if result.draft and not result.draft.get("content_type"):
        for entry in reversed(tool_call_log):
            if entry["name"] == "generate_image" and isinstance(entry.get("input"), dict):
                ct = entry["input"].get("content_type")
                if ct:
                    result.draft["content_type"] = ct
                    break

    # Extract image URLs from tool calls
    result.image_url = _extract_image_url(tool_call_log)
    result.image_urls = _extract_image_urls(tool_call_log)

    logger.info(
        "Unified brain complete: %d turns, %.1fs, %d tool calls, draft=%s, image=%s",
        result.turns_used,
        result.total_time,
        len(result.tool_calls_made),
        bool(result.draft),
        bool(result.image_url),
    )

    return result


def _tool_description(tool_name: str, tool_input: dict) -> str:
    """Brief human-readable description of a tool call."""
    descs = {
        "read_brand_guidelines": "Loading brand guidelines and references...",
        "read_references": "Checking available reference materials...",
        "check_figma_design": f"Checking Figma design ({tool_input.get('action', 'styles')})...",
        "generate_image": "Generating brand image...",
        "read_feedback_history": "Reviewing feedback history...",
        "log_resource_usage": "Logging resources used...",
        "img2img": f"Generating image from reference: {tool_input.get('reference_image_path', 'auto')}...",
        "execute_openclaw_script": f"Running {tool_input.get('script_name', 'script')}...",
        "get_pending_draft": "Checking pending draft...",
        "revise_draft": f"Revising draft: {tool_input.get('feedback', '?')[:60]}...",
        "check_auto_post_status": "Checking auto-post schedule...",
        "web_fetch": f"Fetching {tool_input.get('url', 'URL')[:60]}...",
        "save_session_plan": "Saving content plan...",
        "get_session_plan": "Checking session plan...",
        "update_plan_item": f"Updating plan item #{tool_input.get('item_id', '?')}...",
        "execute_code": f"Running script: {tool_input.get('description', 'computation')}...",
        "send_file": f"Sending file: {Path(tool_input.get('file_path', '')).name if tool_input.get('file_path') else '?'}...",
        "read_state_file": f"Reading {tool_input.get('file_path', 'file')}...",
        "run_self_review": "Analyzing performance and updating preferences...",
        "start_autonomous_plan": "Working through plan autonomously...",
        "show_queued_draft": f"Loading draft #{tool_input.get('item_id', '?')} for review...",
        "post_approved": "Posting approved draft to X...",
        "schedule_post": f"Scheduling post for {tool_input.get('time_description', '?')}...",
        "list_scheduled_posts": "Checking scheduled posts...",
        "cancel_scheduled_post": f"Cancelling scheduled post {tool_input.get('item_id', '?')}...",
    }
    return descs.get(tool_name, f"Executing {tool_name}...")
