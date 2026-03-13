"""
Unified agent brain — single LLM loop with personality + tools.

LEGACY PATH — Not actively maintained.
This module is only used when UNIFIED_BRAIN_ENABLED=true.
The active architecture uses agent/engine.py with think/finish tools.
See agent/heartbeat.py for the scheduler and agent/session.py for memory.
Do not add new features here — they won't benefit from session memory,
conversation continuity, self-critique, or the heartbeat system.

ARCHITECTURE:
This is the core reasoning engine. It replaces the old two-brain architecture
(chat.handle_casual_chat for conversation + engine.run_agent for generation)
with ONE entry point that handles both chat and content generation.

HOW IT WORKS:
1. Builds a system prompt with personality, brand voice, learned preferences,
   current state, and tool descriptions (see unified_prompt.py)
2. Sends the user's message + conversation history to Claude
3. Claude responds with text AND/OR tool calls
4. We execute each tool call and feed results back to Claude
5. Repeat until Claude stops calling tools or we hit max_turns
6. Parse the final response for a JSON draft block (if content was generated)

KEY DESIGN DECISIONS:
- tool_choice="none" on the LAST turn forces Claude to give a final answer
  instead of calling more tools forever (prevents infinite loops)
- Tool results are truncated to 15K chars to prevent context window blowup
- Prompt caching (cache_control: ephemeral) avoids re-tokenizing the ~4K
  system prompt on every turn of the loop
- The LLM decides whether to chat or generate — we don't pre-classify intent

INTERVIEW TALKING POINT:
This is a classic "ReAct" (Reasoning + Acting) agent loop. The model reasons
about what to do, takes an action (tool call), observes the result, and
reasons again. The key insight is constraining the final turn to prevent
runaway tool usage while still allowing multi-step reasoning.
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
    _try_parse_draft,     # Regex parser that extracts JSON draft blocks from response text
    _sanitize_draft,      # Normalizes draft fields (content_type, hashtags, etc.)
    _extract_image_url,   # Pulls first image URL from tool call results
    _extract_image_urls,  # Pulls ALL image URLs from tool call results
    OnToolCall,           # Type alias for progress callback
    OnReasoning,          # Type alias for reasoning trace callback
)
from agent.resource_log import ResourceTracker  # Tracks API calls, files read, tokens used
from agent.unified_prompt import build_unified_system_prompt
from agent.unified_tools import UNIFIED_TOOL_DEFINITIONS, execute_tool
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResult:
    """Result of a unified brain run.

    This is the return type for every brain invocation. Handlers inspect these
    fields to decide what to show the user (text reply, image, draft for approval).

    Fields:
        response_text: Claude's natural language response to display to the user.
        draft: If content was generated, the structured draft dict (caption, image_prompt, etc.).
                None if this was just a chat response.
        image_url: Primary generated image URL (first one found in tool results).
        image_urls: ALL generated image URLs (for multi-option generation like brand_3d).
        resources: Tracks what APIs/files were used and token counts for cost attribution.
        tool_calls_made: List of tool names called (e.g., ["read_brand_guidelines", "generate_image"]).
        turns_used: How many LLM round-trips occurred (1 = simple chat, 3-5 = complex generation).
        total_time: Wall-clock time for the entire run in seconds.
        is_generation: True if a draft was produced (vs. pure chat response).
    """
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
    on_reasoning: OnReasoning | None = None,
    user_id: int | None = None,
    tool_context: dict | None = None,
) -> UnifiedResult:
    """Run the unified brain for any message.

    This is the MAIN ENTRY POINT for all user interactions. The LLM decides
    whether to chat, generate content, or both. It uses personality + memory +
    tools in every call — there's no separate "chat mode" vs "generation mode".

    Args:
        message: User's message text (e.g., "make a post about our launch").
        context: Conversation context with history (past messages for continuity).
        on_tool_call: Optional progress callback — called with (tool_name, description)
                      each time a tool is invoked, so the UI can show "Generating image...".
        on_reasoning: Optional progress callback — called with Claude's reasoning text
                      between tool calls, so the UI can show live thinking traces.
        user_id: Telegram user ID for per-user state (drafts, preferences).
        tool_context: Optional dict with bot/chat_id for tools that need Telegram access
                      (e.g., send_file needs the bot instance to send files to the user).

    Returns:
        UnifiedResult with response text and optional draft.
    """
    t_start = time.time()
    result = UnifiedResult()
    tracker = ResourceTracker()
    result.resources = tracker

    # Lazy import to avoid circular dependency (brain → client → settings → brain)
    from agent._client import get_anthropic
    client = get_anthropic()

    # Build the system prompt dynamically — includes personality, brand voice,
    # learned preferences from past feedback, current draft state, and tool guidance.
    # This changes with every call because state (pending drafts, notes) changes.
    system_prompt = build_unified_system_prompt(context, user_id=user_id)

    # Build messages array from conversation history + the new user message.
    # ConversationContext stores past turns so Claude has memory across messages.
    messages = []
    for turn in context.conversation_history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
        })
    messages.append({"role": "user", "content": message})

    # Max turns = how many LLM round-trips before we force a final answer.
    # Each turn: Claude responds → we execute tools → feed results back.
    # Default is 15 turns (configurable via AGENT_MAX_TURNS env var).
    max_turns = settings.AGENT_MAX_TURNS

    # Log of all tool calls and their results — used for post-processing
    # (extracting image URLs, backfilling content_type).
    tool_call_log = []

    # ── THE AGENT LOOP ──
    # This is the core ReAct loop. Each iteration:
    # 1. Send messages to Claude (with tool definitions)
    # 2. Claude responds with text + optional tool_use blocks
    # 3. Execute each tool, collect results
    # 4. Append results to messages for the next iteration
    # 5. Break if Claude says "end_turn" or we hit max_turns
    for turn in range(max_turns):
        result.turns_used = turn + 1

        # CRITICAL: On the last turn, set tool_choice to "none" to force Claude
        # to give a final text answer instead of calling more tools.
        # Without this, the agent could loop forever.
        # On other turns, "auto" lets Claude decide whether to use tools.
        tool_choice = (
            {"type": "none"} if turn >= max_turns - 1
            else {"type": "auto"}
        )

        try:
            response = await client.messages.create(
                model=settings.SONNET_MODEL,  # Default: claude-sonnet-4-6
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    # cache_control: ephemeral tells Anthropic's API to cache this
                    # system prompt across turns. Since it's ~4K tokens and identical
                    # across turns, this saves significant token costs (only charged once).
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=UNIFIED_TOOL_DEFINITIONS,  # All 36 tool schemas
                tool_choice=tool_choice,
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error("Unified brain API error on turn %d: %s", turn + 1, e)
            result.response_text = f"API error: {e}"
            break

        # Track token usage for cost attribution.
        # input_tokens = tokens sent TO Claude, output_tokens = tokens Claude generated.
        if hasattr(response, "usage") and response.usage:
            tracker.add_tokens(response.usage.input_tokens, response.usage.output_tokens)

        # Parse response into text blocks (Claude's words) and tool_use blocks (tool calls).
        # A single response can contain BOTH text AND tool calls.
        assistant_content = response.content
        tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
        text_blocks = [b for b in assistant_content if b.type == "text"]

        # Accumulate text output across turns (Claude may speak on multiple turns).
        for tb in text_blocks:
            result.response_text += tb.text + "\n"

        # Fire reasoning callback with Claude's text between tool calls
        if on_reasoning and text_blocks:
            combined = " ".join(tb.text.strip() for tb in text_blocks if tb.text.strip())
            if combined:
                await on_reasoning(combined)

        # EXIT CONDITION: If Claude didn't call any tools, or explicitly said "end_turn",
        # the reasoning is complete. Break out of the loop.
        if not tool_use_blocks or response.stop_reason == "end_turn":
            logger.info(
                "Unified brain finished after %d turns (stop_reason=%s)",
                turn + 1, response.stop_reason,
            )
            break

        # Append the assistant's response to the message history so Claude
        # can see what it said/called on the next turn.
        messages.append({"role": "assistant", "content": assistant_content})

        # ── EXECUTE TOOL CALLS ──
        # Process each tool call in the response. Tool calls are executed
        # sequentially (not in parallel) to avoid race conditions on shared state.
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block.name      # e.g., "generate_image"
            tool_input = tool_block.input    # e.g., {"prompt": "...", "content_type": "announcement"}
            result.tool_calls_made.append(tool_name)

            logger.info("Unified brain calling tool: %s (input: %s)", tool_name, str(tool_input)[:200])

            # Fire the progress callback so the Telegram UI can show
            # "Generating brand image..." while the user waits.
            if on_tool_call:
                brief = _tool_description(tool_name, tool_input)
                await on_tool_call(tool_name, brief)

            try:
                # Dispatch to the appropriate handler (see unified_tools.py).
                # execute_tool is async — some tools (like generate_image) take 10-30s.
                tool_result = await execute_tool(
                    tool_name, tool_input, tracker,
                    user_id=user_id, tool_context=tool_context,
                )
                # SAFETY: Truncate huge tool results to prevent context window blowup.
                # Some tools (like read_state_file on a large JSON) could return 50K+ chars.
                # 15K is enough for Claude to extract what it needs.
                if len(tool_result) > 15000:
                    tool_result = tool_result[:15000] + "\n\n[... truncated to 15000 chars ...]"
            except Exception as e:
                # Tool failures shouldn't crash the entire run — return the error
                # as a tool result so Claude can adapt (retry, try a different approach).
                logger.error("Tool %s failed: %s", tool_name, e)
                tool_result = json.dumps({"error": str(e)})

            # Build a log entry for post-processing (image URL extraction).
            # We keep full results for image generation tools (need the URL)
            # but truncate others to save memory.
            log_entry = {
                "name": tool_name,
                "input": tool_input,
                "result": tool_result if tool_name in ("generate_image", "img2img") else tool_result[:500],
            }

            # Pre-extract image URLs from generation tool results so we can
            # attach them to the final result without re-parsing later.
            if tool_name in ("generate_image", "img2img"):
                try:
                    parsed = json.loads(tool_result)
                    if "image_url" in parsed:
                        log_entry["image_url"] = parsed["image_url"]
                except (json.JSONDecodeError, TypeError):
                    pass

            tool_call_log.append(log_entry)

            # Format tool results in Anthropic's expected format.
            # tool_use_id links this result back to the specific tool call.
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": tool_result,
            })

        # Append tool results as a "user" message (Anthropic's API convention:
        # tool results are sent as user messages with type=tool_result).
        messages.append({"role": "user", "content": tool_results})

    # ── POST-PROCESSING ──
    result.response_text = result.response_text.strip()
    result.total_time = round(time.time() - t_start, 1)

    # Try to parse a structured draft from Claude's response text.
    # Claude outputs drafts as ```json blocks with caption, image_prompt, etc.
    # _try_parse_draft uses regex to find and parse these blocks.
    draft = _try_parse_draft(result.response_text)
    if draft:
        result.draft = _sanitize_draft(draft)  # Normalize fields, add defaults
        result.is_generation = True

    # Backfill content_type if Claude forgot to include it in the JSON draft
    # but DID specify it in the generate_image tool call.
    if result.draft and not result.draft.get("content_type"):
        for entry in reversed(tool_call_log):
            if entry["name"] == "generate_image" and isinstance(entry.get("input"), dict):
                ct = entry["input"].get("content_type")
                if ct:
                    result.draft["content_type"] = ct
                    break

    # Extract image URLs from tool call results.
    # These are used to display the generated image(s) to the user.
    result.image_url = _extract_image_url(tool_call_log)     # First image (primary)
    result.image_urls = _extract_image_urls(tool_call_log)   # All images (for multi-option)

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
    """Brief human-readable description of a tool call.

    Used for the progress callback (on_tool_call) so the Telegram UI can show
    the user what's happening: "Generating brand image...", "Fetching URL...", etc.
    Each description is a short present-tense phrase with the most relevant parameter.
    """
    descs = {
        # ── Base tools (from agent/tools.py) ──
        "read_brand_guidelines": "Loading brand guidelines and references...",
        "read_references": "Checking available reference materials...",
        "check_figma_design": f"Checking Figma design ({tool_input.get('action', 'styles')})...",
        "generate_image": "Generating brand image...",
        "read_feedback_history": "Reviewing feedback history...",
        "log_resource_usage": "Logging resources used...",
        "img2img": f"Generating image from reference: {tool_input.get('reference_image_path', 'auto')}...",
        "execute_openclaw_script": f"Running {tool_input.get('script_name', 'script')}...",
        # ── Unified-only tools (from agent/unified_tools.py) ──
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
        # ── New tools (screenshot, image editing, notes, git, channel, snippets) ──
        "take_screenshot": f"Capturing screenshot of {tool_input.get('url', 'page')[:50]}...",
        "edit_image": f"Editing image: {len(tool_input.get('operations', []))} operation(s)...",
        "save_note": f"Saving note: {tool_input.get('key', '?')}...",
        "get_notes": f"Retrieving note(s){': ' + tool_input.get('key', '') if tool_input.get('key') else ''}...",
        "git_info": f"Git {tool_input.get('action', 'info')}...",
        "read_telegram_channel": "Reading channel messages...",
        "save_snippet": f"Saving snippet: {tool_input.get('label', '?')[:40]}...",
        "list_snippets": "Listing saved snippets...",
        "use_snippet": f"Loading snippet {tool_input.get('id', '?')}...",
    }
    return descs.get(tool_name, f"Executing {tool_name}...")
