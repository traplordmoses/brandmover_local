"""
Core agent engine — Claude tool-use loop.
Calls Claude with tools, executes tool calls, feeds results back, repeats until done.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import anthropic

from agent.resource_log import ResourceTracker
from agent.skill_prompt import build_system_prompt
from agent.tools import TOOL_DEFINITIONS, execute_tool
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of an agent run."""
    final_text: str = ""
    draft: dict = field(default_factory=dict)
    image_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    resources: ResourceTracker = field(default_factory=ResourceTracker)
    tool_calls_made: list[str] = field(default_factory=list)
    turns_used: int = 0
    total_time: float = 0.0


def _try_parse_draft(text: str) -> dict | None:
    """
    Try to extract a JSON draft from free-form text.
    Looks for a JSON object containing caption, hashtags, alt_text, image_prompt.
    """
    # Try to find JSON in markdown fences first
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if "caption" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object with the required keys
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if "caption" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    return None


# AI-sounding words to strip from captions (case-insensitive)
_AI_WORDS = re.compile(
    r"\b(?:revolutionizing|leveraging|cutting-edge|seamlessly|dive into|unlock)\b",
    re.IGNORECASE,
)

# Hashtag pattern: # followed by word chars (but not hex color codes like #000000 in image_prompt)
_HASHTAG_RE = re.compile(r"#[A-Za-z]\w*")

_DRAFT_TEXT_FIELDS = ("caption", "title", "subtitle")


def _sanitize_draft(draft: dict) -> dict:
    """Post-process a parsed draft to enforce hard compliance rules.

    Strips hashtags and AI-sounding words from user-facing text fields.
    Returns the (possibly modified) draft. No-op for compliant drafts.
    """
    for field in _DRAFT_TEXT_FIELDS:
        original = draft.get(field)
        if not original or not isinstance(original, str):
            continue

        cleaned = original

        # Strip hashtags (e.g. #brand, #promo) but not hex colors (#AABBCC)
        hashtags = _HASHTAG_RE.findall(cleaned)
        if hashtags:
            cleaned = _HASHTAG_RE.sub("", cleaned)
            logger.warning("Sanitized %d hashtag(s) from draft.%s: %s", len(hashtags), field, hashtags)

        # Strip AI-sounding words
        ai_matches = _AI_WORDS.findall(cleaned)
        if ai_matches:
            cleaned = _AI_WORDS.sub("", cleaned)
            logger.warning("Sanitized AI word(s) from draft.%s: %s", field, ai_matches)

        # Collapse double spaces and strip
        if cleaned != original:
            cleaned = re.sub(r"  +", " ", cleaned).strip()
            draft[field] = cleaned

    return draft


def _extract_image_url(tool_calls_made: list[dict]) -> str | None:
    """Extract image URL from generate_image tool results."""
    for call in tool_calls_made:
        if call.get("name") in ("generate_image", "img2img"):
            # Check the pre-extracted URL first (set during execution)
            if call.get("image_url"):
                return call["image_url"]
            # Fallback: try to parse from result string
            result_str = call.get("result", "")
            try:
                result = json.loads(result_str)
                if "image_url" in result:
                    return result["image_url"]
            except (json.JSONDecodeError, TypeError):
                pass
            # Last resort: regex for URL in the result string
            url_match = re.search(r'https://[^\s"\']+(?:replicate\.delivery|pbxt\.replicate\.com)[^\s"\']*', result_str)
            if url_match:
                return url_match.group()
    return None


def _extract_image_urls(tool_calls_made: list[dict]) -> list[str]:
    """Extract all image URLs (including parallel options) from tool results."""
    urls: list[str] = []
    for call in tool_calls_made:
        if call.get("name") in ("generate_image", "img2img"):
            result_str = call.get("result", "")
            try:
                result = json.loads(result_str)
                # Check for image_urls array (parallel generation)
                if "image_urls" in result:
                    urls.extend(result["image_urls"])
                    continue
                # Single image_url
                if "image_url" in result:
                    urls.append(result["image_url"])
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            # Pre-extracted URL
            if call.get("image_url"):
                urls.append(call["image_url"])
    return urls


# Type for the on_tool_call callback
OnToolCall = Callable[[str, str], Awaitable[None]]


async def run_agent(
    request: str,
    on_tool_call: OnToolCall | None = None,
    revision_context: str | None = None,
) -> AgentResult:
    """
    Run the agent loop for a content request.

    Args:
        request: User's content request.
        on_tool_call: Optional async callback(tool_name, brief_description) for progress updates.
        revision_context: Optional context about a previous draft + feedback for revisions.

    Returns:
        AgentResult with the final draft and metadata.
    """
    t_start = time.time()
    result = AgentResult()
    tracker = ResourceTracker()
    result.resources = tracker

    from agent._client import get_anthropic
    client = get_anthropic()

    system_prompt = build_system_prompt()

    # Build the initial user message
    user_content = request
    if revision_context:
        user_content = f"{revision_context}\n\nNew request: {request}"

    messages = [{"role": "user", "content": user_content}]

    max_turns = settings.AGENT_MAX_TURNS
    tool_call_log = []  # For image URL extraction

    for turn in range(max_turns):
        result.turns_used = turn + 1

        # Force no tools on the last turn to get a final answer
        tool_choice = (
            {"type": "any"} if turn == 0  # force at least one tool call first turn
            else {"type": "none"} if turn >= max_turns - 1
            else {"type": "auto"}
        )

        try:
            response = await client.messages.create(
                model=settings.AGENT_MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOL_DEFINITIONS,
                tool_choice=tool_choice,
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error("Anthropic API error on turn %d: %s", turn + 1, e)
            result.final_text = f"API error: {e}"
            break

        # Process the response content blocks
        assistant_content = response.content
        tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
        text_blocks = [b for b in assistant_content if b.type == "text"]

        # Collect any text output
        for tb in text_blocks:
            result.final_text += tb.text + "\n"

        # If no tool calls, we're done
        if not tool_use_blocks or response.stop_reason == "end_turn":
            logger.info("Agent finished after %d turns (stop_reason=%s)", turn + 1, response.stop_reason)
            break

        # Append the assistant message
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute each tool call and build tool results
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block.name
            tool_input = tool_block.input
            result.tool_calls_made.append(tool_name)

            logger.info("Agent calling tool: %s (input: %s)", tool_name, str(tool_input)[:200])

            # Send progress callback
            if on_tool_call:
                brief = _tool_description(tool_name, tool_input)
                await on_tool_call(tool_name, brief)

            # Execute the tool
            try:
                tool_result = await execute_tool(tool_name, tool_input, tracker)
                # Truncate very long results to avoid context overflow
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

            # Pre-extract image URL immediately when generate_image/img2img succeeds
            if tool_name in ("generate_image", "img2img"):
                try:
                    parsed = json.loads(tool_result)
                    if "image_url" in parsed:
                        log_entry["image_url"] = parsed["image_url"]
                        logger.info("Extracted image URL from tool result: %s", parsed["image_url"][:120])
                except (json.JSONDecodeError, TypeError):
                    pass

            tool_call_log.append(log_entry)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": tool_result,
            })

        # Append tool results as user message
        messages.append({"role": "user", "content": tool_results})

    # Post-processing
    result.final_text = result.final_text.strip()
    result.total_time = round(time.time() - t_start, 1)

    # Try to parse a draft from the final text
    draft = _try_parse_draft(result.final_text)
    if draft:
        result.draft = _sanitize_draft(draft)

    # Backfill content_type from generate_image tool call if missing from draft
    if result.draft and not result.draft.get("content_type"):
        for entry in reversed(tool_call_log):
            if entry["name"] == "generate_image" and isinstance(entry.get("input"), dict):
                ct = entry["input"].get("content_type")
                if ct:
                    result.draft["content_type"] = ct
                    break

    # Extract image URL from tool calls
    result.image_url = _extract_image_url(tool_call_log)
    result.image_urls = _extract_image_urls(tool_call_log)

    logger.info(
        "Agent run complete: %d turns, %.1fs, %d tool calls, draft=%s, image=%s",
        result.turns_used,
        result.total_time,
        len(result.tool_calls_made),
        bool(result.draft),
        bool(result.image_url),
    )

    return result


def _tool_description(tool_name: str, tool_input: dict) -> str:
    """Generate a brief human-readable description of a tool call."""
    descs = {
        "read_brand_guidelines": "Loading brand guidelines and references...",
        "read_references": "Checking available reference materials...",
        "check_figma_design": f"Checking Figma design ({tool_input.get('action', 'styles')})...",
        "generate_image": "Generating brand image...",
        "read_feedback_history": "Reviewing feedback history...",
        "log_resource_usage": "Logging resources used...",
        "img2img": f"Generating image from reference: {tool_input.get('reference_image_path', 'auto')}...",
        "execute_openclaw_script": f"Running {tool_input.get('script_name', 'script')}...",
    }
    return descs.get(tool_name, f"Executing {tool_name}...")
