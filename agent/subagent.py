"""
Subagent delegation — spawn isolated sub-agents for parallel tasks.

Allows the main agent to delegate sub-tasks (research, analysis, etc.)
to a lightweight agent with its own context and tool subset. The result
is returned as a tool result to the main agent.

This is a simplified version of OpenClaw's subagent system — single-level
delegation (no recursive spawning) with a restricted tool set.
"""

import json
import logging
import time

import anthropic

from agent._client import get_anthropic
from agent.resource_log import ResourceTracker
from config import settings

logger = logging.getLogger(__name__)

# Tools available to sub-agents (restricted set — no publishing, no drafts)
_SUBAGENT_TOOLS = [
    "read_brand_guidelines",
    "read_references",
    "read_feedback_history",
    "think",
    "list_skills",
    "use_skill",
]

# Maximum turns for a sub-agent (keep it fast and cheap)
_MAX_SUBAGENT_TURNS = 5


async def delegate_task(
    task: str,
    context: str = "",
    max_turns: int = _MAX_SUBAGENT_TURNS,
    tracker: ResourceTracker | None = None,
) -> dict:
    """Run a sub-agent with a focused task and return its findings.

    Args:
        task: What the sub-agent should do (e.g., "Research competitor X's recent posts")
        context: Additional context to include in the sub-agent's prompt
        max_turns: Maximum LLM round-trips
        tracker: Optional resource tracker for cost logging

    Returns:
        {"result": str, "turns_used": int, "tools_called": list[str]}
    """
    from agent.tools import TOOL_DEFINITIONS, execute_tool

    # Filter to allowed tools only
    tools = [t for t in TOOL_DEFINITIONS if t["name"] in _SUBAGENT_TOOLS]

    system_prompt = (
        f"You are a research sub-agent for {settings.BRAND_NAME}. "
        f"Your task is to gather information and return a concise summary. "
        f"You have a limited set of tools. Work efficiently — you have at most {max_turns} turns.\n\n"
        f"When done, provide your findings as clear, structured text."
    )

    messages = [{"role": "user", "content": f"{task}\n\n{context}" if context else task}]

    if tracker is None:
        tracker = ResourceTracker()

    client = get_anthropic()
    tools_called = []
    start_time = time.time()

    for turn in range(max_turns):
        try:
            response = await client.messages.create(
                model=settings.HAIKU_MODEL,  # Sub-agents use Haiku for speed/cost
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
                tools=tools if tools else anthropic.NOT_GIVEN,
            )
        except Exception as e:
            logger.error("Sub-agent API error on turn %d: %s", turn, e)
            return {
                "result": f"Sub-agent failed: {e}",
                "turns_used": turn + 1,
                "tools_called": tools_called,
            }

        tracker.log_api(f"subagent:haiku:turn{turn}")

        # Check for end turn (no tool use)
        if response.stop_reason == "end_turn":
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            result_text = "\n".join(text_blocks)
            logger.info(
                "Sub-agent completed in %d turns (%.1fs): %s",
                turn + 1, time.time() - start_time, task[:60],
            )
            return {
                "result": result_text,
                "turns_used": turn + 1,
                "tools_called": tools_called,
            }

        # Process tool calls
        if response.stop_reason == "tool_use":
            # Add assistant message
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tools_called.append(tool_name)
                    try:
                        result = await execute_tool(tool_name, block.input, tracker)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:5000],  # Cap result size
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason — extract text and return
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return {
                "result": "\n".join(text_blocks) or "(no output)",
                "turns_used": turn + 1,
                "tools_called": tools_called,
            }

    # Hit max turns
    logger.warning("Sub-agent hit max turns (%d) for: %s", max_turns, task[:60])
    return {
        "result": "(sub-agent reached maximum turns without completing)",
        "turns_used": max_turns,
        "tools_called": tools_called,
    }
