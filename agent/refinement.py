"""
Artifact-scoped refinement mode -- focused editing with lighter context.

Instead of loading full brand context for every edit, refinement mode
scopes the agent to a single piece of content and provides only the
context relevant to that specific artifact.
"""

import json
import logging
import time

from agent.engine import AgentResult, run_agent, run_agent_with_history, OnToolCall, OnReasoning

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Refinement presets — shorthand flags for common refinement operations
# ---------------------------------------------------------------------------

REFINEMENT_PRESETS = {
    "--shorter": "Make the caption significantly shorter. Cut to under 100 characters. Remove filler words.",
    "--longer": "Expand the caption with more detail. Add context and a stronger narrative.",
    "--playful": "Rewrite with a more playful, fun, lighthearted tone. Add personality.",
    "--professional": "Rewrite with a more professional, authoritative tone.",
    "--urgent": "Rewrite with more urgency and a stronger call to action.",
    "--add-cta": "Add a clear call to action at the end (e.g., 'Try it now', 'Learn more', 'Join us').",
    "--add-emoji": "Add 1 well-placed emoji that enhances the message.",
    "--remove-emoji": "Remove all emojis from the caption.",
    "--punchy": "Make every sentence hit harder. Shorter. Punchier. No filler.",
}

# Also accept --tone=<value> syntax
_TONE_PRESETS = {
    "playful": REFINEMENT_PRESETS["--playful"],
    "professional": REFINEMENT_PRESETS["--professional"],
    "urgent": REFINEMENT_PRESETS["--urgent"],
    "casual": "Rewrite with a more casual, conversational tone. Like talking to a friend.",
    "formal": "Rewrite with a more formal, polished tone.",
}


def parse_preset_flags(text: str) -> tuple[str, str]:
    """Parse preset flags from a refinement instruction.

    Extracts flags like --shorter, --punchy, --tone=playful from the text
    and returns (remaining_text, combined_instructions).

    Multiple presets are combined. If no presets found, returns original text
    with empty instructions.

    Returns:
        (remaining_text, preset_instructions) — preset_instructions is empty
        if no presets were found.
    """
    import re

    instructions: list[str] = []
    remaining = text

    # Match --tone=<value> first
    tone_matches = re.findall(r"--tone=(\w+)", remaining)
    for tone_val in tone_matches:
        preset_text = _TONE_PRESETS.get(tone_val.lower())
        if preset_text:
            instructions.append(preset_text)
        remaining = remaining.replace(f"--tone={tone_val}", "").strip()

    # Match simple flags like --shorter, --punchy
    for flag, instruction in REFINEMENT_PRESETS.items():
        if flag in remaining:
            instructions.append(instruction)
            remaining = remaining.replace(flag, "").strip()

    remaining = remaining.strip()
    combined = " ".join(instructions)
    return remaining, combined


# Tools allowed during refinement -- no image gen, no web search, no scripts.
# Keeps the loop fast and focused on text editing.
_REFINEMENT_EXCLUDED_TOOLS = {
    "generate_image",
    "img2img",
    "execute_openclaw_script",
    "check_figma_design",
    "log_resource_usage",
}


def _get_voice_context() -> str:
    """Load minimal brand voice context (voice_traits + avoid_terms only)."""
    try:
        from agent.compositor_config import get_config
        cfg = get_config()
        parts = []
        if cfg.voice_traits:
            parts.append("Voice traits: " + ", ".join(cfg.voice_traits))
        if cfg.avoid_terms:
            parts.append("Avoid terms: " + ", ".join(cfg.avoid_terms))
        if cfg.brand_phrases:
            parts.append("Brand phrases: " + ", ".join(cfg.brand_phrases))
        return "\n".join(parts)
    except Exception as e:
        logger.debug("Failed to load voice context for refinement: %s", e)
        return ""


def _format_artifact(artifact: dict) -> str:
    """Format an artifact dict into a readable block for the prompt."""
    lines = []
    for key in ("caption", "content_type", "title", "subtitle", "alt_text", "image_prompt"):
        val = artifact.get(key)
        if val:
            lines.append(f"  {key}: {val}")
    # Include format-specific data
    if artifact.get("format") and artifact["format"] != "single":
        lines.append(f"  format: {artifact['format']}")
    if artifact.get("thread_posts"):
        for i, post in enumerate(artifact["thread_posts"]):
            lines.append(f"  thread_post[{i}]: {post.get('text', '')}")
    return "\n".join(lines)


def build_refinement_prompt(artifact: dict, instruction: str) -> str:
    """Build a focused system prompt for refining a single artifact.

    Includes the artifact content, minimal brand voice context, and the
    user's edit instruction. Much lighter than the full system prompt.

    Args:
        artifact: Draft dict with caption, image_prompt, content_type, etc.
        instruction: The user's specific edit instruction.

    Returns:
        Formatted prompt string for the refinement agent run.
    """
    from config import settings

    voice_ctx = _get_voice_context()

    prompt_parts = [
        f"You are refining a single piece of content for {settings.BRAND_NAME}.",
        "",
        "## CURRENT DRAFT",
        "",
        _format_artifact(artifact),
        "",
    ]

    if voice_ctx:
        prompt_parts.extend([
            "## BRAND VOICE (summary)",
            "",
            voice_ctx,
            "",
        ])

    prompt_parts.extend([
        "## INSTRUCTION",
        "",
        instruction,
        "",
        "## RULES",
        "",
        "- Apply the instruction to the current draft.",
        "- Preserve everything not mentioned in the instruction.",
        "- No hashtags. No AI buzzwords. No em-dashes.",
        "- Caption must stay under 280 chars.",
        "- Submit the refined draft via the finish tool with all fields.",
    ])

    return "\n".join(prompt_parts)


async def refine_artifact(
    artifact: dict,
    instruction: str,
    history: list | None = None,
    on_tool_call: OnToolCall | None = None,
    on_reasoning: OnReasoning | None = None,
) -> AgentResult:
    """Refine a single artifact with a focused edit instruction.

    Uses a lighter context window than a full agent run -- no full brand
    guidelines, no session context, limited tools.

    Args:
        artifact: Draft dict (from pending state or recent post).
        instruction: The user's edit instruction (e.g. "make it shorter").
        history: If provided, continues the conversation (appends instruction).
        on_tool_call: Optional progress callback.
        on_reasoning: Optional reasoning callback.

    Returns:
        AgentResult with the refined draft.
    """
    t_start = time.time()

    if history:
        # Continue from prior conversation -- append refinement instruction
        history.append({
            "role": "user",
            "content": (
                f"Refine the current draft. Do NOT reject it, just apply this edit:\n\n"
                f"\"{instruction}\"\n\n"
                f"Use think to plan the edit, then submit the updated draft via finish."
            ),
        })
        result = await run_agent_with_history(
            history,
            on_tool_call=on_tool_call,
            on_reasoning=on_reasoning,
            excluded_tools=_REFINEMENT_EXCLUDED_TOOLS,
        )
    else:
        # No history -- build a focused revision context from the artifact
        revision_context = build_refinement_prompt(artifact, instruction)
        original_request = artifact.get("original_request", "")
        # Use the instruction as the request so the agent focuses on it
        result = await run_agent(
            request=instruction,
            on_tool_call=on_tool_call,
            on_reasoning=on_reasoning,
            revision_context=revision_context,
            excluded_tools=_REFINEMENT_EXCLUDED_TOOLS,
        )

    logger.info(
        "Refinement complete: %d turns, %.1fs, draft=%s",
        result.turns_used,
        time.time() - t_start,
        bool(result.draft),
    )
    return result


async def batch_refine(
    artifacts: list[dict],
    instruction: str,
    on_tool_call: OnToolCall | None = None,
    on_reasoning: OnReasoning | None = None,
) -> list[AgentResult]:
    """Apply the same instruction across multiple artifacts sequentially.

    E.g. "make all posts more casual". Runs one at a time to avoid rate limits.

    Args:
        artifacts: List of draft dicts to refine.
        instruction: Edit instruction to apply to each.
        on_tool_call: Optional progress callback.
        on_reasoning: Optional reasoning callback.

    Returns:
        List of AgentResults, one per artifact.
    """
    results = []
    for artifact in artifacts:
        result = await refine_artifact(
            artifact, instruction,
            on_tool_call=on_tool_call,
            on_reasoning=on_reasoning,
        )
        results.append(result)
    return results


def extract_artifact_from_pending(user_id: int | None = None) -> dict | None:
    """Load the current pending draft and return it as an artifact dict.

    Includes conversation_history if available for continuity.

    Args:
        user_id: Telegram user ID. None defaults to admin.

    Returns:
        Artifact dict or None if no pending draft.
    """
    from agent import state
    pending = state.get_pending(user_id=user_id)
    if not pending:
        return None
    return dict(pending)


def extract_artifact_from_recent(index: int = 0) -> dict | None:
    """Load a recent post from session memory by index.

    Args:
        index: 0 = most recent, 1 = second most recent, etc.

    Returns:
        Artifact dict (without conversation_history) or None.
    """
    from agent.session import load_session
    session = load_session()
    if not session.recent_posts:
        return None
    # recent_posts is oldest-first; reverse for newest-first indexing
    reversed_posts = list(reversed(session.recent_posts))
    if index >= len(reversed_posts):
        return None
    return dict(reversed_posts[index])
