"""
Design bridge — handles the Design Agent chat and generation pipeline.

The Design Agent is a lightweight conversational agent (Haiku) that helps
users refine a design brief before sending it to the main generation engine.
"""

import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Design Agent chat
# ---------------------------------------------------------------------------

async def design_agent_chat(
    messages: list[dict],
    brand_context: str,
    session_uploads: list[dict] | None = None,
) -> str:
    """Run the Design Agent for spec refinement.

    Uses Haiku for speed/cost. Has brand context but does NOT generate
    content itself — it helps the user articulate what they want and
    produces a structured spec.

    Args:
        messages: Conversation history [{role, content}].
        brand_context: Pre-assembled brand context string.
        session_uploads: Reference image analyses from this session.

    Returns:
        Agent's response text (may include a JSON spec block).
    """
    from agent._client import get_anthropic
    from config import settings

    system_prompt = _build_design_agent_prompt(brand_context, session_uploads)

    client = get_anthropic()
    response = await client.messages.create(
        model=settings.HAIKU_MODEL,
        max_tokens=1500,
        system=system_prompt,
        messages=messages,
    )

    return response.content[0].text


def _build_design_agent_prompt(
    brand_context: str,
    session_uploads: list[dict] | None = None,
) -> str:
    """Build the Design Agent's system prompt with brand context."""

    uploads_section = ""
    if session_uploads:
        uploads_section = "\n\n## Reference Images Uploaded This Session\n"
        for u in session_uploads:
            uploads_section += f"\n### {u.get('filename', 'uploaded image')}\n"
            colors = ", ".join(
                c.get("hex", "") for c in u.get("dominant_colors", [])
            )
            uploads_section += f"- Colors: {colors}\n"
            uploads_section += (
                f"- Style: {', '.join(u.get('style_keywords', []))}\n"
            )
            uploads_section += f"- Category: {u.get('category', 'unknown')}\n"
            uploads_section += (
                f"- Brand alignment: {u.get('brand_alignment', 'unknown')}\n"
            )

    return f"""You are the Design Agent -- a creative director helping the user refine their design brief before it goes to the generation bot.

## Your Role
- Help the user articulate what they want visually
- Suggest content types, templates, color choices, and layouts based on their brand
- Ask clarifying questions when the intent is vague
- When the user is ready, produce a structured design spec

## What You Know
{brand_context}
{uploads_section}

## Rules
- You do NOT generate content. You help PLAN what to generate.
- Keep responses short and conversational
- When suggesting colors, reference the brand palette by name
- When suggesting templates, reference available templates by name
- Proactively suggest content types and layouts that fit the user's intent

## Design Spec Format
When the user says they're ready (e.g. "send it", "generate", "looks good", "go"), respond with a structured spec in a JSON block:

```json
{{
  "content_type": "announcement",
  "template_id": null,
  "title": "TITLE HERE",
  "subtitle": "subtitle here",
  "caption_guidance": "what the post should convey",
  "image_prompt": "detailed image generation prompt",
  "color_overrides": null,
  "layout_preset": "16:9",
  "style_notes": "any additional style guidance"
}}
```

Only output the spec when the user indicates they're ready. Before that, help them think through what they want."""


# ---------------------------------------------------------------------------
# Spec -> agent request
# ---------------------------------------------------------------------------

async def build_agent_request(spec: dict) -> str:
    """Convert a design spec dict into an agent request string."""
    parts: list[str] = []

    if spec.get("content_type"):
        parts.append(f"[CONTENT_TYPE: {spec['content_type']}]")
    if spec.get("template_id"):
        parts.append(f"[TEMPLATE: {spec['template_id']}]")
    if spec.get("title"):
        parts.append(f"[TITLE: {spec['title']}]")
    if spec.get("subtitle"):
        parts.append(f"[SUBTITLE: {spec['subtitle']}]")
    if spec.get("image_prompt"):
        parts.append(f"[IMAGE_PROMPT: {spec['image_prompt']}]")
    if spec.get("layout_preset"):
        parts.append(f"[LAYOUT: {spec['layout_preset']}]")
    if spec.get("caption_guidance"):
        parts.append(spec["caption_guidance"])
    if spec.get("style_notes"):
        parts.append(f"[STYLE: {spec['style_notes']}]")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation pipeline (SSE)
# ---------------------------------------------------------------------------

async def run_design_generation(
    spec: dict,
    user_id: int,
) -> AsyncIterator[dict]:
    """Run generation from a design spec. Yields SSE event dicts."""
    import asyncio
    from agent import engine

    request = await build_agent_request(spec)

    yield {
        "type": "progress",
        "step": "starting",
        "message": "Building design spec...",
    }

    # Use a queue so tool call callbacks can push progress events
    progress_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def on_tool_call(tool_name: str, description: str):
        await progress_queue.put({
            "type": "progress",
            "step": tool_name,
            "message": description,
        })

    async def on_reasoning(text: str):
        # Truncate reasoning to a short line
        short = text.replace("\n", " ").strip()[:80]
        if short:
            await progress_queue.put({
                "type": "progress",
                "step": "thinking",
                "message": short,
            })

    try:
        # Run the agent in a background task so we can yield progress
        agent_task = asyncio.create_task(
            engine.run_agent(
                request=request,
                on_tool_call=on_tool_call,
                on_reasoning=on_reasoning,
            )
        )

        # Yield progress events until the agent finishes
        while not agent_task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue

        # Drain remaining progress events
        while not progress_queue.empty():
            yield progress_queue.get_nowait()

        result = agent_task.result()

        if result.draft:
            yield {
                "type": "result",
                "draft": result.draft,
                "image_url": result.image_url,
                "image_urls": result.image_urls,
                "turns": result.turns_used,
                "time": result.total_time,
            }
        else:
            yield {
                "type": "result",
                "text": result.final_text or "No draft generated.",
                "draft": None,
            }
    except Exception as e:
        logger.error("Design generation failed: %s", e)
        yield {"type": "error", "message": str(e)}
