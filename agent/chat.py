"""
Handlers for casual chat, greeting, and modify_last intents.

Casual chat uses Haiku for quick, on-brand responses.
Greeting uses templates (no LLM call).
Modify last loads the pending draft and calls Sonnet to revise it.
"""

import json
import logging
import random

import anthropic

from agent.conversation_context import ConversationContext
from agent import state
from config import settings

logger = logging.getLogger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Greeting — templated, no LLM call
# ---------------------------------------------------------------------------

_GREETING_TEMPLATES = [
    "Hey{name_part}! Ready to create some content? Just tell me what you'd like to post about.",
    "Hi{name_part}! What would you like to create today?",
    "Hello{name_part}! Send me a topic or idea and I'll draft something for you.",
    "Hey there{name_part}! I'm ready when you are. What should we work on?",
]


async def handle_greeting(user_name: str = "") -> str:
    """Return a friendly, templated greeting. No LLM call needed."""
    name_part = f" {user_name}" if user_name else ""
    template = random.choice(_GREETING_TEMPLATES)
    return template.format(name_part=name_part)


# ---------------------------------------------------------------------------
# Casual chat — Haiku for quick responses
# ---------------------------------------------------------------------------

_CASUAL_SYSTEM_PROMPT = """\
You are a friendly social media content assistant.
Keep responses brief (1-2 sentences), helpful, and on-brand.
If the user seems to want to create content, suggest they tell you a topic.
Do NOT generate social media posts or drafts — just respond conversationally.
"""


async def handle_casual_chat(message: str, context: ConversationContext) -> str:
    """Generate a brief, conversational response using Haiku."""
    try:
        from agent._client import get_anthropic
        client = get_anthropic()
        response = await client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=150,
            system=_CASUAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("Casual chat failed: %s", e)
        return "I'm here to help with content creation! Send me a topic and I'll draft something."


# ---------------------------------------------------------------------------
# Modify last — revise the pending draft with user feedback
# ---------------------------------------------------------------------------

_MODIFY_SYSTEM_PROMPT = """\
You are a social media content editor. The user has a draft post and wants changes.
Apply the requested changes to the draft while keeping the overall structure.

Return ONLY valid JSON with the modified fields:
{"caption": "...", "alt_text": "...", "image_prompt": "..."}

Only include fields that changed. Keep the tone and style consistent.
"""


async def handle_modify_last(
    feedback: str,
    context: ConversationContext,
) -> dict | None:
    """Load the pending draft, apply feedback via Sonnet, return modified draft.

    Returns the modified draft dict, or None if no pending draft exists.
    """
    pending = state.get_pending()
    if not pending:
        return None

    user_msg = (
        f"Current draft:\n"
        f"Caption: {pending.get('caption', '')}\n"
        f"Alt text: {pending.get('alt_text', '')}\n"
        f"Image prompt: {pending.get('image_prompt', '')}\n\n"
        f"Requested changes: {feedback}"
    )

    try:
        from agent._client import get_anthropic
        client = get_anthropic()
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_MODIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        changes = json.loads(raw)

        # Apply changes to a copy of pending
        modified = dict(pending)
        for key in ("caption", "alt_text", "image_prompt"):
            if key in changes:
                modified[key] = changes[key]

        return modified
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Modify last failed: %s", e)
        return None
