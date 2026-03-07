"""
Handlers for casual chat, greeting, and modify_last intents.

Casual chat uses Sonnet with personality, brand context, conversation history,
and persistent memory for natural, on-brand responses.
Greeting uses the personality-aware chat path (or templates as fallback).
Modify last loads the pending draft and calls Sonnet to revise it.
"""

import json
import logging
import random
from pathlib import Path

import anthropic

from agent.conversation_context import ConversationContext
from agent import state
from config import settings

logger = logging.getLogger(__name__)

_SONNET_MODEL = "claude-sonnet-4-6"

_project_root = Path(__file__).resolve().parent.parent
_PERSONALITY_DIR = _project_root / "brand" / "personality"
_PERSONALITY_FILE = _PERSONALITY_DIR / "system_prompt.md"
_MEMORY_FILE = _PERSONALITY_DIR / "memory.md"

# Cache personality + memory with mtime-based invalidation
_personality_cache: str | None = None
_personality_mtime: float = 0
_memory_cache: str | None = None
_memory_mtime: float = 0


def _load_personality() -> str:
    """Load personality from brand/personality/system_prompt.md with mtime caching."""
    global _personality_cache, _personality_mtime
    if not _PERSONALITY_FILE.exists():
        return ""
    mtime = _PERSONALITY_FILE.stat().st_mtime
    if _personality_cache is not None and mtime == _personality_mtime:
        return _personality_cache
    _personality_cache = _PERSONALITY_FILE.read_text(encoding="utf-8").strip()
    _personality_mtime = mtime
    logger.info("Loaded personality: %d chars", len(_personality_cache))
    return _personality_cache


def _load_memory() -> str:
    """Load persistent memory from brand/personality/memory.md with mtime caching."""
    global _memory_cache, _memory_mtime
    if not _MEMORY_FILE.exists():
        return ""
    mtime = _MEMORY_FILE.stat().st_mtime
    if _memory_cache is not None and mtime == _memory_mtime:
        return _memory_cache
    _memory_cache = _MEMORY_FILE.read_text(encoding="utf-8").strip()
    _memory_mtime = mtime
    logger.info("Loaded memory: %d chars", len(_memory_cache))
    return _memory_cache


def _load_voice_rules() -> str:
    """Extract voice & tone section from brand guidelines for chat context."""
    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
    if not guidelines_path.exists():
        return ""
    text = guidelines_path.read_text(encoding="utf-8")
    # Extract VOICE & TONE section
    sections = []
    for section_name in ("VOICE & TONE", "BRAND PHRASES & SLANG", "NEVER DO"):
        start = text.find(f"## {section_name}")
        if start == -1:
            continue
        end = text.find("\n## ", start + 1)
        chunk = text[start:end] if end != -1 else text[start:]
        sections.append(chunk.strip())
    return "\n\n".join(sections)


def _build_chat_system_prompt(context: ConversationContext) -> str:
    """Build the full system prompt for conversational responses."""
    parts = []

    # Base identity
    parts.append(
        f"You are the conversational assistant for {settings.BRAND_NAME}. "
        f"You handle casual conversation, answer questions, and help the user "
        f"with their brand's social media presence."
    )

    # Personality (from brand/personality/system_prompt.md)
    personality = _load_personality()
    if personality:
        parts.append(f"--- YOUR PERSONALITY ---\n{personality}")

    # Brand voice rules (from guidelines.md)
    voice = _load_voice_rules()
    if voice:
        parts.append(f"--- BRAND VOICE (apply to your own speech too) ---\n{voice}")

    # Persistent memory
    memory = _load_memory()
    if memory:
        parts.append(f"--- MEMORY (things you remember) ---\n{memory}")

    # User context
    if context.user_name:
        parts.append(f"The user's name is {context.user_name}.")

    # Behavioral rules
    parts.append(
        "RULES:\n"
        "- Keep responses brief (1-3 sentences) unless the user asks for detail.\n"
        "- Match the brand's voice and tone in how YOU speak, not just in content you generate.\n"
        "- If the user wants to create content, tell them to describe what they want.\n"
        "- Do NOT generate social media posts or drafts in chat — that happens through the content pipeline.\n"
        "- Be natural. Sound like a person, not a bot.\n"
        "- Never start with \"I'd be happy to help\" or similar AI cliches."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Greeting — personality-aware (falls back to templates if no personality)
# ---------------------------------------------------------------------------

_GREETING_TEMPLATES = [
    "Hey{name_part}! Ready to create some content? Just tell me what you'd like to post about.",
    "Hi{name_part}! What would you like to create today?",
    "Hello{name_part}! Send me a topic or idea and I'll draft something for you.",
    "Hey there{name_part}! I'm ready when you are. What should we work on?",
]


async def handle_greeting(user_name: str = "", context: ConversationContext | None = None) -> str:
    """Return a greeting. Uses personality if available, otherwise templates."""
    personality = _load_personality()
    if personality and context is not None:
        # Use the conversational path for a personality-driven greeting
        return await handle_casual_chat(
            f"{'Hi!' if not user_name else f'Hi, my name is {user_name}!'}",
            context,
        )
    # Fallback to templates
    name_part = f" {user_name}" if user_name else ""
    template = random.choice(_GREETING_TEMPLATES)
    return template.format(name_part=name_part)


# ---------------------------------------------------------------------------
# Casual chat — Sonnet with personality, memory, and conversation history
# ---------------------------------------------------------------------------


async def handle_casual_chat(message: str, context: ConversationContext) -> str:
    """Generate a conversational response with personality and memory."""
    try:
        from agent._client import get_anthropic
        client = get_anthropic()

        system_prompt = _build_chat_system_prompt(context)

        # Build messages with conversation history
        messages = []
        for turn in context.conversation_history:
            messages.append({
                "role": turn["role"],
                "content": turn["content"],
            })
        messages.append({"role": "user", "content": message})

        response = await client.messages.create(
            model=_SONNET_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=messages,
        )
        reply = response.content[0].text.strip()

        # Update conversation history on the context (caller persists it)
        context.conversation_history.append({"role": "user", "content": message})
        context.conversation_history.append({"role": "assistant", "content": reply})

        return reply
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
    user_id: int | None = None,
) -> dict | None:
    """Load the pending draft, apply feedback via Sonnet, return modified draft.

    Returns the modified draft dict, or None if no pending draft exists.
    """
    pending = state.get_pending(user_id=user_id)
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
            model=_SONNET_MODEL,
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
