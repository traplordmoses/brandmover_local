"""
Unified system prompt builder — combines personality, memory, brand voice,
learned preferences, current state, and generation rules into a single prompt.

Used by unified_brain.py so every message (chat or generation) goes through
one LLM call with full context.
"""

import logging
from pathlib import Path

from agent.chat import _load_personality, _load_memory, _load_voice_rules
from agent.skill_prompt import (
    _get_platform_block,
    _get_platform_json_line,
    _get_image_mode_block,
    _get_content_types_block,
)
from agent.conversation_context import ConversationContext
from agent import feedback, self_review, session_plan, state
from config import settings

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_PREFERENCES_FILE = _project_root / "state" / "learned_preferences.md"


def _get_learned_preferences() -> str:
    """Load condensed learned preferences from state/learned_preferences.md."""
    if not _PREFERENCES_FILE.exists():
        return ""
    try:
        text = _PREFERENCES_FILE.read_text(encoding="utf-8").strip()
        return text[:2000]  # cap to avoid prompt bloat
    except OSError:
        return ""


def _get_recent_feedback(n: int = 3) -> str:
    """Return the last N feedback entries as a compact string."""
    ctx = feedback.get_feedback_context()
    if ctx == "No feedback history yet.":
        return ""
    return ctx


def _get_state_context(context: ConversationContext, user_id: int | None = None) -> str:
    """Return current state info: pending draft, schedule status."""
    parts = []

    # Pending draft
    pending = state.get_pending(user_id=user_id)
    if pending:
        caption = pending.get("caption", "")[:100]
        ct = pending.get("content_type", "unknown")
        revision = state.get_draft_revision_count(user_id=user_id)
        parts.append(
            f"PENDING DRAFT (rev {revision}): content_type={ct}, "
            f"caption preview: \"{caption}...\"\n"
            f"The user can approve, reject with feedback, or ask you something else. "
            f"Do NOT block conversation just because a draft is pending."
        )
    else:
        parts.append("No pending draft.")

    # Session plan
    plan_summary = session_plan.get_plan_summary()
    if plan_summary:
        parts.append(plan_summary)

    return "\n".join(parts)


def _get_generation_rules() -> str:
    """Return condensed generation rules from skill_prompt building blocks."""
    platform_block = _get_platform_block()
    platform_json_line = _get_platform_json_line()
    image_mode_block = _get_image_mode_block()
    content_types_block = _get_content_types_block()

    platform_field = f",\n{platform_json_line}" if platform_json_line else ""

    return f"""When you decide to generate content, follow these rules:

## WORKFLOW
1. Call `read_brand_guidelines` to load brand context (voice, tone, colors, visual style). ALWAYS do this first when generating.
2. Call `read_feedback_history` to check past approvals/rejections.
3. Optionally call `check_figma_design` for design precision.
4. Craft the draft: caption (<280 chars for X), alt text, detailed image prompt.
5. Call `generate_image` with your prompt.
6. Call `log_resource_usage` to record what you consulted.
7. Output final draft as a JSON block:

```json
{{
  "caption": "The post caption text",
  "alt_text": "Accessible image description",
  "image_prompt": "The prompt used for image generation",
  "content_type": "announcement",
  "title": "UPPERCASE HEADLINE",
  "subtitle": "Brief explanation"{platform_field}
}}
```

{platform_block}
{image_mode_block}
CONTENT TYPES (pick the best fit):
{content_types_block}

## HARD RULES
1. ZERO HASHTAGS — No #word in caption, title, or subtitle. Ever.
2. NO AI WORDS — Never use: "revolutionizing", "leveraging", "cutting-edge", "seamlessly", "dive into", "unlock".
3. MAX 1 EMOJI — One emoji max. Zero is fine. Never start with an emoji.
4. CAPTION LENGTH — 50-150 chars for most posts. Shorter is better.
5. Sound HUMAN. Be punchy and confident. No passive voice, no corporate jargon.

## IMAGE PROMPT (SPLICE framework)
1. Subject — specific main subject
2. Parameters — style, medium
3. Lighting — how is it lit
4. Image Type — photo, illustration, 3D render
5. Composition — camera angle, framing
6. Enhancers — quality terms (8K, ultra-detailed)

Keep prompts 40-80 words. Front-load important elements.

## REVISION MODE
When revising a rejected draft, you'll receive the previous draft and feedback in the conversation. Focus on addressing the specific feedback."""


def build_unified_system_prompt(
    context: ConversationContext,
    user_id: int | None = None,
) -> str:
    """Build the unified system prompt combining personality + generation capabilities."""
    parts = []

    # 1. Personality (dominates tone)
    personality = _load_personality()
    if personality:
        parts.append(personality)
    else:
        parts.append(
            f"You are the AI assistant for {settings.BRAND_NAME}. "
            f"You help with casual conversation and generate social media content."
        )

    # 2. Brand voice rules
    voice = _load_voice_rules()
    if voice:
        parts.append(f"--- BRAND VOICE (apply to your own speech) ---\n{voice}")

    # 3. Persistent memory
    memory = _load_memory()
    if memory:
        parts.append(f"--- MEMORY ---\n{memory}")

    # 4. Learned preferences
    prefs = _get_learned_preferences()
    if prefs:
        parts.append(f"--- LEARNED PREFERENCES ---\n{prefs}")

    # 4b. Self-review summary
    review_summary = self_review.get_last_review_summary()
    if review_summary:
        parts.append(review_summary)

    # 5. Recent feedback
    fb = _get_recent_feedback()
    if fb:
        parts.append(fb)

    # 6. Current state
    state_ctx = _get_state_context(context, user_id=user_id)
    parts.append(f"--- CURRENT STATE ---\n{state_ctx}")

    # 7. User context
    if context.user_name:
        parts.append(f"The user's name is {context.user_name}.")

    # 8. Capabilities
    parts.append(
        "--- CAPABILITIES ---\n"
        "You can do TWO things:\n"
        "1. CHAT — Have natural conversations. Be brief (1-3 sentences unless asked for detail). "
        "Sound like a person, not a bot. Never start with \"I'd be happy to help\".\n"
        "2. GENERATE — Create social media post drafts with images using your tools. "
        "When you decide the user wants content, use your tools and output a JSON draft block.\n\n"
        "You decide which to do based on the message. You can also chat AND generate in the same turn — "
        "e.g. comment on the request in your voice, then call tools and output a draft.\n\n"
        "If a pending draft exists, you can still chat about other topics. "
        "The user can approve/reject the draft separately.\n\n"
        "SESSION PLANS: When you have an active plan and the operator approves a draft, "
        "acknowledge it and naturally propose moving to the next plan item. Don't auto-generate — "
        "ask if they want you to proceed or adjust the angle first. "
        "Plans are optional — most requests are standalone, not part of a plan.\n\n"
        "AUTONOMOUS MODE: If the operator tells you to cook everything, work through the plan, "
        "or says they'll review later, use start_autonomous_plan. This generates all remaining "
        "plan items in sequence and queues drafts for review. When done, the operator can say "
        "'show me #1' to review each draft at their own pace. Use show_queued_draft to load "
        "a specific draft into the pending state for approve/reject."
    )

    # 9. Generation rules (present but not forced)
    parts.append(f"--- GENERATION RULES (when you generate content) ---\n{_get_generation_rules()}")

    return "\n\n".join(parts)
