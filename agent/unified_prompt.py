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


def _get_recent_feedback() -> str:
    """Return recent feedback entries (without learned preferences, which are loaded separately)."""
    ctx = feedback.get_feedback_context()
    if ctx == "No feedback history yet.":
        return ""
    # Strip out the LEARNED PREFERENCES section — it's already included
    # separately via _get_learned_preferences(). Only keep RECENT FEEDBACK.
    marker = "--- RECENT FEEDBACK"
    idx = ctx.find(marker)
    if idx != -1:
        return ctx[idx:]
    # If no RECENT FEEDBACK section found, the context is just preferences — skip it
    if "LEARNED PREFERENCES" in ctx:
        return ""
    return ctx


def _get_state_context(context: ConversationContext, user_id: int | None = None) -> str:
    """Return current state info: approved draft, pending draft, schedule status."""
    parts = []

    # Approved draft (approved but not yet posted)
    approved = state.get_approved(user_id=user_id)
    if approved:
        caption = approved.get("caption", "")[:100]
        parts.append(
            f"APPROVED DRAFT awaiting post/schedule: \"{caption}...\"\n"
            f"Use post_approved to publish now, or schedule_post to schedule."
        )

    # Pending draft
    pending = state.get_pending(user_id=user_id)
    if pending:
        caption = pending.get("caption", "")[:100]
        ct = pending.get("content_type", "unknown")
        revision = state.get_draft_revision_count(user_id=user_id)
        parts.append(
            f"PENDING DRAFT (rev {revision}): content_type={ct}, "
            f"caption preview: \"{caption}...\""
        )

    if not approved and not pending:
        parts.append("No pending or approved drafts.")

    # Reference image
    ref = state.get_reference_image()
    if ref:
        parts.append(f"Reference image loaded: {Path(ref).name}")

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

## GENERATION STEPS
1. Call `read_brand_guidelines` to load brand context. ALWAYS do this first.
2. Call `read_feedback_history` to check past approvals/rejections.
3. Optionally call `check_figma_design` for design precision.
4. Craft the draft: caption (<280 chars for X), alt text, detailed image prompt.
5. Call `generate_image` (or `img2img` if a reference image is loaded).
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
Subject → Parameters → Lighting → Image Type → Composition → Enhancers.
Keep prompts 40-80 words. Front-load important elements."""


def build_unified_system_prompt(
    context: ConversationContext,
    user_id: int | None = None,
) -> str:
    """Build the unified system prompt combining personality + generation capabilities."""
    parts = []

    # 1. Personality (dominates tone — always first)
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

    # 4. Learned preferences (from feedback summarization)
    prefs = _get_learned_preferences()
    if prefs:
        parts.append(f"--- LEARNED PREFERENCES ---\n{prefs}")

    # 4b. Self-review summary
    review_summary = self_review.get_last_review_summary()
    if review_summary:
        parts.append(review_summary)

    # 5. Recent feedback entries (preferences already loaded above — this is just entries)
    fb = _get_recent_feedback()
    if fb:
        parts.append(fb)

    # 6. Current state
    state_ctx = _get_state_context(context, user_id=user_id)
    parts.append(f"--- CURRENT STATE ---\n{state_ctx}")

    # 7. User context
    if context.user_name:
        parts.append(f"The user's name is {context.user_name}.")

    # 8. Capabilities + tool guidance
    parts.append(_build_capabilities_section())

    # 9. Common workflows (multi-tool patterns)
    parts.append(_build_workflows_section())

    # 10. Generation rules (present but not forced)
    parts.append(f"--- GENERATION RULES (when you generate content) ---\n{_get_generation_rules()}")

    return "\n\n".join(parts)


def _build_capabilities_section() -> str:
    """Build structured capabilities section with tool guidance."""
    return (
        "--- CAPABILITIES ---\n"
        "You have two modes. You decide which to use based on the message, "
        "and can combine them in one turn.\n\n"

        "**CHAT** — Natural conversation. 1-3 sentences unless asked for detail. "
        "Sound like a person, not a bot.\n\n"

        "**GENERATE** — Create social media post drafts with images. "
        "Use your tools, then output a JSON draft block.\n\n"

        "## APPROVE / POST / SCHEDULE\n"
        "Approving a draft does NOT post it. After approval:\n"
        "- \"post it\" / \"send it\" → use `post_approved` to publish to X now\n"
        "- \"schedule for 3pm\" → use `schedule_post` with natural language time\n"
        "- Always ask: \"Want me to post now or schedule for later?\"\n\n"

        "## REVISIONS\n"
        "When the user gives feedback on a pending draft (e.g. 'change the image', "
        "'make it shorter'), call `revise_draft` with their feedback, then generate "
        "a revised version. Don't ask them to formally reject first.\n\n"

        "## SESSION PLANS\n"
        "After a draft is approved, propose the next plan item. Don't auto-generate — "
        "ask first. Use `start_autonomous_plan` if the operator wants batch generation. "
        "Use `show_queued_draft` to load a specific draft for review.\n\n"

        "## TOOL REFERENCE\n"
        "Content creation: `read_brand_guidelines`, `read_references`, `read_feedback_history`, "
        "`check_figma_design`, `generate_image`, `img2img` (from reference photo), `log_resource_usage`\n"
        "Draft management: `get_pending_draft`, `revise_draft`, `approve_draft`\n"
        "Publishing: `post_approved`, `schedule_post`, `list_scheduled_posts`, `cancel_scheduled_post`\n"
        "Planning: `save_session_plan`, `get_session_plan`, `update_plan_item`, "
        "`start_autonomous_plan`, `show_queued_draft`\n"
        "Research: `web_fetch` (read URLs), `read_state_file` (read state/brand data)\n"
        "Utilities: `execute_code` (run Python scripts), `send_file` (deliver files to user), "
        "`check_auto_post_status`, `run_self_review`\n\n"

        "You can chain tools freely. Read a URL, then use what you learned in a draft. "
        "Read state data, run a script to analyze it, send the result as a file."
    )


def _build_workflows_section() -> str:
    """Build common multi-tool workflow patterns."""
    return (
        "--- WORKFLOWS (common multi-tool patterns) ---\n"

        "**Content creation**: read_brand_guidelines → read_feedback_history → "
        "generate_image → output JSON draft block → user approves → post_approved or schedule_post\n\n"

        "**Photo-based content**: User sends photo (stored as reference) → "
        "read_brand_guidelines → img2img with reference → output draft → "
        "if user says 'use that photo again', reference persists across revisions\n\n"

        "**Web-informed content**: web_fetch URL → extract key info → "
        "use it in your caption/image prompt → generate as normal\n\n"

        "**Approve → publish**: User approves draft → ask 'post now or schedule?' → "
        "post_approved (immediate) or schedule_post with time\n\n"

        "**Content session**: Discuss strategy → save_session_plan → "
        "generate item #1 → approve → next item → ... → all done. "
        "Or: start_autonomous_plan to batch generate, then show_queued_draft to review each.\n\n"

        "**Report / analysis**: read_state_file (feedback.json, generation_history.json, etc.) → "
        "execute_code to process data or build HTML/charts → send_file to deliver\n\n"

        "**Schedule queue**: schedule_post to add → list_scheduled_posts to check → "
        "cancel_scheduled_post to remove. Times: '3pm', 'tomorrow 9am', 'in 2 hours', 'friday 3:30pm'.\n\n"

        "**Self-improvement**: run_self_review analyzes approval rates, rejection patterns, "
        "and updates learned preferences. Use when asked about performance."
    )
