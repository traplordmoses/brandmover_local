"""
Unified system prompt builder — combines personality, memory, brand voice,
learned preferences, current state, and generation rules into a single prompt.

LEGACY PATH — Not actively maintained.
This module is only used when UNIFIED_BRAIN_ENABLED=true (via unified_brain.py).
The active architecture uses agent/skill_prompt.py for system prompts.
See agent/session.py for the current memory/preferences system.
Do not add new features here — they won't benefit from session memory,
conversation continuity, self-critique, or the heartbeat system.

ARCHITECTURE:
This is the "brain configuration" module. Every time unified_brain.py makes an
LLM call, it calls build_unified_system_prompt() to construct a dynamic system
prompt that includes ALL context the agent needs:

1. Personality (from brand/personality.md — defines the agent's tone/character)
2. Brand voice (from brand/voice_rules.md — how the brand speaks)
3. Memory (from brand/memory.md — persistent facts the agent should know)
4. Learned preferences (from state/learned_preferences.md — feedback patterns)
5. Self-review summary (from the last performance review)
6. Recent feedback entries (last 10 approvals/rejections)
7. Current state (pending drafts, approved drafts, reference images, session plans, notes)
8. Capabilities (what tools are available and how to use them)
9. Creative coding context (available fonts, assets, Playwright patterns)
10. Workflows (common multi-tool patterns like content creation, scheduling)
11. Generation rules (content format, hard rules, image prompt framework)

WHY DYNAMIC:
The system prompt changes with EVERY call because:
- Draft state changes (pending → approved → posted)
- Learned preferences update as the agent gets more feedback
- Agent notes are added/removed by the operator
- Session plans progress

INTERVIEW TALKING POINT:
"The system prompt is assembled dynamically from ~11 sources. This means the
agent's behavior naturally adapts as context changes — new feedback shifts
its style, pending drafts inform its suggestions, and agent notes give it
persistent memory. It's basically a living configuration document."
"""

import json
import logging
from pathlib import Path

# These imports pull in the personality/memory/voice loaders from the chat module
# and the generation-specific prompt blocks from skill_prompt.
from agent.chat import _load_personality, _load_memory, _load_voice_rules
from agent.skill_prompt import (
    _get_platform_block,       # Platform-specific rules (X char limits, etc.)
    _get_platform_json_line,   # Platform field for the JSON draft schema
    _get_image_mode_block,     # Image generation mode instructions
    _get_content_types_block,  # List of available content types
)
from agent.conversation_context import ConversationContext
from agent import feedback, self_review, session_plan, state
from config import settings

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_PREFERENCES_FILE = _project_root / "state" / "learned_preferences.md"


def _get_learned_preferences() -> str:
    """Load condensed learned preferences from state/learned_preferences.md.

    These are Claude-generated summaries of what the operator approves/rejects.
    Capped at 2000 chars to prevent prompt bloat — the summary should be concise.
    """
    if not _PREFERENCES_FILE.exists():
        return ""
    try:
        text = _PREFERENCES_FILE.read_text(encoding="utf-8").strip()
        return text[:2000]  # Cap to avoid prompt bloat
    except OSError:
        return ""


def _get_recent_feedback() -> str:
    """Return recent feedback entries (without learned preferences).

    Learned preferences are loaded separately via _get_learned_preferences()
    to avoid duplication in the prompt. This function strips out the preferences
    section and only returns raw recent feedback entries.
    """
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


def _get_agent_notes_summary() -> str:
    """Load agent notes and return a compact summary for the system prompt.

    Agent notes are persistent key-value pairs saved by the save_note tool.
    They're things like "launch_date: March 15th" or "posting_style: memes preferred".
    We inject them into the system prompt so the agent always has them in context.
    Values are truncated to 100 chars each to keep the prompt compact.
    """
    notes_file = _project_root / "state" / "agent_notes.json"
    if not notes_file.exists():
        return ""
    try:
        notes = json.loads(notes_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    if not notes:
        return ""
    lines = [f"- {k}: {str(v)[:100]}" for k, v in notes.items()]
    return "NOTES:\n" + "\n".join(lines)


def _get_state_context(context: ConversationContext, user_id: int | None = None) -> str:
    """Return current state info: approved draft, pending draft, schedule status.

    This gives the agent awareness of what's happening RIGHT NOW:
    - Is there a draft waiting for approval?
    - Is there an approved draft ready to post?
    - Is a reference image loaded?
    - What's the current session plan?
    - What persistent notes exist?
    """
    parts = []

    # Approved draft (approved but not yet posted)
    approved = state.get_approved(user_id=user_id)
    if approved:
        caption = approved.get("caption", "")[:100]
        parts.append(
            f"APPROVED DRAFT awaiting post/schedule: \"{caption}...\"\n"
            f"Use post_approved to publish now, or schedule_post to schedule."
        )

    # Pending draft (generated but not yet approved)
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

    # Reference image (for img2img generation)
    ref = state.get_reference_image()
    if ref:
        parts.append(f"Reference image loaded: {Path(ref).name}")

    # Session plan (multi-item content plan for batch generation)
    plan_summary = session_plan.get_plan_summary()
    if plan_summary:
        parts.append(plan_summary)

    # Agent notes (persistent key-value memory)
    notes = _get_agent_notes_summary()
    if notes:
        parts.append(notes)

    return "\n".join(parts)


def _get_generation_rules() -> str:
    """Return condensed generation rules from skill_prompt building blocks.

    These rules tell the agent HOW to generate content:
    - The step-by-step workflow (read guidelines → generate image → output JSON)
    - The exact JSON schema for draft output
    - Hard rules (no hashtags, no AI words, caption length limits)
    - The SPLICE framework for image prompt construction
    """
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
    """Build the unified system prompt combining personality + generation capabilities.

    This is the MASTER PROMPT BUILDER. It assembles ~11 sections into one system
    prompt that gets sent to Claude on every brain call. The order matters:
    - Personality comes FIRST because it dominates the agent's tone
    - State comes before tools so the agent knows what's happening before deciding what to do
    - Generation rules come LAST because they're only relevant when generating

    The prompt is typically ~4K tokens. With prompt caching (cache_control: ephemeral),
    this only costs tokens on the first turn of each conversation.
    """
    parts = []

    # 1. Personality (dominates tone — always first)
    # Loaded from brand/personality.md — defines who the agent IS.
    personality = _load_personality()
    if personality:
        parts.append(personality)
    else:
        parts.append(
            f"You are the AI assistant for {settings.BRAND_NAME}. "
            f"You help with casual conversation and generate social media content."
        )

    # 2. Brand voice rules (from brand/voice_rules.md)
    # How the brand speaks — vocabulary, tone, dos/don'ts.
    voice = _load_voice_rules()
    if voice:
        parts.append(f"--- BRAND VOICE (apply to your own speech) ---\n{voice}")

    # 3. Persistent memory (from brand/memory.md)
    # Facts the agent should always know (founder names, key dates, etc.)
    memory = _load_memory()
    if memory:
        parts.append(f"--- MEMORY ---\n{memory}")

    # 4. Learned preferences (from state/learned_preferences.md)
    # Claude-generated summary of feedback patterns. This is the OUTPUT
    # of the feedback loop (see feedback.py:summarize_preferences).
    prefs = _get_learned_preferences()
    if prefs:
        parts.append(f"--- LEARNED PREFERENCES ---\n{prefs}")

    # 4b. Self-review summary (from the last run_self_review call)
    review_summary = self_review.get_last_review_summary()
    if review_summary:
        parts.append(review_summary)

    # 5. Recent feedback entries (raw approval/rejection log)
    # Preferences loaded above are the SUMMARY; these are the RAW last 10 entries.
    fb = _get_recent_feedback()
    if fb:
        parts.append(fb)

    # 6. Current state (what's happening right now)
    state_ctx = _get_state_context(context, user_id=user_id)
    parts.append(f"--- CURRENT STATE ---\n{state_ctx}")

    # 7. User context (personalization)
    if context.user_name:
        parts.append(f"The user's name is {context.user_name}.")

    # 8. Capabilities + tool guidance (what the agent CAN do)
    parts.append(_build_capabilities_section())

    # 9. Creative coding context (execute_code power-ups)
    # Available fonts, assets, Playwright patterns, image downloading.
    parts.append(_build_creative_coding_section())

    # 10. Common workflows (multi-tool patterns)
    # Teaches the agent common tool chains like "generate → approve → post".
    parts.append(_build_workflows_section())

    # 11. Generation rules (only relevant when generating content)
    # Step-by-step generation workflow, JSON schema, hard rules.
    parts.append(f"--- GENERATION RULES (when you generate content) ---\n{_get_generation_rules()}")

    return "\n\n".join(parts)


def _build_capabilities_section() -> str:
    """Build structured capabilities section with tool guidance.

    This section tells the agent:
    - What modes it has (CHAT vs GENERATE)
    - Its problem-solving mindset (NEVER say "I can't")
    - How approve/post/schedule flow works
    - How to handle revisions
    - Complete tool reference organized by category
    - Self-modification awareness (can read its own source code)
    """
    return (
        "<capabilities>\n"
        "You have two modes. You decide which to use based on the message, "
        "and can combine them in one turn.\n\n"

        "<mode name='chat'>\n"
        "Natural conversation. Be helpful and proactive. If the user "
        "describes a problem, propose a solution and offer to implement it. "
        "Don't just answer — act.\n"
        "</mode>\n\n"

        "<mode name='generate'>\n"
        "Create social media post drafts with images. "
        "Use your tools, then output a JSON draft block.\n"
        "</mode>\n\n"

        "<problem_solving>\n"
        "You have execute_code (full Python with "
        "Pillow, httpx, playwright, numpy), web_fetch, read_state_file, and send_file. "
        "Between these tools, you can figure out almost anything:\n"
        "- Need data you don't have? → web_fetch it or read_state_file\n"
        "- Need to create something visual? → execute_code with Playwright HTML→PNG\n"
        "- Need to analyze something? → execute_code with Python\n"
        "- Need to deliver a file? → execute_code to create it, then send_file\n"
        "- Don't know how? → Try anyway. Write code, run it, iterate if it fails.\n"
        "If a user asks for something and you don't have a specific tool, use execute_code. "
        "You are a developer with a full Python runtime — act like one.\n"
        "</problem_solving>\n\n"

        "<draft_lifecycle>\n"
        "Approving a draft does NOT post it. After approval:\n"
        "- \"post it\" / \"send it\" → use `post_approved` to publish to X now\n"
        "- \"schedule for 3pm\" → use `schedule_post` with natural language time\n"
        "- Always ask: \"Want me to post now or schedule for later?\"\n\n"

        "When the user gives feedback on a pending draft (e.g. 'change the image', "
        "'make it shorter'), call `revise_draft` with their feedback, then generate "
        "a revised version. Don't ask them to formally reject first.\n\n"

        "After a draft is approved, propose the next plan item. Don't auto-generate — "
        "ask first. Use `start_autonomous_plan` if the operator wants batch generation. "
        "Use `show_queued_draft` to load a specific draft for review.\n"
        "</draft_lifecycle>\n\n"

        "<tool_reference>\n"
        "Content creation: `read_brand_guidelines`, `read_references`, `read_feedback_history`, "
        "`check_figma_design`, `generate_image`, `img2img` (from reference photo), `log_resource_usage`\n"
        "Draft management: `get_pending_draft`, `revise_draft`, `approve_draft`\n"
        "Publishing: `post_approved`, `schedule_post`, `list_scheduled_posts`, `cancel_scheduled_post`\n"
        "Planning: `save_session_plan`, `get_session_plan`, `update_plan_item`, "
        "`start_autonomous_plan`, `show_queued_draft`\n"
        "Research: `web_fetch` (read URLs), `read_state_file` (read state/brand data), "
        "`take_screenshot` (capture web pages)\n"
        "Image editing: `edit_image` (text overlay, resize, crop, composite, border), "
        "`generate_image`, `img2img`\n"
        "Memory: `save_note` / `get_notes` (persistent key-value notes), "
        "`save_snippet` / `list_snippets` / `use_snippet` (content library)\n"
        "Utilities: `execute_code` (run Python scripts), `register_draft` (link execute_code output "
        "into draft pipeline), `send_file` (deliver files to user), "
        "`check_auto_post_status`, `run_self_review`\n"
        "Dev tools: `git_info` (log/diff/show/status), `read_telegram_channel` (community messages)\n"
        "Video: `smart_record` (vision-guided browser recording), `edit_video` (cut/stitch/style), "
        "`style_video` (phone mockup + gradient), `review_video` (self-review quality gate), "
        "`analyze_video_scenes` (scene classification), `edit_by_intent` (natural language editing), "
        "`generate_video` (Remotion motion graphics from brief)\n"
        "Analytics: `check_post_performance` (engagement metrics for posted content)\n"
        "</tool_reference>\n\n"

        "You can chain tools freely. Read a URL, then use what you learned in a draft. "
        "Read state data, run a script to analyze it, send the result as a file.\n\n"

        "<video_production_workflow>\n"
        "When recording or editing demo videos, follow this pipeline:\n\n"

        "<preferred_pipeline name='scene_analysis'>\n"
        "1. Record — Use `smart_record` to capture the walkthrough\n"
        "2. Analyze — Call `analyze_video_scenes` on the raw recording. This extracts frames "
        "at 0.5s intervals, classifies each (static/animation/loading/transition/interaction), "
        "and returns a structured alignment map of scene tokens.\n"
        "   The analysis pipeline automatically applies three smart compression passes:\n"
        "   - Static compression: Typewriter animations trimmed to 2.5s (keeps full-text state)\n"
        "   - Seam detection: Take-stitching artifacts (home screen flashes, loading glitches) removed\n"
        "   - Repetitive UI compression: Modal sequences (wallet setup, multi-step dialogs) compressed to 4s max\n"
        "3. Edit — Call `edit_by_intent` with the alignment map and a natural language instruction. "
        "The system translates your intent into structured edit operations (delete loading, trim dead time, "
        "reorder sections, add narration) and renders the final video.\n"
        "   `edit_by_intent` has auto_review=true by default: it automatically reviews the output "
        "and re-edits if score < 9/10, appending the reviewer's feedback to the intent. Max 1 retry.\n"
        "   You can set auto_review=false if you want manual control.\n"
        "4. Final check — The auto-review handles most quality issues. Call `review_video` manually "
        "only if you want an additional check or if auto_review was disabled.\n"
        "5. Send — Only send after review passes (score >= 8)\n"
        "</preferred_pipeline>\n\n"

        "<fallback_pipeline name='manual_editing'>\n"
        "If `analyze_video_scenes` fails or you need precise control:\n"
        "1. Use `smart_record` step_timeline and dead_time_hints to build segments manually\n"
        "2. Call `edit_video` with explicit {start, end} segments\n"
        "3. Rules: delete loading >3s, keep swipe sequences, keep first/last steps, "
        "trim static waits, cut error steps\n"
        "</fallback_pipeline>\n\n"

        "You are autonomous and have time to think. Take multiple review+edit passes if needed. "
        "Quality matters more than speed.\n"
        "</video_production_workflow>\n\n"

        "<self_modification>\n"
        "You can read your own source code with read_state_file and execute_code. "
        "If you need to understand how something works internally, read the relevant Python file. "
        "Your own code lives in: agent/, bot/, config/, scripts/\n"
        "Key files: agent/unified_tools.py (your tools), agent/unified_prompt.py (this prompt), "
        "agent/unified_brain.py (your reasoning loop), config/settings.py (configuration).\n"
        "</self_modification>\n"
        "</capabilities>"
    )


def _build_creative_coding_section() -> str:
    """Build creative coding context for execute_code — fonts, assets, patterns.

    This section teaches the agent about its most powerful capability:
    execute_code with Playwright (headless Chrome). It can write HTML, render
    it to pixel-perfect PNGs, and use them as social media images.

    The section dynamically lists available brand fonts and assets so the agent
    knows what it can use without guessing.
    """
    brand_folder = Path(settings.BRAND_FOLDER)
    fonts_dir = brand_folder / "assets" / "fonts"
    logo_path = brand_folder / "assets" / "logo.png"

    # Dynamically list available brand fonts
    font_list = ""
    if fonts_dir.exists():
        fonts = sorted(f.name for f in fonts_dir.iterdir() if f.suffix in (".ttf", ".otf"))
        if fonts:
            font_list = ", ".join(fonts[:10])
            if len(fonts) > 10:
                font_list += f" (+{len(fonts) - 10} more)"

    # Check for logo file
    logo_info = f"Logo: `{logo_path}` (RGBA PNG)" if logo_path.exists() else "No logo file found"

    return (
        "--- CREATIVE CODING (execute_code power) ---\n"
        "execute_code runs Python with full packages: Pillow, httpx, numpy, **playwright** "
        "(headless Chrome). 60s timeout. Write to `state/outputs/`. Call `register_draft` after.\n\n"

        "## AVAILABLE ASSETS\n"
        f"Fonts dir: `{fonts_dir}/`\n"
        f"Fonts: {font_list}\n"
        f"{logo_info}\n"
        f"Assets: `{brand_folder}/assets/` (check for .png files)\n\n"

        # Playwright HTML→PNG is the agent's most powerful creative tool.
        # It can create pixel-perfect branded cards, tables, dashboards, etc.
        "## HTML → PNG (Playwright — your most powerful creative tool)\n"
        "Write styled HTML, render to pixel-perfect PNG with headless Chrome:\n"
        "```python\n"
        "from playwright.sync_api import sync_playwright\n"
        "import os\n"
        "os.makedirs('state/outputs', exist_ok=True)\n"
        "html = '<html><body style=\"background:#0D1B2A;color:#fff;...\">...</body></html>'\n"
        "with sync_playwright() as p:\n"
        "    browser = p.chromium.launch()\n"
        "    page = browser.new_page(viewport={'width': 1280, 'height': 720})\n"
        "    page.set_content(html)\n"
        "    page.screenshot(path='state/outputs/card.png')\n"
        "    browser.close()\n"
        "```\n"
        "Use this for: schedule cards, data tables, listing banners, reports, "
        "any layout that needs CSS precision. You can use Google Fonts via "
        "`@import url(...)` in the HTML `<style>` tag, or reference local fonts.\n\n"

        "## IMAGE DOWNLOADING\n"
        "Fetch images from the web (logos, avatars, screenshots):\n"
        "```python\n"
        "import httpx\n"
        "from PIL import Image\n"
        "from io import BytesIO\n"
        "resp = httpx.get('https://example.com/logo.png', timeout=15)\n"
        "img = Image.open(BytesIO(resp.content))\n"
        "```\n\n"

        # Decision guide: which tool to use for what
        "## WHEN TO USE WHAT\n"
        "- **generate_image**: AI-generated artwork, photos, illustrations\n"
        "- **execute_code + Pillow**: Simple graphics, image manipulation, overlays\n"
        "- **execute_code + Playwright**: Complex layouts, tables, cards, CSS-styled content, "
        "schedule views, reports, data-rich graphics, anything with precise typography\n\n"

        "For polished branded cards, dashboards, or any layout with tables/grids → "
        "**always use Playwright HTML→PNG**. It's faster and better than Pillow for layouts."
    )


def _build_workflows_section() -> str:
    """Build common multi-tool workflow patterns.

    These are "recipes" that teach the agent how to chain tools together
    for common tasks. Without these, the agent might call tools in the wrong
    order or miss steps (like forgetting to call register_draft after execute_code).

    Each workflow is a named pattern with a → chain showing the tool sequence.
    """
    return (
        "--- WORKFLOWS (common multi-tool patterns) ---\n"

        # Standard content creation workflow
        "**Content creation**: read_brand_guidelines → read_feedback_history → "
        "generate_image → output JSON draft block → user approves → post_approved or schedule_post\n\n"

        # Photo-based content (user sends a reference photo)
        "**Photo-based content**: User sends photo (stored as reference) → "
        "read_brand_guidelines → img2img with reference → output draft → "
        "if user says 'use that photo again', reference persists across revisions\n\n"

        # Web-informed content (use info from a URL)
        "**Web-informed content**: web_fetch URL → extract key info → "
        "use it in your caption/image prompt → generate as normal\n\n"

        # Post-approval publishing
        "**Approve → publish**: User approves draft → ask 'post now or schedule?' → "
        "post_approved (immediate) or schedule_post with time\n\n"

        # Multi-item content sessions
        "**Content session**: Discuss strategy → save_session_plan → "
        "generate item #1 → approve → next item → ... → all done. "
        "Or: start_autonomous_plan to batch generate, then show_queued_draft to review each.\n\n"

        # Code-generated content (memes, banners, templates via execute_code)
        "**Code-generated content** (memes, banners, templates): execute_code creates image → "
        "register_draft with file path AND caption → approve → post/schedule. "
        "ALWAYS include a caption in register_draft. ALWAYS call register_draft after execute_code.\n\n"

        # Dynamic brand graphics using web data + code
        "**Dynamic brand graphics** (listing banners, partnership announcements): "
        "web_fetch to get info → execute_code to download logo + create branded template "
        "with Pillow (use brand fonts, colors, logo) → register_draft → approve → post. "
        "Example: fetch coin logo from CoinGecko API, create listing banner with brand template.\n\n"

        # Reports and analysis
        "**Report / analysis**: read_state_file (feedback.json, generation_history.json, etc.) → "
        "execute_code to process data or build HTML/charts → send_file to deliver\n\n"

        # Schedule management
        "**Schedule queue**: schedule_post to add → list_scheduled_posts to check → "
        "cancel_scheduled_post to remove. Times: '3pm', 'tomorrow 9am', 'in 2 hours', 'friday 3:30pm'.\n\n"

        # Self-improvement
        "**Self-improvement**: run_self_review analyzes approval rates, rejection patterns, "
        "and updates learned preferences. Use when asked about performance.\n\n"

        # Freestyle — the key "figure it out" workflow
        "**Freestyle problem-solving**: User asks for something with no pre-built tool → "
        "think about what tools you DO have → combine web_fetch + execute_code + send_file "
        "to accomplish it. Example: 'What\\'s our engagement rate?' → read_state_file to get "
        "generation_history → execute_code to analyze → respond with insights. "
        "Example: 'Make me a comparison chart' → execute_code with Playwright HTML→PNG → "
        "register_draft or send_file."
    )
