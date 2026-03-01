"""
Conversational onboarding — Claude-driven discovery + state machine for
uploads/audit/strategy/confirm phases.

Persisted in state/onboarding.json. Survives bot restarts.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import anthropic

from config import settings

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "onboarding.json"


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class OnboardingState(str, Enum):
    IDLE = "idle"
    DISCOVERY = "discovery"         # Claude-driven conversation
    # Legacy states kept for backward compat with existing sessions
    PROJECT_NAME = "project_name"
    DESCRIPTION = "description"
    PLATFORMS = "platforms"
    ASSET_CHECK = "asset_check"
    # Active states
    UPLOADS = "uploads"
    AUDITING = "auditing"
    VISUAL_PREF = "visual_pref"
    TEMPLATE_CHOICE = "template_choice"
    STRATEGY = "strategy"
    CONFIRM = "confirm"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Session data
# ---------------------------------------------------------------------------

@dataclass
class OnboardingSession:
    user_id: int = 0
    state: str = OnboardingState.IDLE.value
    brand_name: str = ""
    description: str = ""
    platforms: list[str] = field(default_factory=list)
    uploaded_assets: list[dict] = field(default_factory=list)  # [{path, type}]
    asset_audit: dict = field(default_factory=dict)
    visual_preferences: dict = field(default_factory=dict)
    strategy: dict = field(default_factory=dict)
    started_at: float = 0.0
    updated_at: float = 0.0
    # Smart onboarding fields
    conversation_history: list[dict] = field(default_factory=list)  # [{role, content}]
    collected_fields: dict = field(default_factory=dict)


# Required fields that must be collected during discovery
REQUIRED_FIELDS = (
    "project_name", "description", "platforms",
    "has_assets", "visual_preference",
)

# Optional fields Claude may discover through conversation
OPTIONAL_FIELDS = (
    "target_audience", "posting_frequency", "tone_preference",
    "competitors_or_references",
    "cultural_references", "what_they_hate", "secret_weapon",
)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_sessions() -> dict:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sessions(sessions: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def get_session(user_id: int) -> OnboardingSession | None:
    """Load an onboarding session for a user. Returns None if not found."""
    sessions = _load_sessions()
    key = str(user_id)
    if key not in sessions:
        return None
    data = sessions[key]
    # Filter out unknown fields for backward compat
    known = {f.name for f in OnboardingSession.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    return OnboardingSession(**filtered)


def save_session(session: OnboardingSession) -> None:
    """Persist an onboarding session."""
    sessions = _load_sessions()
    session.updated_at = time.time()
    sessions[str(session.user_id)] = {
        "user_id": session.user_id,
        "state": session.state,
        "brand_name": session.brand_name,
        "description": session.description,
        "platforms": session.platforms,
        "uploaded_assets": session.uploaded_assets,
        "asset_audit": session.asset_audit,
        "visual_preferences": session.visual_preferences,
        "strategy": session.strategy,
        "started_at": session.started_at,
        "updated_at": session.updated_at,
        "conversation_history": session.conversation_history,
        "collected_fields": session.collected_fields,
    }
    _save_sessions(sessions)


def delete_session(user_id: int) -> None:
    """Remove an onboarding session."""
    sessions = _load_sessions()
    sessions.pop(str(user_id), None)
    _save_sessions(sessions)


# ---------------------------------------------------------------------------
# Claude-driven discovery
# ---------------------------------------------------------------------------

_DISCOVERY_SYSTEM = """\
You are a creative collaborator onboarding a new client for BrandMover, an AI content engine \
that generates branded social media posts.

Your job is to genuinely understand their brand — not just collect facts, but feel what \
makes it special. React to specifics. If they say "we hand-draw our logo on butcher paper," \
don't just note "illustration style" — ask about the story behind it.

INFORMATION STILL NEEDED:
{missing_fields}

INFORMATION ALREADY COLLECTED:
{collected_fields}

RULES:
- Be a curious creative, not a form processor. React to what they tell you with genuine interest.
- If they mention something distinctive (hand-drawn assets, pixel art, brutalist design), \
dig into THAT — ask why, ask about the story, ask what it means to them.
- Don't ask more than 1-2 questions per message.
- When you have enough info for a field, extract it.
- When all required fields are collected, set all_required_complete to true.
- Keep messages concise. This is Telegram, not email.
- Use HTML formatting sparingly (<b> for emphasis).

FIELD EXTRACTION GUIDE:
- project_name: The brand/project name
- description: What the brand does (1-2 sentences)
- platforms: List of social platforms (x, telegram, linkedin, instagram, threads, bluesky)
- has_assets: Whether they have brand assets to upload (true/false)
- visual_preference: Their preferred visual style or aesthetic description
- target_audience: Who their audience is (optional)
- posting_frequency: How often they want to post (optional)
- tone_preference: Communication tone/voice (optional)
- competitors_or_references: Brands they admire or compete with (optional)
- cultural_references: Films, music, art movements, aesthetics they love (optional)
- what_they_hate: Brands or aesthetics they actively dislike (optional)
- secret_weapon: The one thing that makes them unlike anything else (optional)

Respond with ONLY valid JSON (no markdown fences):
{{"message": "your response to the user", "fields_collected": {{}}, "all_required_complete": false, "suggest_upload": false}}"""


async def _call_discovery(session: OnboardingSession, user_message: str) -> dict:
    """Send conversation to Claude and get the next response + extracted fields."""
    missing = [f for f in REQUIRED_FIELDS if f not in session.collected_fields]
    collected_str = json.dumps(session.collected_fields, indent=2) if session.collected_fields else "(none yet)"
    missing_str = ", ".join(missing) if missing else "(all collected)"

    system = _DISCOVERY_SYSTEM.format(
        missing_fields=missing_str,
        collected_fields=collected_str,
    )

    # Build message history
    messages = list(session.conversation_history)
    messages.append({"role": "user", "content": user_message})

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system,
        messages=messages,
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Discovery response was not JSON: %s", raw[:200])
        data = {"message": raw, "fields_collected": {}, "all_required_complete": False, "suggest_upload": False}

    return data


def _apply_collected_fields(session: OnboardingSession, fields: dict) -> None:
    """Apply extracted fields to the session."""
    for key, value in fields.items():
        session.collected_fields[key] = value

        # Also map to the session's typed fields
        if key == "project_name" and value:
            session.brand_name = str(value)
        elif key == "description" and value:
            session.description = str(value)
        elif key == "platforms" and value:
            if isinstance(value, list):
                session.platforms = value
            elif isinstance(value, str):
                session.platforms = [p.strip().lower() for p in value.split(",") if p.strip()]
        elif key == "visual_preference" and value:
            session.visual_preferences = {"style": "custom", "description": str(value)}


# ---------------------------------------------------------------------------
# State machine transitions
# ---------------------------------------------------------------------------

_VISUAL_STYLES = {"modern", "playful", "corporate", "minimal", "bold", "elegant"}


def advance(session: OnboardingSession, user_input: str | None) -> tuple[OnboardingSession, str]:
    """Advance the state machine based on user input (sync transitions only).

    Returns (updated_session, response_message).
    The DISCOVERY state returns a placeholder — the actual Claude call
    happens in advance_async().
    """
    st = session.state
    inp = (user_input or "").strip()
    inp_lower = inp.lower()

    if st == OnboardingState.IDLE.value:
        session.state = OnboardingState.DISCOVERY.value
        session.started_at = time.time()
        session.collected_fields = {}
        session.conversation_history = []
        return session, (
            "Let's set up your brand! I'll ask you a few questions — "
            "just chat naturally and I'll figure out the rest.\n\n"
            "What's your brand name?"
        )

    # Legacy states — redirect to DISCOVERY for sessions started pre-upgrade
    elif st in (OnboardingState.PROJECT_NAME.value,
                OnboardingState.DESCRIPTION.value,
                OnboardingState.PLATFORMS.value,
                OnboardingState.ASSET_CHECK.value):
        return _advance_legacy(session, inp, inp_lower)

    elif st == OnboardingState.DISCOVERY.value:
        # Placeholder — actual processing happens in advance_async()
        return session, "_NEEDS_ASYNC_"

    elif st == OnboardingState.UPLOADS.value:
        if inp_lower in ("/onboard_skip", "done", "skip"):
            if session.uploaded_assets:
                session.state = OnboardingState.AUDITING.value
                return session, (
                    f"Analyzing {len(session.uploaded_assets)} asset(s) with Claude Vision..."
                )
            else:
                session.state = OnboardingState.VISUAL_PREF.value
                return session, (
                    "No assets uploaded. Let's pick a visual style instead.\n\n"
                    "Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
                    "<b>minimal</b> / <b>bold</b> / <b>elegant</b>"
                )
        return session, (
            "Upload your brand assets as photos. "
            "Use /onboard_skip when you're done uploading."
        )

    elif st == OnboardingState.AUDITING.value:
        return session, "Still analyzing your assets... please wait."

    elif st == OnboardingState.VISUAL_PREF.value:
        if not inp:
            return session, "Please describe your preferred visual style."
        if inp_lower in _VISUAL_STYLES:
            session.visual_preferences = {"style": inp_lower}
        else:
            session.visual_preferences = {"style": "custom", "description": inp}
        session.state = OnboardingState.STRATEGY.value
        return session, "Generating your brand strategy..."

    elif st == OnboardingState.TEMPLATE_CHOICE.value:
        session.state = OnboardingState.STRATEGY.value
        return session, "Generating your brand strategy based on your assets..."

    elif st == OnboardingState.STRATEGY.value:
        return session, "Still generating strategy... please wait."

    elif st == OnboardingState.CONFIRM.value:
        if inp_lower in ("yes", "y", "confirm", "ok", "looks good"):
            session.state = OnboardingState.COMPLETE.value
            return session, "Setting up your brand..."
        elif inp_lower in ("no", "n", "restart", "redo"):
            session.state = OnboardingState.DISCOVERY.value
            session.brand_name = ""
            session.description = ""
            session.platforms = []
            session.uploaded_assets = []
            session.asset_audit = {}
            session.visual_preferences = {}
            session.strategy = {}
            session.collected_fields = {}
            session.conversation_history = []
            return session, "Let's start over.\n\nWhat's your brand name?"
        return session, "Reply <b>yes</b> to confirm or <b>no</b> to restart."

    elif st == OnboardingState.COMPLETE.value:
        return session, "Onboarding already complete! Your brand is set up."

    return session, "Something went wrong. Use /onboard_cancel to restart."


async def advance_async(session: OnboardingSession, user_input: str) -> tuple[OnboardingSession, str]:
    """Async version for the DISCOVERY state — calls Claude for conversation."""
    if session.state != OnboardingState.DISCOVERY.value:
        return advance(session, user_input)

    inp = (user_input or "").strip()
    if not inp:
        return session, "Tell me about your brand!"

    # Call Claude
    result = await _call_discovery(session, inp)

    # Update conversation history
    session.conversation_history.append({"role": "user", "content": inp})
    bot_msg = result.get("message", "Tell me more about your brand.")
    session.conversation_history.append({"role": "assistant", "content": bot_msg})

    # Trim history to last 20 messages to prevent unbounded growth
    if len(session.conversation_history) > 20:
        session.conversation_history = session.conversation_history[-20:]

    # Apply extracted fields
    fields = result.get("fields_collected", {})
    if fields:
        _apply_collected_fields(session, fields)

    # Check if discovery is complete
    if result.get("all_required_complete"):
        has_assets = session.collected_fields.get("has_assets")
        if has_assets in (True, "true", "yes"):
            session.state = OnboardingState.UPLOADS.value
            bot_msg += (
                "\n\nUpload your brand assets now — logos, images, style guides. "
                "Send them as photos or documents. Use /onboard_skip when done."
            )
        else:
            # Check if visual preference was already collected
            if session.visual_preferences:
                session.state = OnboardingState.STRATEGY.value
                bot_msg += "\n\nGenerating your brand strategy..."
            else:
                session.state = OnboardingState.VISUAL_PREF.value
                bot_msg += (
                    "\n\nWhat visual style suits your brand?\n"
                    "Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
                    "<b>minimal</b> / <b>bold</b> / <b>elegant</b>"
                )

    return session, bot_msg


# ---------------------------------------------------------------------------
# Legacy state transitions (for sessions started before smart onboarding)
# ---------------------------------------------------------------------------

def _advance_legacy(session: OnboardingSession, inp: str, inp_lower: str) -> tuple[OnboardingSession, str]:
    """Handle legacy fixed-state transitions for backward compat."""
    st = session.state

    if st == OnboardingState.PROJECT_NAME.value:
        if not inp:
            return session, "Please enter your brand name."
        session.brand_name = inp
        session.state = OnboardingState.DESCRIPTION.value
        return session, f"Got it — <b>{inp}</b>.\n\nDescribe what {inp} does in 1-2 sentences."

    elif st == OnboardingState.DESCRIPTION.value:
        if not inp:
            return session, "Please describe what your brand does."
        session.description = inp
        session.state = OnboardingState.PLATFORMS.value
        return session, (
            "Which platforms will you post to?\n\n"
            "Options: twitter, linkedin, instagram, threads, bluesky\n"
            "(comma-separated, e.g. 'twitter, linkedin')"
        )

    elif st == OnboardingState.PLATFORMS.value:
        if not inp:
            return session, "Please list your target platforms."
        platforms = [p.strip().lower() for p in inp.split(",") if p.strip()]
        session.platforms = platforms if platforms else ["twitter"]
        session.state = OnboardingState.ASSET_CHECK.value
        return session, (
            "Do you have brand assets to upload? (logos, color palettes, style guides)\n\n"
            "Reply <b>yes</b> or <b>no</b>"
        )

    elif st == OnboardingState.ASSET_CHECK.value:
        if inp_lower in ("yes", "y", "yeah", "yep"):
            session.state = OnboardingState.UPLOADS.value
            return session, (
                "Upload your brand assets now — logos, color palettes, style guides, etc.\n\n"
                "Send them as photos or documents. When you're done, "
                "use /onboard_skip to continue."
            )
        else:
            session.state = OnboardingState.VISUAL_PREF.value
            return session, (
                "What visual style suits your brand?\n\n"
                "Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
                "<b>minimal</b> / <b>bold</b> / <b>elegant</b>\n\n"
                "Or describe your own style."
            )

    return session, "Something went wrong. Use /onboard_cancel to restart."


# ---------------------------------------------------------------------------
# Audit finalization (called after Claude Vision audit completes)
# ---------------------------------------------------------------------------

def finalize_audit(session: OnboardingSession, audit_result: dict) -> tuple[OnboardingSession, str]:
    """Called after the async audit_batch completes. Advances to VISUAL_PREF or TEMPLATE_CHOICE."""
    session.asset_audit = audit_result
    archetype = audit_result.get("archetype", "starting_fresh")

    if archetype == "full_brand":
        session.state = OnboardingState.TEMPLATE_CHOICE.value
        return session, (
            f"<b>Asset Analysis Complete</b>\n\n"
            f"Archetype: <b>Full Brand</b> — you have comprehensive brand assets!\n"
            f"Colors found: {len(audit_result.get('consolidated_colors', []))}\n"
            f"Style: {', '.join(audit_result.get('consolidated_style', [])[:5])}\n\n"
            f"Press enter or send any message to continue with strategy generation."
        )
    else:
        session.state = OnboardingState.VISUAL_PREF.value
        missing = audit_result.get("missing_items", [])
        missing_str = ", ".join(missing) if missing else "none"
        return session, (
            f"<b>Asset Analysis Complete</b>\n\n"
            f"Archetype: <b>{archetype.replace('_', ' ').title()}</b>\n"
            f"Missing: {missing_str}\n\n"
            f"What visual style suits your brand?\n"
            f"Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
            f"<b>minimal</b> / <b>bold</b> / <b>elegant</b>"
        )


# ---------------------------------------------------------------------------
# Strategy finalization
# ---------------------------------------------------------------------------

def finalize_strategy(session: OnboardingSession, strategy_data: dict) -> tuple[OnboardingSession, str]:
    """Called after strategy recommendation completes. Shows summary for confirmation."""
    session.strategy = strategy_data
    session.state = OnboardingState.CONFIRM.value

    rec = strategy_data
    compositor_str = "ON" if rec.get("compositor_enabled") else "OFF"
    mode = rec.get("default_mode", "image_optional")
    types = rec.get("recommended_content_types", [])
    types_str = ", ".join(types[:6])
    reasoning = rec.get("reasoning", "")

    summary = (
        f"<b>Setup Summary for {session.brand_name}</b>\n\n"
        f"<b>Description:</b> {session.description}\n"
        f"<b>Platforms:</b> {', '.join(session.platforms)}\n"
        f"<b>Compositor:</b> {compositor_str}\n"
        f"<b>Mode:</b> {mode}\n"
        f"<b>Content types:</b> {types_str}\n"
    )
    if reasoning:
        summary += f"\n<i>{reasoning}</i>\n"
    summary += "\nReply <b>yes</b> to confirm or <b>no</b> to restart."

    return session, summary


# ---------------------------------------------------------------------------
# Guidelines generation from audit data
# ---------------------------------------------------------------------------

_GUIDELINES_MERGE_PROMPT = """\
You are a brand strategist performing a TARGETED UPDATE of an existing brand guidelines \
document. You have two inputs: the existing guidelines and fresh asset audit data.

YOUR TASK: Merge the audit data INTO the existing guidelines. This is a surgical update, \
not a rewrite. Most of the document stays exactly as-is.

=== SECTIONS TO PRESERVE VERBATIM (copy word-for-word from existing guidelines) ===
- BRAND IDENTITY (name, tagline, positioning, website, X handle, posting cadence, themes)
- VOICE & TONE (personality traits, writing style, tone by content type, emoji rules, never-use list)
- CONTENT CATEGORIES (all content type descriptions and examples)
- BRAND PHRASES & SLANG (established phrases, community slang, CTAs)
- HASHTAGS (all hashtag rules)
- TARGET AUDIENCE (all audience descriptions)
- VISUAL EFFECTS (compositor glass/orb/noise values — table)
- LAYOUT PROFILES (canvas dimensions, logo position — table)
- LAYOUT MAPPINGS (content type to profile mapping — table)
- Any other sections NOT listed below as "replace" sections

=== SECTIONS TO REPLACE WITH AUDIT DATA ===

1. COLORS / COLOR PALETTE:
   Replace the ENTIRE color table and color rules with audit-extracted colors.
   - Use ONLY hex values from the consolidated_colors list below
   - Format: Markdown table with | Role | Name | Hex | RGB | Notes |
   - Assign roles (Primary, Secondary, Accent 1-3, Background, Text) from the audit role field
   - Write new color usage rules based on the actual color relationships in the audit data
   - DO NOT keep any hex values from the old guidelines — the old colors were incorrect

2. ILLUSTRATION STYLE:
   Replace with a description faithful to the ACTUAL assets.
   - Describe what the audit data shows: if anime-influenced, say anime-influenced. \
If hand-drawn, say hand-drawn. Do NOT reinterpret into a different aesthetic.
   - Use the style keywords and collection analysis as your source
   - Include: image generation prompt guidance based on actual asset style, avoid list
   - Keep the Technical specs line if present in the old guidelines

3. TYPOGRAPHY:
   Replace ONLY if the existing fonts are contradicted by the available_fonts list below.
   - If the existing guidelines list fonts that ARE in available_fonts, KEEP them
   - If the existing guidelines list fonts NOT in available_fonts, replace with fonts \
that ARE available, or write "Typography: defined by brand assets" if no match
   - Keep any typography rules (hierarchy, mixing rules) that still make sense

4. MASCOT / CHARACTER SYSTEM:
   Replace with descriptions derived from the creative audit data.
   - Describe the ACTUAL characters/mascots observed in the assets
   - Use first_impression, creative_dna, character_system data directly
   - Include appearance details, personality, and usage context from the audit

5. Add ## CREATIVE BRIEF section (if creative audit data is provided):
   Synthesize creative_dna, overall_energy, first_impression into a vivid 2-3 sentence \
   creative direction statement. Include character_system insights if present.

6. Add ## NEVER DO section (if creative audit data provides never_do entries):
   Bulleted list of creative prohibitions from the audit data. \
   Merge with any existing voice/tone contradictions. 3-8 items.

=== EXISTING GUIDELINES (preserve everything not listed above) ===

{existing_guidelines}

=== ASSET AUDIT DATA (source for replacement sections) ===

Colors extracted from assets: {colors_json}
Style keywords from assets: {style_keywords}
Available fonts in project: {available_fonts}

{collection_section}
{insights_section}
{creative_section}

=== RULES ===
- EVERY color hex in the output must come from the audit data. ZERO old colors retained.
- EVERY non-visual section must be copied VERBATIM from the existing guidelines.
- Illustration style must faithfully describe the actual assets, not aspirational aesthetics.
- Do NOT invent font names. Only use fonts from available_fonts or write "defined by brand assets."
- The output is parsed by code — preserve exact table formats, section headers, and structure.
- Output ONLY the complete Markdown document, no preamble or explanation.
"""

_GUIDELINES_FRESH_PROMPT = """\
You are a brand strategist. Generate a comprehensive brand guidelines document in Markdown \
from the brand information and asset audit data below.

BRAND INFO:
- Name: {brand_name}
- Description: {description}
- Platforms: {platforms}
- Visual preferences (user-stated): {visual_prefs}

ASSET AUDIT DATA:
- Colors extracted from assets: {colors_json}
- Style keywords from assets: {style_keywords}
- Available fonts in project: {available_fonts}

{collection_section}
{insights_section}
{creative_section}

STRATEGY:
- Compositor enabled: {compositor_enabled}
- Badge text: {badge_text}
- Default mode: {default_mode}

Generate a Markdown document with these sections (use ## headers):

1. ## BRAND IDENTITY — Brand Name, Tagline, Product description
2. ## VOICE & TONE — Core personality traits, Writing style, Emoji usage, Never-use list
3. ## COLOR PALETTE — ONLY hex values from audit data. Table: | Role | Name | Hex | RGB |
4. ## TYPOGRAPHY — Use ONLY fonts from available_fonts list. If empty, write "defined by brand assets."
5. ## ILLUSTRATION STYLE — Faithful to actual assets. Prompt guidance. Avoid list.
6. ## COMPOSITOR — Table: Enabled={compositor_enabled}, Badge={badge_text}, Mode={default_mode}
7. ## CREATIVE BRIEF (if creative audit data provided) — 2-3 sentence creative direction
8. ## NEVER DO (if creative audit data provided) — Bulleted prohibitions, 3-6 items
9. ## MASCOT / CHARACTER SYSTEM (if character_system data present)

Rules:
- EVERY color must come from the audit data. Do NOT invent colors.
- Do NOT invent font names. Only use fonts from available_fonts or "defined by brand assets."
- Style descriptions must faithfully represent the actual assets.
- Keep it concise but complete. This file is parsed by code — follow table formats exactly.
- Output ONLY the Markdown document, no preamble or explanation.
"""


def _detect_available_fonts() -> list[str]:
    """Scan brand/assets/fonts/ and return list of font family names found."""
    fonts_dir = Path(settings.BRAND_FOLDER) / "assets" / "fonts"
    if not fonts_dir.exists():
        return []
    # Extract unique family names from filenames
    families: set[str] = set()
    for f in fonts_dir.iterdir():
        if f.suffix.lower() in (".ttf", ".otf", ".woff", ".woff2"):
            # Heuristic: strip weight/style suffixes from filename
            name = f.stem.lower()
            # Known patterns: "inter-variable", "orbitron-variable", "poppins-bold"
            for known in ("poppins", "orbitron", "inter", "vt323", "aeonikextendedpro"):
                if name.startswith(known):
                    families.add(known.replace("aeonikextendedpro", "Aeonik Extended Pro")
                                 .replace("poppins", "Poppins")
                                 .replace("orbitron", "Orbitron")
                                 .replace("inter", "Inter")
                                 .replace("vt323", "VT323"))
                    break
    return sorted(families)


async def generate_guidelines_from_audit(
    session: "OnboardingSession",
    strategy_rec: "StrategyRecommendation",
    existing_guidelines: str = "",
) -> str:
    """Call Claude to generate/update guidelines.md using audit data.

    If existing_guidelines is provided, uses a MERGE strategy: preserves voice/tone/
    positioning/content sections verbatim, replaces only colors/style/typography/mascot
    with audit-derived data.

    If no existing guidelines, generates fresh from audit + brand info.
    """
    import json as _json

    audit = session.asset_audit or {}
    colors = audit.get("consolidated_colors", [])
    style_kw = audit.get("consolidated_style", [])
    collection = audit.get("collection_analysis", {})
    insights = audit.get("brand_insights", {})
    visual_prefs = session.visual_preferences or {}
    available_fonts = _detect_available_fonts()

    # Build common sections
    collection_section = ""
    if collection:
        collection_section = f"COLLECTION ANALYSIS:\n{_json.dumps(collection, indent=2)}"

    insights_section = ""
    if insights:
        insights_section = f"BRAND INSIGHTS:\n{_json.dumps(insights, indent=2)}"

    entries_creative = audit.get("entries_creative", [])
    creative_section = ""
    if entries_creative:
        creative_section = f"CREATIVE AUDIT DATA:\n{_json.dumps(entries_creative, indent=2)}"

    if existing_guidelines:
        # Merge mode — preserve existing, replace visual sections only
        prompt = _GUIDELINES_MERGE_PROMPT.format(
            existing_guidelines=existing_guidelines,
            colors_json=_json.dumps(colors, indent=2) if colors else "(none extracted)",
            style_keywords=", ".join(style_kw) if style_kw else "(none detected)",
            available_fonts=", ".join(available_fonts) if available_fonts else "(none found)",
            collection_section=collection_section,
            insights_section=insights_section,
            creative_section=creative_section,
        )
    else:
        # Fresh mode — generate from scratch
        prompt = _GUIDELINES_FRESH_PROMPT.format(
            brand_name=session.brand_name,
            description=session.description,
            platforms=", ".join(session.platforms or ["x"]),
            visual_prefs=_json.dumps(visual_prefs) if visual_prefs else "(none stated)",
            colors_json=_json.dumps(colors, indent=2) if colors else "(none extracted)",
            style_keywords=", ".join(style_kw) if style_kw else "(none detected)",
            available_fonts=", ".join(available_fonts) if available_fonts else "(none found)",
            collection_section=collection_section,
            insights_section=insights_section,
            creative_section=creative_section,
            compositor_enabled="true" if strategy_rec.compositor_enabled else "false",
            badge_text=strategy_rec.badge_text or "(none)",
            default_mode=strategy_rec.default_mode,
        )

    # Merge mode needs more tokens since the existing guidelines are included in output
    max_tokens = 8000 if existing_guidelines else 4000

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def _guidelines_template_fallback(
    session: "OnboardingSession",
    rec: "StrategyRecommendation",
) -> str:
    """Minimal template-based guidelines.md — used when Claude generation fails."""
    style_info = session.visual_preferences.get(
        "description", session.visual_preferences.get("style", "modern")
    )
    audit_colors = session.asset_audit.get("consolidated_colors", [])
    color_rows = ""
    for i, c in enumerate(audit_colors[:6]):
        role = ["Primary", "Accent 1", "Accent 2", "Background", "Text", "Accent 3"][min(i, 5)]
        color_rows += f"| {role} | {c.get('name', 'Color')} | {c.get('hex', '#000000')} | |\n"

    md = (
        f"# {session.brand_name} Brand Guidelines\n\n"
        f"**Brand Name:** {session.brand_name}\n"
        f"**Product:** {session.description}\n\n"
        f"## VOICE & TONE\n\n"
        f"**Core personality traits:**\n"
        f"- Authentic and approachable\n"
        f"- Clear and concise\n\n"
        f"## COLOR PALETTE\n\n"
        f"| Role | Name | Hex | RGB |\n"
        f"|------|------|-----|-----|\n"
        f"{color_rows}\n"
        f"## ILLUSTRATION STYLE\n\n"
        f"- **{style_info.title()}** aesthetic\n\n"
        f"## COMPOSITOR\n\n"
        f"| Setting | Value |\n"
        f"|---------|-------|\n"
        f"| Enabled | {'true' if rec.compositor_enabled else 'false'} |\n"
    )
    if rec.badge_text:
        md += f"| Badge text | {rec.badge_text} |\n"
    md += f"| Default mode | {rec.default_mode} |\n"
    return md


# ---------------------------------------------------------------------------
# Finalize onboarding (writes guidelines.md + config.json)
# ---------------------------------------------------------------------------

async def finalize_onboarding(session: OnboardingSession) -> str:
    """Write all config files and return a completion summary."""
    from agent import strategy as strategy_mod
    from agent import compositor_config

    brand_path = Path(settings.BRAND_FOLDER)
    brand_path.mkdir(parents=True, exist_ok=True)

    # Build strategy recommendation from session data
    from agent.strategy import StrategyRecommendation
    rec_data = session.strategy or {}
    rec = StrategyRecommendation(
        archetype=rec_data.get("archetype", "starting_fresh"),
        compositor_enabled=rec_data.get("compositor_enabled", False),
        badge_text=rec_data.get("badge_text"),
        default_mode=rec_data.get("default_mode", "image_optional"),
        recommended_content_types=rec_data.get("recommended_content_types", ["announcement", "community"]),
        platforms=session.platforms or ["x"],
        visual_style_notes=rec_data.get("visual_style_notes", ""),
        reasoning=rec_data.get("reasoning", ""),
    )

    # Save config.json + strategy.md
    strategy_mod.save_strategy(rec, session.brand_name)

    # Extract creative data for content calendar
    entries_creative = (session.asset_audit or {}).get("entries_creative", [])
    creative_brief = ""
    never_do_list: list[str] = []
    for ec in entries_creative:
        if ec.get("overall_energy"):
            creative_brief += ec["overall_energy"] + ". "
        for dna in ec.get("creative_dna", []):
            creative_brief += dna + ". "
        never_do_list.extend(ec.get("never_do", []))
    creative_brief = creative_brief.strip()

    # Generate content calendar
    try:
        posting_freq = session.collected_fields.get("posting_frequency", "")
        await strategy_mod.generate_content_calendar(
            brand_name=session.brand_name,
            description=session.description,
            platforms=session.platforms or ["x"],
            rec=rec,
            posting_frequency=posting_freq,
            creative_brief=creative_brief,
            never_do=never_do_list or None,
        )
    except Exception as e:
        logger.warning("Content calendar generation failed: %s", e)

    # Generate guidelines.md ONLY if none exists — never overwrite manually curated guidelines.
    # Use /regen_guidelines for intentional updates (which uses merge strategy).
    guidelines_path = brand_path / "guidelines.md"
    if not guidelines_path.exists():
        try:
            guidelines_md = await generate_guidelines_from_audit(session, rec)
            logger.info("Generated guidelines.md from audit data for %s", session.brand_name)
        except Exception as e:
            logger.warning("Claude guidelines generation failed, using template: %s", e)
            guidelines_md = _guidelines_template_fallback(session, rec)

        guidelines_path.write_text(guidelines_md, encoding="utf-8")

    # Invalidate caches
    compositor_config.invalidate_cache()

    # Mark session complete
    session.state = OnboardingState.COMPLETE.value
    save_session(session)

    return (
        f"<b>Onboarding Complete!</b>\n\n"
        f"Brand <b>{session.brand_name}</b> is ready.\n\n"
        f"Files created:\n"
        f"- <code>brand/config.json</code>\n"
        f"- <code>brand/strategy.md</code>\n"
        f"- <code>brand/guidelines.md</code>\n"
        f"- <code>brand/content_calendar.md</code>\n\n"
        f"Use /strategy to view your setup. "
        f"Send me a content request to generate your first post!"
    )
