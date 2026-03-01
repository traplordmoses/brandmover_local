"""
Conversational onboarding state machine — guides new brands through setup.

Persisted in state/onboarding.json. Survives bot restarts.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "onboarding.json"


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class OnboardingState(str, Enum):
    IDLE = "idle"
    PROJECT_NAME = "project_name"
    DESCRIPTION = "description"
    PLATFORMS = "platforms"
    ASSET_CHECK = "asset_check"
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
    return OnboardingSession(**data)


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
    }
    _save_sessions(sessions)


def delete_session(user_id: int) -> None:
    """Remove an onboarding session."""
    sessions = _load_sessions()
    sessions.pop(str(user_id), None)
    _save_sessions(sessions)


# ---------------------------------------------------------------------------
# State machine transitions
# ---------------------------------------------------------------------------

_VISUAL_STYLES = {"modern", "playful", "corporate", "minimal", "bold", "elegant"}


def advance(session: OnboardingSession, user_input: str | None) -> tuple[OnboardingSession, str]:
    """Advance the state machine based on user input.

    Returns (updated_session, response_message).
    """
    st = session.state
    inp = (user_input or "").strip()
    inp_lower = inp.lower()

    if st == OnboardingState.IDLE.value:
        # Starting onboarding
        session.state = OnboardingState.PROJECT_NAME.value
        session.started_at = time.time()
        return session, (
            "Let's set up your brand! This takes about 2 minutes.\n\n"
            "What's your brand name?"
        )

    elif st == OnboardingState.PROJECT_NAME.value:
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

    elif st == OnboardingState.UPLOADS.value:
        # Accumulate uploads — photos are handled by handle_photo
        # This state handles text messages during upload phase
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
        # This state is transitioned through programmatically after Claude Vision
        # audit completes — see finalize_audit() below
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
        # After audit, show archetype recommendation
        session.state = OnboardingState.STRATEGY.value
        return session, "Generating your brand strategy based on your assets..."

    elif st == OnboardingState.STRATEGY.value:
        # This state is transitioned through programmatically after strategy generation
        return session, "Still generating strategy... please wait."

    elif st == OnboardingState.CONFIRM.value:
        if inp_lower in ("yes", "y", "confirm", "ok", "looks good"):
            session.state = OnboardingState.COMPLETE.value
            return session, "Setting up your brand..."
        elif inp_lower in ("no", "n", "restart", "redo"):
            session.state = OnboardingState.PROJECT_NAME.value
            session.brand_name = ""
            session.description = ""
            session.platforms = []
            session.uploaded_assets = []
            session.asset_audit = {}
            session.visual_preferences = {}
            session.strategy = {}
            return session, "Let's start over.\n\nWhat's your brand name?"
        return session, "Reply <b>yes</b> to confirm or <b>no</b> to restart."

    elif st == OnboardingState.COMPLETE.value:
        return session, "Onboarding already complete! Your brand is set up."

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
        visual_style_notes=rec_data.get("visual_style_notes", ""),
        reasoning=rec_data.get("reasoning", ""),
    )

    # Save config.json + strategy.md
    strategy_mod.save_strategy(rec, session.brand_name)

    # Generate minimal guidelines.md if not present
    guidelines_path = brand_path / "guidelines.md"
    if not guidelines_path.exists():
        style_info = session.visual_preferences.get("description", session.visual_preferences.get("style", "modern"))
        audit_colors = session.asset_audit.get("consolidated_colors", [])
        color_rows = ""
        for i, c in enumerate(audit_colors[:6]):
            role = ["Primary", "Accent 1", "Accent 2", "Background", "Text", "Accent 3"][min(i, 5)]
            color_rows += f"| {role} | {c.get('name', 'Color')} | {c.get('hex', '#000000')} | |\n"

        guidelines_md = (
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
            guidelines_md += f"| Badge text | {rec.badge_text} |\n"
        guidelines_md += f"| Default mode | {rec.default_mode} |\n"

        guidelines_path.write_text(guidelines_md, encoding="utf-8")
        logger.info("Generated guidelines.md for %s", session.brand_name)

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
        f"- <code>brand/guidelines.md</code>\n\n"
        f"Send me a content request to generate your first post!"
    )
