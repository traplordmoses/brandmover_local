"""
Configuration loaded from .env file.

All settings are module-level variables for simple import. This means any module
can do `from config import settings` and access `settings.BRAND_NAME` directly.

ARCHITECTURE:
- Uses python-dotenv to load .env from project root at import time
- All values come from environment variables with sensible defaults
- Boolean settings use the pattern: os.getenv(...).lower() in ("true", "1", "yes")
- validate() checks critical settings on startup and exits if missing

INTERVIEW TALKING POINT:
"We use a flat configuration module instead of a config class or YAML files.
Every setting is a module-level constant loaded from environment variables.
This keeps configuration simple, testable (monkeypatch any setting), and
12-factor app compliant (config from environment, not code)."
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root — this must happen before any os.getenv() calls below.
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

_logger = logging.getLogger(__name__)

# ── LLM Provider ──
# Anthropic is the primary provider (Claude for reasoning + generation).
# OpenAI is used for Whisper (voice transcription).
# Gemini is an alternative LLM option.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Image Generation ──
# Replicate hosts the image models (Flux, Seedream, Nano-Banana, Recraft).
# IMAGE_MODEL: "auto" = smart routing based on content type (see image_gen.py),
# or a specific model name to override.
REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
IMAGE_MODEL: str = os.getenv("IMAGE_MODEL", "auto")

# ── Telegram ──
# TELEGRAM_ALLOWED_USER_ID: The admin user (full access to all commands).
# TELEGRAM_OPERATOR_IDS: Additional users who can generate content but not configure.
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USER_ID: int = int(os.getenv("TELEGRAM_ALLOWED_USER_ID", "0"))
_raw_operator_ids = os.getenv("TELEGRAM_OPERATOR_IDS", "")
TELEGRAM_OPERATOR_IDS: list[int] = [
    int(uid.strip()) for uid in _raw_operator_ids.split(",")
    if uid.strip().isdigit()
]

# ── X / Twitter ──
# Full OAuth 1.0a credentials for tweepy v2.
# Both consumer keys (API_KEY/SECRET) and access tokens are required for posting.
X_API_KEY: str = os.getenv("X_API_KEY", "")
X_API_SECRET: str = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN: str = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_SECRET: str = os.getenv("X_ACCESS_SECRET", "")
X_BEARER_TOKEN: str = os.getenv("X_BEARER_TOKEN", "")

# ── Brand ──
# BRAND_FOLDER: Root directory for all brand assets (guidelines.md, prompts/, assets/, etc.)
# BRAND_NAME: Used in system prompts and generation context.
BRAND_FOLDER: str = os.getenv("BRAND_FOLDER", str(_project_root / "brand"))
BRAND_NAME: str = os.getenv("BRAND_NAME", "MyBrand")

# ── Pipeline ──
# Legacy pipeline mode (4-step: analyze → plan → verify → generate).
# "full" runs all 4 steps, "fast" merges plan+verify into one step.
PIPELINE_MODE: str = os.getenv("PIPELINE_MODE", "full")
MAX_REFERENCE_CHARS: int = int(os.getenv("MAX_REFERENCE_CHARS", "50000"))
REFERENCES_FOLDER: str = os.getenv("REFERENCES_FOLDER", str(Path(BRAND_FOLDER) / "references"))

# ── Agent Mode ──
# AGENT_MODE: "pipeline" uses the legacy multi-step pipeline, "agent" uses
# a Claude tool-use loop (the modern approach — see unified_brain.py).
# AGENT_MAX_TURNS: Max LLM round-trips per request. More turns = more tool calls
# but higher cost. 15 is generous for complex multi-tool workflows.
# SONNET_MODEL: The main model used for all brain calls. Sonnet for speed+cost.
# HAIKU_MODEL: Used for lightweight tasks (intent classification, etc.)
AGENT_MODE: str = os.getenv("AGENT_MODE", "pipeline")
AGENT_MAX_TURNS: int = int(os.getenv("AGENT_MAX_TURNS", "15"))
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
SONNET_MODEL: str = os.getenv("SONNET_MODEL", "claude-sonnet-4-6")
HAIKU_MODEL: str = os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001")
# FEEDBACK_SUMMARIZE_EVERY: After this many feedback entries, auto-trigger
# Claude to summarize patterns into learned_preferences.md.
FEEDBACK_SUMMARIZE_EVERY: int = int(os.getenv("FEEDBACK_SUMMARIZE_EVERY", "10"))
CHAT_MAX_TOKENS: int = int(os.getenv("CHAT_MAX_TOKENS", "600"))

# ── Discord ──
# Optional Discord bot for cross-posting content to Discord channels.
DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID: int = int(os.getenv("DISCORD_GUILD_ID", "0"))

# ── Figma ──
# Optional Figma integration for design-accurate content generation.
# The agent can check Figma designs for colors, typography, and layout.
FIGMA_ACCESS_TOKEN: str = os.getenv("FIGMA_ACCESS_TOKEN", "")
FIGMA_FILE_KEY: str = os.getenv("FIGMA_FILE_KEY", "")
FIGMA_NODE_ID: str = os.getenv("FIGMA_NODE_ID", "")

# ── OpenClaw ──
# Path to OpenClaw scripts directory. OpenClaw is an external tool
# that provides Node.js scripts for specialized tasks.
OPENCLAW_SCRIPTS_DIR: str = os.getenv(
    "OPENCLAW_SCRIPTS_DIR",
    str(Path.home() / ".openclaw" / "skills" / "brand-mover" / "scripts"),
)

# ── Intent Router ──
# When enabled, classifies incoming messages (approve/reject/edit/reroll/generate)
# before routing to the appropriate handler. Saves cost by avoiding full brain
# calls for simple commands like "yes" or "looks good".
INTENT_ROUTER_ENABLED: bool = os.getenv("INTENT_ROUTER_ENABLED", "true").lower() in ("true", "1", "yes")

# ── Whisper (voice transcription) ──
# Uses OpenAI's Whisper API to transcribe voice messages into text.
# Auto-enabled if OPENAI_API_KEY is set.
WHISPER_ENABLED: bool = os.getenv(
    "WHISPER_ENABLED",
    "true" if os.getenv("OPENAI_API_KEY") else "false",
).lower() in ("true", "1", "yes")

# ── Unified Brain ──
# When True, ALL messages go through unified_brain.py (single LLM loop).
# When False, legacy routing is used (separate chat + generation paths).
UNIFIED_BRAIN_ENABLED: bool = os.getenv("UNIFIED_BRAIN_ENABLED", "false").lower() in ("true", "1", "yes")

# ── Auto Post ──
# Background scheduler that generates and posts content at predefined times.
# AUTO_POST_DRY_RUN: If True, generates content but doesn't actually post to X.
AUTO_POST_ENABLED: bool = os.getenv("AUTO_POST_ENABLED", "false").lower() in ("true", "1", "yes")
AUTO_POST_SCHEDULE_FILE: str = os.getenv(
    "AUTO_POST_SCHEDULE_FILE",
    str(_project_root / "config" / "schedule.json"),
)
AUTO_POST_STATE_FILE: str = os.getenv(
    "AUTO_POST_STATE_FILE",
    str(_project_root / "state" / "auto_post_state.json"),
)
AUTO_POST_DRY_RUN: bool = os.getenv("AUTO_POST_DRY_RUN", "false").lower() in ("true", "1", "yes")

# ── Timezone ──
# IANA timezone name for interpreting user-provided times (e.g. "America/Chicago").
# Used by schedule_queue.py to convert "3pm" into the correct UTC timestamp.
# If not set, uses the system's local timezone automatically.
TIMEZONE: str = os.getenv("TIMEZONE", "")

# ── Telegram channel monitoring ──
# Comma-separated channel/group IDs to silently log messages from.
# The bot records text messages from these channels into state/channel_messages.json.
# Used by the read_telegram_channel tool for community sentiment analysis.
TELEGRAM_MONITOR_CHANNELS: str = os.getenv("TELEGRAM_MONITOR_CHANNELS", "")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate(exit_on_error: bool = True) -> list[str]:
    """Check required settings and warn about optional ones.

    Called once at startup by telegram_bot.py. Separates errors (bot won't work)
    from warnings (features degraded). This catches misconfiguration early
    instead of failing at runtime when a user tries to generate content.

    Returns list of error messages. If exit_on_error is True, prints errors
    and calls sys.exit(1) when critical settings are missing.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── Critical (bot won't start without these) ──
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN is required")
    if not TELEGRAM_ALLOWED_USER_ID:
        errors.append("TELEGRAM_ALLOWED_USER_ID is required (must be non-zero)")
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required for content generation")

    # ── Important (features degraded but bot runs) ──
    if not REPLICATE_API_TOKEN:
        warnings.append("REPLICATE_API_TOKEN not set — image generation will be disabled")
    if not X_API_KEY or not X_ACCESS_TOKEN:
        warnings.append("X/Twitter credentials incomplete — posting to X will fail")

    # ── Informational ──
    if not DISCORD_BOT_TOKEN:
        warnings.append("DISCORD_BOT_TOKEN not set — Discord posting disabled")
    if not FIGMA_ACCESS_TOKEN:
        warnings.append("FIGMA_ACCESS_TOKEN not set — Figma design checks disabled")

    # ── Value validation ──
    if AGENT_MAX_TURNS < 1:
        errors.append(f"AGENT_MAX_TURNS must be >= 1, got {AGENT_MAX_TURNS}")
    if FEEDBACK_SUMMARIZE_EVERY < 1:
        errors.append(f"FEEDBACK_SUMMARIZE_EVERY must be >= 1, got {FEEDBACK_SUMMARIZE_EVERY}")
    if MAX_REFERENCE_CHARS < 0:
        errors.append(f"MAX_REFERENCE_CHARS must be >= 0, got {MAX_REFERENCE_CHARS}")

    for w in warnings:
        _logger.warning("Config: %s", w)

    if errors:
        for e in errors:
            _logger.error("Config: %s", e)
        if exit_on_error:
            print("\n".join(f"  ERROR: {e}" for e in errors), file=sys.stderr)
            sys.exit(1)

    return errors
