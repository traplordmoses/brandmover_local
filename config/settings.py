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

# Load .env — supports BRANDMOVER_ENV_FILE override for multi-brand instances.
# Default: .env from project root. Override: set BRANDMOVER_ENV_FILE=/path/to/.env.brand2
_project_root = Path(__file__).resolve().parent.parent
_env_file = os.getenv("BRANDMOVER_ENV_FILE", str(_project_root / ".env"))
load_dotenv(_env_file)

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
# STATE_FOLDER: Root for all runtime state (pending drafts, feedback, transcripts, etc.)
# Defaults to state/ in project root. Override for multi-brand isolation.
STATE_FOLDER: str = os.getenv("STATE_FOLDER", str(_project_root / "state"))

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
AGENT_MODE: str = os.getenv("AGENT_MODE", "agent")
AGENT_MAX_TURNS: int = int(os.getenv("AGENT_MAX_TURNS", "15"))
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
AGENT_SELF_CRITIQUE: bool = os.getenv("AGENT_SELF_CRITIQUE", "true").lower() in ("true", "1", "yes")
SONNET_MODEL: str = os.getenv("SONNET_MODEL", "claude-sonnet-4-6")
HAIKU_MODEL: str = os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001")
# FEEDBACK_SUMMARIZE_EVERY: After this many feedback entries, auto-trigger
# Claude to summarize patterns into learned_preferences.md.
FEEDBACK_SUMMARIZE_EVERY: int = int(os.getenv("FEEDBACK_SUMMARIZE_EVERY", "10"))
CHAT_MAX_TOKENS: int = int(os.getenv("CHAT_MAX_TOKENS", "600"))
# Model fallback — comma-separated fallback chain for agent calls.
# When primary model fails (429/500/503), tries the next model in the chain.
AGENT_FALLBACK_MODELS: str = os.getenv("AGENT_FALLBACK_MODELS", "")

# ── Discord ──
# Optional Discord bot for cross-posting content to Discord channels.
DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID: int = int(os.getenv("DISCORD_GUILD_ID", "0"))

# ── Multi-Platform Publishing ──
# PUBLISH_PLATFORMS: Comma-separated list of platforms to publish to on /approve.
# Supported: x, discord, telegram. Default: ["x"].
# DISCORD_CROSSPOST_ENABLED: Automatically cross-post to Discord when DISCORD_BOT_TOKEN is set.
_raw_platforms = os.getenv("PUBLISH_PLATFORMS", "x")
PUBLISH_PLATFORMS: list[str] = [
    p.strip().lower() for p in _raw_platforms.split(",") if p.strip()
]
DISCORD_CROSSPOST_ENABLED: bool = os.getenv(
    "DISCORD_CROSSPOST_ENABLED",
    "true" if os.getenv("DISCORD_BOT_TOKEN") else "false",
).lower() in ("true", "1", "yes")
# Auto-add discord to PUBLISH_PLATFORMS when cross-post is enabled
if DISCORD_CROSSPOST_ENABLED and "discord" not in PUBLISH_PLATFORMS:
    PUBLISH_PLATFORMS.append("discord")

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

# ── ElevenLabs TTS ──
# Used for video voiceover. Falls back to OpenAI TTS if not set.
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

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

# ── Heartbeat ──
# HEARTBEAT_ENABLED: When True, the scheduler uses the heartbeat reasoning layer
# (assess → reason → dispatch). When False, falls back to the original cron loop.
# HEARTBEAT_PROACTIVE_HOURS: After this many hours without a post, the heartbeat
# triggers a proactive content generation (Claude decides what to post about).
HEARTBEAT_ENABLED: bool = os.getenv("HEARTBEAT_ENABLED", "true").lower() in ("true", "1", "yes")
HEARTBEAT_PROACTIVE_HOURS: int = int(os.getenv("HEARTBEAT_PROACTIVE_HOURS", "8"))

# ── Content Mix Ratios ──
# Configurable content type distribution for auto-posting (proactive/scheduled).
# Format: "type:weight,type:weight,..." — weights are relative, not percentages.
# Default: educational 25%, community 20%, announcement 20%, engagement 15%,
#          lifestyle 10%, meme 10%. Adjust via CONTENT_MIX_RATIOS env var.
_raw_mix = os.getenv(
    "CONTENT_MIX_RATIOS",
    "educational:25,community:20,announcement:20,engagement:15,lifestyle:10,meme:10"
)
CONTENT_MIX_RATIOS: dict[str, int] = {}
for _pair in _raw_mix.split(","):
    _pair = _pair.strip()
    if ":" in _pair:
        _k, _v = _pair.split(":", 1)
        _k = _k.strip()
        _v = _v.strip()
        if _k and _v.isdigit():
            CONTENT_MIX_RATIOS[_k] = int(_v)

# ── Topic Bank ──
# How often (hours) to refresh the topic bank with new Claude-generated angles.
TOPIC_BANK_REFRESH_INTERVAL_HOURS: int = int(os.getenv("TOPIC_BANK_REFRESH_INTERVAL_HOURS", "72"))

# ── Auto Preference Extraction ──
# Periodically analyzes approval/rejection patterns to auto-generate preferences.
PREF_EXTRACTION_ENABLED: bool = os.getenv("PREF_EXTRACTION_ENABLED", "true").lower() in ("true", "1", "yes")
PREF_EXTRACTION_MIN_EVENTS: int = int(os.getenv("PREF_EXTRACTION_MIN_EVENTS", "5"))
PREF_EXTRACTION_INTERVAL_HOURS: int = int(os.getenv("PREF_EXTRACTION_INTERVAL_HOURS", "24"))

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

# ── Cost Gate ──
# Daily budget cap for image generation spend (estimated costs from Replicate).
# When cumulative daily spend reaches this limit, check_cost_budget() returns allowed=False.
DAILY_COST_BUDGET_USD: float = float(os.getenv("DAILY_COST_BUDGET_USD", "5.0"))

# ── Content Planner ──
# Rolling 7-day content plan with automatic type balancing.
# CONTENT_PLANNER_ENABLED: opt-in; when True, auto_post uses the planner
# instead of slot-based scheduling.
# PLAN_HORIZON_DAYS: how many days ahead the planner maintains.
CONTENT_PLANNER_ENABLED: bool = os.getenv("CONTENT_PLANNER_ENABLED", "false").lower() in ("true", "1", "yes")
PLAN_HORIZON_DAYS: int = int(os.getenv("PLAN_HORIZON_DAYS", "7"))

# ── Skeleton Library & Diversity Tracking ──
# Structural skeletons ensure content variety by assigning structure templates
# (hook type, body flow, CTA style) before generation. The diversity tracker
# checks proposed structures against recent posts to prevent repetition.
SKELETON_LIBRARY_ENABLED: bool = os.getenv("SKELETON_LIBRARY_ENABLED", "true").lower() in ("true", "1", "yes")
DIVERSITY_TRACKER_ENABLED: bool = os.getenv("DIVERSITY_TRACKER_ENABLED", "true").lower() in ("true", "1", "yes")

# ── Performance Tracking ──
# Tracks post engagement metrics (likes, retweets, impressions) from X API
# and feeds data back into skeleton selection and content planning.
PERFORMANCE_TRACKING_ENABLED: bool = os.getenv("PERFORMANCE_TRACKING_ENABLED", "true").lower() in ("true", "1", "yes")
PERFORMANCE_REFRESH_HOURS: int = int(os.getenv("PERFORMANCE_REFRESH_HOURS", "6"))

# ── Draft Scoring ──
# Preference engine scores drafts against learned approval/rejection patterns.
# DRAFT_SCORE_THRESHOLD: Minimum score (1-10) to pass. Drafts below are flagged.
# DRAFT_SCORE_ENABLED: Master toggle for the scoring system.
DRAFT_SCORE_THRESHOLD: float = float(os.getenv("DRAFT_SCORE_THRESHOLD", "6.0"))
DRAFT_SCORE_ENABLED: bool = os.getenv("DRAFT_SCORE_ENABLED", "true").lower() in ("true", "1", "yes")

# ── Context Feed ──
# Real-time context aggregator that feeds on-chain events and X mentions
# into auto-post content generation prompts.
CONTEXT_FEED_ENABLED: bool = os.getenv("CONTEXT_FEED_ENABLED", "true").lower() in ("true", "1", "yes")
X_MENTIONS_ENABLED: bool = os.getenv("X_MENTIONS_ENABLED", "false").lower() in ("true", "1", "yes")
X_MENTIONS_POLL_MINUTES: int = int(os.getenv("X_MENTIONS_POLL_MINUTES", "30"))
EVENT_TRIGGER_ENABLED: bool = os.getenv("EVENT_TRIGGER_ENABLED", "true").lower() in ("true", "1", "yes")

# ── Monitoring ──
# Daily digest: summary of bot performance sent at DAILY_DIGEST_HOUR (UTC).
DAILY_DIGEST_ENABLED: bool = os.getenv("DAILY_DIGEST_ENABLED", "true").lower() in ("true", "1", "yes")
DAILY_DIGEST_HOUR: int = int(os.getenv("DAILY_DIGEST_HOUR", "21"))
# Health monitor: periodic system health checks with Telegram alerts.
HEALTH_CHECK_ENABLED: bool = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() in ("true", "1", "yes")
HEALTH_ALERT_ENABLED: bool = os.getenv("HEALTH_ALERT_ENABLED", "true").lower() in ("true", "1", "yes")


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

    # ── Deprecation checks ──
    if AGENT_MODE == "pipeline":
        _logger.warning(
            "AGENT_MODE=pipeline is deprecated. Switch to AGENT_MODE=agent."
        )

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
