"""
Configuration loaded from .env file.
All settings are module-level variables for simple import.
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

_logger = logging.getLogger(__name__)

# --- LLM ---
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# --- Image Generation ---
REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
IMAGE_MODEL: str = os.getenv("IMAGE_MODEL", "auto")  # "auto" or specific model name

# --- Telegram ---
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USER_ID: int = int(os.getenv("TELEGRAM_ALLOWED_USER_ID", "0"))

# --- X / Twitter ---
X_API_KEY: str = os.getenv("X_API_KEY", "")
X_API_SECRET: str = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN: str = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_SECRET: str = os.getenv("X_ACCESS_SECRET", "")
X_BEARER_TOKEN: str = os.getenv("X_BEARER_TOKEN", "")

# --- Brand ---
BRAND_FOLDER: str = os.getenv("BRAND_FOLDER", str(_project_root / "brand"))
BRAND_NAME: str = os.getenv("BRAND_NAME", "MyBrand")

# --- Pipeline ---
PIPELINE_MODE: str = os.getenv("PIPELINE_MODE", "full")  # "full" (4-step) or "fast" (3-step, merges plan+verify)
MAX_REFERENCE_CHARS: int = int(os.getenv("MAX_REFERENCE_CHARS", "50000"))
REFERENCES_FOLDER: str = os.getenv("REFERENCES_FOLDER", str(Path(BRAND_FOLDER) / "references"))

# --- Agent Mode ---
AGENT_MODE: str = os.getenv("AGENT_MODE", "pipeline")  # "pipeline" (existing) or "agent" (tool-use loop)
AGENT_MAX_TURNS: int = int(os.getenv("AGENT_MAX_TURNS", "15"))
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "claude-opus-4-6")
FEEDBACK_SUMMARIZE_EVERY: int = int(os.getenv("FEEDBACK_SUMMARIZE_EVERY", "10"))

# --- Figma ---
FIGMA_ACCESS_TOKEN: str = os.getenv("FIGMA_ACCESS_TOKEN", "")
FIGMA_FILE_KEY: str = os.getenv("FIGMA_FILE_KEY", "")
FIGMA_NODE_ID: str = os.getenv("FIGMA_NODE_ID", "")

# --- OpenClaw ---
OPENCLAW_SCRIPTS_DIR: str = os.getenv(
    "OPENCLAW_SCRIPTS_DIR",
    str(Path.home() / ".openclaw" / "skills" / "brand-mover" / "scripts"),
)

# --- Auto Post ---
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


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate(exit_on_error: bool = True) -> list[str]:
    """Check required settings and warn about optional ones.

    Returns list of error messages. If exit_on_error is True, prints errors
    and calls sys.exit(1) when critical settings are missing.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Critical (bot won't function) ---
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN is required")
    if not TELEGRAM_ALLOWED_USER_ID:
        errors.append("TELEGRAM_ALLOWED_USER_ID is required (must be non-zero)")
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required for content generation")

    # --- Important (features degraded) ---
    if not REPLICATE_API_TOKEN:
        warnings.append("REPLICATE_API_TOKEN not set — image generation will be disabled")
    if not X_API_KEY or not X_ACCESS_TOKEN:
        warnings.append("X/Twitter credentials incomplete — posting to X will fail")

    # --- Informational ---
    if not FIGMA_ACCESS_TOKEN:
        warnings.append("FIGMA_ACCESS_TOKEN not set — Figma design checks disabled")

    # --- Value validation ---
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
