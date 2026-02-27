"""
Configuration loaded from .env file.
All settings are module-level variables for simple import.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

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
