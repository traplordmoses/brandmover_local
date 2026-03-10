"""
BrandMover Local — entry point.

This is the top-level entry point for the entire application. It does three things:
1. Configures logging (stdout + persistent file) BEFORE any other imports
2. Runs a startup asset scan to index new brand files
3. Launches the Telegram bot polling loop (which runs forever)

Architecture note: Logging is configured first because many modules log during
import (e.g., settings.py validates config, state.py runs file migrations).
If we imported those modules before configuring logging, their messages would
be lost or use Python's default format.
"""

import logging
import sys
import tempfile
from pathlib import Path

# Persistent log file so we can debug issues even after the terminal is closed.
# Lives in /tmp/brandmover_bot.log on macOS/Linux.
_LOG_FILE = str(Path(tempfile.gettempdir()) / "brandmover_bot.log")

# Log to both stdout (for real-time monitoring) and a file (for post-mortem debugging).
# File handler is created separately so we can set its level independently if needed.
_file_handler = logging.FileHandler(_LOG_FILE)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
)

# basicConfig sets up the root logger — all child loggers (agent.*, bot.*, etc.)
# inherit this configuration automatically.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), _file_handler],
)

# Import AFTER logging setup — telegram_bot imports settings.py which validates
# config and logs warnings. We want those messages captured.
from bot import telegram_bot  # noqa: E402 — after logging setup

if __name__ == "__main__":
    logging.getLogger().info("Bot starting — logs also written to %s", _LOG_FILE)

    # Startup asset scan — index new files added to brand/assets/ since last run.
    # This populates an in-memory index used by generate_image for brand-specific
    # assets (logos, mascots, background textures, etc.).
    try:
        from agent import asset_library
        count = asset_library.index_directory()
        if count:
            logging.getLogger().info("Asset library: indexed %d new files on startup", count)
    except Exception as e:
        # Non-fatal — the bot can still run without the asset library.
        logging.getLogger().warning("Asset library startup scan failed: %s", e)

    # Start the Telegram bot polling loop. This call blocks forever —
    # it connects to Telegram's API and waits for messages.
    # The bot's background tasks (auto-poster, scheduler) are launched
    # inside telegram_bot.run() via a post_init hook.
    telegram_bot.run()
