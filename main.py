"""
BrandMover Local — entry point.
Starts the Telegram bot polling loop.
"""

import logging
import sys
import tempfile
from pathlib import Path

_LOG_FILE = str(Path(tempfile.gettempdir()) / "brandmover_bot.log")

# Log to both stdout and a persistent file
_file_handler = logging.FileHandler(_LOG_FILE)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), _file_handler],
)

from bot import telegram_bot  # noqa: E402 — after logging setup

if __name__ == "__main__":
    logging.getLogger().info("Bot starting — logs also written to %s", _LOG_FILE)

    # Startup asset scan — index new files in brand/assets/
    try:
        from agent import asset_library
        count = asset_library.index_directory()
        if count:
            logging.getLogger().info("Asset library: indexed %d new files on startup", count)
    except Exception as e:
        logging.getLogger().warning("Asset library startup scan failed: %s", e)

    telegram_bot.run()
