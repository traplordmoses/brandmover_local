"""
BrandMover Local — entry point.
Starts the Telegram bot polling loop.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

from bot import telegram_bot  # noqa: E402 — after logging setup

if __name__ == "__main__":
    telegram_bot.run()
