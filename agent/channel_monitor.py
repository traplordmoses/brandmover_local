"""
Telegram channel message logger.

Stores recent messages from monitored channels in state/channel_messages.json
for community sentiment analysis via the read_telegram_channel tool.
"""

import json
import logging
import os
import threading
from pathlib import Path

from agent.paths import STATE_DIR

logger = logging.getLogger(__name__)

_CHANNEL_MESSAGES_FILE = STATE_DIR / "channel_messages.json"


def _read_channel_messages() -> list[dict]:
    """Read stored channel messages. Returns empty list on error."""
    if not _CHANNEL_MESSAGES_FILE.exists():
        return []
    try:
        return json.loads(_CHANNEL_MESSAGES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def log_channel_message(chat_id: int, author: str, text: str, timestamp: float) -> None:
    """Called from the Telegram message handler to log channel messages."""
    messages = _read_channel_messages()
    messages.append({
        "chat_id": chat_id,
        "author": author,
        "text": text[:500],
        "timestamp": timestamp,
    })
    # Keep last 100 messages
    messages = messages[-100:]
    _CHANNEL_MESSAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _CHANNEL_MESSAGES_FILE.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}")
    tmp_path.write_text(
        json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(str(tmp_path), str(_CHANNEL_MESSAGES_FILE))


def read_channel_messages(channel_id: str = "", limit: int = 20) -> list[dict]:
    """Read stored channel messages, optionally filtered by channel ID."""
    messages = _read_channel_messages()
    if channel_id:
        try:
            cid = int(channel_id)
            messages = [m for m in messages if m.get("chat_id") == cid]
        except ValueError:
            pass
    return messages[-min(limit, 50):]
