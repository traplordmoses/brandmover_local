"""
Audit log — append-only log for sensitive operations.

Logs: post approvals, code execution, script execution, tool calls with user attribution.
Format: newline-delimited JSON (NDJSON) in state/audit.log.
"""

import json
import logging
import time
from pathlib import Path

from agent.paths import STATE_DIR

logger = logging.getLogger(__name__)

_AUDIT_FILE = STATE_DIR / "audit.log"


def audit(action: str, user_id: int | None = None, **details) -> None:
    """Append an audit entry for a sensitive operation.

    Args:
        action: What happened (e.g. "approve_draft", "execute_code", "post_to_x").
        user_id: Telegram user ID who triggered it.
        **details: Additional context (tool_name, caption, script, etc.)
    """
    entry = {
        "ts": time.time(),
        "action": action,
        "user_id": user_id,
        **details,
    }
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning("Failed to write audit log: %s", e)
