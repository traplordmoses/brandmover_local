"""
Per-user conversation context tracking for intent routing.

Persists to state/conversation.json. Auto-prunes entries older than 24 hours.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_STATE_DIR = _project_root / "state"
_CONTEXT_FILE = _STATE_DIR / "conversation.json"

_PRUNE_AGE_SECONDS = 24 * 60 * 60  # 24 hours
_MAX_RECENT_INTENTS = 5
_MAX_CONVERSATION_HISTORY = 20


@dataclass
class ConversationContext:
    user_id: int
    last_bot_action: str = "idle"  # "sent_draft" | "asked_question" | "sent_content" | "idle"
    last_bot_message: str = ""
    pending_draft_exists: bool = False
    last_content_type: str = ""
    last_command: str = ""
    recent_intents: list[str] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    user_name: str = ""
    updated_at: float = 0.0


def _load_all() -> dict[str, dict]:
    """Load all contexts from disk, pruning stale entries."""
    if not _CONTEXT_FILE.exists():
        return {}
    try:
        raw = json.loads(_CONTEXT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read conversation.json: %s", e)
        return {}

    now = time.time()
    pruned = {
        uid: ctx
        for uid, ctx in raw.items()
        if now - ctx.get("updated_at", 0) < _PRUNE_AGE_SECONDS
    }
    if len(pruned) < len(raw):
        _save_all(pruned)
    return pruned


def _save_all(data: dict[str, dict]) -> None:
    """Write all contexts to disk."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _CONTEXT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_context(user_id: int) -> ConversationContext:
    """Return context for a user, creating a fresh one if none exists."""
    all_ctx = _load_all()
    key = str(user_id)
    if key in all_ctx:
        data = all_ctx[key]
        return ConversationContext(
            user_id=data.get("user_id", user_id),
            last_bot_action=data.get("last_bot_action", "idle"),
            last_bot_message=data.get("last_bot_message", ""),
            pending_draft_exists=data.get("pending_draft_exists", False),
            last_content_type=data.get("last_content_type", ""),
            last_command=data.get("last_command", ""),
            recent_intents=data.get("recent_intents", []),
            conversation_history=data.get("conversation_history", []),
            user_name=data.get("user_name", ""),
            updated_at=data.get("updated_at", 0.0),
        )
    return ConversationContext(user_id=user_id, updated_at=time.time())


def update_context(user_id: int, **fields) -> ConversationContext:
    """Update specific fields on a user's context and persist."""
    ctx = get_context(user_id)
    for key, value in fields.items():
        if hasattr(ctx, key):
            setattr(ctx, key, value)
    # Trim recent_intents to max size
    if len(ctx.recent_intents) > _MAX_RECENT_INTENTS:
        ctx.recent_intents = ctx.recent_intents[-_MAX_RECENT_INTENTS:]
    # Trim conversation history
    if len(ctx.conversation_history) > _MAX_CONVERSATION_HISTORY:
        ctx.conversation_history = ctx.conversation_history[-_MAX_CONVERSATION_HISTORY:]
    ctx.updated_at = time.time()

    all_ctx = _load_all()
    all_ctx[str(user_id)] = asdict(ctx)
    _save_all(all_ctx)
    return ctx


def clear_context(user_id: int) -> None:
    """Remove a user's context entirely."""
    all_ctx = _load_all()
    key = str(user_id)
    if key in all_ctx:
        del all_ctx[key]
        _save_all(all_ctx)
