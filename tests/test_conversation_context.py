"""Tests for agent.conversation_context — per-user conversation state tracking."""

import json
import time
from unittest.mock import patch

import pytest

from agent.conversation_context import (
    ConversationContext,
    get_context,
    update_context,
    clear_context,
    _load_all,
    _PRUNE_AGE_SECONDS,
    _MAX_RECENT_INTENTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _use_tmp_state(tmp_path, monkeypatch):
    """Point conversation context to a temp directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ctx_file = state_dir / "conversation.json"
    monkeypatch.setattr("agent.conversation_context._STATE_DIR", state_dir)
    monkeypatch.setattr("agent.conversation_context._CONTEXT_FILE", ctx_file)


# ---------------------------------------------------------------------------
# ConversationContext dataclass
# ---------------------------------------------------------------------------

class TestConversationContext:
    def test_defaults(self):
        ctx = ConversationContext(user_id=123)
        assert ctx.user_id == 123
        assert ctx.last_bot_action == "idle"
        assert ctx.last_bot_message == ""
        assert ctx.pending_draft_exists is False
        assert ctx.last_content_type == ""
        assert ctx.last_command == ""
        assert ctx.recent_intents == []
        assert ctx.updated_at == 0.0


# ---------------------------------------------------------------------------
# get_context
# ---------------------------------------------------------------------------

class TestGetContext:
    def test_returns_fresh_context_for_new_user(self):
        ctx = get_context(42)
        assert ctx.user_id == 42
        assert ctx.last_bot_action == "idle"
        assert ctx.updated_at > 0

    def test_returns_existing_context(self):
        update_context(42, last_bot_action="sent_draft", pending_draft_exists=True)
        ctx = get_context(42)
        assert ctx.last_bot_action == "sent_draft"
        assert ctx.pending_draft_exists is True

    def test_separate_users(self):
        update_context(1, last_bot_action="sent_draft")
        update_context(2, last_bot_action="idle")
        assert get_context(1).last_bot_action == "sent_draft"
        assert get_context(2).last_bot_action == "idle"


# ---------------------------------------------------------------------------
# update_context
# ---------------------------------------------------------------------------

class TestUpdateContext:
    def test_updates_single_field(self):
        ctx = update_context(10, last_bot_action="sent_content")
        assert ctx.last_bot_action == "sent_content"
        assert ctx.user_id == 10

    def test_updates_multiple_fields(self):
        ctx = update_context(
            10,
            last_bot_action="sent_draft",
            pending_draft_exists=True,
            last_content_type="announcement",
        )
        assert ctx.last_bot_action == "sent_draft"
        assert ctx.pending_draft_exists is True
        assert ctx.last_content_type == "announcement"

    def test_ignores_unknown_fields(self):
        ctx = update_context(10, nonexistent_field="value")
        assert not hasattr(ctx, "nonexistent_field") or ctx.last_bot_action == "idle"

    def test_sets_updated_at(self):
        before = time.time()
        ctx = update_context(10, last_bot_action="idle")
        assert ctx.updated_at >= before

    def test_persists_to_disk(self, tmp_path):
        update_context(99, last_bot_action="sent_draft")
        ctx_file = tmp_path / "state" / "conversation.json"
        assert ctx_file.exists()
        data = json.loads(ctx_file.read_text())
        assert "99" in data
        assert data["99"]["last_bot_action"] == "sent_draft"

    def test_trims_recent_intents(self):
        intents = [f"intent_{i}" for i in range(_MAX_RECENT_INTENTS + 3)]
        ctx = update_context(10, recent_intents=intents)
        assert len(ctx.recent_intents) == _MAX_RECENT_INTENTS
        # Should keep the most recent ones
        assert ctx.recent_intents[-1] == intents[-1]


# ---------------------------------------------------------------------------
# clear_context
# ---------------------------------------------------------------------------

class TestClearContext:
    def test_clears_existing(self):
        update_context(50, last_bot_action="sent_draft")
        clear_context(50)
        ctx = get_context(50)
        assert ctx.last_bot_action == "idle"

    def test_clear_nonexistent_is_safe(self):
        clear_context(999)  # should not raise


# ---------------------------------------------------------------------------
# Auto-prune
# ---------------------------------------------------------------------------

class TestAutoPrune:
    def test_prunes_old_entries(self, tmp_path):
        ctx_file = tmp_path / "state" / "conversation.json"
        old_time = time.time() - _PRUNE_AGE_SECONDS - 100
        data = {
            "1": {"user_id": 1, "last_bot_action": "idle", "updated_at": old_time},
            "2": {"user_id": 2, "last_bot_action": "sent_draft", "updated_at": time.time()},
        }
        ctx_file.write_text(json.dumps(data))

        loaded = _load_all()
        assert "1" not in loaded
        assert "2" in loaded

    def test_keeps_recent_entries(self, tmp_path):
        ctx_file = tmp_path / "state" / "conversation.json"
        now = time.time()
        data = {
            "1": {"user_id": 1, "last_bot_action": "idle", "updated_at": now - 100},
            "2": {"user_id": 2, "last_bot_action": "idle", "updated_at": now - 200},
        }
        ctx_file.write_text(json.dumps(data))

        loaded = _load_all()
        assert "1" in loaded
        assert "2" in loaded


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_round_trip(self):
        update_context(
            77,
            last_bot_action="sent_draft",
            pending_draft_exists=True,
            last_content_type="meme",
            last_command="/generate",
            recent_intents=["approve", "generate_content"],
            last_bot_message="Here's your draft",
        )
        ctx = get_context(77)
        assert ctx.user_id == 77
        assert ctx.last_bot_action == "sent_draft"
        assert ctx.pending_draft_exists is True
        assert ctx.last_content_type == "meme"
        assert ctx.last_command == "/generate"
        assert ctx.recent_intents == ["approve", "generate_content"]
        assert ctx.last_bot_message == "Here's your draft"

    def test_handles_corrupt_file(self, tmp_path):
        ctx_file = tmp_path / "state" / "conversation.json"
        ctx_file.write_text("not valid json{{{")
        ctx = get_context(1)
        assert ctx.user_id == 1
        assert ctx.last_bot_action == "idle"

    def test_handles_missing_file(self):
        ctx = get_context(1)
        assert ctx.user_id == 1
        assert ctx.last_bot_action == "idle"
