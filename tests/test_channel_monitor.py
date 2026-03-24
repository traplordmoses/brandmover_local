"""Tests for agent/channel_monitor.py — Telegram channel message logger."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_channel_file(tmp_path: Path):
    """Return a context manager that patches _CHANNEL_MESSAGES_FILE to tmp_path."""
    return patch(
        "agent.channel_monitor._CHANNEL_MESSAGES_FILE",
        tmp_path / "channel_messages.json",
    )


# ---------------------------------------------------------------------------
# log_channel_message()
# ---------------------------------------------------------------------------


class TestLogChannelMessage:
    def test_appends_message_to_empty_file(self, tmp_path):
        """First message creates the file and writes one entry."""
        from agent.channel_monitor import log_channel_message

        with _patch_channel_file(tmp_path):
            log_channel_message(chat_id=-1001, author="alice", text="hello", timestamp=100.0)

        data = json.loads((tmp_path / "channel_messages.json").read_text())
        assert len(data) == 1
        assert data[0]["chat_id"] == -1001
        assert data[0]["author"] == "alice"
        assert data[0]["text"] == "hello"
        assert data[0]["timestamp"] == 100.0

    def test_appends_multiple_messages(self, tmp_path):
        """Multiple calls accumulate messages."""
        from agent.channel_monitor import log_channel_message

        with _patch_channel_file(tmp_path):
            log_channel_message(chat_id=-1001, author="alice", text="msg1", timestamp=1.0)
            log_channel_message(chat_id=-1001, author="bob", text="msg2", timestamp=2.0)
            log_channel_message(chat_id=-1002, author="carol", text="msg3", timestamp=3.0)

        data = json.loads((tmp_path / "channel_messages.json").read_text())
        assert len(data) == 3
        assert data[2]["author"] == "carol"

    def test_cap_at_100_messages(self, tmp_path):
        """Messages are capped at 100, oldest evicted."""
        from agent.channel_monitor import log_channel_message

        with _patch_channel_file(tmp_path):
            # Write 100 messages
            for i in range(100):
                log_channel_message(chat_id=-1001, author=f"user_{i}", text=f"msg_{i}", timestamp=float(i))

            # Write one more — should evict the oldest
            log_channel_message(chat_id=-1001, author="new_user", text="new_msg", timestamp=200.0)

        data = json.loads((tmp_path / "channel_messages.json").read_text())
        assert len(data) == 100
        # Oldest message (user_0) should be gone
        authors = [m["author"] for m in data]
        assert "user_0" not in authors
        # Newest should be present
        assert data[-1]["author"] == "new_user"

    def test_text_truncated_to_500_chars(self, tmp_path):
        """Long text is truncated to 500 characters."""
        from agent.channel_monitor import log_channel_message

        long_text = "x" * 1000

        with _patch_channel_file(tmp_path):
            log_channel_message(chat_id=-1001, author="alice", text=long_text, timestamp=1.0)

        data = json.loads((tmp_path / "channel_messages.json").read_text())
        assert len(data[0]["text"]) == 500


# ---------------------------------------------------------------------------
# read_channel_messages()
# ---------------------------------------------------------------------------


class TestReadChannelMessages:
    def test_filter_by_channel_id(self, tmp_path):
        """Filtering by channel_id returns only matching messages."""
        from agent.channel_monitor import log_channel_message, read_channel_messages

        with _patch_channel_file(tmp_path):
            log_channel_message(chat_id=-1001, author="alice", text="chan1", timestamp=1.0)
            log_channel_message(chat_id=-1002, author="bob", text="chan2", timestamp=2.0)
            log_channel_message(chat_id=-1001, author="carol", text="chan1b", timestamp=3.0)

            result = read_channel_messages(channel_id="-1001")

        assert len(result) == 2
        assert all(m["chat_id"] == -1001 for m in result)

    def test_respects_limit(self, tmp_path):
        """Limit parameter caps the returned messages."""
        from agent.channel_monitor import log_channel_message, read_channel_messages

        with _patch_channel_file(tmp_path):
            for i in range(10):
                log_channel_message(chat_id=-1001, author=f"user_{i}", text=f"msg_{i}", timestamp=float(i))

            result = read_channel_messages(limit=3)

        assert len(result) == 3
        # Should return the last 3 messages
        assert result[0]["author"] == "user_7"
        assert result[2]["author"] == "user_9"

    def test_limit_capped_at_50(self, tmp_path):
        """Even if limit > 50, at most 50 messages are returned."""
        from agent.channel_monitor import log_channel_message, read_channel_messages

        with _patch_channel_file(tmp_path):
            for i in range(60):
                log_channel_message(chat_id=-1001, author=f"u{i}", text=f"m{i}", timestamp=float(i))

            result = read_channel_messages(limit=100)

        # min(100, 50) = 50 — capped at 50 in the code
        assert len(result) == 50

    def test_returns_empty_for_unknown_channel(self, tmp_path):
        """Filtering by a non-existent channel returns empty list."""
        from agent.channel_monitor import log_channel_message, read_channel_messages

        with _patch_channel_file(tmp_path):
            log_channel_message(chat_id=-1001, author="alice", text="hi", timestamp=1.0)
            result = read_channel_messages(channel_id="-9999")

        assert result == []

    def test_no_filter_returns_all(self, tmp_path):
        """Without channel_id filter, returns messages from all channels."""
        from agent.channel_monitor import log_channel_message, read_channel_messages

        with _patch_channel_file(tmp_path):
            log_channel_message(chat_id=-1001, author="a", text="1", timestamp=1.0)
            log_channel_message(chat_id=-1002, author="b", text="2", timestamp=2.0)

            result = read_channel_messages()

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestChannelMonitorErrorHandling:
    def test_corrupt_json_returns_empty(self, tmp_path):
        """Corrupt JSON file returns empty list."""
        from agent.channel_monitor import read_channel_messages

        corrupt_file = tmp_path / "channel_messages.json"
        corrupt_file.write_text("{not valid json!!!", encoding="utf-8")

        with _patch_channel_file(tmp_path):
            result = read_channel_messages()

        assert result == []

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing file returns empty list."""
        from agent.channel_monitor import read_channel_messages

        with _patch_channel_file(tmp_path):
            result = read_channel_messages()

        assert result == []

    def test_log_after_corrupt_file_recovers(self, tmp_path):
        """Logging a message after a corrupt file recovers gracefully."""
        from agent.channel_monitor import log_channel_message, read_channel_messages

        corrupt_file = tmp_path / "channel_messages.json"
        corrupt_file.write_text("CORRUPT DATA", encoding="utf-8")

        with _patch_channel_file(tmp_path):
            # _read_channel_messages will return [] due to corrupt JSON,
            # then log_channel_message will write a fresh list with one entry
            log_channel_message(chat_id=-1001, author="alice", text="recovered", timestamp=1.0)
            result = read_channel_messages()

        assert len(result) == 1
        assert result[0]["text"] == "recovered"

    def test_invalid_channel_id_string_returns_all(self, tmp_path):
        """Non-numeric channel_id string falls through without filtering."""
        from agent.channel_monitor import log_channel_message, read_channel_messages

        with _patch_channel_file(tmp_path):
            log_channel_message(chat_id=-1001, author="a", text="msg", timestamp=1.0)
            # "not_a_number" triggers ValueError in int() — passes through
            result = read_channel_messages(channel_id="not_a_number")

        # ValueError caught — returns unfiltered messages
        assert len(result) == 1
