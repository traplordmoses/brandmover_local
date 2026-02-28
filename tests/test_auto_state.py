"""Tests for agent.auto_state — auto-post scheduler state."""

import time
from unittest.mock import patch

from agent import auto_state


def test_can_post_when_paused(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        auto_state.set_paused(True)
        allowed, reason = auto_state.can_post()
        assert not allowed
        assert "paused" in reason.lower()


def test_can_post_when_ok(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        allowed, reason = auto_state.can_post()
        assert allowed
        assert reason == "OK"


def test_record_and_check_slot(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        assert not auto_state.is_slot_posted("morning")

        auto_state.record_post(
            slot_name="morning",
            caption="Good morning!",
            tweet_url="https://x.com/test/1",
        )

        assert auto_state.is_slot_posted("morning")


def test_duplicate_caption_detection(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        assert not auto_state.is_duplicate_caption("Hello world today")

        auto_state.record_post("slot", "Hello world today")
        assert auto_state.is_duplicate_caption("Hello world today")
        assert not auto_state.is_duplicate_caption("Something completely different")


def test_rotation_index(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        assert auto_state.get_rotation_index("general") == 0

        next_idx = auto_state.advance_rotation("general", 5)
        assert next_idx == 1

        next_idx = auto_state.advance_rotation("general", 5)
        assert next_idx == 2


def test_rotation_wraps(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        for _ in range(4):
            auto_state.advance_rotation("cat", 3)

        assert auto_state.get_rotation_index("cat") == 1  # 4 % 3 = 1


def test_pause_resume(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        assert not auto_state.is_paused()
        auto_state.set_paused(True)
        assert auto_state.is_paused()
        auto_state.set_paused(False)
        assert not auto_state.is_paused()


def test_status_summary(tmp_path):
    state_file = tmp_path / "auto_state.json"

    with patch.object(auto_state, "_STATE_FILE", state_file):
        summary = auto_state.get_status_summary()
        assert "paused" in summary
        assert "posts_today" in summary
        assert summary["posts_today"] == 0
