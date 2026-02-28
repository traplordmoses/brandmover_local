"""Tests for agent.state — pending draft management and draft versioning."""

import json
from unittest.mock import patch
from pathlib import Path

from agent import state


def test_save_and_get_pending(tmp_path):
    state_file = tmp_path / "state.json"

    with patch.object(state, "_STATE_FILE", state_file):
        state.save_pending(
            caption="Test caption",
            hashtags=["#test"],
            image_url="https://example.com/img.png",
            alt_text="alt",
            image_prompt="a test image",
            original_request="make a test post",
        )

        pending = state.get_pending()
        assert pending is not None
        assert pending["caption"] == "Test caption"
        assert pending["hashtags"] == ["#test"]


def test_clear_pending(tmp_path):
    state_file = tmp_path / "state.json"

    with patch.object(state, "_STATE_FILE", state_file):
        state.save_pending("cap", [], None, "", "", "req")
        assert state.has_pending()

        state.clear_pending()
        assert not state.has_pending()


def test_draft_versioning(tmp_path):
    state_file = tmp_path / "state.json"

    with patch.object(state, "_STATE_FILE", state_file):
        # First draft — no history
        state.save_pending("v1", [], None, "", "", "req")
        assert state.get_draft_revision_count() == 1
        assert len(state.get_draft_history()) == 0

        # Second draft — v1 archived
        state.save_pending("v2", [], None, "", "", "req")
        assert state.get_draft_revision_count() == 2
        history = state.get_draft_history()
        assert len(history) == 1
        assert history[0]["caption"] == "v1"

        # Third draft
        state.save_pending("v3", [], None, "", "", "req")
        assert state.get_draft_revision_count() == 3
        assert len(state.get_draft_history()) == 2


def test_clear_draft_history(tmp_path):
    state_file = tmp_path / "state.json"

    with patch.object(state, "_STATE_FILE", state_file):
        state.save_pending("v1", [], None, "", "", "req")
        state.save_pending("v2", [], None, "", "", "req")
        assert len(state.get_draft_history()) == 1

        state.clear_draft_history()
        assert len(state.get_draft_history()) == 0


def test_content_type_saved(tmp_path):
    state_file = tmp_path / "state.json"

    with patch.object(state, "_STATE_FILE", state_file):
        state.save_pending("cap", [], None, "", "", "req", content_type="brand_3d")
        pending = state.get_pending()
        assert pending["content_type"] == "brand_3d"


def test_reference_image(tmp_path):
    state_file = tmp_path / "state.json"

    with patch.object(state, "_STATE_FILE", state_file):
        assert state.get_reference_image() is None
        state.set_reference_image("/tmp/test.png")
        assert state.get_reference_image() == "/tmp/test.png"
        state.clear_reference_image()
        assert state.get_reference_image() is None


def test_last_generated(tmp_path):
    state_file = tmp_path / "state.json"

    with patch.object(state, "_STATE_FILE", state_file):
        url, ct = state.get_last_generated()
        assert url is None

        state.save_last_generated("https://example.com/img.png", "announcement")
        url, ct = state.get_last_generated()
        assert url == "https://example.com/img.png"
        assert ct == "announcement"
