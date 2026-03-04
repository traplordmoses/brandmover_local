"""Tests for agent.state — pending draft management and draft versioning."""

import json
from unittest.mock import patch
from pathlib import Path

from config import settings
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


# ---------------------------------------------------------------------------
# Per-user state isolation tests
# ---------------------------------------------------------------------------


def _setup_per_user(tmp_path):
    """Patch state module to use tmp_path as STATE_DIR and clear caches."""
    state.invalidate_state_cache()
    return {
        "_STATE_DIR": tmp_path,
        "_STATE_FILE": tmp_path / "state.json",
    }


def test_per_user_pending_isolation(tmp_path):
    """Two users have independent pending drafts."""
    patches = _setup_per_user(tmp_path)

    with patch.object(state, "_STATE_DIR", patches["_STATE_DIR"]), \
         patch.object(state, "_STATE_FILE", patches["_STATE_FILE"]), \
         patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 100):

        state.invalidate_state_cache()

        # Admin saves a draft
        state.save_pending("admin draft", [], None, "", "", "req", user_id=100)
        assert state.has_pending(user_id=100)
        assert state.get_pending(user_id=100)["caption"] == "admin draft"

        # Operator saves a draft
        state.save_pending("operator draft", [], None, "", "", "req", user_id=200)
        assert state.has_pending(user_id=200)
        assert state.get_pending(user_id=200)["caption"] == "operator draft"

        # Both are independent
        assert state.get_pending(user_id=100)["caption"] == "admin draft"
        assert state.get_pending(user_id=200)["caption"] == "operator draft"


def test_clear_one_user_doesnt_affect_other(tmp_path):
    """Clearing one user's draft doesn't affect another."""
    patches = _setup_per_user(tmp_path)

    with patch.object(state, "_STATE_DIR", patches["_STATE_DIR"]), \
         patch.object(state, "_STATE_FILE", patches["_STATE_FILE"]), \
         patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 100):

        state.invalidate_state_cache()

        state.save_pending("admin", [], None, "", "", "req", user_id=100)
        state.save_pending("operator", [], None, "", "", "req", user_id=200)

        state.clear_pending(user_id=200)

        assert state.has_pending(user_id=100)
        assert not state.has_pending(user_id=200)


def test_user_id_none_defaults_to_admin(tmp_path):
    """user_id=None routes to admin state file."""
    patches = _setup_per_user(tmp_path)

    with patch.object(state, "_STATE_DIR", patches["_STATE_DIR"]), \
         patch.object(state, "_STATE_FILE", patches["_STATE_FILE"]), \
         patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 100):

        state.invalidate_state_cache()

        state.save_pending("default admin", [], None, "", "", "req", user_id=None)
        assert state.get_pending(user_id=100)["caption"] == "default admin"
        assert state.get_pending(user_id=None)["caption"] == "default admin"


def test_per_user_state_files(tmp_path):
    """Admin uses state.json, operators use draft_{uid}.json."""
    patches = _setup_per_user(tmp_path)

    with patch.object(state, "_STATE_DIR", patches["_STATE_DIR"]), \
         patch.object(state, "_STATE_FILE", patches["_STATE_FILE"]), \
         patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 100):

        state.invalidate_state_cache()

        state.save_pending("admin", [], None, "", "", "req", user_id=100)
        state.save_pending("op", [], None, "", "", "req", user_id=200)

        assert (tmp_path / "state.json").exists()
        assert (tmp_path / "draft_200.json").exists()
        assert not (tmp_path / "draft_100.json").exists()  # admin uses state.json


def test_per_user_last_composed(tmp_path):
    """Last composed is per-user."""
    patches = _setup_per_user(tmp_path)

    with patch.object(state, "_STATE_DIR", patches["_STATE_DIR"]), \
         patch.object(state, "_STATE_FILE", patches["_STATE_FILE"]), \
         patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 100):

        state.invalidate_state_cache()

        state.set_last_composed("/admin.png", "brand_3d", user_id=100)
        state.set_last_composed("/op.png", "announcement", user_id=200)

        path_a, ct_a = state.get_last_composed(user_id=100)
        path_o, ct_o = state.get_last_composed(user_id=200)

        assert path_a == "/admin.png"
        assert path_o == "/op.png"
        assert ct_a == "brand_3d"
        assert ct_o == "announcement"
