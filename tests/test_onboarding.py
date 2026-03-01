"""Tests for agent.onboarding — state machine transitions and persistence."""

import json
from unittest.mock import patch

import pytest

from agent.onboarding import (
    OnboardingState,
    OnboardingSession,
    get_session,
    save_session,
    delete_session,
    advance,
    finalize_audit,
    finalize_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_path(tmp_path):
    p = tmp_path / "onboarding.json"
    with patch("agent.onboarding._STATE_PATH", p):
        yield p


def _session(state=OnboardingState.IDLE.value, **kwargs):
    defaults = dict(user_id=12345, state=state, started_at=1000.0)
    defaults.update(kwargs)
    return OnboardingSession(**defaults)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestAdvance:
    def test_idle_to_project_name(self):
        s = _session(OnboardingState.IDLE.value)
        s, msg = advance(s, None)
        assert s.state == OnboardingState.PROJECT_NAME.value
        assert "brand name" in msg.lower()

    def test_project_name_to_description(self):
        s = _session(OnboardingState.PROJECT_NAME.value)
        s, msg = advance(s, "ACME Corp")
        assert s.state == OnboardingState.DESCRIPTION.value
        assert s.brand_name == "ACME Corp"
        assert "ACME Corp" in msg

    def test_empty_brand_name_stays(self):
        s = _session(OnboardingState.PROJECT_NAME.value)
        s, msg = advance(s, "")
        assert s.state == OnboardingState.PROJECT_NAME.value

    def test_description_to_platforms(self):
        s = _session(OnboardingState.DESCRIPTION.value, brand_name="ACME")
        s, msg = advance(s, "A widget company")
        assert s.state == OnboardingState.PLATFORMS.value
        assert s.description == "A widget company"

    def test_platforms_to_asset_check(self):
        s = _session(OnboardingState.PLATFORMS.value)
        s, msg = advance(s, "twitter, linkedin")
        assert s.state == OnboardingState.ASSET_CHECK.value
        assert s.platforms == ["twitter", "linkedin"]

    def test_asset_check_yes_to_uploads(self):
        s = _session(OnboardingState.ASSET_CHECK.value)
        s, msg = advance(s, "yes")
        assert s.state == OnboardingState.UPLOADS.value

    def test_asset_check_no_to_visual_pref(self):
        s = _session(OnboardingState.ASSET_CHECK.value)
        s, msg = advance(s, "no")
        assert s.state == OnboardingState.VISUAL_PREF.value

    def test_uploads_skip_with_assets_to_auditing(self):
        s = _session(OnboardingState.UPLOADS.value, uploaded_assets=[{"path": "/tmp/logo.png"}])
        s, msg = advance(s, "/onboard_skip")
        assert s.state == OnboardingState.AUDITING.value

    def test_uploads_skip_without_assets_to_visual_pref(self):
        s = _session(OnboardingState.UPLOADS.value)
        s, msg = advance(s, "done")
        assert s.state == OnboardingState.VISUAL_PREF.value

    def test_uploads_text_stays_in_uploads(self):
        s = _session(OnboardingState.UPLOADS.value)
        s, msg = advance(s, "here's some text")
        assert s.state == OnboardingState.UPLOADS.value

    def test_visual_pref_known_style(self):
        s = _session(OnboardingState.VISUAL_PREF.value)
        s, msg = advance(s, "modern")
        assert s.state == OnboardingState.STRATEGY.value
        assert s.visual_preferences == {"style": "modern"}

    def test_visual_pref_custom_style(self):
        s = _session(OnboardingState.VISUAL_PREF.value)
        s, msg = advance(s, "retro synthwave with neon")
        assert s.state == OnboardingState.STRATEGY.value
        assert s.visual_preferences["style"] == "custom"
        assert "retro synthwave" in s.visual_preferences["description"]

    def test_confirm_yes_to_complete(self):
        s = _session(OnboardingState.CONFIRM.value)
        s, msg = advance(s, "yes")
        assert s.state == OnboardingState.COMPLETE.value

    def test_confirm_no_restarts(self):
        s = _session(OnboardingState.CONFIRM.value, brand_name="Test")
        s, msg = advance(s, "no")
        assert s.state == OnboardingState.PROJECT_NAME.value
        assert s.brand_name == ""

    def test_complete_state_is_terminal(self):
        s = _session(OnboardingState.COMPLETE.value)
        s, msg = advance(s, "anything")
        assert s.state == OnboardingState.COMPLETE.value


# ---------------------------------------------------------------------------
# Full flow test
# ---------------------------------------------------------------------------

class TestFullFlow:
    def test_full_no_assets_flow(self):
        s = _session()

        # idle → project_name
        s, msg = advance(s, None)
        assert s.state == OnboardingState.PROJECT_NAME.value

        # project_name → description
        s, msg = advance(s, "TestBrand")
        assert s.state == OnboardingState.DESCRIPTION.value

        # description → platforms
        s, msg = advance(s, "A test product")
        assert s.state == OnboardingState.PLATFORMS.value

        # platforms → asset_check
        s, msg = advance(s, "twitter")
        assert s.state == OnboardingState.ASSET_CHECK.value

        # asset_check (no) → visual_pref
        s, msg = advance(s, "no")
        assert s.state == OnboardingState.VISUAL_PREF.value

        # visual_pref → strategy
        s, msg = advance(s, "minimal")
        assert s.state == OnboardingState.STRATEGY.value

        # strategy → confirm (via finalize_strategy)
        s, msg = finalize_strategy(s, {
            "archetype": "starting_fresh",
            "compositor_enabled": False,
            "default_mode": "image_optional",
            "recommended_content_types": ["announcement"],
        })
        assert s.state == OnboardingState.CONFIRM.value

        # confirm → complete
        s, msg = advance(s, "yes")
        assert s.state == OnboardingState.COMPLETE.value


# ---------------------------------------------------------------------------
# finalize_audit
# ---------------------------------------------------------------------------

class TestFinalizeAudit:
    def test_full_brand_goes_to_template_choice(self):
        s = _session(OnboardingState.AUDITING.value)
        audit = {"archetype": "full_brand", "consolidated_colors": [], "consolidated_style": ["bold"]}
        s, msg = finalize_audit(s, audit)
        assert s.state == OnboardingState.TEMPLATE_CHOICE.value
        assert "Full Brand" in msg

    def test_has_identity_goes_to_visual_pref(self):
        s = _session(OnboardingState.AUDITING.value)
        audit = {"archetype": "has_identity", "missing_items": ["style_guide"]}
        s, msg = finalize_audit(s, audit)
        assert s.state == OnboardingState.VISUAL_PREF.value
        assert "Has Identity" in msg

    def test_starting_fresh_goes_to_visual_pref(self):
        s = _session(OnboardingState.AUDITING.value)
        audit = {"archetype": "starting_fresh", "missing_items": ["logo"]}
        s, msg = finalize_audit(s, audit)
        assert s.state == OnboardingState.VISUAL_PREF.value


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, state_path):
        s = OnboardingSession(user_id=99, state="project_name", brand_name="Test")
        save_session(s)

        loaded = get_session(99)
        assert loaded is not None
        assert loaded.brand_name == "Test"
        assert loaded.state == "project_name"

    def test_get_missing_returns_none(self, state_path):
        assert get_session(999) is None

    def test_delete_session(self, state_path):
        s = OnboardingSession(user_id=99, state="idle")
        save_session(s)
        delete_session(99)
        assert get_session(99) is None

    def test_resume_after_restart(self, state_path):
        """Session persists across save/load cycles (simulating restart)."""
        s = _session(OnboardingState.DESCRIPTION.value, brand_name="Persisted")
        save_session(s)

        # Simulate bot restart — load fresh
        loaded = get_session(12345)
        assert loaded is not None
        assert loaded.brand_name == "Persisted"
        assert loaded.state == OnboardingState.DESCRIPTION.value

        # Continue
        loaded, msg = advance(loaded, "A description")
        assert loaded.state == OnboardingState.PLATFORMS.value
