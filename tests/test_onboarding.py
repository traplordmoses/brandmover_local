"""Tests for agent.onboarding — smart discovery + state machine transitions."""

import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from agent.onboarding import (
    OnboardingState,
    OnboardingSession,
    get_session,
    save_session,
    delete_session,
    advance,
    advance_async,
    finalize_audit,
    finalize_strategy,
    _apply_collected_fields,
    REQUIRED_FIELDS,
    OPTIONAL_FIELDS,
    _DISCOVERY_SYSTEM,
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
# IDLE → DISCOVERY transition
# ---------------------------------------------------------------------------

class TestIdleToDiscovery:
    def test_idle_starts_discovery(self):
        s = _session(OnboardingState.IDLE.value)
        s, msg = advance(s, None)
        assert s.state == OnboardingState.DISCOVERY.value
        assert "brand" in msg.lower()

    def test_discovery_returns_async_marker(self):
        s = _session(OnboardingState.DISCOVERY.value)
        s, msg = advance(s, "hello")
        assert msg == "_NEEDS_ASYNC_"


# ---------------------------------------------------------------------------
# Smart discovery (mocked Claude)
# ---------------------------------------------------------------------------

class TestSmartDiscovery:
    def _mock_claude_response(self, message, fields=None, complete=False, suggest_upload=False):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "message": message,
            "fields_collected": fields or {},
            "all_required_complete": complete,
            "suggest_upload": suggest_upload,
        }))]
        return mock_response

    def test_extracts_fields_from_conversation(self):
        s = _session(OnboardingState.DISCOVERY.value)
        response = self._mock_claude_response(
            "Great name! What does ACME do?",
            fields={"project_name": "ACME Corp"},
        )

        async def _run():
            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=response)
                mock_cls.return_value = mock_client
                return await advance_async(s, "We're called ACME Corp")

        s, msg = asyncio.run(_run())
        assert s.state == OnboardingState.DISCOVERY.value
        assert s.brand_name == "ACME Corp"
        assert s.collected_fields["project_name"] == "ACME Corp"
        assert "ACME" in msg

    def test_multiple_fields_in_one_turn(self):
        s = _session(OnboardingState.DISCOVERY.value)
        response = self._mock_claude_response(
            "Got it! Do you have brand assets?",
            fields={
                "project_name": "TestBrand",
                "description": "A test product",
                "platforms": ["x", "telegram"],
            },
        )

        async def _run():
            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=response)
                mock_cls.return_value = mock_client
                return await advance_async(s, "I'm building TestBrand, a test product. We post on X and Telegram")

        s, msg = asyncio.run(_run())
        assert s.brand_name == "TestBrand"
        assert s.description == "A test product"
        assert s.platforms == ["x", "telegram"]

    def test_discovery_complete_with_assets(self):
        s = _session(OnboardingState.DISCOVERY.value, collected_fields={
            "project_name": "Test", "description": "A test",
            "platforms": ["x"], "visual_preference": "modern",
        })
        s.brand_name = "Test"
        response = self._mock_claude_response(
            "Let's get those uploaded!",
            fields={"has_assets": True},
            complete=True,
            suggest_upload=True,
        )

        async def _run():
            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=response)
                mock_cls.return_value = mock_client
                return await advance_async(s, "yes I have logos")

        s, msg = asyncio.run(_run())
        assert s.state == OnboardingState.UPLOADS.value

    def test_discovery_complete_no_assets_with_visual_pref(self):
        s = _session(OnboardingState.DISCOVERY.value)
        s.visual_preferences = {"style": "modern"}
        response = self._mock_claude_response(
            "Perfect!",
            fields={
                "project_name": "Test", "description": "A test",
                "platforms": ["x"], "has_assets": False,
                "visual_preference": "modern",
            },
            complete=True,
        )

        async def _run():
            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=response)
                mock_cls.return_value = mock_client
                return await advance_async(s, "no assets, modern style")

        s, msg = asyncio.run(_run())
        assert s.state == OnboardingState.STRATEGY.value

    def test_discovery_complete_no_assets_no_visual_pref(self):
        s = _session(OnboardingState.DISCOVERY.value)
        response = self._mock_claude_response(
            "Almost there!",
            fields={
                "project_name": "Test", "description": "A test",
                "platforms": ["x"], "has_assets": False,
            },
            complete=True,
        )

        async def _run():
            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=response)
                mock_cls.return_value = mock_client
                return await advance_async(s, "no assets")

        s, msg = asyncio.run(_run())
        assert s.state == OnboardingState.VISUAL_PREF.value

    def test_conversation_history_stored(self):
        s = _session(OnboardingState.DISCOVERY.value)
        response = self._mock_claude_response("Tell me more!")

        async def _run():
            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=response)
                mock_cls.return_value = mock_client
                return await advance_async(s, "hello")

        s, _ = asyncio.run(_run())
        assert len(s.conversation_history) == 2
        assert s.conversation_history[0]["role"] == "user"
        assert s.conversation_history[1]["role"] == "assistant"

    def test_handles_non_json_response(self):
        s = _session(OnboardingState.DISCOVERY.value)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Just a plain text response")]

        async def _run():
            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await advance_async(s, "hello")

        s, msg = asyncio.run(_run())
        assert s.state == OnboardingState.DISCOVERY.value
        assert msg == "Just a plain text response"


# ---------------------------------------------------------------------------
# _apply_collected_fields
# ---------------------------------------------------------------------------

class TestApplyCollectedFields:
    def test_applies_project_name(self):
        s = _session()
        _apply_collected_fields(s, {"project_name": "ACME"})
        assert s.brand_name == "ACME"
        assert s.collected_fields["project_name"] == "ACME"

    def test_applies_platforms_list(self):
        s = _session()
        _apply_collected_fields(s, {"platforms": ["x", "telegram"]})
        assert s.platforms == ["x", "telegram"]

    def test_applies_platforms_string(self):
        s = _session()
        _apply_collected_fields(s, {"platforms": "x, telegram"})
        assert s.platforms == ["x", "telegram"]

    def test_applies_visual_preference(self):
        s = _session()
        _apply_collected_fields(s, {"visual_preference": "retro neon"})
        assert s.visual_preferences["style"] == "custom"
        assert "retro neon" in s.visual_preferences["description"]


# ---------------------------------------------------------------------------
# Legacy state transitions (backward compat)
# ---------------------------------------------------------------------------

class TestLegacyAdvance:
    def test_project_name_to_description(self):
        s = _session(OnboardingState.PROJECT_NAME.value)
        s, msg = advance(s, "ACME Corp")
        assert s.state == OnboardingState.DESCRIPTION.value
        assert s.brand_name == "ACME Corp"

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


# ---------------------------------------------------------------------------
# Non-discovery states (shared by both paths)
# ---------------------------------------------------------------------------

class TestSharedStates:
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

    def test_confirm_no_restarts_to_discovery(self):
        s = _session(OnboardingState.CONFIRM.value, brand_name="Test")
        s, msg = advance(s, "no")
        assert s.state == OnboardingState.DISCOVERY.value
        assert s.brand_name == ""
        assert s.collected_fields == {}

    def test_complete_state_is_terminal(self):
        s = _session(OnboardingState.COMPLETE.value)
        s, msg = advance(s, "anything")
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
        s = OnboardingSession(user_id=99, state="discovery", brand_name="Test")
        save_session(s)

        loaded = get_session(99)
        assert loaded is not None
        assert loaded.brand_name == "Test"
        assert loaded.state == "discovery"

    def test_get_missing_returns_none(self, state_path):
        assert get_session(999) is None

    def test_delete_session(self, state_path):
        s = OnboardingSession(user_id=99, state="idle")
        save_session(s)
        delete_session(99)
        assert get_session(99) is None

    def test_resume_after_restart(self, state_path):
        """Session persists across save/load cycles (simulating restart)."""
        s = _session(OnboardingState.DISCOVERY.value, brand_name="Persisted")
        s.collected_fields = {"project_name": "Persisted"}
        save_session(s)

        loaded = get_session(12345)
        assert loaded is not None
        assert loaded.brand_name == "Persisted"
        assert loaded.state == OnboardingState.DISCOVERY.value
        assert loaded.collected_fields["project_name"] == "Persisted"

    def test_new_optional_fields_in_tuple(self):
        assert "cultural_references" in OPTIONAL_FIELDS
        assert "what_they_hate" in OPTIONAL_FIELDS
        assert "secret_weapon" in OPTIONAL_FIELDS

    def test_required_fields_unchanged(self):
        assert "project_name" in REQUIRED_FIELDS
        assert "description" in REQUIRED_FIELDS
        assert "platforms" in REQUIRED_FIELDS
        assert "has_assets" in REQUIRED_FIELDS
        assert "visual_preference" in REQUIRED_FIELDS

    def test_discovery_prompt_has_creative_framing(self):
        assert "creative collaborator" in _DISCOVERY_SYSTEM.lower() or "creative" in _DISCOVERY_SYSTEM.lower()
        assert "cultural_references" in _DISCOVERY_SYSTEM
        assert "what_they_hate" in _DISCOVERY_SYSTEM
        assert "secret_weapon" in _DISCOVERY_SYSTEM

    def test_new_optional_fields_collected(self):
        s = _session()
        _apply_collected_fields(s, {
            "cultural_references": "Blade Runner, synthwave",
            "what_they_hate": "Corporate Memphis",
            "secret_weapon": "Hand-drawn everything",
        })
        assert s.collected_fields["cultural_references"] == "Blade Runner, synthwave"
        assert s.collected_fields["what_they_hate"] == "Corporate Memphis"
        assert s.collected_fields["secret_weapon"] == "Hand-drawn everything"

    def test_conversation_history_persists(self, state_path):
        s = _session(OnboardingState.DISCOVERY.value)
        s.conversation_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        save_session(s)

        loaded = get_session(12345)
        assert len(loaded.conversation_history) == 2
        assert loaded.conversation_history[0]["content"] == "hello"
