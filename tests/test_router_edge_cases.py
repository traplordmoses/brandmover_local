"""Tests for intent router edge cases and safety mechanisms."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.conversation_context import ConversationContext
from agent.intent_router import (
    RoutingResult,
    classify_intent,
    clear_cache,
    reset_rate_limits,
    _RATE_LIMIT_PER_HOUR,
    _record_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state():
    reset_rate_limits()
    clear_cache()
    yield
    reset_rate_limits()
    clear_cache()


def _ctx(pending=False, last_action="idle", user_id=1):
    return ConversationContext(
        user_id=user_id,
        last_bot_action=last_action,
        pending_draft_exists=pending,
        updated_at=time.time(),
    )


def _mock_update(user_id=123, text="hello"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.message.text = text
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    return update


def _routing_result(intent, confidence=0.95, parameters=None, via="table"):
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        parameters=parameters or {},
        raw_message="test",
        routed_via=via,
    )


# ---------------------------------------------------------------------------
# Timeout fallback
# ---------------------------------------------------------------------------

class TestTimeoutFallback:
    def test_haiku_timeout_returns_generate_content(self):
        """When Haiku times out, classify_intent returns generate_content as fallback."""
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def _slow(**kwargs):
                    await asyncio.sleep(10)
                mock_client.messages.create = _slow
                mock_cls.return_value = mock_client
                return await classify_intent("some complex message", _ctx())

        with patch("agent.intent_router._HAIKU_TIMEOUT_SECONDS", 0.1):
            result = asyncio.run(_run())

        assert result.intent == "generate_content"
        assert result.routed_via == "fallback"
        assert result.confidence == 0.5


# ---------------------------------------------------------------------------
# Rate limiting edge cases
# ---------------------------------------------------------------------------

class TestRateLimitEdgeCases:
    def test_rate_limit_hit_returns_fallback(self):
        """When rate limit is exceeded, classify_intent returns generate_content fallback."""
        for _ in range(_RATE_LIMIT_PER_HOUR):
            _record_rate(1)

        result = asyncio.run(classify_intent("write a post about cats", _ctx(user_id=1)))
        assert result.intent == "generate_content"
        assert result.routed_via == "fallback"

    def test_table_lookups_unaffected_by_rate_limit(self):
        """Short message table bypasses rate limiting."""
        for _ in range(_RATE_LIMIT_PER_HOUR + 10):
            _record_rate(1)

        result = asyncio.run(classify_intent("hello", _ctx(user_id=1)))
        assert result.intent == "greeting"
        assert result.routed_via == "table"

    def test_different_users_have_separate_limits(self):
        """Rate limits are per-user."""
        for _ in range(_RATE_LIMIT_PER_HOUR):
            _record_rate(1)

        # User 1 is rate limited
        result1 = asyncio.run(classify_intent("complex message", _ctx(user_id=1)))
        assert result1.routed_via == "fallback"

        # User 2 is NOT rate limited
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                resp = MagicMock()
                resp.content = [MagicMock(text='{"intent":"casual_chat","confidence":0.9,"parameters":{}}')]
                mock_client.messages.create = AsyncMock(return_value=resp)
                mock_cls.return_value = mock_client
                return await classify_intent("complex message", _ctx(user_id=2))

        result2 = asyncio.run(_run())
        assert result2.routed_via == "haiku"


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------

class TestErrorRecovery:
    def test_api_exception_returns_fallback(self):
        """Any API exception falls back to generate_content."""
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    side_effect=ConnectionError("Network down")
                )
                mock_cls.return_value = mock_client
                return await classify_intent("write a post", _ctx())

        result = asyncio.run(_run())
        assert result.intent == "generate_content"
        assert result.routed_via == "fallback"

    def test_router_exception_in_handle_message_falls_through(self):
        """When _route_intent raises, handle_message falls through to generation."""
        from bot.handlers import handle_message

        async def _run():
            with patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.onboarding") as mock_onboard, \
                 patch("bot.handlers.settings") as mock_settings, \
                 patch("bot.handlers._route_intent", side_effect=RuntimeError("boom")), \
                 patch("bot.handlers._rate_limited", return_value=False), \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._handle_pipeline_mode") as mock_pipeline:
                mock_settings.INTENT_ROUTER_ENABLED = True
                mock_settings.AGENT_MODE = "pipeline"
                mock_settings.TELEGRAM_ALLOWED_USER_ID = 123
                mock_onboard.get_session.return_value = None
                mock_state.has_pending.return_value = False
                mock_pipeline.return_value = None

                update = _mock_update(text="some message")
                await handle_message(update, MagicMock())

                # Should have fallen through to pipeline
                mock_pipeline.assert_called_once()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Repeated / cache behavior
# ---------------------------------------------------------------------------

class TestRepeatedMessages:
    def test_identical_messages_hit_cache(self):
        """Second identical Haiku call returns cached result."""
        call_count = 0

        async def _run():
            nonlocal call_count
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def _create(**kwargs):
                    nonlocal call_count
                    call_count += 1
                    resp = MagicMock()
                    resp.content = [MagicMock(
                        text='{"intent":"generate_content","confidence":0.9,"parameters":{"topic":"test"}}'
                    )]
                    return resp
                mock_client.messages.create = _create
                mock_cls.return_value = mock_client

                ctx = _ctx()
                r1 = await classify_intent("write about our product", ctx)
                r2 = await classify_intent("write about our product", ctx)
                return r1, r2

        r1, r2 = asyncio.run(_run())
        assert call_count == 1  # Only one API call
        assert r1.intent == r2.intent

    def test_different_context_bypasses_cache(self):
        """Same message but different context → different cache keys."""
        call_count = 0

        async def _run():
            nonlocal call_count
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def _create(**kwargs):
                    nonlocal call_count
                    call_count += 1
                    resp = MagicMock()
                    resp.content = [MagicMock(
                        text='{"intent":"generate_content","confidence":0.9,"parameters":{}}'
                    )]
                    return resp
                mock_client.messages.create = _create
                mock_cls.return_value = mock_client

                ctx1 = _ctx(pending=False)
                ctx2 = _ctx(pending=True)
                await classify_intent("ambiguous message", ctx1)
                await classify_intent("ambiguous message", ctx2)

        asyncio.run(_run())
        assert call_count == 2  # Different context = different cache key


# ---------------------------------------------------------------------------
# Context-aware edge cases
# ---------------------------------------------------------------------------

class TestContextAwareEdgeCases:
    def test_yes_without_draft_is_casual(self):
        """'yes' without a pending draft → casual_chat, not approve."""
        result = asyncio.run(classify_intent("yes", _ctx(pending=False)))
        assert result.intent == "casual_chat"

    def test_yes_with_draft_is_approve(self):
        """'yes' with a pending draft → approve."""
        result = asyncio.run(classify_intent("yes", _ctx(pending=True)))
        assert result.intent == "approve"

    def test_no_without_draft_is_casual(self):
        """'no' without a pending draft → casual_chat, not reject."""
        result = asyncio.run(classify_intent("no", _ctx(pending=False)))
        assert result.intent == "casual_chat"

    def test_no_with_draft_is_reject(self):
        """'no' with a pending draft → reject."""
        result = asyncio.run(classify_intent("no", _ctx(pending=True)))
        assert result.intent == "reject"

    def test_help_always_works_regardless_of_draft(self):
        """'help' should always route to show_help."""
        result1 = asyncio.run(classify_intent("help", _ctx(pending=False)))
        result2 = asyncio.run(classify_intent("help", _ctx(pending=True)))
        assert result1.intent == "show_help"
        assert result2.intent == "show_help"

    def test_greeting_always_works_regardless_of_draft(self):
        """'hi' should always route to greeting."""
        result1 = asyncio.run(classify_intent("hi", _ctx(pending=False)))
        result2 = asyncio.run(classify_intent("hi", _ctx(pending=True)))
        assert result1.intent == "greeting"
        assert result2.intent == "greeting"


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_disabled_router_skips_all_classification(self):
        """INTENT_ROUTER_ENABLED=False means handle_message skips router entirely."""
        from bot.handlers import handle_message

        async def _run():
            with patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.onboarding") as mock_onboard, \
                 patch("bot.handlers.settings") as mock_settings, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers._rate_limited", return_value=False), \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._handle_pipeline_mode") as mock_pipeline:
                mock_settings.INTENT_ROUTER_ENABLED = False
                mock_settings.AGENT_MODE = "pipeline"
                mock_settings.TELEGRAM_ALLOWED_USER_ID = 123
                mock_onboard.get_session.return_value = None
                mock_state.has_pending.return_value = False
                mock_pipeline.return_value = None

                update = _mock_update(text="hello")
                await handle_message(update, MagicMock())

                mock_ir.classify_intent.assert_not_called()
                mock_pipeline.assert_called_once()

        asyncio.run(_run())
