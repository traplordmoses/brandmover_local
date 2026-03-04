"""Tests for intent routing integration in bot/handlers.py."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agent.conversation_context import ConversationContext
from agent.intent_router import RoutingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_update(user_id=123, text="hello"):
    """Create a mock Telegram Update."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.message.text = text
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    return update


def _mock_context():
    return MagicMock()


def _routing_result(intent, confidence=0.95, parameters=None, via="table"):
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        parameters=parameters or {},
        raw_message="test",
        routed_via=via,
    )


# ---------------------------------------------------------------------------
# _route_intent dispatch
# ---------------------------------------------------------------------------

class TestRouteIntent:
    """Test that _route_intent dispatches intents to the correct handlers."""

    def test_greeting_dispatched(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers.chat") as mock_chat:
                mock_cc.get_context.return_value = ConversationContext(user_id=123, updated_at=time.time())
                mock_state.has_pending.return_value = False
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("greeting")
                )
                mock_chat.handle_greeting = AsyncMock(return_value="Hey Test!")

                update = _mock_update(text="hi")
                result = await _route_intent(update, _mock_context(), "hi")
                assert result is True
                update.message.reply_text.assert_called_once_with("Hey Test!")

        asyncio.run(_run())

    def test_casual_chat_dispatched(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers.chat") as mock_chat:
                mock_cc.get_context.return_value = ConversationContext(user_id=123, updated_at=time.time())
                mock_state.has_pending.return_value = False
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("casual_chat", 0.8, via="table")
                )
                mock_chat.handle_casual_chat = AsyncMock(return_value="I'm here to help!")

                update = _mock_update(text="how are you")
                result = await _route_intent(update, _mock_context(), "how are you")
                assert result is True
                mock_chat.handle_casual_chat.assert_called_once()

        asyncio.run(_run())

    def test_approve_dispatched(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._do_approve") as mock_approve:
                mock_cc.get_context.return_value = ConversationContext(
                    user_id=123, pending_draft_exists=True, updated_at=time.time()
                )
                mock_state.has_pending.return_value = True
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("approve", 0.95)
                )
                mock_approve.return_value = None  # AsyncMock auto

                update = _mock_update(text="yes")
                ctx = _mock_context()
                result = await _route_intent(update, ctx, "yes")
                assert result is True
                mock_approve.assert_called_once_with(update, ctx, source="router")

        asyncio.run(_run())

    def test_reject_dispatched(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._do_reject") as mock_reject:
                mock_cc.get_context.return_value = ConversationContext(
                    user_id=123, pending_draft_exists=True, updated_at=time.time()
                )
                mock_state.has_pending.return_value = True
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("reject", 0.95, {"feedback": "too formal"})
                )
                mock_reject.return_value = None

                update = _mock_update(text="no")
                ctx = _mock_context()
                result = await _route_intent(update, ctx, "no")
                assert result is True
                mock_reject.assert_called_once_with(
                    update, ctx, feedback_text="too formal", source="router"
                )

        asyncio.run(_run())

    def test_show_status_dispatched(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers.status_command") as mock_status:
                mock_cc.get_context.return_value = ConversationContext(user_id=123, updated_at=time.time())
                mock_state.has_pending.return_value = False
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("show_status", 0.95)
                )
                mock_status.return_value = None

                update = _mock_update(text="status")
                ctx = _mock_context()
                result = await _route_intent(update, ctx, "status")
                assert result is True
                mock_status.assert_called_once_with(update, ctx)

        asyncio.run(_run())

    def test_unknown_falls_through(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state:
                mock_cc.get_context.return_value = ConversationContext(user_id=123, updated_at=time.time())
                mock_state.has_pending.return_value = False
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("unknown", 0.3)
                )

                update = _mock_update(text="something random")
                result = await _route_intent(update, _mock_context(), "something random")
                assert result is False

        asyncio.run(_run())

    def test_generate_content_falls_through(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state:
                mock_cc.get_context.return_value = ConversationContext(user_id=123, updated_at=time.time())
                mock_state.has_pending.return_value = False
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("generate_content", 0.9, {"topic": "product launch"})
                )

                update = _mock_update(text="write about product launch")
                result = await _route_intent(update, _mock_context(), "write about product launch")
                assert result is False

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_router_disabled_skips_classification(self):
        from bot.handlers import handle_message

        async def _run():
            with patch("bot.handlers.settings") as mock_settings, \
                 patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.onboarding") as mock_onboard, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers._rate_limited", return_value=False), \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._handle_pipeline_mode") as mock_pipeline:
                mock_settings.INTENT_ROUTER_ENABLED = False
                mock_settings.AGENT_MODE = "pipeline"
                mock_settings.TELEGRAM_ALLOWED_USER_ID = 123
                mock_settings._RATE_LIMIT_SECONDS = 10
                mock_onboard.get_session.return_value = None
                mock_state.has_pending.return_value = False
                mock_pipeline.return_value = None

                update = _mock_update(text="write a post")
                await handle_message(update, _mock_context())

                # Router should NOT have been called
                mock_ir.classify_intent.assert_not_called()
                # Should fall through to pipeline
                mock_pipeline.assert_called_once()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Reroll intent
# ---------------------------------------------------------------------------

class TestRerollIntent:
    def test_reroll_clears_and_regenerates(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._handle_pipeline_mode") as mock_pipeline, \
                 patch("bot.handlers.settings") as mock_settings:
                mock_cc.get_context.return_value = ConversationContext(user_id=123, updated_at=time.time())
                mock_state.has_pending.return_value = True
                mock_state.get_pending.return_value = {
                    "original_request": "weekly update post",
                    "caption": "Old draft",
                }
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("reroll", 0.95)
                )
                mock_settings.AGENT_MODE = "pipeline"
                mock_pipeline.return_value = None

                update = _mock_update(text="try again")
                result = await _route_intent(update, _mock_context(), "try again")

                assert result is True
                mock_state.clear_pending.assert_called_once_with(user_id=123)
                mock_state.clear_draft_history.assert_called_once_with(user_id=123)
                mock_pipeline.assert_called_once_with(update, "weekly update post", user_id=123)

        asyncio.run(_run())

    def test_reroll_without_pending_falls_through(self):
        from bot.handlers import _route_intent

        async def _run():
            with patch("bot.handlers.conversation_context") as mock_cc, \
                 patch("bot.handlers.intent_router") as mock_ir, \
                 patch("bot.handlers.state") as mock_state:
                mock_cc.get_context.return_value = ConversationContext(user_id=123, updated_at=time.time())
                mock_state.has_pending.return_value = False
                mock_state.get_pending.return_value = None
                mock_ir.classify_intent = AsyncMock(
                    return_value=_routing_result("reroll", 0.95)
                )

                update = _mock_update(text="try again")
                result = await _route_intent(update, _mock_context(), "try again")
                assert result is False

        asyncio.run(_run())
