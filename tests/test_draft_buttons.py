"""Tests for inline draft button callbacks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.handlers import draft_callback, _CallbackProxy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_callback_update(action, user_id=123):
    """Create a mock Update with a callback_query."""
    update = MagicMock()
    update.callback_query.data = f"draft_{action}"
    update.callback_query.from_user.id = user_id
    update.callback_query.from_user.first_name = "Test"
    update.callback_query.answer = AsyncMock()
    update.callback_query.message.reply_text = AsyncMock()
    update.callback_query.message.reply_photo = AsyncMock()
    update.callback_query.message.chat.send_action = AsyncMock()
    return update


def _mock_context():
    return MagicMock()


# ---------------------------------------------------------------------------
# _CallbackProxy
# ---------------------------------------------------------------------------

class TestCallbackProxy:
    def test_proxy_message(self):
        update = MagicMock()
        query = MagicMock()
        query.message = MagicMock()
        query.from_user = MagicMock()
        proxy = _CallbackProxy(update, query)
        assert proxy.message is query.message
        assert proxy.effective_user is query.from_user

    def test_proxy_delegates(self):
        update = MagicMock()
        update.some_attr = "test"
        query = MagicMock()
        proxy = _CallbackProxy(update, query)
        assert proxy.some_attr == "test"


# ---------------------------------------------------------------------------
# draft_callback dispatch
# ---------------------------------------------------------------------------

class TestDraftCallback:
    def test_approve_button(self):
        async def _run():
            with patch("bot.handlers._can_operate", return_value=True), \
                 patch("bot.handlers._do_approve") as mock_approve:
                mock_approve.return_value = None
                update = _mock_callback_update("approve")
                ctx = _mock_context()
                await draft_callback(update, ctx)
                update.callback_query.answer.assert_called_once()
                mock_approve.assert_called_once()
                call_args = mock_approve.call_args
                assert call_args.kwargs.get("source") == "button"

        asyncio.run(_run())

    def test_reject_button_asks_for_feedback(self):
        async def _run():
            with patch("bot.handlers._can_operate", return_value=True):
                update = _mock_callback_update("reject")
                await draft_callback(update, _mock_context())
                update.callback_query.message.reply_text.assert_called_once()
                msg = update.callback_query.message.reply_text.call_args[0][0]
                assert "feedback" in msg.lower()

        asyncio.run(_run())

    def test_edit_button_asks_for_feedback(self):
        async def _run():
            with patch("bot.handlers._can_operate", return_value=True):
                update = _mock_callback_update("edit")
                await draft_callback(update, _mock_context())
                update.callback_query.message.reply_text.assert_called_once()
                msg = update.callback_query.message.reply_text.call_args[0][0]
                assert "edit" in msg.lower() or "feedback" in msg.lower()

        asyncio.run(_run())

    def test_reroll_button_with_pending(self):
        async def _run():
            with patch("bot.handlers._can_operate", return_value=True), \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._handle_pipeline_mode") as mock_pipeline, \
                 patch("bot.handlers.settings") as mock_settings:
                mock_state.get_pending.return_value = {
                    "original_request": "test topic",
                    "caption": "Old",
                }
                mock_settings.AGENT_MODE = "pipeline"
                mock_pipeline.return_value = None

                update = _mock_callback_update("reroll")
                await draft_callback(update, _mock_context())

                mock_state.clear_pending.assert_called_once_with(user_id=123)
                mock_state.clear_draft_history.assert_called_once_with(user_id=123)
                mock_pipeline.assert_called_once()

        asyncio.run(_run())

    def test_reroll_button_without_pending(self):
        async def _run():
            with patch("bot.handlers._can_operate", return_value=True), \
                 patch("bot.handlers.state") as mock_state:
                mock_state.get_pending.return_value = None

                update = _mock_callback_update("reroll")
                await draft_callback(update, _mock_context())
                # Should not crash, just do nothing

        asyncio.run(_run())

    def test_unauthorized_user_ignored(self):
        async def _run():
            with patch("bot.handlers._can_operate", return_value=False), \
                 patch("bot.handlers._do_approve") as mock_approve:
                update = _mock_callback_update("approve", user_id=999)
                await draft_callback(update, _mock_context())
                mock_approve.assert_not_called()

        asyncio.run(_run())
