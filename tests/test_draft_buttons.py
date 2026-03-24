"""Tests for inline draft button callbacks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.handlers import draft_callback, _CallbackProxy

_DRAFT = "bot.handlers.draft"

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
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}._do_approve") as mock_approve:
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
            with patch(f"{_DRAFT}._can_operate", return_value=True):
                update = _mock_callback_update("reject")
                await draft_callback(update, _mock_context())
                update.callback_query.message.reply_text.assert_called_once()
                msg = update.callback_query.message.reply_text.call_args[0][0]
                assert "feedback" in msg.lower()

        asyncio.run(_run())

    def test_edit_button_asks_for_feedback(self):
        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True):
                update = _mock_callback_update("edit")
                await draft_callback(update, _mock_context())
                update.callback_query.message.reply_text.assert_called_once()
                msg = update.callback_query.message.reply_text.call_args[0][0]
                assert "edit" in msg.lower() or "feedback" in msg.lower()

        asyncio.run(_run())

    def test_reroll_button_with_pending(self):
        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state, \
                 patch("bot.handlers.generation._handle_agent_mode") as mock_agent, \
                 patch(f"{_DRAFT}.settings") as mock_settings:
                mock_state.get_pending.return_value = {
                    "original_request": "test topic",
                    "caption": "Old",
                }
                mock_settings.AGENT_MODE = "agent"
                mock_agent.return_value = None

                update = _mock_callback_update("reroll")
                await draft_callback(update, _mock_context())

                mock_state.clear_pending.assert_called_once_with(user_id=123)
                mock_state.clear_draft_history.assert_called_once_with(user_id=123)
                mock_agent.assert_called_once()

        asyncio.run(_run())

    def test_reroll_button_without_pending(self):
        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state:
                mock_state.get_pending.return_value = None

                update = _mock_callback_update("reroll")
                await draft_callback(update, _mock_context())
                # Should not crash, just do nothing

        asyncio.run(_run())

    def test_unauthorized_user_ignored(self):
        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=False), \
                 patch(f"{_DRAFT}._do_approve") as mock_approve:
                update = _mock_callback_update("approve", user_id=999)
                await draft_callback(update, _mock_context())
                mock_approve.assert_not_called()

        asyncio.run(_run())

    def test_edit_caption_button_dispatches(self):
        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}._do_edit_caption") as mock_edit_caption:
                mock_edit_caption.return_value = None
                update = _mock_callback_update("edit_caption")
                await draft_callback(update, _mock_context())
                update.callback_query.answer.assert_called_once()
                mock_edit_caption.assert_called_once()
                call_args = mock_edit_caption.call_args
                assert call_args.kwargs.get("user_id") == 123

        asyncio.run(_run())

    def test_edit_image_button_dispatches(self):
        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}._do_edit_image") as mock_edit_image:
                mock_edit_image.return_value = None
                update = _mock_callback_update("edit_image")
                await draft_callback(update, _mock_context())
                update.callback_query.answer.assert_called_once()
                mock_edit_image.assert_called_once()
                call_args = mock_edit_image.call_args
                assert call_args.kwargs.get("user_id") == 123

        asyncio.run(_run())

    def test_shorten_button_dispatches(self):
        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}._do_shorten") as mock_shorten:
                mock_shorten.return_value = None
                update = _mock_callback_update("shorten")
                await draft_callback(update, _mock_context())
                update.callback_query.answer.assert_called_once()
                mock_shorten.assert_called_once()
                call_args = mock_shorten.call_args
                assert call_args.kwargs.get("user_id") == 123

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Granular edit handlers
# ---------------------------------------------------------------------------


class TestDoShorten:
    """Tests for the _do_shorten handler."""

    def test_shorten_long_caption(self):
        from bot.handlers.draft import _do_shorten

        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state, \
                 patch(f"{_DRAFT}._send_draft") as mock_send:
                mock_state.get_pending.return_value = {
                    "caption": "A" * 200,
                    "hashtags": [],
                    "image_url": "https://example.com/img.png",
                    "alt_text": "",
                    "image_prompt": "test prompt",
                    "original_request": "test",
                    "content_type": "announcement",
                }
                mock_send.return_value = None

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.message.chat.send_action = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_shorten(update, _mock_context(), user_id=123)

                # Should have saved a shortened caption
                mock_state.save_pending.assert_called_once()
                saved_caption = mock_state.save_pending.call_args.kwargs.get("caption", "")
                assert len(saved_caption) <= 100
                assert saved_caption.endswith("...")

                # Should have sent the draft
                mock_send.assert_called_once()

        asyncio.run(_run())

    def test_shorten_already_short_caption(self):
        from bot.handlers.draft import _do_shorten

        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state:
                mock_state.get_pending.return_value = {
                    "caption": "Short caption",
                    "hashtags": [],
                }

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_shorten(update, _mock_context(), user_id=123)

                # Should not have called save_pending since it's already short
                mock_state.save_pending.assert_not_called()
                # Should have informed the user
                update.message.reply_text.assert_called_once()
                msg = update.message.reply_text.call_args[0][0]
                assert "already" in msg.lower()

        asyncio.run(_run())

    def test_shorten_no_pending(self):
        from bot.handlers.draft import _do_shorten

        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state:
                mock_state.get_pending.return_value = None

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_shorten(update, _mock_context(), user_id=123)

                update.message.reply_text.assert_called_once()
                msg = update.message.reply_text.call_args[0][0]
                assert "no pending" in msg.lower()

        asyncio.run(_run())


class TestDoEditCaption:
    """Tests for the _do_edit_caption handler."""

    def test_edit_caption_no_pending(self):
        from bot.handlers.draft import _do_edit_caption

        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state:
                mock_state.get_pending.return_value = None

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_edit_caption(update, _mock_context(), user_id=123)

                update.message.reply_text.assert_called_once()
                msg = update.message.reply_text.call_args[0][0]
                assert "no pending" in msg.lower()

        asyncio.run(_run())

    def test_edit_caption_calls_agent(self):
        from bot.handlers.draft import _do_edit_caption

        async def _run():
            mock_result = MagicMock()
            mock_result.draft = {"caption": "New shiny caption", "hashtags": ["#test"]}
            mock_result.image_url = None
            mock_result.resources = []

            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state, \
                 patch(f"{_DRAFT}.engine") as mock_engine, \
                 patch(f"{_DRAFT}._send_draft") as mock_send, \
                 patch(f"{_DRAFT}._rate_limited", return_value=False):
                mock_state.get_pending.return_value = {
                    "caption": "Old caption",
                    "hashtags": [],
                    "image_url": "https://example.com/img.png",
                    "alt_text": "",
                    "image_prompt": "test prompt",
                    "original_request": "test request",
                    "content_type": "announcement",
                }
                mock_engine.run_agent = AsyncMock(return_value=mock_result)
                mock_send.return_value = None

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.message.chat.send_action = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_edit_caption(update, _mock_context(), user_id=123)

                mock_engine.run_agent.assert_called_once()
                # Instruction should mention rewriting caption
                instruction = mock_engine.run_agent.call_args[0][0]
                assert "caption" in instruction.lower()
                # Should send draft with new caption
                mock_send.assert_called_once()

        asyncio.run(_run())


class TestDoEditImage:
    """Tests for the _do_edit_image handler."""

    def test_edit_image_no_pending(self):
        from bot.handlers.draft import _do_edit_image

        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state:
                mock_state.get_pending.return_value = None

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_edit_image(update, _mock_context(), user_id=123)

                update.message.reply_text.assert_called_once()
                msg = update.message.reply_text.call_args[0][0]
                assert "no pending" in msg.lower()

        asyncio.run(_run())

    def test_edit_image_no_prompt(self):
        from bot.handlers.draft import _do_edit_image

        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state, \
                 patch(f"{_DRAFT}._rate_limited", return_value=False):
                mock_state.get_pending.return_value = {
                    "caption": "Some caption",
                    "image_prompt": "",
                    "content_type": "default",
                }

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.message.chat.send_action = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_edit_image(update, _mock_context(), user_id=123)

                update.message.reply_text.assert_called_once()
                msg = update.message.reply_text.call_args[0][0]
                assert "no image prompt" in msg.lower()

        asyncio.run(_run())

    def test_edit_image_calls_generate(self):
        from bot.handlers.draft import _do_edit_image

        async def _run():
            with patch(f"{_DRAFT}._can_operate", return_value=True), \
                 patch(f"{_DRAFT}.state") as mock_state, \
                 patch(f"{_DRAFT}.image_gen") as mock_img_gen, \
                 patch(f"{_DRAFT}._send_draft") as mock_send, \
                 patch(f"{_DRAFT}._rate_limited", return_value=False):
                mock_state.get_pending.return_value = {
                    "caption": "Keep this caption",
                    "hashtags": ["#tag"],
                    "image_url": "https://example.com/old.png",
                    "alt_text": "old alt",
                    "image_prompt": "a beautiful brand hero shot",
                    "original_request": "hero image",
                    "content_type": "hero",
                }
                mock_img_gen.generate_image = AsyncMock(return_value="https://example.com/new.png")
                mock_send.return_value = None

                update = MagicMock()
                update.message.reply_text = AsyncMock()
                update.message.chat.send_action = AsyncMock()
                update.effective_user = MagicMock()
                update.effective_user.id = 123

                await _do_edit_image(update, _mock_context(), user_id=123)

                mock_img_gen.generate_image.assert_called_once_with(
                    "a beautiful brand hero shot", "hero",
                )
                # Should save the new image url
                mock_state.save_pending.assert_called_once()
                saved_url = mock_state.save_pending.call_args.kwargs.get("image_url")
                assert saved_url == "https://example.com/new.png"
                # Caption should be preserved
                saved_caption = mock_state.save_pending.call_args.kwargs.get("caption")
                assert saved_caption == "Keep this caption"
                mock_send.assert_called_once()

        asyncio.run(_run())
