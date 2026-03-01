"""Tests for agent.chat — casual chat, greeting, and modify_last handlers."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.chat import (
    handle_greeting,
    handle_casual_chat,
    handle_modify_last,
    _GREETING_TEMPLATES,
)
from agent.conversation_context import ConversationContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(pending=False, last_action="idle", user_id=1):
    return ConversationContext(
        user_id=user_id,
        last_bot_action=last_action,
        pending_draft_exists=pending,
        updated_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Greeting
# ---------------------------------------------------------------------------

class TestHandleGreeting:
    def test_returns_string(self):
        result = asyncio.run(handle_greeting("Alice"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_name_when_provided(self):
        result = asyncio.run(handle_greeting("Bob"))
        assert "Bob" in result

    def test_works_without_name(self):
        result = asyncio.run(handle_greeting(""))
        assert isinstance(result, str)
        # Should not have double spaces
        assert "  " not in result

    def test_all_templates_valid(self):
        for template in _GREETING_TEMPLATES:
            result = template.format(name_part=" Test")
            assert "Test" in result
            result_empty = template.format(name_part="")
            assert isinstance(result_empty, str)


# ---------------------------------------------------------------------------
# Casual chat (mocked Haiku)
# ---------------------------------------------------------------------------

class TestHandleCasualChat:
    def test_returns_haiku_response(self):
        async def _run():
            with patch("agent.chat.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_resp = MagicMock()
                mock_resp.content = [MagicMock(text="Sure, I can help with that!")]
                mock_client.messages.create = AsyncMock(return_value=mock_resp)
                mock_cls.return_value = mock_client
                return await handle_casual_chat("how are you?", _ctx())

        result = asyncio.run(_run())
        assert result == "Sure, I can help with that!"

    def test_fallback_on_error(self):
        async def _run():
            with patch("agent.chat.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))
                mock_cls.return_value = mock_client
                return await handle_casual_chat("hello", _ctx())

        result = asyncio.run(_run())
        assert "content creation" in result.lower() or "topic" in result.lower()


# ---------------------------------------------------------------------------
# Modify last
# ---------------------------------------------------------------------------

class TestHandleModifyLast:
    def test_modifies_pending_draft(self):
        pending = {
            "caption": "Check out our new product!",
            "alt_text": "Product photo",
            "image_prompt": "modern product on white background",
            "hashtags": ["#launch"],
        }
        modified_json = json.dumps({
            "caption": "Exciting news: our new product is here!",
        })

        async def _run():
            with patch("agent.chat.state.get_pending", return_value=pending):
                with patch("agent.chat.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_resp = MagicMock()
                    mock_resp.content = [MagicMock(text=modified_json)]
                    mock_client.messages.create = AsyncMock(return_value=mock_resp)
                    mock_cls.return_value = mock_client
                    return await handle_modify_last("make it more exciting", _ctx(pending=True))

        result = asyncio.run(_run())
        assert result is not None
        assert result["caption"] == "Exciting news: our new product is here!"
        # Unchanged fields preserved
        assert result["alt_text"] == "Product photo"
        assert result["hashtags"] == ["#launch"]

    def test_returns_none_when_no_pending(self):
        async def _run():
            with patch("agent.chat.state.get_pending", return_value=None):
                return await handle_modify_last("change something", _ctx(pending=False))

        result = asyncio.run(_run())
        assert result is None

    def test_handles_api_error(self):
        pending = {"caption": "Test", "alt_text": "Test", "image_prompt": "test"}

        async def _run():
            with patch("agent.chat.state.get_pending", return_value=pending):
                with patch("agent.chat.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))
                    mock_cls.return_value = mock_client
                    return await handle_modify_last("change it", _ctx(pending=True))

        result = asyncio.run(_run())
        assert result is None

    def test_handles_bad_json_response(self):
        pending = {"caption": "Test", "alt_text": "Test", "image_prompt": "test"}

        async def _run():
            with patch("agent.chat.state.get_pending", return_value=pending):
                with patch("agent.chat.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_resp = MagicMock()
                    mock_resp.content = [MagicMock(text="Not valid JSON")]
                    mock_client.messages.create = AsyncMock(return_value=mock_resp)
                    mock_cls.return_value = mock_client
                    return await handle_modify_last("change it", _ctx(pending=True))

        result = asyncio.run(_run())
        assert result is None

    def test_strips_markdown_code_fences(self):
        pending = {"caption": "Old caption", "alt_text": "Test", "image_prompt": "test"}
        fenced_json = '```json\n{"caption": "New caption"}\n```'

        async def _run():
            with patch("agent.chat.state.get_pending", return_value=pending):
                with patch("agent.chat.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_resp = MagicMock()
                    mock_resp.content = [MagicMock(text=fenced_json)]
                    mock_client.messages.create = AsyncMock(return_value=mock_resp)
                    mock_cls.return_value = mock_client
                    return await handle_modify_last("update caption", _ctx(pending=True))

        result = asyncio.run(_run())
        assert result is not None
        assert result["caption"] == "New caption"

    def test_only_applies_known_fields(self):
        pending = {"caption": "Test", "alt_text": "Test", "image_prompt": "test"}
        modified_json = json.dumps({
            "caption": "Updated",
            "malicious_field": "should be ignored",
        })

        async def _run():
            with patch("agent.chat.state.get_pending", return_value=pending):
                with patch("agent.chat.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_resp = MagicMock()
                    mock_resp.content = [MagicMock(text=modified_json)]
                    mock_client.messages.create = AsyncMock(return_value=mock_resp)
                    mock_cls.return_value = mock_client
                    return await handle_modify_last("update it", _ctx(pending=True))

        result = asyncio.run(_run())
        assert result is not None
        assert "malicious_field" not in result
        assert result["caption"] == "Updated"
