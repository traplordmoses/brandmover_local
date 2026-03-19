"""Tests for agent/model_fallback.py — LLM fallback chains."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from agent.model_fallback import (
    DEFAULT_CHAINS,
    _RETRIABLE_STATUS,
    _detect_provider,
    _wrap_as_anthropic,
    call_with_fallback,
    get_fallback_chain,
)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestDetectProvider:
    def test_claude_is_anthropic(self):
        assert _detect_provider("claude-sonnet-4-6") == "anthropic"

    def test_gpt_is_openai(self):
        assert _detect_provider("gpt-4o") == "openai"

    def test_o1_is_openai(self):
        assert _detect_provider("o1-mini") == "openai"

    def test_gemini_is_google(self):
        assert _detect_provider("gemini-2.0-flash") == "google"

    def test_unknown_defaults_to_anthropic(self):
        assert _detect_provider("some-random-model") == "anthropic"


# ---------------------------------------------------------------------------
# Fallback chain construction
# ---------------------------------------------------------------------------

class TestGetFallbackChain:
    def test_primary_in_default_chain_returns_tail(self):
        """If primary is in a default chain, returns from that point onward."""
        chain = get_fallback_chain("claude-sonnet-4-6")
        assert chain[0] == "claude-sonnet-4-6"
        assert len(chain) > 1

    def test_haiku_chain_is_single(self):
        chain = get_fallback_chain("claude-haiku-4-5-20251001")
        # Haiku appears in both agent and haiku chains; should start from haiku
        assert chain[0] == "claude-haiku-4-5-20251001"

    def test_unknown_model_returns_singleton(self):
        chain = get_fallback_chain("custom-model-v1")
        assert chain == ["custom-model-v1"]

    def test_env_override_takes_priority(self):
        with patch.dict(os.environ, {"AGENT_FALLBACK_MODELS": "model-a, model-b, model-c"}):
            chain = get_fallback_chain("model-b")
        assert chain == ["model-b", "model-c"]

    def test_env_override_prepends_primary_if_not_in_list(self):
        with patch.dict(os.environ, {"AGENT_FALLBACK_MODELS": "model-a, model-b"}):
            chain = get_fallback_chain("my-primary")
        assert chain == ["my-primary", "model-a", "model-b"]

    def test_empty_env_ignored(self):
        with patch.dict(os.environ, {"AGENT_FALLBACK_MODELS": ""}):
            chain = get_fallback_chain("claude-sonnet-4-6")
        assert chain[0] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Primary model succeeds — no fallback
# ---------------------------------------------------------------------------

class TestPrimarySucceeds:
    @pytest.mark.asyncio
    async def test_returns_response_on_first_try(self):
        """When primary Anthropic model succeeds, return directly."""
        mock_response = MagicMock(spec=anthropic.types.Message)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await call_with_fallback(
            client=mock_client,
            models=["claude-sonnet-4-6"],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )

        assert result is mock_response
        mock_client.messages.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Primary fails with 429 — falls back to next
# ---------------------------------------------------------------------------

class TestFallbackOn429:
    @pytest.mark.asyncio
    async def test_falls_back_on_rate_limit(self):
        """429 from primary triggers fallback to next Anthropic model."""
        error_429 = anthropic.APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_response = MagicMock(spec=anthropic.types.Message)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[error_429, mock_response],
        )

        with patch("agent.model_fallback._provider_available", return_value=True):
            result = await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

        assert result is mock_response
        assert mock_client.messages.create.await_count == 2

    @pytest.mark.asyncio
    async def test_non_retriable_status_raises_immediately(self):
        """A 400 error does not trigger fallback — raised immediately."""
        error_400 = anthropic.APIStatusError(
            message="bad request",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=error_400)

        with patch("agent.model_fallback._provider_available", return_value=True), \
             pytest.raises(anthropic.APIStatusError):
            await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

        # Only called once — no fallback attempted
        assert mock_client.messages.create.await_count == 1


# ---------------------------------------------------------------------------
# Non-Anthropic provider strips tools (verify warning logged)
# ---------------------------------------------------------------------------

class TestCrossProviderFallback:
    @pytest.mark.asyncio
    async def test_tools_stripped_with_warning(self):
        """Falling back to OpenAI strips tools and logs a warning."""
        error_429 = anthropic.APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=error_429)

        openai_result = {
            "provider": "openai",
            "model": "gpt-4o",
            "content": [{"type": "text", "text": "fallback text"}],
            "stop_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        with patch("agent.model_fallback._provider_available", return_value=True), \
             patch.dict("agent.model_fallback._PROVIDER_CALLERS", {"openai": AsyncMock(return_value=openai_result)}), \
             patch("agent.model_fallback.logger") as mock_logger:
            result = await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "gpt-4o"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                tools=[{"name": "test_tool"}],
            )

        # Should have logged a warning about tools being stripped
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "tools" in warning_msg.lower() or "ARCH-03" in warning_msg

        # Result should be an Anthropic Message wrapper
        assert isinstance(result, anthropic.types.Message)
        assert "FALLBACK NOTICE" in result.content[0].text


# ---------------------------------------------------------------------------
# All models fail — final error raised
# ---------------------------------------------------------------------------

class TestAllModelsFail:
    @pytest.mark.asyncio
    async def test_raises_last_error(self):
        """When all models fail, the last error is raised."""
        error = anthropic.APIConnectionError(request=MagicMock())
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=error)

        with patch("agent.model_fallback._provider_available", return_value=True), \
             pytest.raises(anthropic.APIConnectionError):
            await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )


# ---------------------------------------------------------------------------
# Response wrapping (_wrap_as_anthropic)
# ---------------------------------------------------------------------------

class TestWrapAsAnthropic:
    def test_basic_wrapping(self):
        """Wraps a normalized dict into an Anthropic Message."""
        result = {
            "provider": "openai",
            "model": "gpt-4o",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        msg = _wrap_as_anthropic(result)
        assert isinstance(msg, anthropic.types.Message)
        assert msg.content[0].text == "hello"
        assert msg.model == "gpt-4o"
        assert msg.usage.input_tokens == 5

    def test_tools_degraded_prepends_notice(self):
        """When tools_degraded=True, the text includes a FALLBACK NOTICE."""
        result = {
            "provider": "google",
            "model": "gemini-2.0-flash",
            "content": [{"type": "text", "text": "response text"}],
            "usage": {},
        }
        msg = _wrap_as_anthropic(result, tools_degraded=True)
        assert "FALLBACK NOTICE" in msg.content[0].text
        assert "response text" in msg.content[0].text

    def test_empty_content_still_works(self):
        result = {
            "provider": "openai",
            "model": "gpt-4o",
            "content": [],
            "usage": {},
        }
        msg = _wrap_as_anthropic(result)
        assert msg.content[0].text == ""
