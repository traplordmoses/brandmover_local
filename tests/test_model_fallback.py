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
        """Falling back to OpenAI passes tools through (full tool-use support)."""
        error_429 = anthropic.APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=error_429)

        openai_result = {
            "provider": "openai",
            "model": "gpt-5.4",
            "content": [{"type": "text", "text": "fallback text"}],
            "stop_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        with patch("agent.model_fallback._provider_available", return_value=True), \
             patch.dict("agent.model_fallback._PROVIDER_CALLERS", {"openai": AsyncMock(return_value=openai_result)}), \
             patch("agent.model_fallback.logger") as mock_logger:
            result = await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "gpt-5.4"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                tools=[{"name": "test_tool"}],
            )

        # OpenAI now gets tools passed through — should log info, not warning about stripping
        mock_logger.info.assert_called()

        # Result should be an Anthropic Message wrapper without FALLBACK NOTICE
        assert isinstance(result, anthropic.types.Message)
        assert "fallback text" in result.content[0].text


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


# ---------------------------------------------------------------------------
# OpenAI tool conversion
# ---------------------------------------------------------------------------

class TestAnthropicToolsToOpenAI:
    """Test _anthropic_tools_to_openai() tool definition conversion."""

    def test_basic_tool_conversion(self):
        from agent.model_fallback import _anthropic_tools_to_openai

        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            }
        ]
        oai_tools = _anthropic_tools_to_openai(anthropic_tools)

        assert len(oai_tools) == 1
        assert oai_tools[0]["type"] == "function"
        assert oai_tools[0]["function"]["name"] == "get_weather"
        assert oai_tools[0]["function"]["description"] == "Get current weather for a location"
        assert oai_tools[0]["function"]["parameters"]["properties"]["location"]["type"] == "string"

    def test_missing_description_defaults_empty(self):
        from agent.model_fallback import _anthropic_tools_to_openai

        tools = [{"name": "my_tool", "input_schema": {"type": "object", "properties": {}}}]
        oai_tools = _anthropic_tools_to_openai(tools)

        assert oai_tools[0]["function"]["description"] == ""

    def test_missing_input_schema_defaults(self):
        from agent.model_fallback import _anthropic_tools_to_openai

        tools = [{"name": "simple_tool", "description": "A tool"}]
        oai_tools = _anthropic_tools_to_openai(tools)

        assert oai_tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_multiple_tools(self):
        from agent.model_fallback import _anthropic_tools_to_openai

        tools = [
            {"name": "tool_a", "description": "A", "input_schema": {"type": "object", "properties": {}}},
            {"name": "tool_b", "description": "B", "input_schema": {"type": "object", "properties": {}}},
            {"name": "tool_c", "description": "C", "input_schema": {"type": "object", "properties": {}}},
        ]
        oai_tools = _anthropic_tools_to_openai(tools)
        assert len(oai_tools) == 3
        assert [t["function"]["name"] for t in oai_tools] == ["tool_a", "tool_b", "tool_c"]


class TestAnthropicToolChoiceToOpenAI:
    """Test _anthropic_tool_choice_to_openai() conversion for all cases."""

    def test_auto(self):
        from agent.model_fallback import _anthropic_tool_choice_to_openai

        assert _anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"

    def test_none(self):
        from agent.model_fallback import _anthropic_tool_choice_to_openai

        assert _anthropic_tool_choice_to_openai({"type": "none"}) == "none"

    def test_any_becomes_required(self):
        from agent.model_fallback import _anthropic_tool_choice_to_openai

        assert _anthropic_tool_choice_to_openai({"type": "any"}) == "required"

    def test_tool_becomes_function_dict(self):
        from agent.model_fallback import _anthropic_tool_choice_to_openai

        result = _anthropic_tool_choice_to_openai({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_missing_type_defaults_to_auto(self):
        from agent.model_fallback import _anthropic_tool_choice_to_openai

        assert _anthropic_tool_choice_to_openai({}) == "auto"

    def test_unknown_type_defaults_to_auto(self):
        from agent.model_fallback import _anthropic_tool_choice_to_openai

        assert _anthropic_tool_choice_to_openai({"type": "something_else"}) == "auto"


# ---------------------------------------------------------------------------
# OpenAI message format conversion (mock httpx POST)
# ---------------------------------------------------------------------------

class TestCallOpenAIMessageConversion:
    """Test that _call_openai() correctly converts Anthropic message format."""

    @pytest.mark.asyncio
    async def test_system_and_user_messages(self):
        """System prompt and simple user messages are converted correctly."""
        import json as _json

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello back"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        captured_body = {}

        async def mock_post(url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return mock_resp

        mock_httpx = AsyncMock()
        mock_httpx.post = mock_post

        with patch("agent.model_fallback.get_httpx", return_value=mock_httpx), \
             patch("config.settings.OPENAI_API_KEY", "test-key"):
            from agent.model_fallback import _call_openai
            result = await _call_openai(
                "gpt-4o",
                system="You are helpful",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

        # Verify system message was prepended
        assert captured_body["messages"][0]["role"] == "system"
        assert captured_body["messages"][0]["content"] == "You are helpful"
        # Verify user message
        assert captured_body["messages"][1]["role"] == "user"
        assert captured_body["messages"][1]["content"] == "hi"
        # Verify result
        assert result["provider"] == "openai"
        assert result["content"][0]["text"] == "hello back"

    @pytest.mark.asyncio
    async def test_tool_use_messages_converted(self):
        """Assistant tool_use blocks and user tool_result blocks are converted."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {},
        }

        captured_body = {}

        async def mock_post(url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return mock_resp

        mock_httpx = AsyncMock()
        mock_httpx.post = mock_post

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"city": "NYC"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "72F sunny"},
                ],
            },
        ]

        with patch("agent.model_fallback.get_httpx", return_value=mock_httpx), \
             patch("config.settings.OPENAI_API_KEY", "test-key"):
            from agent.model_fallback import _call_openai
            await _call_openai("gpt-4o", messages=messages, max_tokens=100)

        oai_msgs = captured_body["messages"]
        # User message
        assert oai_msgs[0]["role"] == "user"
        assert oai_msgs[0]["content"] == "What's the weather?"
        # Assistant message with tool_calls
        assert oai_msgs[1]["role"] == "assistant"
        assert oai_msgs[1]["tool_calls"][0]["id"] == "call_123"
        assert oai_msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        # Tool result message
        assert oai_msgs[2]["role"] == "tool"
        assert oai_msgs[2]["tool_call_id"] == "call_123"
        assert oai_msgs[2]["content"] == "72F sunny"

    @pytest.mark.asyncio
    async def test_tool_calls_in_response_parsed(self):
        """OpenAI tool_calls in response are parsed as tool_use blocks."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "model": "gpt-4o",
            "usage": {},
        }

        async def mock_post(url, **kwargs):
            return mock_resp

        mock_httpx = AsyncMock()
        mock_httpx.post = mock_post

        with patch("agent.model_fallback.get_httpx", return_value=mock_httpx), \
             patch("config.settings.OPENAI_API_KEY", "test-key"):
            from agent.model_fallback import _call_openai
            result = await _call_openai("gpt-4o", messages=[{"role": "user", "content": "weather?"}], max_tokens=100)

        assert result["stop_reason"] == "tool_use"
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "get_weather"
        assert result["content"][0]["input"] == {"city": "NYC"}


# ---------------------------------------------------------------------------
# Gemini tool stripping with ARCH-03 warning
# ---------------------------------------------------------------------------

class TestGeminiToolSupport:
    """Test that Gemini fallback passes tools through with function-calling support."""

    @pytest.mark.asyncio
    async def test_gemini_receives_tools_and_strips_tool_choice(self):
        """When falling back to Gemini with tools, tools are passed but tool_choice is stripped."""
        error_429 = anthropic.APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=error_429)

        gemini_result = {
            "provider": "google",
            "model": "gemini-2.0-flash",
            "content": [{"type": "text", "text": "gemini response"}],
            "stop_reason": "STOP",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        captured_kwargs = {}

        async def fake_gemini(model, **kwargs):
            captured_kwargs.update(kwargs)
            return gemini_result

        with patch("agent.model_fallback._provider_available", return_value=True), \
             patch.dict("agent.model_fallback._PROVIDER_CALLERS", {"google": fake_gemini}), \
             patch("agent.model_fallback.logger") as mock_logger:
            result = await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "gemini-2.0-flash"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                tools=[{"name": "my_tool", "description": "A tool"}],
                tool_choice={"type": "auto"},
            )

        # Verify tools are passed through to Gemini (function-calling support)
        assert "tools" in captured_kwargs
        # Verify tool_choice is stripped (Gemini doesn't support it)
        assert "tool_choice" not in captured_kwargs

        # Verify no ARCH-03 warning (Gemini now supports tools)
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert not any("ARCH-03" in w for w in warning_calls)

        # Verify info-level log about tool-use support
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("tool-use support" in i for i in info_calls)

        # Verify result does NOT include FALLBACK NOTICE (tools not degraded)
        assert isinstance(result, anthropic.types.Message)
        assert "FALLBACK NOTICE" not in result.content[0].text
        assert "gemini response" in result.content[0].text

    @pytest.mark.asyncio
    async def test_gemini_function_call_response_parsed(self):
        """When Gemini returns a functionCall, it is parsed into a tool_use block."""
        from agent.model_fallback import _call_gemini, _wrap_as_anthropic

        gemini_result = {
            "provider": "google",
            "model": "gemini-2.0-flash",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_gemini_my_tool",
                    "name": "my_tool",
                    "input": {"arg1": "value1"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        wrapped = _wrap_as_anthropic(gemini_result)
        assert wrapped.stop_reason == "tool_use"
        assert len(wrapped.content) == 1
        assert wrapped.content[0].type == "tool_use"
        assert wrapped.content[0].name == "my_tool"
        assert wrapped.content[0].input == {"arg1": "value1"}


# ---------------------------------------------------------------------------
# Credit balance error treated as retriable (special case at line 465)
# ---------------------------------------------------------------------------

class TestCreditBalanceFallback:
    """Test that 400 + 'credit balance' in message IS retriable."""

    @pytest.mark.asyncio
    async def test_credit_balance_400_triggers_fallback(self):
        """A 400 error with 'credit balance' in the message triggers fallback."""
        error_credit = anthropic.APIStatusError(
            message="Your credit balance is too low to access this model",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        mock_response = MagicMock(spec=anthropic.types.Message)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[error_credit, mock_response],
        )

        with patch("agent.model_fallback._provider_available", return_value=True):
            result = await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

        # Should have fallen back to second model
        assert result is mock_response
        assert mock_client.messages.create.await_count == 2

    @pytest.mark.asyncio
    async def test_regular_400_does_not_trigger_fallback(self):
        """A normal 400 without 'credit balance' does NOT trigger fallback."""
        error_400 = anthropic.APIStatusError(
            message="invalid request: max_tokens too large",
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

        assert mock_client.messages.create.await_count == 1


# ---------------------------------------------------------------------------
# Provider availability filtering
# ---------------------------------------------------------------------------

class TestProviderAvailabilityFiltering:
    """Test that models whose providers lack API keys are skipped."""

    @pytest.mark.asyncio
    async def test_skips_unavailable_provider(self):
        """Models whose providers lack API keys are skipped in the chain."""
        error_429 = anthropic.APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_response = MagicMock(spec=anthropic.types.Message)
        mock_client = AsyncMock()
        # First call (Claude Sonnet) fails with 429, second call (Claude Haiku) succeeds
        mock_client.messages.create = AsyncMock(
            side_effect=[error_429, mock_response],
        )

        def fake_available(provider):
            # Anthropic available, OpenAI not
            return provider == "anthropic"

        with patch("agent.model_fallback._provider_available", side_effect=fake_available):
            result = await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "gpt-5.4", "claude-haiku-4-5-20251001"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

        # gpt-5.4 should have been skipped; went from sonnet -> haiku
        assert result is mock_response
        assert mock_client.messages.create.await_count == 2
        # Verify the second call used haiku
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        assert second_call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_all_providers_unavailable_tries_first(self):
        """When ALL providers lack keys, tries the first model anyway."""
        mock_response = MagicMock(spec=anthropic.types.Message)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("agent.model_fallback._provider_available", return_value=False):
            result = await call_with_fallback(
                client=mock_client,
                models=["claude-sonnet-4-6", "gpt-5.4"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

        # Should try first model even though no providers are "available"
        assert result is mock_response
        assert mock_client.messages.create.await_count == 1
