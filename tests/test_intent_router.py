"""Tests for agent.intent_router — NL intent classification."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.conversation_context import ConversationContext
from agent.intent_router import (
    KNOWN_INTENTS,
    RoutingResult,
    _SHORT_MESSAGE_TABLE,
    _DRAFT_DEPENDENT_INTENTS,
    _check_rate_limit,
    _record_rate,
    _cache_put,
    _cache_get,
    _cache_key,
    classify_intent,
    clear_cache,
    reset_rate_limits,
    _RATE_LIMIT_PER_HOUR,
    _CACHE_TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state():
    """Reset rate limits and cache between tests."""
    reset_rate_limits()
    clear_cache()
    yield
    reset_rate_limits()
    clear_cache()


def _ctx(pending=False, last_action="idle", user_id=1, last_content_type=""):
    return ConversationContext(
        user_id=user_id,
        last_bot_action=last_action,
        pending_draft_exists=pending,
        last_content_type=last_content_type,
        updated_at=time.time(),
    )


def _mock_haiku_response(intent, confidence=0.9, parameters=None):
    """Create a mock Anthropic response."""
    data = {"intent": intent, "confidence": confidence, "parameters": parameters or {}}
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(data))]
    return mock_response


# ---------------------------------------------------------------------------
# Short message table
# ---------------------------------------------------------------------------

class TestShortMessageTable:
    def test_approve_variants(self):
        for msg in ("yes", "yep", "looks good", "lgtm", "post it", "ship it", "approve"):
            assert _SHORT_MESSAGE_TABLE[msg] == "approve", f"Failed for: {msg}"

    def test_reject_variants(self):
        for msg in ("no", "nah", "nope", "skip", "reject", "cancel", "discard"):
            assert _SHORT_MESSAGE_TABLE[msg] == "reject", f"Failed for: {msg}"

    def test_reroll_variants(self):
        for msg in ("try again", "again", "another", "reroll", "redo", "regenerate"):
            assert _SHORT_MESSAGE_TABLE[msg] == "reroll", f"Failed for: {msg}"

    def test_greeting_variants(self):
        for msg in ("hi", "hello", "hey", "gm", "good morning"):
            assert _SHORT_MESSAGE_TABLE[msg] == "greeting", f"Failed for: {msg}"

    def test_upload_variants(self):
        for msg in ("upload", "ingest", "send images", "send pictures", "add images"):
            assert _SHORT_MESSAGE_TABLE[msg] == "upload_assets", f"Failed for: {msg}"

    def test_utility_variants(self):
        assert _SHORT_MESSAGE_TABLE["help"] == "show_help"
        assert _SHORT_MESSAGE_TABLE["status"] == "show_status"
        assert _SHORT_MESSAGE_TABLE["analytics"] == "show_analytics"
        assert _SHORT_MESSAGE_TABLE["history"] == "show_history"


# ---------------------------------------------------------------------------
# Table-based classification
# ---------------------------------------------------------------------------

class TestTableClassification:
    def test_approve_with_pending_draft(self):
        result = asyncio.run(classify_intent("yes", _ctx(pending=True)))
        assert result.intent == "approve"
        assert result.confidence == 0.95
        assert result.routed_via == "table"

    def test_approve_without_pending_draft_remaps_to_casual(self):
        result = asyncio.run(classify_intent("yes", _ctx(pending=False)))
        assert result.intent == "casual_chat"
        assert result.routed_via == "table"

    def test_reject_with_pending_draft(self):
        result = asyncio.run(classify_intent("no", _ctx(pending=True)))
        assert result.intent == "reject"
        assert result.confidence == 0.95

    def test_reject_without_pending_draft_remaps(self):
        result = asyncio.run(classify_intent("nah", _ctx(pending=False)))
        assert result.intent == "casual_chat"

    def test_reroll_with_pending_draft(self):
        result = asyncio.run(classify_intent("try again", _ctx(pending=True)))
        assert result.intent == "reroll"

    def test_reroll_without_pending_draft_remaps(self):
        result = asyncio.run(classify_intent("try again", _ctx(pending=False)))
        assert result.intent == "casual_chat"

    def test_greeting_always_works(self):
        result = asyncio.run(classify_intent("hi", _ctx(pending=False)))
        assert result.intent == "greeting"
        assert result.confidence == 0.95

    def test_case_insensitive(self):
        result = asyncio.run(classify_intent("YES", _ctx(pending=True)))
        assert result.intent == "approve"

    def test_whitespace_stripped(self):
        result = asyncio.run(classify_intent("  hello  ", _ctx()))
        assert result.intent == "greeting"

    def test_help_no_draft_dependency(self):
        result = asyncio.run(classify_intent("help", _ctx(pending=False)))
        assert result.intent == "show_help"
        assert result.confidence == 0.95


# ---------------------------------------------------------------------------
# Haiku classification (mocked)
# ---------------------------------------------------------------------------

class TestHaikuClassification:
    def test_routes_to_haiku_for_complex_messages(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=_mock_haiku_response("generate_content", 0.9, {"topic": "weekly update"})
                )
                mock_cls.return_value = mock_client
                return await classify_intent(
                    "write a post about our weekly update",
                    _ctx(),
                )

        result = asyncio.run(_run())
        assert result.intent == "generate_content"
        assert result.confidence == 0.9
        assert result.parameters.get("topic") == "weekly update"
        assert result.routed_via == "haiku"

    def test_edit_request_with_feedback(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=_mock_haiku_response(
                        "edit_request", 0.85, {"feedback": "make it more playful"}
                    )
                )
                mock_cls.return_value = mock_client
                return await classify_intent(
                    "make it more playful and fun",
                    _ctx(pending=True),
                )

        result = asyncio.run(_run())
        assert result.intent == "edit_request"
        assert result.parameters["feedback"] == "make it more playful"

    def test_invalid_intent_remapped_to_unknown(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=_mock_haiku_response("nonexistent_intent", 0.8)
                )
                mock_cls.return_value = mock_client
                return await classify_intent("something weird", _ctx())

        result = asyncio.run(_run())
        assert result.intent == "unknown"
        assert result.confidence == 0.0

    def test_low_confidence_forced_to_unknown(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=_mock_haiku_response("approve", 0.2)
                )
                mock_cls.return_value = mock_client
                return await classify_intent("ambiguous message", _ctx())

        result = asyncio.run(_run())
        assert result.intent == "unknown"

    def test_bad_json_response(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_resp = MagicMock()
                mock_resp.content = [MagicMock(text="Not valid JSON")]
                mock_client.messages.create = AsyncMock(return_value=mock_resp)
                mock_cls.return_value = mock_client
                return await classify_intent("some message", _ctx())

        result = asyncio.run(_run())
        assert result.intent == "unknown"
        assert result.routed_via == "haiku"

    def test_strips_markdown_code_fences(self):
        """Haiku sometimes wraps JSON in ```json ... ``` — should still parse."""
        fenced = '```json\n{"intent": "generate_content", "confidence": 0.85, "parameters": {"topic": "brand vision"}}\n```'

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_resp = MagicMock()
                mock_resp.content = [MagicMock(text=fenced)]
                mock_client.messages.create = AsyncMock(return_value=mock_resp)
                mock_cls.return_value = mock_client
                return await classify_intent("help me with brand vision", _ctx())

        result = asyncio.run(_run())
        assert result.intent == "generate_content"
        assert result.confidence == 0.85
        assert result.parameters.get("topic") == "brand vision"
        assert result.routed_via == "haiku"

    def test_upload_assets_classification(self):
        """Upload-related messages should classify as upload_assets."""
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=_mock_haiku_response("upload_assets", 0.9)
                )
                mock_cls.return_value = mock_client
                return await classify_intent(
                    "can i give you my brand assets?", _ctx()
                )

        result = asyncio.run(_run())
        assert result.intent == "upload_assets"
        assert result.routed_via == "haiku"

    def test_upload_table_lookup(self):
        result = asyncio.run(classify_intent("upload", _ctx()))
        assert result.intent == "upload_assets"
        assert result.routed_via == "table"

    def test_haiku_context_aware_approve_without_draft(self):
        """Haiku returns approve but no draft pending → remap to casual_chat."""
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=_mock_haiku_response("approve", 0.9)
                )
                mock_cls.return_value = mock_client
                return await classify_intent("that sounds great", _ctx(pending=False))

        result = asyncio.run(_run())
        assert result.intent == "casual_chat"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_under_limit(self):
        assert _check_rate_limit(1) is True

    def test_over_limit(self):
        for _ in range(_RATE_LIMIT_PER_HOUR):
            _record_rate(1)
        assert _check_rate_limit(1) is False

    def test_old_entries_expire(self):
        import agent.intent_router as mod
        # Inject old timestamps
        mod._rate_counts[1] = [time.time() - 3700] * _RATE_LIMIT_PER_HOUR
        assert _check_rate_limit(1) is True

    def test_rate_limit_falls_back_to_generate(self):
        for _ in range(_RATE_LIMIT_PER_HOUR):
            _record_rate(1)

        async def _run():
            return await classify_intent("write a post about cats", _ctx(user_id=1))

        result = asyncio.run(_run())
        assert result.intent == "generate_content"
        assert result.routed_via == "fallback"

    def test_table_lookups_bypass_rate_limit(self):
        for _ in range(_RATE_LIMIT_PER_HOUR):
            _record_rate(1)

        result = asyncio.run(classify_intent("hello", _ctx(user_id=1)))
        assert result.intent == "greeting"
        assert result.routed_via == "table"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_hit(self):
        ctx = _ctx()
        key = _cache_key("test message", ctx)
        result = RoutingResult(intent="greeting", confidence=0.9, routed_via="haiku")
        _cache_put(key, result)
        cached = _cache_get(key)
        assert cached is not None
        assert cached.intent == "greeting"

    def test_cache_miss(self):
        assert _cache_get("nonexistent") is None

    def test_cache_ttl_expiry(self):
        import agent.intent_router as mod
        key = "test_ttl"
        result = RoutingResult(intent="greeting", confidence=0.9)
        mod._cache[key] = (result, time.time() - _CACHE_TTL_SECONDS - 10)
        assert _cache_get(key) is None

    def test_cache_eviction(self):
        from agent.intent_router import _CACHE_MAX_SIZE
        for i in range(_CACHE_MAX_SIZE + 5):
            _cache_put(f"key_{i}", RoutingResult(intent="unknown", confidence=0.0))
        import agent.intent_router as mod
        assert len(mod._cache) <= _CACHE_MAX_SIZE

    def test_haiku_result_cached(self):
        """Second identical call should hit cache, not call Haiku again."""
        call_count = 0

        async def _run():
            nonlocal call_count
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def _create(**kwargs):
                    nonlocal call_count
                    call_count += 1
                    return _mock_haiku_response("generate_content", 0.9, {"topic": "test"})
                mock_client.messages.create = _create
                mock_cls.return_value = mock_client

                ctx = _ctx()
                r1 = await classify_intent("write about our product launch", ctx)
                r2 = await classify_intent("write about our product launch", ctx)
                return r1, r2

        r1, r2 = asyncio.run(_run())
        assert call_count == 1
        assert r1.intent == r2.intent == "generate_content"


# ---------------------------------------------------------------------------
# Timeout and error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_timeout_falls_back(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def _slow(**kwargs):
                    await asyncio.sleep(10)
                mock_client.messages.create = _slow
                mock_cls.return_value = mock_client
                return await classify_intent("test message", _ctx())

        # Override timeout for test speed
        with patch("agent.intent_router._HAIKU_TIMEOUT_SECONDS", 0.1):
            result = asyncio.run(_run())

        assert result.intent == "generate_content"
        assert result.routed_via == "fallback"

    def test_api_error_falls_back(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    side_effect=Exception("API error")
                )
                mock_cls.return_value = mock_client
                return await classify_intent("test message", _ctx())

        result = asyncio.run(_run())
        assert result.intent == "generate_content"
        assert result.routed_via == "fallback"


# ---------------------------------------------------------------------------
# Known intents validation
# ---------------------------------------------------------------------------

class TestKnownIntents:
    def test_all_table_intents_are_known(self):
        for intent in _SHORT_MESSAGE_TABLE.values():
            assert intent in KNOWN_INTENTS, f"Table intent '{intent}' not in KNOWN_INTENTS"

    def test_draft_dependent_intents_are_known(self):
        for intent in _DRAFT_DEPENDENT_INTENTS:
            assert intent in KNOWN_INTENTS

    def test_known_intents_tuple(self):
        assert len(KNOWN_INTENTS) == 17
        assert "approve" in KNOWN_INTENTS
        assert "schedule_post" in KNOWN_INTENTS
        assert "upload_assets" in KNOWN_INTENTS
        assert "unknown" in KNOWN_INTENTS
