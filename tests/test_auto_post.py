"""Tests for scripts/auto_post.py — auto-post scheduler logic."""

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _notify_telegram
# ---------------------------------------------------------------------------

class TestNotifyTelegram:
    """Tests for the standalone Telegram notification helper."""

    @pytest.mark.asyncio
    async def test_sends_correct_payload(self):
        """_notify_telegram posts the right chat_id and text."""
        mock_response = MagicMock(status_code=200)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as _httpx_mod
        with patch("scripts.auto_post.settings") as mock_settings, \
             patch.object(_httpx_mod, "AsyncClient", return_value=mock_client):
            mock_settings.TELEGRAM_BOT_TOKEN = "tok123"
            mock_settings.TELEGRAM_ALLOWED_USER_ID = "99"

            from scripts.auto_post import _notify_telegram
            await _notify_telegram("hello world")

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.telegram.org/bottok123/sendMessage"
        payload = call_args[1]["json"]
        assert payload["chat_id"] == "99"
        assert payload["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_skips_when_no_token(self):
        """_notify_telegram is a no-op when token is missing."""
        with patch("scripts.auto_post.settings") as mock_settings:
            mock_settings.TELEGRAM_BOT_TOKEN = ""
            mock_settings.TELEGRAM_ALLOWED_USER_ID = "99"

            from scripts.auto_post import _notify_telegram
            # Should complete without error or HTTP call
            await _notify_telegram("ignored")

    @pytest.mark.asyncio
    async def test_skips_when_no_user_id(self):
        """_notify_telegram is a no-op when user ID is missing."""
        with patch("scripts.auto_post.settings") as mock_settings:
            mock_settings.TELEGRAM_BOT_TOKEN = "tok123"
            mock_settings.TELEGRAM_ALLOWED_USER_ID = ""

            from scripts.auto_post import _notify_telegram
            await _notify_telegram("ignored")

    @pytest.mark.asyncio
    async def test_handles_http_error_gracefully(self):
        """_notify_telegram logs a warning on non-200 response."""
        mock_response = MagicMock(status_code=500)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as _httpx_mod
        with patch("scripts.auto_post.settings") as mock_settings, \
             patch.object(_httpx_mod, "AsyncClient", return_value=mock_client):
            mock_settings.TELEGRAM_BOT_TOKEN = "tok123"
            mock_settings.TELEGRAM_ALLOWED_USER_ID = "99"

            from scripts.auto_post import _notify_telegram
            # Should not raise
            await _notify_telegram("test")

    @pytest.mark.asyncio
    async def test_handles_network_exception_gracefully(self):
        """_notify_telegram catches and logs exceptions."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("no network"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as _httpx_mod
        with patch("scripts.auto_post.settings") as mock_settings, \
             patch.object(_httpx_mod, "AsyncClient", return_value=mock_client):
            mock_settings.TELEGRAM_BOT_TOKEN = "tok123"
            mock_settings.TELEGRAM_ALLOWED_USER_ID = "99"

            from scripts.auto_post import _notify_telegram
            # Should not raise
            await _notify_telegram("test")


# ---------------------------------------------------------------------------
# Rate limiting (via auto_state.can_post)
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Verify process_slot respects rate limiting."""

    @pytest.mark.asyncio
    async def test_skips_slot_when_rate_limited(self):
        """process_slot returns False when can_post says no."""
        with patch("scripts.auto_post.auto_state") as mock_auto_state, \
             patch("scripts.auto_post.context_feed") as mock_ctx:
            mock_auto_state.can_post.return_value = (False, "Too many posts today")
            mock_ctx.aggregate_context = AsyncMock(return_value=MagicMock(
                summary="", has_urgent=False, signals=[],
            ))

            from scripts.auto_post import process_slot
            result = await process_slot(
                "test_slot", {"type": "engagement"}, {"min_gap_minutes": 60, "max_posts_per_day": 4},
            )

        assert result is False
        mock_auto_state.can_post.assert_called_once_with(60, 4)

    @pytest.mark.asyncio
    async def test_ignores_rate_limit_on_dry_run(self):
        """process_slot proceeds past rate limit when dry_run=True."""
        mock_result = MagicMock()
        mock_result.draft = {"caption": "test", "hashtags": [], "content_type": "engagement"}
        mock_result.image_url = None
        mock_result.image_urls = []
        mock_result.turns_used = 1
        mock_result.total_time = 0.5
        mock_result.tool_calls_made = []
        mock_result.conversation_history = []

        with patch("scripts.auto_post.auto_state") as mock_auto_state, \
             patch("scripts.auto_post.context_feed") as mock_ctx, \
             patch("scripts.auto_post.engine") as mock_engine, \
             patch("scripts.auto_post.scheduler") as mock_sched, \
             patch("scripts.auto_post.state") as mock_state, \
             patch("scripts.auto_post.preference_engine") as mock_pref:
            mock_auto_state.can_post.return_value = (False, "rate limited")
            mock_auto_state.is_duplicate_caption.return_value = False
            mock_ctx.aggregate_context = AsyncMock(return_value=MagicMock(
                summary="", has_urgent=False, signals=[],
            ))
            mock_engine.run_agent = AsyncMock(return_value=mock_result)
            mock_sched.build_prompt_for_slot = AsyncMock(return_value=("test prompt", []))
            mock_state.async_has_pending = AsyncMock(return_value=False)
            mock_pref.score_draft = AsyncMock(return_value=MagicMock(
                score=8.0, should_reject=False, reasoning="good", flags=[],
            ))

            from scripts.auto_post import process_slot
            result = await process_slot(
                "test_slot", {"type": "engagement"}, {},
                dry_run=True,
            )

        assert result is True


# ---------------------------------------------------------------------------
# Duplicate caption detection (via auto_state.is_duplicate_caption)
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    """Verify process_slot rejects duplicate captions."""

    @pytest.mark.asyncio
    async def test_skips_duplicate_caption(self):
        """process_slot returns False when caption is duplicate."""
        mock_result = MagicMock()
        mock_result.draft = {"caption": "same old same old", "hashtags": []}
        mock_result.image_url = None
        mock_result.image_urls = []
        mock_result.turns_used = 1
        mock_result.total_time = 0.5
        mock_result.conversation_history = []

        with patch("scripts.auto_post.auto_state") as mock_auto_state, \
             patch("scripts.auto_post.context_feed") as mock_ctx, \
             patch("scripts.auto_post.engine") as mock_engine, \
             patch("scripts.auto_post.scheduler") as mock_sched, \
             patch("scripts.auto_post.state") as mock_state, \
             patch("scripts.auto_post.preference_engine") as mock_pref, \
             patch("scripts.auto_post._notify_telegram", new_callable=AsyncMock):
            mock_auto_state.can_post.return_value = (True, "")
            mock_auto_state.is_duplicate_caption.return_value = True
            mock_ctx.aggregate_context = AsyncMock(return_value=MagicMock(
                summary="", has_urgent=False, signals=[],
            ))
            mock_engine.run_agent = AsyncMock(return_value=mock_result)
            mock_sched.build_prompt_for_slot = AsyncMock(return_value=("prompt", []))
            mock_state.async_has_pending = AsyncMock(return_value=False)
            mock_pref.score_draft = AsyncMock(return_value=MagicMock(
                score=8.0, should_reject=False, reasoning="fine", flags=[],
            ))

            from scripts.auto_post import process_slot
            result = await process_slot(
                "test_slot", {"type": "engagement"}, {},
            )

        assert result is False
        mock_auto_state.is_duplicate_caption.assert_called_once_with("same old same old")


# ---------------------------------------------------------------------------
# Overall flow — process_slot orchestration
# ---------------------------------------------------------------------------

class TestProcessSlotFlow:
    """Verify the end-to-end flow of process_slot."""

    @pytest.mark.asyncio
    async def test_happy_path_saves_draft_and_notifies(self):
        """Full flow: agent succeeds, draft saved, notification sent."""
        mock_result = MagicMock()
        mock_result.draft = {
            "caption": "New post about FOID",
            "hashtags": ["#foid"],
            "alt_text": "logo",
            "image_prompt": "futuristic logo",
            "content_type": "engagement",
        }
        mock_result.image_url = "https://example.com/img.png"
        mock_result.image_urls = ["https://example.com/img.png"]
        mock_result.turns_used = 2
        mock_result.total_time = 3.0
        mock_result.tool_calls_made = ["generate_image"]
        mock_result.conversation_history = []

        notify_mock = AsyncMock()

        with patch("scripts.auto_post.auto_state") as mock_auto_state, \
             patch("scripts.auto_post.context_feed") as mock_ctx, \
             patch("scripts.auto_post.engine") as mock_engine, \
             patch("scripts.auto_post.scheduler") as mock_sched, \
             patch("scripts.auto_post.state") as mock_state, \
             patch("scripts.auto_post.preference_engine") as mock_pref, \
             patch("scripts.auto_post._notify_telegram", notify_mock):
            mock_auto_state.can_post.return_value = (True, "")
            mock_auto_state.is_duplicate_caption.return_value = False
            mock_ctx.aggregate_context = AsyncMock(return_value=MagicMock(
                summary="", has_urgent=False, signals=[],
            ))
            mock_engine.run_agent = AsyncMock(return_value=mock_result)
            mock_sched.build_prompt_for_slot = AsyncMock(return_value=("make a post", []))
            mock_state.async_has_pending = AsyncMock(return_value=False)
            mock_state.async_save_pending = AsyncMock()
            mock_state.save_last_generated = MagicMock()
            mock_pref.score_draft = AsyncMock(return_value=MagicMock(
                score=8.0, should_reject=False, reasoning="looks good", flags=[],
            ))

            from scripts.auto_post import process_slot
            result = await process_slot(
                "engagement_am", {"type": "engagement"}, {},
                bot=None,
            )

        assert result is True
        mock_state.async_save_pending.assert_awaited_once()
        notify_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_failure_after_retries_returns_false(self):
        """process_slot returns False when agent fails all retries."""
        with patch("scripts.auto_post.auto_state") as mock_auto_state, \
             patch("scripts.auto_post.context_feed") as mock_ctx, \
             patch("scripts.auto_post.engine") as mock_engine, \
             patch("scripts.auto_post.scheduler") as mock_sched, \
             patch("scripts.auto_post.state") as mock_state, \
             patch("scripts.auto_post._notify_telegram", new_callable=AsyncMock), \
             patch("scripts.auto_post.asyncio") as mock_asyncio, \
             patch("scripts.auto_post._MAX_RETRIES", 0):
            mock_auto_state.can_post.return_value = (True, "")
            mock_ctx.aggregate_context = AsyncMock(return_value=MagicMock(
                summary="", has_urgent=False, signals=[],
            ))
            mock_engine.run_agent = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_sched.build_prompt_for_slot = AsyncMock(return_value=("prompt", []))
            mock_state.async_has_pending = AsyncMock(return_value=False)
            mock_asyncio.sleep = AsyncMock()

            from scripts.auto_post import process_slot
            result = await process_slot(
                "test_slot", {"type": "engagement"}, {},
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_skips_when_pending_draft_exists(self):
        """process_slot returns False when there is already a pending draft."""
        with patch("scripts.auto_post.auto_state") as mock_auto_state, \
             patch("scripts.auto_post.context_feed") as mock_ctx, \
             patch("scripts.auto_post.state") as mock_state:
            mock_auto_state.can_post.return_value = (True, "")
            mock_ctx.aggregate_context = AsyncMock(return_value=MagicMock(
                summary="", has_urgent=False, signals=[],
            ))
            mock_state.async_has_pending = AsyncMock(return_value=True)

            from scripts.auto_post import process_slot
            result = await process_slot(
                "test_slot", {"type": "engagement"}, {},
            )

        assert result is False
