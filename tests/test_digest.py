"""Tests for agent.digest -- daily and weekly digest generation."""

import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# collect_daily_metrics
# ---------------------------------------------------------------------------

class TestCollectDailyMetrics:
    """Test the collect_daily_metrics function."""

    def test_empty_history(self):
        """Returns zeroed metrics when no data exists."""
        from agent import digest, generation_history

        with patch.object(generation_history, "_read_history", return_value=[]), \
             patch("agent.preference_engine.get_daily_approval_rate", return_value={
                 "date": "2026-03-18", "total": 0, "approved": 0,
                 "rejected": 0, "rate": 0.0, "by_content_type": {},
             }):
            metrics = digest.collect_daily_metrics("2026-03-18")

        assert metrics["posts_published"] == 0
        assert metrics["total_generations"] == 0
        assert metrics["approval_rate"] == 0.0
        assert metrics["rejections"] == 0
        assert metrics["failures"] == 0
        assert metrics["content_type_breakdown"] == {}

    def test_with_entries(self):
        """Correctly aggregates entries for a specific day."""
        from agent import digest, generation_history

        today = "2026-03-18"
        ts_today = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc).timestamp()

        entries = [
            {"timestamp": ts_today, "status": "approved", "content_type": "meme"},
            {"timestamp": ts_today + 100, "status": "approved", "content_type": "educational"},
            {"timestamp": ts_today + 200, "status": "rejected", "content_type": "meme"},
            {"timestamp": ts_today + 300, "status": "failed", "content_type": "educational"},
        ]

        with patch.object(generation_history, "_read_history", return_value=entries), \
             patch("agent.preference_engine.get_daily_approval_rate", return_value={
                 "date": today, "total": 3, "approved": 2,
                 "rejected": 1, "rate": 66.7, "by_content_type": {},
             }):
            metrics = digest.collect_daily_metrics(today)

        assert metrics["posts_published"] == 2
        assert metrics["total_generations"] == 4
        assert metrics["failures"] == 1
        assert metrics["approval_rate"] == 66.7
        assert metrics["content_type_breakdown"]["meme"] == 2
        assert metrics["content_type_breakdown"]["educational"] == 2


# ---------------------------------------------------------------------------
# Daily trigger gating
# ---------------------------------------------------------------------------

class TestDailyTriggerGating:
    """Test that maybe_trigger_daily_digest only fires at the right time."""

    @pytest.mark.asyncio
    async def test_disabled_setting(self, monkeypatch):
        """Does not trigger when DAILY_DIGEST_ENABLED is False."""
        from config import settings
        from agent import digest

        monkeypatch.setattr(settings, "DAILY_DIGEST_ENABLED", False)
        result = await digest.maybe_trigger_daily_digest()
        assert result is False

    @pytest.mark.asyncio
    async def test_wrong_hour(self, monkeypatch):
        """Does not trigger when current hour != DAILY_DIGEST_HOUR."""
        from config import settings
        from agent import digest

        monkeypatch.setattr(settings, "DAILY_DIGEST_ENABLED", True)
        monkeypatch.setattr(settings, "DAILY_DIGEST_HOUR", 21)

        # Simulate it being hour 10
        fake_now = datetime(2026, 3, 18, 10, 0)
        with patch("agent.digest.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = await digest.maybe_trigger_daily_digest()

        assert result is False

    @pytest.mark.asyncio
    async def test_too_recent(self, tmp_path, monkeypatch):
        """Does not trigger if last daily was less than 20h ago."""
        from config import settings
        from agent import digest
        from agent.state_manager import FileStore

        monkeypatch.setattr(settings, "DAILY_DIGEST_ENABLED", True)
        monkeypatch.setattr(settings, "DAILY_DIGEST_HOUR", 21)

        # Write state with recent last_daily_at
        store = FileStore(tmp_path / "digest_state.json", default_factory=dict)
        store.write({"last_daily_at": time.time() - 3600, "last_weekly_at": 0.0})

        fake_now = datetime(2026, 3, 18, 21, 0)

        with patch.object(digest, "_store", store), \
             patch("agent.digest.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = await digest.maybe_trigger_daily_digest()

        assert result is False
