"""Tests for agent.health_monitor -- health checks and alert rate limiting."""

import time
from unittest.mock import patch, AsyncMock

import pytest

from agent.state_manager import FileStore


# ---------------------------------------------------------------------------
# Health check status
# ---------------------------------------------------------------------------

class TestHealthChecks:
    """Test run_health_checks returns correct status."""

    @pytest.mark.asyncio
    async def test_healthy_when_no_issues(self, tmp_path):
        """Returns healthy status when all checks pass."""
        from agent import health_monitor

        # Create fake state dir with required files
        fake_state = tmp_path / "state"
        fake_state.mkdir()
        (fake_state / "state.json").write_text("{}")
        (fake_state / "feedback.json").write_text("[]")
        (fake_state / "generation_history.json").write_text("[]")

        now = time.time()

        with patch("agent.auto_state._read_state", return_value={
                 "posts_today": [{"timestamp": now - 3600}],
                 "last_post_at": now - 3600,
             }), \
             patch("agent.generation_history._read_history", return_value=[
                 {"timestamp": now - 100, "status": "approved"},
                 {"timestamp": now - 200, "status": "approved"},
             ]), \
             patch.object(health_monitor, "STATE_DIR", fake_state), \
             patch("agent.state.get_pending", return_value=None):

            health = await health_monitor.run_health_checks()

        assert health.status == "healthy"
        assert all(c["ok"] for c in health.checks.values())

    @pytest.mark.asyncio
    async def test_degraded_on_high_error_rate(self, tmp_path):
        """Returns degraded when error rate exceeds 30%."""
        from agent import health_monitor

        fake_state = tmp_path / "state"
        fake_state.mkdir()
        (fake_state / "state.json").write_text("{}")
        (fake_state / "feedback.json").write_text("[]")
        (fake_state / "generation_history.json").write_text("[]")

        now = time.time()
        entries = [
            {"timestamp": now - 100, "status": "failed"},
            {"timestamp": now - 200, "status": "failed"},
            {"timestamp": now - 300, "status": "approved"},
        ]

        with patch("agent.auto_state._read_state", return_value={
                 "posts_today": [{"timestamp": now - 1800}],
                 "last_post_at": now - 1800,
             }), \
             patch("agent.generation_history._read_history", return_value=entries), \
             patch.object(health_monitor, "STATE_DIR", fake_state), \
             patch("agent.state.get_pending", return_value=None):

            health = await health_monitor.run_health_checks()

        assert health.status == "degraded"
        assert health.error_rate_24h > 30

    @pytest.mark.asyncio
    async def test_degraded_on_stale_draft(self, tmp_path):
        """Returns degraded when a pending draft is older than 2h."""
        from agent import health_monitor

        fake_state = tmp_path / "state"
        fake_state.mkdir()
        (fake_state / "state.json").write_text("{}")
        (fake_state / "feedback.json").write_text("[]")
        (fake_state / "generation_history.json").write_text("[]")

        now = time.time()
        stale_pending = {"timestamp": now - 8000, "caption": "test"}

        with patch("agent.auto_state._read_state", return_value={
                 "posts_today": [{"timestamp": now - 1800}],
                 "last_post_at": now - 1800,
             }), \
             patch("agent.generation_history._read_history", return_value=[
                 {"timestamp": now - 100, "status": "approved"},
             ]), \
             patch.object(health_monitor, "STATE_DIR", fake_state), \
             patch("agent.state.get_pending", return_value=stale_pending):

            health = await health_monitor.run_health_checks()

        assert health.status == "degraded"
        assert not health.checks["stale_drafts"]["ok"]


# ---------------------------------------------------------------------------
# Alert rate limiting
# ---------------------------------------------------------------------------

class TestAlertRateLimiting:
    """Test that alerts are rate-limited per error_type."""

    @pytest.mark.asyncio
    async def test_first_alert_sends(self, tmp_path, monkeypatch):
        """First alert for an error_type is sent."""
        from config import settings
        from agent import health_monitor

        store = FileStore(tmp_path / "health_state.json", default_factory=dict)
        store.write({"last_alert_timestamps": {}, "last_check_at": 0.0})

        monkeypatch.setattr(settings, "HEALTH_ALERT_ENABLED", True)
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_USER_ID", 123)
        mock_bot = AsyncMock()

        with patch.object(health_monitor, "_store", store):
            await health_monitor.alert_on_failure("test_error", "Something broke", bot=mock_bot)

        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_repeated_alert_blocked(self, tmp_path, monkeypatch):
        """Second alert for same error_type within 1h is blocked."""
        from config import settings
        from agent import health_monitor

        store = FileStore(tmp_path / "health_state.json", default_factory=dict)
        store.write({
            "last_alert_timestamps": {"test_error": time.time() - 600},
            "last_check_at": 0.0,
        })

        monkeypatch.setattr(settings, "HEALTH_ALERT_ENABLED", True)
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_USER_ID", 123)
        mock_bot = AsyncMock()

        with patch.object(health_monitor, "_store", store):
            await health_monitor.alert_on_failure("test_error", "Something broke", bot=mock_bot)

        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_different_error_types_independent(self, tmp_path, monkeypatch):
        """Alerts for different error_types have independent cooldowns."""
        from config import settings
        from agent import health_monitor

        store = FileStore(tmp_path / "health_state.json", default_factory=dict)
        store.write({
            "last_alert_timestamps": {"error_a": time.time() - 600},
            "last_check_at": 0.0,
        })

        monkeypatch.setattr(settings, "HEALTH_ALERT_ENABLED", True)
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_USER_ID", 123)
        mock_bot = AsyncMock()

        with patch.object(health_monitor, "_store", store):
            # error_a should be blocked (recent)
            await health_monitor.alert_on_failure("error_a", "Still broken", bot=mock_bot)
            assert mock_bot.send_message.call_count == 0

            # error_b should send (never alerted)
            await health_monitor.alert_on_failure("error_b", "New issue", bot=mock_bot)
            assert mock_bot.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_disabled_setting(self, tmp_path, monkeypatch):
        """No alerts when HEALTH_ALERT_ENABLED is False."""
        from config import settings
        from agent import health_monitor

        store = FileStore(tmp_path / "health_state.json", default_factory=dict)
        store.write({"last_alert_timestamps": {}, "last_check_at": 0.0})

        monkeypatch.setattr(settings, "HEALTH_ALERT_ENABLED", False)
        mock_bot = AsyncMock()

        with patch.object(health_monitor, "_store", store):
            await health_monitor.alert_on_failure("test_error", "Something broke", bot=mock_bot)

        mock_bot.send_message.assert_not_called()
