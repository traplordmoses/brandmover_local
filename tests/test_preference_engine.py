"""Tests for agent.preference_engine -- preference clustering, draft scoring, and approval trends."""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feedback_entry(
    content_type: str = "tweet",
    accepted: bool = True,
    feedback_text: str = "",
    timestamp: float | None = None,
) -> dict:
    """Create a feedback entry matching the schema in agent/feedback.py."""
    return {
        "request": "write a post",
        "draft": {"caption": "Test caption", "content_type": content_type},
        "accepted": accepted,
        "feedback_text": feedback_text,
        "resources_used": [],
        "timestamp": timestamp or time.time(),
    }


def _mock_haiku_response(text: str) -> MagicMock:
    """Create a mock Anthropic response with the given text content."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    return mock_response


# ---------------------------------------------------------------------------
# cluster_preferences
# ---------------------------------------------------------------------------

class TestClusterPreferences:
    def test_groups_by_content_type(self, tmp_path):
        entries = [
            _make_feedback_entry(content_type="tweet", accepted=True),
            _make_feedback_entry(content_type="tweet", accepted=True),
            _make_feedback_entry(content_type="meme", accepted=False),
            _make_feedback_entry(content_type="meme", accepted=True),
            _make_feedback_entry(content_type="thread", accepted=True),
        ]
        with patch("agent.preference_engine._read_feedback", return_value=entries), \
             patch("agent.preference_engine._cluster_store") as mock_store:
            mock_store.write = MagicMock()
            from agent.preference_engine import cluster_preferences
            clusters = cluster_preferences()

        assert "tweet" in clusters
        assert "meme" in clusters
        assert "thread" in clusters
        assert clusters["tweet"].sample_size == 2
        assert clusters["meme"].sample_size == 2
        assert clusters["thread"].sample_size == 1

    def test_calculates_approval_rate(self):
        entries = (
            [_make_feedback_entry(content_type="tweet", accepted=True) for _ in range(8)]
            + [_make_feedback_entry(content_type="tweet", accepted=False) for _ in range(2)]
        )
        with patch("agent.preference_engine._read_feedback", return_value=entries), \
             patch("agent.preference_engine._cluster_store") as mock_store:
            mock_store.write = MagicMock()
            from agent.preference_engine import cluster_preferences
            clusters = cluster_preferences()

        assert clusters["tweet"].approval_rate == pytest.approx(80.0)

    def test_empty_feedback(self):
        with patch("agent.preference_engine._read_feedback", return_value=[]), \
             patch("agent.preference_engine._cluster_store"):
            from agent.preference_engine import cluster_preferences
            clusters = cluster_preferences()

        assert clusters == {}

    def test_extracts_patterns(self):
        # Need duplicate feedback text for patterns to register (count >= 2)
        entries = [
            _make_feedback_entry(content_type="tweet", accepted=True, feedback_text="love the casual tone"),
            _make_feedback_entry(content_type="tweet", accepted=True, feedback_text="great casual tone here"),
            _make_feedback_entry(content_type="tweet", accepted=False, feedback_text="too formal and stiff"),
            _make_feedback_entry(content_type="tweet", accepted=False, feedback_text="way too formal again"),
        ]
        with patch("agent.preference_engine._read_feedback", return_value=entries), \
             patch("agent.preference_engine._cluster_store") as mock_store:
            mock_store.write = MagicMock()
            from agent.preference_engine import cluster_preferences
            clusters = cluster_preferences()

        tweet = clusters["tweet"]
        # "casual" and "tone" appear in both approved entries
        assert len(tweet.approved_patterns) > 0
        assert len(tweet.rejected_patterns) > 0


# ---------------------------------------------------------------------------
# score_draft
# ---------------------------------------------------------------------------

class TestScoreDraft:
    def test_above_threshold(self):
        async def _run():
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                return_value=_mock_haiku_response(
                    json.dumps({"score": 8.0, "reasoning": "On-brand, good tone", "flags": []})
                )
            )
            with patch("agent.preference_engine.get_anthropic", return_value=mock_client), \
                 patch("agent.preference_engine._load_clusters", return_value={}):
                from agent.preference_engine import score_draft
                return await score_draft(
                    draft={"caption": "the grid remembers.", "content_type": "tweet"},
                    request="write a post about loreboard",
                )

        result = asyncio.run(_run())
        assert result.score == pytest.approx(8.0)
        assert result.should_reject is False

    def test_below_threshold(self):
        async def _run():
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                return_value=_mock_haiku_response(
                    json.dumps({"score": 4.0, "reasoning": "Off-brand, too formal", "flags": ["formal"]})
                )
            )
            with patch("agent.preference_engine.get_anthropic", return_value=mock_client), \
                 patch("agent.preference_engine._load_clusters", return_value={}):
                from agent.preference_engine import score_draft
                return await score_draft(
                    draft={"caption": "We are pleased to announce...", "content_type": "tweet"},
                    request="announce something",
                )

        result = asyncio.run(_run())
        assert result.score == pytest.approx(4.0)
        assert result.should_reject is True

    def test_disabled_returns_max_score(self):
        async def _run():
            with patch("agent.preference_engine.settings") as mock_settings:
                mock_settings.DRAFT_SCORE_ENABLED = False
                from agent.preference_engine import score_draft
                return await score_draft(
                    draft={"caption": "anything", "content_type": "tweet"},
                    request="test",
                )

        result = asyncio.run(_run())
        assert result.score == 10.0
        assert result.should_reject is False


# ---------------------------------------------------------------------------
# daily_approval_rate & approval_trend
# ---------------------------------------------------------------------------

class TestApprovalMetrics:
    def test_daily_approval_rate(self):
        target_date = datetime(2026, 3, 15, tzinfo=timezone.utc)
        entries = [
            _make_feedback_entry(accepted=True, timestamp=target_date.replace(hour=10).timestamp()),
            _make_feedback_entry(accepted=True, timestamp=target_date.replace(hour=14).timestamp()),
            _make_feedback_entry(accepted=False, timestamp=target_date.replace(hour=18).timestamp()),
            # Different day, should not count
            _make_feedback_entry(accepted=False, timestamp=(target_date - timedelta(days=1)).timestamp()),
        ]
        with patch("agent.preference_engine._read_feedback", return_value=entries):
            from agent.preference_engine import get_daily_approval_rate
            result = get_daily_approval_rate("2026-03-15")

        assert result["total"] == 3
        assert result["approved"] == 2
        assert result["rejected"] == 1
        # 2/3 * 100 = 66.7
        assert result["rate"] == pytest.approx(66.7)

    def test_approval_trend_multiple_days(self):
        base = datetime(2026, 3, 14, hour=12, tzinfo=timezone.utc)
        entries = []
        for day_offset in range(5):
            dt = base + timedelta(days=day_offset)
            for _ in range(3):
                entries.append(_make_feedback_entry(accepted=True, timestamp=dt.timestamp()))
            entries.append(_make_feedback_entry(accepted=False, timestamp=dt.timestamp()))

        with patch("agent.preference_engine._read_feedback", return_value=entries), \
             patch("agent.preference_engine.datetime") as mock_dt:
            # Fix "today" to 2026-03-18 so the 5 days (14-18) are within range
            mock_dt.now.return_value = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
            mock_dt.fromtimestamp = datetime.fromtimestamp
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            from agent.preference_engine import get_approval_trend
            trend = get_approval_trend(days=5)

        assert len(trend) == 5
