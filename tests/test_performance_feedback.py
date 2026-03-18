"""Tests for the performance feedback loop wiring.

Covers: record_post, refresh_recent_metrics, get_skeleton_performance,
get_performance_context.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# record_post stores entry correctly
# ---------------------------------------------------------------------------

def test_record_post_stores_entry():
    """record_post should append an entry and call _save_performance."""
    from agent.performance import record_post

    fake_data: list[dict] = []

    with (
        patch("agent.performance._load_performance", return_value=fake_data),
        patch("agent.performance._save_performance") as mock_save,
    ):
        record_post(tweet_id="123456", content_type="meme", caption="Hello world")

    mock_save.assert_called_once()
    saved = mock_save.call_args[0][0]
    assert len(saved) == 1
    entry = saved[0]
    assert entry["tweet_id"] == "123456"
    assert entry["content_type"] == "meme"
    assert entry["caption_preview"] == "Hello world"
    assert entry["likes"] == 0
    assert entry["last_checked"] == 0.0
    assert entry["posted_at"] > 0


def test_record_post_truncates_caption():
    """record_post should truncate caption to 100 chars."""
    from agent.performance import record_post

    long_caption = "x" * 200

    with (
        patch("agent.performance._load_performance", return_value=[]),
        patch("agent.performance._save_performance") as mock_save,
    ):
        record_post(tweet_id="999", caption=long_caption)

    saved = mock_save.call_args[0][0]
    assert len(saved[0]["caption_preview"]) == 100


def test_record_post_caps_at_200():
    """record_post should keep only the last 200 entries."""
    from agent.performance import record_post

    existing = [{"tweet_id": str(i), "posted_at": i} for i in range(200)]

    with (
        patch("agent.performance._load_performance", return_value=existing),
        patch("agent.performance._save_performance") as mock_save,
    ):
        record_post(tweet_id="new_one", content_type="test")

    saved = mock_save.call_args[0][0]
    assert len(saved) == 200
    assert saved[-1]["tweet_id"] == "new_one"


# ---------------------------------------------------------------------------
# refresh_recent_metrics skips recently checked posts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_skips_recently_checked():
    """refresh_recent_metrics should skip posts checked within refresh interval."""
    from agent.performance import refresh_recent_metrics

    now = time.time()
    data = [
        {
            "tweet_id": "fresh",
            "posted_at": now - 100,
            "last_checked": now - 1000,  # checked 16 min ago (< 6 hours)
            "likes": 5,
            "retweets": 1,
            "replies": 0,
            "impressions": 100,
            "engagement_rate": 6.0,
        },
    ]

    # last_checked is within 6 hours, so it should be skipped
    # (now - 1000) is 16 minutes ago, well within 6 hours
    with (
        patch("agent.performance._load_performance", return_value=data),
        patch("agent.performance.fetch_post_metrics", new_callable=AsyncMock) as mock_fetch,
    ):
        result = await refresh_recent_metrics(max_posts=10)

    assert result == 0
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_updates_stale_posts():
    """refresh_recent_metrics should fetch metrics for stale posts."""
    from agent.performance import refresh_recent_metrics

    now = time.time()
    data = [
        {
            "tweet_id": "stale_one",
            "posted_at": now - 86400,
            "last_checked": now - 25000,  # checked ~7 hours ago
            "likes": 0,
            "retweets": 0,
            "replies": 0,
            "impressions": 0,
            "engagement_rate": 0.0,
        },
    ]

    mock_metrics = {"likes": 10, "retweets": 2, "replies": 1, "impressions": 500}

    with (
        patch("agent.performance._load_performance", return_value=data),
        patch("agent.performance.fetch_post_metrics", new_callable=AsyncMock, return_value=mock_metrics),
        patch("agent.performance.update_metrics") as mock_update,
    ):
        result = await refresh_recent_metrics(max_posts=10)

    assert result == 1
    mock_update.assert_called_once_with("stale_one", **mock_metrics)


# ---------------------------------------------------------------------------
# get_skeleton_performance cross-references data correctly
# ---------------------------------------------------------------------------

def test_get_skeleton_performance_cross_references():
    """get_skeleton_performance should match perf entries to structure entries by timestamp."""
    from agent.skeleton_library import get_skeleton_performance
    from agent.diversity_tracker import StructureEntry

    now = time.time()

    perf_data = [
        {
            "tweet_id": "t1",
            "posted_at": now - 100,
            "engagement_rate": 5.0,
            "likes": 10,
            "retweets": 2,
        },
        {
            "tweet_id": "t2",
            "posted_at": now - 200,
            "engagement_rate": 8.0,
            "likes": 20,
            "retweets": 5,
        },
        {
            "tweet_id": "t3",
            "posted_at": now - 300,
            "engagement_rate": 3.0,
            "likes": 5,
            "retweets": 1,
        },
    ]

    structures = [
        StructureEntry(
            skeleton_id="quiet_value",
            hook_type="cold_open",
            body_structure=["single_insight"],
            cta_type="none",
            tone="understated",
            content_type="meme",
            timestamp=now - 100,  # matches t1
        ),
        StructureEntry(
            skeleton_id="data_punch",
            hook_type="statistic",
            body_structure=["context_for_stat"],
            cta_type="save_this",
            tone="educational",
            content_type="educational",
            timestamp=now - 200,  # matches t2
        ),
        StructureEntry(
            skeleton_id="quiet_value",
            hook_type="cold_open",
            body_structure=["single_insight"],
            cta_type="none",
            tone="understated",
            content_type="meme",
            timestamp=now - 300,  # matches t3
        ),
    ]

    with (
        patch("agent.performance._load_performance", return_value=perf_data),
        patch("agent.diversity_tracker.get_recent_structures", return_value=structures),
    ):
        result = get_skeleton_performance(last_n=50)

    # quiet_value: avg of 5.0 and 3.0 = 4.0
    assert "quiet_value" in result
    assert abs(result["quiet_value"] - 4.0) < 0.01

    # data_punch: 8.0
    assert "data_punch" in result
    assert abs(result["data_punch"] - 8.0) < 0.01


def test_get_skeleton_performance_empty_data():
    """get_skeleton_performance should return empty dict when no data."""
    from agent.skeleton_library import get_skeleton_performance

    with (
        patch("agent.performance._load_performance", return_value=[]),
        patch("agent.diversity_tracker.get_recent_structures", return_value=[]),
    ):
        result = get_skeleton_performance()

    assert result == {}


# ---------------------------------------------------------------------------
# get_performance_context returns formatted string
# ---------------------------------------------------------------------------

def test_get_performance_context_with_data():
    """get_performance_context should return a formatted summary string."""
    from agent.performance import get_performance_context

    fake_summary = {
        "total_posts": 10,
        "recent_count": 10,
        "avg_likes": 15.2,
        "avg_retweets": 3.5,
        "avg_engagement_rate": 4.8,
        "by_content_type": {
            "meme": {"count": 5, "total_likes": 100, "total_retweets": 20, "avg_likes": 20.0, "avg_retweets": 4.0},
            "educational": {"count": 5, "total_likes": 50, "total_retweets": 15, "avg_likes": 10.0, "avg_retweets": 3.0},
        },
    }

    with patch("agent.performance.get_performance_summary", return_value=fake_summary):
        result = get_performance_context()

    assert "15.2 likes" in result
    assert "3.5 RTs" in result
    assert "4.8% engagement" in result
    assert "meme" in result  # best performing type


def test_get_performance_context_empty():
    """get_performance_context should return empty string when no data."""
    from agent.performance import get_performance_context

    with patch("agent.performance.get_performance_summary", return_value={"total_posts": 0}):
        result = get_performance_context()

    assert result == ""
