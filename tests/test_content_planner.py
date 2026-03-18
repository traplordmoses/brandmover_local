"""Tests for agent.content_planner — rolling content plan management."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import content_planner
from agent.content_planner import (
    ContentPlan,
    PlannedPost,
    _dict_to_plan,
    _plan_to_dict,
    get_content_type_distribution,
    get_next_planned_post,
    identify_gaps,
    insert_event_post,
    load_plan,
    mark_post_status,
    save_plan,
)


@pytest.fixture(autouse=True)
def _isolate_plan(tmp_path, monkeypatch):
    """Point content_planner at a temp directory for each test."""
    plan_file = tmp_path / "content_plan.json"
    monkeypatch.setattr(content_planner, "_PLAN_FILE", plan_file)
    from agent.state_manager import FileStore
    monkeypatch.setattr(content_planner, "_store", FileStore(plan_file, default_factory=dict))
    yield


# ---------------------------------------------------------------------------
# test_load_save_roundtrip
# ---------------------------------------------------------------------------

def test_load_save_roundtrip():
    """Plan survives a save/load cycle with all fields intact."""
    posts = [
        PlannedPost(
            date="2026-03-18",
            time_slot="morning",
            content_type="meme",
            prompt_hint="funny take on gas fees",
        ),
        PlannedPost(
            date="2026-03-18",
            time_slot="afternoon",
            content_type="educational",
            prompt_hint="explain staking",
            status="posted",
            posted_at=1234567890.0,
        ),
    ]
    plan = ContentPlan(week_start="2026-03-18", posts=posts, version=3)
    save_plan(plan)

    loaded = load_plan()
    assert loaded.week_start == "2026-03-18"
    assert loaded.version == 3
    assert len(loaded.posts) == 2
    assert loaded.posts[0].content_type == "meme"
    assert loaded.posts[0].prompt_hint == "funny take on gas fees"
    assert loaded.posts[1].status == "posted"
    assert loaded.posts[1].posted_at == 1234567890.0


def test_load_empty():
    """Loading when no file exists returns a default empty plan."""
    plan = load_plan()
    assert plan.posts == []
    assert plan.version == 1


# ---------------------------------------------------------------------------
# test_get_content_type_distribution
# ---------------------------------------------------------------------------

def test_get_content_type_distribution():
    """Distribution counts recent entries by content_type."""
    now = time.time()
    fake_history = [
        {"content_type": "meme", "status": "approved", "timestamp": now - 3600},
        {"content_type": "meme", "status": "approved", "timestamp": now - 7200},
        {"content_type": "educational", "status": "draft", "timestamp": now - 1000},
        {"content_type": "announcement", "status": "rejected", "timestamp": now - 500},
        # Old entry outside 7-day window
        {"content_type": "meme", "status": "approved", "timestamp": now - 700000},
    ]
    with patch("agent.content_planner.get_recent_generations", return_value=fake_history):
        dist = get_content_type_distribution(days=7)
    assert dist["meme"] == 2
    assert dist["educational"] == 1
    # Rejected entries are not counted
    assert "announcement" not in dist
    # Old entry outside window is not counted
    assert dist.get("meme") == 2


# ---------------------------------------------------------------------------
# test_identify_gaps
# ---------------------------------------------------------------------------

def test_identify_gaps():
    """identify_gaps returns underrepresented types sorted by deficit."""
    distribution = {"meme": 3, "announcement": 0, "community": 1}
    target = {"meme": 3, "announcement": 2, "community": 2, "educational": 1}
    gaps = identify_gaps(distribution, target_mix=target)
    # announcement deficit=2, community deficit=1, educational deficit=1
    assert gaps[0] == "announcement"
    assert "meme" not in gaps  # already at target
    assert "community" in gaps
    assert "educational" in gaps


def test_identify_gaps_none_when_all_met():
    """No gaps when all types meet or exceed their target."""
    distribution = {"meme": 5, "announcement": 3, "community": 3}
    target = {"meme": 3, "announcement": 2, "community": 2}
    gaps = identify_gaps(distribution, target_mix=target)
    assert gaps == []


# ---------------------------------------------------------------------------
# test_get_next_planned_post
# ---------------------------------------------------------------------------

def test_get_next_planned_post():
    """Returns the earliest planned post for today or earlier."""
    today = content_planner._today_iso()
    posts = [
        PlannedPost(date=today, time_slot="afternoon", content_type="meme", status="planned"),
        PlannedPost(date=today, time_slot="morning", content_type="educational", status="planned"),
        PlannedPost(date=today, time_slot="evening", content_type="community", status="posted"),
    ]
    plan = ContentPlan(week_start=today, posts=posts)
    save_plan(plan)

    nxt = get_next_planned_post()
    assert nxt is not None
    assert nxt.time_slot == "morning"
    assert nxt.content_type == "educational"


def test_get_next_planned_post_none_when_all_posted():
    """Returns None when no planned posts remain."""
    today = content_planner._today_iso()
    posts = [
        PlannedPost(date=today, time_slot="morning", content_type="meme", status="posted"),
    ]
    plan = ContentPlan(week_start=today, posts=posts)
    save_plan(plan)

    assert get_next_planned_post() is None


# ---------------------------------------------------------------------------
# test_mark_post_status
# ---------------------------------------------------------------------------

def test_mark_post_status():
    """mark_post_status updates the correct post and sets posted_at for 'posted'."""
    today = content_planner._today_iso()
    posts = [
        PlannedPost(date=today, time_slot="morning", content_type="meme", status="planned"),
        PlannedPost(date=today, time_slot="afternoon", content_type="educational", status="planned"),
    ]
    plan = ContentPlan(week_start=today, posts=posts)
    save_plan(plan)

    mark_post_status(today, "morning", "posted")

    updated = load_plan()
    morning = [p for p in updated.posts if p.time_slot == "morning"][0]
    afternoon = [p for p in updated.posts if p.time_slot == "afternoon"][0]
    assert morning.status == "posted"
    assert morning.posted_at is not None
    assert afternoon.status == "planned"


def test_mark_post_status_skipped():
    """mark_post_status can set status to 'skipped' without setting posted_at."""
    today = content_planner._today_iso()
    posts = [
        PlannedPost(date=today, time_slot="midday", content_type="community", status="planned"),
    ]
    save_plan(ContentPlan(week_start=today, posts=posts))

    mark_post_status(today, "midday", "skipped")

    updated = load_plan()
    post = updated.posts[0]
    assert post.status == "skipped"
    assert post.posted_at is None


# ---------------------------------------------------------------------------
# test_insert_event_post
# ---------------------------------------------------------------------------

def test_insert_event_post():
    """insert_event_post adds a new post with event_source set."""
    today = content_planner._today_iso()
    # Pre-populate with a morning post so the event goes to a different slot
    posts = [
        PlannedPost(date=today, time_slot="morning", content_type="meme", status="planned"),
    ]
    save_plan(ContentPlan(week_start=today, posts=posts))

    event_post = insert_event_post(
        title="Token Launch",
        content_type="announcement",
        prompt_hint="Big token launch happening now",
    )
    assert event_post.event_source == "Token Launch"
    assert event_post.content_type == "announcement"
    assert event_post.status == "planned"
    # Should pick a slot other than morning (which is taken)
    assert event_post.time_slot != "morning"

    # Verify it was persisted
    plan = load_plan()
    assert len(plan.posts) == 2
    event_posts = [p for p in plan.posts if p.event_source == "Token Launch"]
    assert len(event_posts) == 1


def test_insert_event_post_invalid_type_defaults():
    """insert_event_post falls back to 'announcement' for unknown content types."""
    today = content_planner._today_iso()
    save_plan(ContentPlan(week_start=today, posts=[]))

    event_post = insert_event_post(
        title="Mystery Event",
        content_type="totally_fake_type",
        prompt_hint="test",
    )
    assert event_post.content_type == "announcement"
