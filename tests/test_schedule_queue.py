"""Tests for agent.schedule_queue — user-driven scheduling."""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from agent import schedule_queue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tmp_queue(tmp_path, monkeypatch):
    """Redirect the queue file to a temp directory."""
    queue_file = tmp_path / "schedule_queue.json"
    monkeypatch.setattr(schedule_queue, "_QUEUE_FILE", queue_file)
    yield queue_file


# ---------------------------------------------------------------------------
# Time parser tests
# ---------------------------------------------------------------------------


class TestParseTime:
    """Tests for schedule_queue.parse_time()."""

    def _now(self):
        """Fixed 'now' for deterministic tests: 2026-03-02 14:00 UTC (Monday)."""
        return datetime(2026, 3, 2, 14, 0, 0, tzinfo=timezone.utc)

    def test_in_hours(self):
        ts, display = schedule_queue.parse_time("in 2 hours", now=self._now())
        assert ts is not None
        expected = self._now() + timedelta(hours=2)
        assert abs(ts - expected.timestamp()) < 1
        assert "16:00" in display

    def test_in_minutes(self):
        ts, display = schedule_queue.parse_time("in 30 min", now=self._now())
        assert ts is not None
        expected = self._now() + timedelta(minutes=30)
        assert abs(ts - expected.timestamp()) < 1

    def test_in_hours_and_minutes(self):
        ts, display = schedule_queue.parse_time("in 1 hour 30 min", now=self._now())
        assert ts is not None
        expected = self._now() + timedelta(hours=1, minutes=30)
        assert abs(ts - expected.timestamp()) < 1

    def test_time_today_future(self):
        # 3pm is after 2pm (our now), so should be today
        ts, display = schedule_queue.parse_time("3pm", now=self._now())
        assert ts is not None
        expected = self._now().replace(hour=15, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_time_today_past_rolls_to_tomorrow(self):
        # 9am already passed (we're at 2pm), so rolls to tomorrow
        ts, display = schedule_queue.parse_time("9am", now=self._now())
        assert ts is not None
        expected = (self._now() + timedelta(days=1)).replace(hour=9, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_time_with_minutes(self):
        ts, display = schedule_queue.parse_time("3:30pm", now=self._now())
        assert ts is not None
        expected = self._now().replace(hour=15, minute=30)
        assert abs(ts - expected.timestamp()) < 1

    def test_24h_format(self):
        ts, display = schedule_queue.parse_time("15:00", now=self._now())
        assert ts is not None
        expected = self._now().replace(hour=15, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_tomorrow(self):
        ts, display = schedule_queue.parse_time("tomorrow 9am", now=self._now())
        assert ts is not None
        expected = (self._now() + timedelta(days=1)).replace(hour=9, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_tomorrow_at(self):
        ts, display = schedule_queue.parse_time("tomorrow at 3pm", now=self._now())
        assert ts is not None
        expected = (self._now() + timedelta(days=1)).replace(hour=15, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_weekday_future(self):
        # now is Monday 2pm, Wednesday should be 2 days ahead
        ts, display = schedule_queue.parse_time("wednesday 9am", now=self._now())
        assert ts is not None
        expected = (self._now() + timedelta(days=2)).replace(hour=9, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_weekday_same_day_future_time(self):
        # Monday at 5pm — same weekday, time hasn't passed yet
        ts, display = schedule_queue.parse_time("monday 5pm", now=self._now())
        assert ts is not None
        expected = self._now().replace(hour=17, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_weekday_same_day_past_time_rolls_to_next_week(self):
        # Monday at 9am — same weekday, time already passed → next Monday
        ts, display = schedule_queue.parse_time("monday 9am", now=self._now())
        assert ts is not None
        expected = (self._now() + timedelta(days=7)).replace(hour=9, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_iso_datetime(self):
        ts, display = schedule_queue.parse_time("2026-03-05 14:00", now=self._now())
        assert ts is not None
        expected = datetime(2026, 3, 5, 14, 0, tzinfo=timezone.utc)
        assert abs(ts - expected.timestamp()) < 1

    def test_iso_datetime_past_returns_none(self):
        ts, display = schedule_queue.parse_time("2026-03-01 14:00", now=self._now())
        assert ts is None
        assert "past" in display.lower()

    def test_unparseable_returns_none(self):
        ts, display = schedule_queue.parse_time("whenever", now=self._now())
        assert ts is None

    def test_ambiguous_hour_assumes_pm(self):
        # "3" without am/pm and hour < 8 → assumes PM
        ts, display = schedule_queue.parse_time("3", now=self._now())
        assert ts is not None
        expected = self._now().replace(hour=15, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_today_keyword(self):
        ts, display = schedule_queue.parse_time("today 5pm", now=self._now())
        assert ts is not None
        expected = self._now().replace(hour=17, minute=0)
        assert abs(ts - expected.timestamp()) < 1

    def test_abbreviated_weekday(self):
        ts, _ = schedule_queue.parse_time("fri 3pm", now=self._now())
        assert ts is not None
        # Monday → Friday = 4 days ahead
        expected = (self._now() + timedelta(days=4)).replace(hour=15, minute=0)
        assert abs(ts - expected.timestamp()) < 1


# ---------------------------------------------------------------------------
# parse_schedule_command tests
# ---------------------------------------------------------------------------


class TestParseScheduleCommand:

    def test_basic(self):
        prompt, ts, rec, display = schedule_queue.parse_schedule_command(
            "tomorrow 9am post about stuff"
        )
        assert prompt == "post about stuff"
        assert ts is not None
        assert rec == "once"

    def test_daily_recurrence(self):
        prompt, ts, rec, display = schedule_queue.parse_schedule_command(
            "daily tomorrow 9am afternoon update"
        )
        assert rec == "daily"
        assert prompt == "afternoon update"
        assert ts is not None

    def test_weekly_recurrence(self):
        prompt, ts, rec, display = schedule_queue.parse_schedule_command(
            "weekly friday 9am week in review"
        )
        assert rec == "weekly"
        assert "week in review" in prompt

    def test_in_hours_format(self):
        prompt, ts, rec, display = schedule_queue.parse_schedule_command(
            "in 2 hours check on the loreboard"
        )
        assert prompt == "check on the loreboard"
        assert ts is not None

    def test_no_prompt_returns_error(self):
        # "3pm" parses as time but nothing left for prompt
        prompt, ts, rec, display = schedule_queue.parse_schedule_command("3pm")
        assert prompt is None
        assert "include what to post" in display.lower()

    def test_empty_returns_error(self):
        prompt, ts, rec, display = schedule_queue.parse_schedule_command("")
        assert prompt is None


# ---------------------------------------------------------------------------
# Queue CRUD tests
# ---------------------------------------------------------------------------


class TestQueueCRUD:

    def test_add_and_list(self):
        item = schedule_queue.add_scheduled("post about something", time.time() + 3600)
        assert item["id"]
        assert item["status"] == "pending"

        items = schedule_queue.list_scheduled()
        assert len(items) == 1
        assert items[0]["id"] == item["id"]

    def test_cancel(self):
        item = schedule_queue.add_scheduled("test", time.time() + 3600)
        assert schedule_queue.cancel_scheduled(item["id"]) is True

        items = schedule_queue.list_scheduled()
        assert len(items) == 0

    def test_cancel_nonexistent(self):
        assert schedule_queue.cancel_scheduled("nonexistent") is False

    def test_get_due_items(self):
        # Past item (should be due)
        schedule_queue.add_scheduled("past item", time.time() - 60)
        # Future item (should not be due)
        schedule_queue.add_scheduled("future item", time.time() + 7200)

        due = schedule_queue.get_due_items(window_seconds=300)
        assert len(due) == 1
        assert due[0]["prompt"] == "past item"

    def test_get_due_items_within_window(self):
        # Item due in 2 minutes — within default 5-minute window
        schedule_queue.add_scheduled("soon item", time.time() + 120)

        due = schedule_queue.get_due_items(window_seconds=300)
        assert len(due) == 1

    def test_mark_generating(self):
        item = schedule_queue.add_scheduled("test", time.time() + 3600)
        schedule_queue.mark_generating(item["id"])

        items = schedule_queue.list_scheduled()
        assert items[0]["status"] == "generating"

    def test_mark_done(self):
        item = schedule_queue.add_scheduled("test", time.time() + 3600)
        schedule_queue.mark_done(item["id"], tweet_url="https://x.com/test")

        # Should not appear in active list
        items = schedule_queue.list_scheduled()
        assert len(items) == 0

        # Should appear in full list
        all_items = schedule_queue.list_scheduled(include_done=True)
        assert all_items[0]["status"] == "posted"
        assert all_items[0]["tweet_url"] == "https://x.com/test"

    def test_mark_failed(self):
        item = schedule_queue.add_scheduled("test", time.time() + 3600)
        schedule_queue.mark_failed(item["id"], "content generation failed")

        items = schedule_queue.list_scheduled()
        assert len(items) == 0

        all_items = schedule_queue.list_scheduled(include_done=True)
        assert all_items[0]["status"] == "failed"
        assert all_items[0]["failure_reason"] == "content generation failed"

    def test_daily_recurrence_creates_next(self):
        now = time.time()
        item = schedule_queue.add_scheduled("daily post", now + 3600, recurrence="daily")
        schedule_queue.mark_done(item["id"])

        items = schedule_queue.list_scheduled()
        assert len(items) == 1
        assert items[0]["recurrence"] == "daily"
        # Next occurrence should be ~24h after the original
        assert items[0]["scheduled_utc"] > now + 3600

    def test_weekly_recurrence_creates_next(self):
        now = time.time()
        item = schedule_queue.add_scheduled("weekly post", now + 3600, recurrence="weekly")
        schedule_queue.mark_done(item["id"])

        items = schedule_queue.list_scheduled()
        assert len(items) == 1
        assert items[0]["recurrence"] == "weekly"
        assert items[0]["scheduled_utc"] > now + 3600

    def test_prune_old(self):
        # Add and complete an item, then backdate its posted_at
        item = schedule_queue.add_scheduled("old post", time.time() - 100000)
        schedule_queue.mark_done(item["id"])

        # Manually backdate the posted_at to make it old enough to prune
        items = schedule_queue._read_queue()
        for i in items:
            if i["id"] == item["id"]:
                i["posted_at"] = time.time() - 100000
        schedule_queue._write_queue(items)

        # Add a pending item
        schedule_queue.add_scheduled("new post", time.time() + 3600)

        pruned = schedule_queue.prune_old(max_age_hours=1)
        assert pruned == 1

        all_items = schedule_queue.list_scheduled(include_done=True)
        assert len(all_items) == 1
        assert all_items[0]["prompt"] == "new post"

    def test_multiple_items_sorted_by_time(self):
        now = time.time()
        schedule_queue.add_scheduled("third", now + 3600)
        schedule_queue.add_scheduled("first", now + 1200)
        schedule_queue.add_scheduled("second", now + 2400)

        items = schedule_queue.list_scheduled()
        assert len(items) == 3

        due = schedule_queue.get_due_items(window_seconds=999999)
        assert due[0]["prompt"] == "first"
        assert due[1]["prompt"] == "second"
        assert due[2]["prompt"] == "third"

    def test_empty_queue_file(self, _tmp_queue):
        """Handle empty or missing queue file gracefully."""
        items = schedule_queue.list_scheduled()
        assert items == []

    def test_corrupt_queue_file(self, _tmp_queue):
        """Handle corrupt JSON gracefully."""
        _tmp_queue.write_text("not valid json")
        items = schedule_queue.list_scheduled()
        assert items == []
