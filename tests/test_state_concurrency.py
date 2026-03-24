"""Tests for agent/state.py under concurrent access."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_state_dir(tmp_path: Path):
    """Patch state module to use tmp_path for state files."""
    state_file = tmp_path / "state.json"
    return patch("agent.state._STATE_DIR", tmp_path), \
           patch("agent.state._STATE_FILE", state_file)


# ---------------------------------------------------------------------------
# Concurrent writes don't corrupt data
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    def test_multiple_threads_writing_same_user(self, tmp_path):
        """Multiple threads writing to the same user's state file don't corrupt JSON."""
        from agent.state import (
            _STATE_CACHE_MAX,
            _state_caches,
            _write_state,
            _read_state,
            invalidate_state_cache,
        )

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            errors = []
            num_threads = 10
            writes_per_thread = 20

            def writer(thread_id):
                try:
                    for i in range(writes_per_thread):
                        data = {
                            "pending": {
                                "caption": f"thread_{thread_id}_write_{i}",
                                "timestamp": time.time(),
                            }
                        }
                        _write_state(data, user_id=12345)
                except Exception as e:
                    errors.append(e)

            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = [pool.submit(writer, tid) for tid in range(num_threads)]
                for f in as_completed(futures):
                    f.result()  # re-raises exceptions

            assert not errors, f"Errors during concurrent writes: {errors}"

            # File should still be valid JSON
            raw = state_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            assert "pending" in data
            assert "caption" in data["pending"]

    def test_concurrent_save_and_clear(self, tmp_path):
        """Concurrent save_pending and clear_pending don't corrupt state."""
        from agent.state import (
            save_pending,
            clear_pending,
            _read_state,
            invalidate_state_cache,
        )

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            errors = []

            def saver(i):
                try:
                    save_pending(
                        caption=f"draft_{i}",
                        hashtags=["#test"],
                        image_url=None,
                        alt_text="",
                        image_prompt="",
                        original_request=f"request_{i}",
                        user_id=12345,
                    )
                except Exception as e:
                    errors.append(e)

            def clearer():
                try:
                    clear_pending(user_id=12345)
                except Exception as e:
                    errors.append(e)

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = []
                for i in range(20):
                    futures.append(pool.submit(saver, i))
                    if i % 3 == 0:
                        futures.append(pool.submit(clearer))
                for f in as_completed(futures):
                    f.result()

            assert not errors, f"Errors during concurrent save/clear: {errors}"

            # File should still be valid JSON
            raw = state_file.read_text(encoding="utf-8")
            json.loads(raw)  # should not raise


# ---------------------------------------------------------------------------
# Cache invalidation across threads
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_invalidate_cache_clears_all_entries(self, tmp_path):
        """invalidate_state_cache() clears all cached entries."""
        from agent.state import (
            _state_caches,
            _write_state,
            _read_state,
            invalidate_state_cache,
        )

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            # Populate cache entries for multiple users
            _write_state({"pending": {"caption": "a"}}, user_id=12345)
            _read_state(user_id=12345)  # populates cache

            assert len(_state_caches) > 0

            # Invalidate from a different thread
            done = threading.Event()

            def invalidator():
                invalidate_state_cache()
                done.set()

            t = threading.Thread(target=invalidator)
            t.start()
            done.wait(timeout=5)
            t.join(timeout=5)

            assert len(_state_caches) == 0

    def test_read_after_invalidation_hits_disk(self, tmp_path):
        """After invalidation, _read_state reads from disk, not stale cache."""
        from agent.state import (
            _write_state,
            _read_state,
            invalidate_state_cache,
        )

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            # Write and cache
            _write_state({"pending": {"caption": "original"}}, user_id=12345)
            data1 = _read_state(user_id=12345)
            assert data1["pending"]["caption"] == "original"

            # Modify file directly (simulate external change)
            state_file.write_text(
                json.dumps({"pending": {"caption": "modified_externally"}}),
                encoding="utf-8",
            )

            # Cache still returns old value
            data2 = _read_state(user_id=12345)
            assert data2["pending"]["caption"] == "original"  # still cached

            # After invalidation, reads from disk
            invalidate_state_cache()
            data3 = _read_state(user_id=12345)
            assert data3["pending"]["caption"] == "modified_externally"


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_eviction_when_cache_exceeds_50(self, tmp_path):
        """LRU eviction removes oldest entries when cache exceeds 50."""
        from agent.state import (
            _STATE_CACHE_MAX,
            _state_caches,
            _sync_lock,
            _evict_state_cache,
            invalidate_state_cache,
        )

        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            # Confirm the max is 50
            assert _STATE_CACHE_MAX == 50

            # Manually populate cache with 55 entries
            with _sync_lock:
                for uid in range(55):
                    # Older entries get smaller timestamps
                    _state_caches[uid] = ({"test": uid}, float(uid))

            assert len(_state_caches) == 55

            # Trigger eviction
            with _sync_lock:
                _evict_state_cache()

            assert len(_state_caches) == _STATE_CACHE_MAX

            # Oldest entries (uid 0-4) should be evicted
            for uid in range(5):
                assert uid not in _state_caches

            # Newest entries should remain
            for uid in range(5, 55):
                assert uid in _state_caches

    def test_write_state_triggers_eviction(self, tmp_path):
        """_write_state triggers eviction when cache exceeds max."""
        from agent.state import (
            _STATE_CACHE_MAX,
            _state_caches,
            _sync_lock,
            _write_state,
            invalidate_state_cache,
        )

        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            # Fill cache to the max with fake entries
            with _sync_lock:
                for uid in range(_STATE_CACHE_MAX):
                    _state_caches[uid] = ({"test": uid}, float(uid))

            assert len(_state_caches) == _STATE_CACHE_MAX

            # Write state for a new user — should trigger eviction
            new_uid = 99999
            # Create the user-specific state file path
            user_state = tmp_path / f"draft_{new_uid}.json"
            _write_state({"pending": {"caption": "new"}}, user_id=new_uid)

            # Cache should not exceed max
            assert len(_state_caches) <= _STATE_CACHE_MAX


# ---------------------------------------------------------------------------
# has_pending() with cache/file disagreement
# ---------------------------------------------------------------------------


class TestHasPendingCacheFileMismatch:
    def test_has_pending_from_cache_when_cached(self, tmp_path):
        """has_pending uses cache when available."""
        from agent.state import (
            _write_state,
            has_pending,
            invalidate_state_cache,
        )

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            _write_state({"pending": {"caption": "test"}}, user_id=12345)
            assert has_pending(user_id=12345) is True

            _write_state({}, user_id=12345)
            assert has_pending(user_id=12345) is False

    def test_has_pending_falls_back_to_file_when_not_cached(self, tmp_path):
        """When not cached, has_pending reads from file."""
        from agent.state import has_pending, invalidate_state_cache

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            # Write file directly (bypass cache)
            state_file.write_text(
                json.dumps({"pending": {"caption": "from file"}}),
                encoding="utf-8",
            )

            assert has_pending(user_id=12345) is True

    def test_has_pending_stale_cache_vs_file(self, tmp_path):
        """Cache says pending exists, but file was cleared externally."""
        from agent.state import (
            _write_state,
            _read_state,
            has_pending,
            invalidate_state_cache,
        )

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            # Write with pending and populate cache
            _write_state({"pending": {"caption": "cached"}}, user_id=12345)
            _ = _read_state(user_id=12345)  # populate cache

            # Externally clear the file (simulate another process)
            state_file.write_text(json.dumps({}), encoding="utf-8")

            # has_pending checks cache first — will see stale data
            assert has_pending(user_id=12345) is True

            # After invalidation, correctly reports no pending
            invalidate_state_cache()
            assert has_pending(user_id=12345) is False

    def test_has_pending_missing_file_no_cache(self, tmp_path):
        """has_pending returns False when file doesn't exist and nothing cached."""
        from agent.state import has_pending, invalidate_state_cache

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            assert has_pending(user_id=12345) is False

    def test_has_pending_corrupt_file_no_cache(self, tmp_path):
        """has_pending returns False when file is corrupt and nothing cached."""
        from agent.state import has_pending, invalidate_state_cache

        state_file = tmp_path / "state.json"
        p1, p2 = _patch_state_dir(tmp_path)
        with p1, p2:
            invalidate_state_cache()

            state_file.write_text("NOT VALID JSON!!!", encoding="utf-8")

            assert has_pending(user_id=12345) is False
