"""Tests for agent.asset_library — indexed asset storage."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.asset_library import (
    LibraryEntry,
    add,
    find,
    suggest,
    mark_used,
    list_all,
    get_library_path,
    _load_index,
    _save_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def library_env(tmp_path):
    """Patch library dir and index path to a temp directory."""
    lib_dir = tmp_path / "assets" / "library"
    idx_path = tmp_path / "asset_library.json"
    with patch("agent.asset_library._LIBRARY_DIR", lib_dir), \
         patch("agent.asset_library._INDEX_PATH", idx_path):
        yield lib_dir, idx_path


@pytest.fixture
def sample_image(tmp_path):
    """Create a small dummy PNG file."""
    img = tmp_path / "test_logo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return img


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

class TestIndexIO:
    def test_load_empty(self, library_env):
        assert _load_index() == []

    def test_save_and_load_round_trip(self, library_env):
        entries = [{"id": "abc", "path": "abc.png", "source": "generated"}]
        _save_index(entries)
        loaded = _load_index()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "abc"

    def test_load_malformed_json(self, library_env):
        _, idx_path = library_env
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text("not json!", encoding="utf-8")
        assert _load_index() == []

    def test_load_flat_list_format(self, library_env):
        """Handles legacy flat-list format."""
        _, idx_path = library_env
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")
        assert _load_index() == [{"id": "x"}]


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_local_file(self, library_env, sample_image):
        lib_dir, _ = library_env
        entry = add(str(sample_image), "uploaded", "logo", prompt="brand logo", tags=["logo", "primary"])

        assert entry.id
        assert entry.source == "uploaded"
        assert entry.content_type == "logo"
        assert entry.prompt == "brand logo"
        assert "logo" in entry.tags
        assert entry.created_at > 0

        # File was copied
        dest = lib_dir / entry.path
        assert dest.exists()

        # Index was written
        entries = _load_index()
        assert len(entries) == 1
        assert entries[0]["id"] == entry.id

    def test_add_url_based(self, library_env):
        entry = add("https://example.com/img.png", "generated", "social_post", prompt="sunset")

        assert entry.path == "https://example.com/img.png"
        assert entry.source == "generated"

    def test_add_multiple(self, library_env, sample_image):
        e1 = add(str(sample_image), "uploaded", "logo")
        e2 = add(str(sample_image), "generated", "social_post")

        entries = _load_index()
        assert len(entries) == 2
        assert entries[0]["id"] != entries[1]["id"]


# ---------------------------------------------------------------------------
# find()
# ---------------------------------------------------------------------------

class TestFind:
    def _seed(self, library_env, sample_image):
        add(str(sample_image), "uploaded", "logo", prompt="minimalist brand logo", tags=["logo", "minimal"])
        add(str(sample_image), "generated", "social_post", prompt="sunset beach vibes", tags=["summer"])
        add(str(sample_image), "approved", "logo", prompt="bold logo variant", tags=["logo", "bold"])

    def test_find_by_content_type(self, library_env, sample_image):
        self._seed(library_env, sample_image)
        results = find(content_type="logo")
        assert len(results) == 2
        assert all(r.content_type == "logo" for r in results)

    def test_find_by_query_in_prompt(self, library_env, sample_image):
        self._seed(library_env, sample_image)
        results = find(query="sunset")
        assert len(results) == 1
        assert "sunset" in results[0].prompt

    def test_find_by_query_in_tags(self, library_env, sample_image):
        self._seed(library_env, sample_image)
        results = find(query="summer")
        assert len(results) == 1

    def test_find_combined_filters(self, library_env, sample_image):
        self._seed(library_env, sample_image)
        results = find(query="bold", content_type="logo")
        assert len(results) == 1
        assert results[0].prompt == "bold logo variant"

    def test_find_no_match(self, library_env, sample_image):
        self._seed(library_env, sample_image)
        results = find(query="nonexistent")
        assert len(results) == 0

    def test_find_limit(self, library_env, sample_image):
        self._seed(library_env, sample_image)
        results = find(limit=1)
        assert len(results) == 1

    def test_find_returns_newest_first(self, library_env, sample_image):
        self._seed(library_env, sample_image)
        results = find()
        # Last added should be first (newest)
        assert results[0].prompt == "bold logo variant"


# ---------------------------------------------------------------------------
# suggest()
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_suggest_with_overlap(self, library_env, sample_image):
        add(str(sample_image), "generated", "social_post",
            prompt="beautiful sunset beach summer vibes warm", tags=["summer", "beach"])

        result = suggest("sunset beach warm golden hour", "social_post")
        assert result is not None
        assert "sunset" in result.prompt

    def test_suggest_below_threshold(self, library_env, sample_image):
        add(str(sample_image), "generated", "social_post",
            prompt="corporate headshot professional", tags=["corporate"])

        # Only 1 word overlap ("professional") — below 3-word minimum
        result = suggest("professional photo", "social_post")
        assert result is None

    def test_suggest_wrong_content_type(self, library_env, sample_image):
        add(str(sample_image), "generated", "logo",
            prompt="beautiful sunset beach summer vibes warm", tags=["summer"])

        result = suggest("sunset beach warm golden hour", "social_post")
        assert result is None

    def test_suggest_empty_library(self, library_env):
        result = suggest("anything", "social_post")
        assert result is None

    def test_suggest_picks_best_match(self, library_env, sample_image):
        add(str(sample_image), "generated", "social_post",
            prompt="sunset warm orange sky", tags=[])
        add(str(sample_image), "generated", "social_post",
            prompt="sunset warm orange sky beach golden hour", tags=["sunset"])

        result = suggest("sunset warm orange beach golden", "social_post")
        assert result is not None
        # The second entry has more overlap
        assert "golden hour" in result.prompt


# ---------------------------------------------------------------------------
# mark_used()
# ---------------------------------------------------------------------------

class TestMarkUsed:
    def test_mark_used_increments(self, library_env, sample_image):
        entry = add(str(sample_image), "generated", "social_post")
        assert entry.used_count == 0

        mark_used(entry.id)
        entries = _load_index()
        e = [x for x in entries if x["id"] == entry.id][0]
        assert e["used_count"] == 1
        assert e["last_used"] > 0

        mark_used(entry.id)
        entries = _load_index()
        e = [x for x in entries if x["id"] == entry.id][0]
        assert e["used_count"] == 2

    def test_mark_used_nonexistent_no_error(self, library_env):
        mark_used("nonexistent_id")  # Should not raise


# ---------------------------------------------------------------------------
# list_all()
# ---------------------------------------------------------------------------

class TestListAll:
    def test_list_all(self, library_env, sample_image):
        add(str(sample_image), "uploaded", "logo")
        add(str(sample_image), "generated", "social_post")

        results = list_all()
        assert len(results) == 2

    def test_list_all_filtered(self, library_env, sample_image):
        add(str(sample_image), "uploaded", "logo")
        add(str(sample_image), "generated", "social_post")

        results = list_all(content_type="logo")
        assert len(results) == 1
        assert results[0].content_type == "logo"


# ---------------------------------------------------------------------------
# get_library_path()
# ---------------------------------------------------------------------------

class TestGetLibraryPath:
    def test_local_file_path(self, library_env, sample_image):
        entry = add(str(sample_image), "uploaded", "logo")
        path = get_library_path(entry)
        assert path is not None
        assert path.exists()

    def test_url_returns_none(self, library_env):
        entry = add("https://example.com/img.png", "generated", "social_post")
        path = get_library_path(entry)
        assert path is None

    def test_missing_file_returns_none(self, library_env):
        entry = LibraryEntry(id="x", path="nonexistent.png")
        path = get_library_path(entry)
        assert path is None
