"""Tests for agent.diversity_tracker -- structure logging, diversity checks, and summaries."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.diversity_tracker import (
    StructureEntry,
    check_structural_diversity,
    get_diversity_summary,
    get_recent_skeleton_ids,
    get_recent_structures,
    log_structure,
)


def _make_entry(
    skeleton_id: str = "quiet_value",
    hook_type: str = "cold_open",
    body_structure: list[str] | None = None,
    cta_type: str = "none",
    content_type: str = "announcement",
    timestamp: float | None = None,
) -> StructureEntry:
    return StructureEntry(
        skeleton_id=skeleton_id,
        hook_type=hook_type,
        body_structure=body_structure or ["single_insight"],
        cta_type=cta_type,
        tone="neutral",
        content_type=content_type,
        timestamp=timestamp or time.time(),
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TestStructureEntry:
    def test_to_dict_roundtrip(self):
        entry = _make_entry()
        restored = StructureEntry.from_dict(entry.to_dict())
        assert restored.skeleton_id == entry.skeleton_id
        assert restored.hook_type == entry.hook_type
        assert restored.body_structure == entry.body_structure

    def test_from_dict_defaults(self):
        entry = StructureEntry.from_dict({})
        assert entry.skeleton_id == "unknown"
        assert entry.hook_type == ""


# ---------------------------------------------------------------------------
# Logging and retrieval
# ---------------------------------------------------------------------------

class TestLogAndRetrieve:
    def test_log_and_retrieve(self):
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": []}

        with patch("agent.diversity_tracker._store", mock_store):
            entry = _make_entry(skeleton_id="data_punch")
            log_structure(entry)

        # Verify write was called with the entry
        written = mock_store.write.call_args[0][0]
        assert len(written["entries"]) == 1
        assert written["entries"][0]["skeleton_id"] == "data_punch"

    def test_log_trims_to_max(self):
        existing = [_make_entry(skeleton_id=f"s_{i}").to_dict() for i in range(25)]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": existing}

        with patch("agent.diversity_tracker._store", mock_store):
            log_structure(_make_entry(skeleton_id="new_one"))

        written = mock_store.write.call_args[0][0]
        assert len(written["entries"]) == 20  # _MAX_ENTRIES

    def test_get_recent_structures(self):
        entries = [
            _make_entry(skeleton_id="a", timestamp=100.0).to_dict(),
            _make_entry(skeleton_id="b", timestamp=200.0).to_dict(),
            _make_entry(skeleton_id="c", timestamp=300.0).to_dict(),
        ]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": entries}

        with patch("agent.diversity_tracker._store", mock_store):
            result = get_recent_structures(3)

        assert len(result) == 3
        # Should be newest first
        assert result[0].skeleton_id == "c"
        assert result[2].skeleton_id == "a"

    def test_get_recent_skeleton_ids(self):
        entries = [
            _make_entry(skeleton_id="x", timestamp=100.0).to_dict(),
            _make_entry(skeleton_id="y", timestamp=200.0).to_dict(),
        ]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": entries}

        with patch("agent.diversity_tracker._store", mock_store):
            ids = get_recent_skeleton_ids(5)

        assert ids == ["y", "x"]


# ---------------------------------------------------------------------------
# Diversity checking
# ---------------------------------------------------------------------------

class TestDiversityCheck:
    def test_no_history_max_score(self):
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": []}

        with patch("agent.diversity_tracker._store", mock_store):
            result = check_structural_diversity(
                skeleton_id="quiet_value",
                hook_type="cold_open",
                body_structure=["single_insight"],
                cta_type="none",
            )

        assert result["diversity_score"] == 10.0
        assert result["should_reject"] is False
        assert result["reasons"] == []

    def test_same_skeleton_recently_penalized(self):
        entries = [
            _make_entry(skeleton_id="quiet_value", timestamp=time.time()).to_dict(),
        ]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": entries}

        with patch("agent.diversity_tracker._store", mock_store):
            result = check_structural_diversity(
                skeleton_id="quiet_value",
                hook_type="cold_open",
                body_structure=["single_insight"],
                cta_type="none",
                variation_aggressiveness=0.6,
            )

        assert result["diversity_score"] < 10.0
        assert any("quiet_value" in r for r in result["reasons"])

    def test_same_hook_penalized(self):
        entries = [
            _make_entry(hook_type="question", skeleton_id="a", timestamp=time.time()).to_dict(),
            _make_entry(hook_type="question", skeleton_id="b", timestamp=time.time() - 1).to_dict(),
        ]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": entries}

        with patch("agent.diversity_tracker._store", mock_store):
            result = check_structural_diversity(
                skeleton_id="new_skeleton",
                hook_type="question",
                body_structure=["listicle"],
                cta_type="save_this",
                variation_aggressiveness=0.8,
            )

        assert any("hook" in r for r in result["reasons"])

    def test_zero_aggressiveness_never_rejects(self):
        entries = [
            _make_entry(skeleton_id="quiet_value", hook_type="cold_open", timestamp=time.time()).to_dict(),
        ]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": entries}

        with patch("agent.diversity_tracker._store", mock_store):
            result = check_structural_diversity(
                skeleton_id="quiet_value",
                hook_type="cold_open",
                body_structure=["single_insight"],
                cta_type="none",
                variation_aggressiveness=0.0,
            )

        # At 0.0 aggressiveness, rejection threshold is 0, so never rejects
        assert result["should_reject"] is False

    def test_high_aggressiveness_rejects_repetition(self):
        # Fill recent with same skeleton
        entries = [
            _make_entry(
                skeleton_id="quiet_value",
                hook_type="cold_open",
                body_structure=["single_insight", "brief_context"],
                cta_type="none",
                timestamp=time.time() - i,
            ).to_dict()
            for i in range(5)
        ]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": entries}

        with patch("agent.diversity_tracker._store", mock_store):
            result = check_structural_diversity(
                skeleton_id="quiet_value",
                hook_type="cold_open",
                body_structure=["single_insight", "brief_context"],
                cta_type="none",
                variation_aggressiveness=1.0,
            )

        assert result["should_reject"] is True
        assert len(result["reasons"]) > 0


# ---------------------------------------------------------------------------
# Diversity summary
# ---------------------------------------------------------------------------

class TestDiversitySummary:
    def test_empty_history(self):
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": []}

        with patch("agent.diversity_tracker._store", mock_store):
            summary = get_diversity_summary(days=7)

        assert summary["total_posts"] == 0
        assert summary["unique_skeletons"] == 0

    def test_summary_counts(self):
        now = time.time()
        entries = [
            _make_entry(skeleton_id="a", hook_type="question", timestamp=now).to_dict(),
            _make_entry(skeleton_id="b", hook_type="question", timestamp=now - 100).to_dict(),
            _make_entry(skeleton_id="a", hook_type="bold_claim", timestamp=now - 200).to_dict(),
        ]
        mock_store = MagicMock()
        mock_store.read.return_value = {"entries": entries}

        with patch("agent.diversity_tracker._store", mock_store):
            summary = get_diversity_summary(days=7)

        assert summary["total_posts"] == 3
        assert summary["unique_skeletons"] == 2
        assert summary["unique_hooks"] == 2
        assert summary["skeleton_distribution"]["a"] == 2
        assert summary["skeleton_distribution"]["b"] == 1
