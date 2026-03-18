"""Tests for agent.skeleton_library -- skeleton selection, diversity scoring, and prompt formatting."""

import pytest

from agent.skeleton_library import (
    SEED_SKELETONS,
    Skeleton,
    get_all_skeletons,
    get_skeleton,
    get_skeletons_for_format,
    select_skeleton,
    format_skeleton_for_prompt,
    _compute_diversity_score,
    _map_content_type_to_format,
)


# ---------------------------------------------------------------------------
# Skeleton data model
# ---------------------------------------------------------------------------

class TestSkeletonDataModel:
    def test_seed_library_not_empty(self):
        assert len(SEED_SKELETONS) >= 15

    def test_all_skeletons_have_required_fields(self):
        for s in SEED_SKELETONS:
            assert s.id, f"Skeleton missing id"
            assert s.format in ("single_post", "thread", "long_form"), f"{s.id} has invalid format: {s.format}"
            assert s.hook, f"{s.id} missing hook"
            assert len(s.body) > 0, f"{s.id} has empty body"
            assert s.cta, f"{s.id} missing cta"
            assert s.tone, f"{s.id} missing tone"

    def test_unique_ids(self):
        ids = [s.id for s in SEED_SKELETONS]
        assert len(ids) == len(set(ids)), f"Duplicate skeleton IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_to_dict_roundtrip(self):
        original = SEED_SKELETONS[0]
        restored = Skeleton.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.format == original.format
        assert restored.hook == original.hook
        assert restored.body == original.body
        assert restored.cta == original.cta
        assert restored.tone == original.tone

    def test_from_dict_defaults(self):
        s = Skeleton.from_dict({})
        assert s.id == "unknown"
        assert s.format == "single_post"
        assert s.hook == "cold_open"


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

class TestLookup:
    def test_get_all_skeletons(self):
        all_s = get_all_skeletons()
        assert len(all_s) == len(SEED_SKELETONS)

    def test_get_skeleton_by_id(self):
        s = get_skeleton("quiet_value")
        assert s is not None
        assert s.id == "quiet_value"

    def test_get_skeleton_missing(self):
        assert get_skeleton("nonexistent_skeleton") is None

    def test_get_skeletons_for_single_post(self):
        singles = get_skeletons_for_format("single_post")
        assert len(singles) >= 5
        assert all(s.format == "single_post" for s in singles)

    def test_get_skeletons_for_thread(self):
        threads = get_skeletons_for_format("thread")
        assert len(threads) >= 3
        assert all(s.format == "thread" for s in threads)

    def test_get_skeletons_for_long_form(self):
        long_forms = get_skeletons_for_format("long_form")
        assert len(long_forms) >= 2
        assert all(s.format == "long_form" for s in long_forms)


# ---------------------------------------------------------------------------
# Content type to format mapping
# ---------------------------------------------------------------------------

class TestFormatMapping:
    def test_default_is_single_post(self):
        assert _map_content_type_to_format("announcement") == "single_post"
        assert _map_content_type_to_format("meme") == "single_post"

    def test_thread_types(self):
        assert _map_content_type_to_format("thread") == "thread"
        assert _map_content_type_to_format("educational") == "thread"

    def test_long_form_types(self):
        assert _map_content_type_to_format("report") == "long_form"


# ---------------------------------------------------------------------------
# Diversity scoring
# ---------------------------------------------------------------------------

class TestDiversityScore:
    def test_no_recent_returns_max(self):
        s = SEED_SKELETONS[0]
        score = _compute_diversity_score(s, [], 0.6)
        assert score == 1.0

    def test_same_skeleton_recently_penalized(self):
        s = get_skeleton("quiet_value")
        score = _compute_diversity_score(s, ["quiet_value"], 0.6)
        assert score < 1.0

    def test_recent_position_matters(self):
        s = get_skeleton("quiet_value")
        score_recent = _compute_diversity_score(s, ["quiet_value", "data_punch"], 0.6)
        score_older = _compute_diversity_score(s, ["data_punch", "quiet_value"], 0.6)
        # More recent usage should get a stronger penalty
        assert score_recent < score_older

    def test_zero_aggressiveness_no_penalty(self):
        s = get_skeleton("quiet_value")
        score = _compute_diversity_score(s, ["quiet_value"], 0.0)
        assert score == 1.0

    def test_max_aggressiveness_strong_penalty(self):
        s = get_skeleton("quiet_value")
        score = _compute_diversity_score(s, ["quiet_value"], 1.0)
        assert score < 0.6

    def test_different_skeleton_not_penalized_by_id(self):
        s = get_skeleton("data_punch")
        score = _compute_diversity_score(s, ["quiet_value", "bold_declaration"], 1.0)
        # Should not be penalized for skeleton ID (different IDs)
        # But might still be penalized for hook/CTA similarity
        assert score >= 0.5


# ---------------------------------------------------------------------------
# Skeleton selection
# ---------------------------------------------------------------------------

class TestSelectSkeleton:
    def test_returns_skeleton(self):
        s = select_skeleton("announcement", [], 0.6)
        assert isinstance(s, Skeleton)

    def test_respects_exclusions(self):
        # Exclude all single_post skeletons except one
        singles = get_skeletons_for_format("single_post")
        excluded = [s.id for s in singles[1:]]
        s = select_skeleton("announcement", [], 0.6, excluded=excluded)
        # Should pick the only non-excluded one or fall back
        assert s.id not in excluded or len(excluded) == len(singles)

    def test_boosts_preferred(self):
        # Run many times, preferred should appear more often
        preferred = ["quiet_value"]
        counts = {"quiet_value": 0, "other": 0}
        for _ in range(100):
            s = select_skeleton("announcement", [], 0.6, preferred=preferred)
            if s.id == "quiet_value":
                counts["quiet_value"] += 1
            else:
                counts["other"] += 1
        # Preferred should appear at least sometimes
        assert counts["quiet_value"] > 0

    def test_avoids_recent_at_high_aggressiveness(self):
        recent = ["quiet_value", "data_punch", "bold_declaration"]
        results = set()
        for _ in range(20):
            s = select_skeleton("announcement", recent, 1.0)
            results.add(s.id)
        # Should prefer skeletons NOT in the recent list
        # At least one result should be different from recent
        non_recent = results - set(recent)
        assert len(non_recent) > 0 or len(get_skeletons_for_format("single_post")) <= len(recent)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    def test_contains_skeleton_id(self):
        s = get_skeleton("quiet_value")
        result = format_skeleton_for_prompt(s)
        assert "quiet_value" in result

    def test_contains_hook(self):
        s = get_skeleton("data_punch")
        result = format_skeleton_for_prompt(s)
        assert "statistic" in result

    def test_contains_body_flow(self):
        s = get_skeleton("contrarian_thread")
        result = format_skeleton_for_prompt(s)
        assert "bold_claim" in result
        assert "reframe" in result

    def test_contains_description(self):
        s = get_skeleton("quiet_value")
        result = format_skeleton_for_prompt(s)
        assert "Guide:" in result
