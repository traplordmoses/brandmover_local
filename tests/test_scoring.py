"""Tests for agent.scoring — weighted assertion scoring framework."""

from agent.scoring import score_draft, format_score_report, DEFAULT_ASSERTIONS


class TestScoreDraft:
    def test_good_draft_scores_higher_than_bad(self):
        good_draft = {
            "caption": "Our community built something incredible this week. Here's the story behind the latest drop.",
            "image_prompt": "A vibrant community gathering scene with neon lighting, glass morphism UI elements floating in dark space, cyberpunk aesthetic with warm undertones",
            "content_type": "community",
        }
        bad_draft = {
            "caption": "Buy now! #crypto #moon",
            "image_prompt": "thing",
            "content_type": "unknown_type_xyz",
        }
        good_result = score_draft(good_draft)
        bad_result = score_draft(bad_draft)
        assert good_result["total_score"] > bad_result["total_score"]
        assert good_result["passed_threshold"] is True
        assert bad_result["passed_threshold"] is False

    def test_empty_draft(self):
        result = score_draft({})
        assert result["total_score"] < 60  # below threshold
        assert result["grade"] in ("D", "F")

    def test_result_structure(self):
        result = score_draft({"caption": "Test caption for scoring", "image_prompt": "A detailed scene"})
        assert "total_score" in result
        assert "grade" in result
        assert "results" in result
        assert "passed_threshold" in result
        assert isinstance(result["results"], list)

    def test_custom_threshold(self):
        draft = {"caption": "Short", "image_prompt": "x"}
        result = score_draft(draft, threshold=10.0)
        # Even a bad draft might pass a very low threshold
        assert "passed_threshold" in result

    def test_grade_boundaries(self):
        # Verify grade assignment works
        draft = {
            "caption": "A perfectly crafted caption that hits the sweet spot for length and engagement.",
            "image_prompt": "Detailed cinematic shot of a product on a dark reflective surface with volumetric lighting from the upper right corner",
            "content_type": "announcement",
        }
        result = score_draft(draft)
        assert result["grade"] in ("A", "B", "C", "D", "F")

    def test_ai_words_lower_score(self):
        clean = {
            "caption": "Check out our latest community update with fresh designs.",
            "image_prompt": "A clean modern design scene with soft lighting",
            "content_type": "community",
        }
        ai_heavy = {
            "caption": "Revolutionizing the game-changing seamless experience to unlock potential.",
            "image_prompt": "A clean modern design scene with soft lighting",
            "content_type": "community",
        }
        clean_score = score_draft(clean)["total_score"]
        ai_score = score_draft(ai_heavy)["total_score"]
        assert clean_score > ai_score


class TestFormatScoreReport:
    def test_format_contains_grade(self):
        result = score_draft({"caption": "Test caption here", "image_prompt": "A scene"})
        formatted = format_score_report(result)
        assert "Grade" in formatted

    def test_format_contains_score(self):
        result = score_draft({"caption": "Test caption here", "image_prompt": "A scene"})
        formatted = format_score_report(result)
        assert "/100" in formatted
