"""Tests for agent.risk_score — content safety risk scoring."""

from agent.risk_score import score_risk


class TestScoreRisk:
    def test_clean_text_is_low_risk(self):
        result = score_risk("Our community is growing stronger every day", load_avoid_terms=False)
        assert result["risk_level"] == "low"
        assert result["safe_to_post"] is True
        assert result["risk_score"] < 0.2

    def test_spam_signals_detected(self):
        result = score_risk("Buy now! Limited time offer! Act fast!", load_avoid_terms=False)
        assert result["risk_level"] in ("medium", "high")
        assert len(result["flags"]) > 0
        assert any(f["category"] == "spam_signals" for f in result["flags"])

    def test_financial_risk_detected(self):
        result = score_risk("Guaranteed returns! Get rich quick with 100x gains!", load_avoid_terms=False)
        assert result["risk_level"] in ("medium", "high")
        assert any(f["category"] == "financial_risk" for f in result["flags"])

    def test_controversial_content_detected(self):
        result = score_risk("This racist content is terrible", load_avoid_terms=False)
        assert len(result["flags"]) > 0
        assert any(f["category"] == "controversial" for f in result["flags"])

    def test_empty_text(self):
        result = score_risk("", load_avoid_terms=False)
        assert result["risk_level"] == "low"
        assert result["safe_to_post"] is True
        assert result["flags"] == []

    def test_result_structure(self):
        result = score_risk("test text", load_avoid_terms=False)
        assert "risk_level" in result
        assert "risk_score" in result
        assert "flags" in result
        assert "safe_to_post" in result

    def test_risk_score_capped_at_one(self):
        # Even with many matches, score should cap at 1.0
        result = score_risk(
            "buy now act fast limited time guaranteed returns get rich 100x pump",
            load_avoid_terms=False
        )
        assert result["risk_score"] <= 1.0

    def test_case_insensitive(self):
        result = score_risk("BUY NOW! ACT FAST!", load_avoid_terms=False)
        assert len(result["flags"]) > 0
