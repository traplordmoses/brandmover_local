"""
Tests for agent/brand_alignment.py — brand alignment scoring and drift detection.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from agent.brand_alignment import (
    score_brand_alignment,
    detect_brand_drift,
    get_alignment_context,
    brand_alignment_assertion,
    _score_brand_alignment_for_assertion,
    _check_voice_match,
    _check_avoid_terms,
    _check_tone_consistency,
    _check_length_compliance,
)
from agent.compositor_config import BrandConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> BrandConfig:
    """Build a BrandConfig with test defaults."""
    defaults = {
        "brand_name": "TestBrand",
        "tagline": "Build the future",
        "voice_traits": ["bold", "technical", "concise"],
        "style_keywords": ["futuristic", "minimal"],
        "avoid_terms": ["blockchain", "web3", "synergy"],
        "brand_phrases": ["build the future", "on-chain innovation"],
        "content_themes": ["technology", "community"],
    }
    defaults.update(overrides)
    return BrandConfig(**defaults)


def _make_draft(**overrides) -> dict:
    """Build a draft dict with test defaults."""
    defaults = {
        "caption": "A bold technical step forward for on-chain innovation in the ecosystem.",
        "title": "TestBrand Update",
        "subtitle": "Build the future with us",
        "image_prompt": "futuristic minimal design",
        "content_type": "announcement",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------

class TestVoiceMatch:
    def test_good_voice_match(self):
        config = _make_config()
        text = "A bold technical approach to build the future with on-chain innovation"
        score, flags = _check_voice_match(text, config)
        assert score > 0.5
        assert "low_voice_match" not in " ".join(flags)

    def test_no_voice_match(self):
        config = _make_config()
        text = "Just a random sentence about nothing relevant"
        score, flags = _check_voice_match(text, config)
        assert score < 0.5
        assert any("low_voice_match" in f for f in flags)

    def test_empty_text(self):
        config = _make_config()
        score, flags = _check_voice_match("", config)
        assert score == 0.0
        assert "empty_text" in flags

    def test_no_config_traits(self):
        config = _make_config(voice_traits=[], brand_phrases=[])
        score, flags = _check_voice_match("anything goes", config)
        assert score == 1.0
        assert flags == []

    def test_brand_phrases_detected(self):
        config = _make_config()
        text = "We build the future every day"
        score, flags = _check_voice_match(text, config)
        assert "no_brand_phrases_used" not in flags

    def test_brand_phrases_missing(self):
        config = _make_config()
        text = "A bold technical approach to something"
        score, flags = _check_voice_match(text, config)
        assert "no_brand_phrases_used" in flags


class TestAvoidTerms:
    def test_clean_text(self):
        config = _make_config()
        text = "A great product for the modern age"
        score, flags = _check_avoid_terms(text, config)
        assert score == 1.0
        assert flags == []

    def test_single_violation(self):
        config = _make_config()
        text = "This blockchain technology is amazing"
        score, flags = _check_avoid_terms(text, config)
        assert score == pytest.approx(0.7)
        assert len(flags) == 1
        assert "blockchain" in flags[0]

    def test_multiple_violations(self):
        config = _make_config()
        text = "The blockchain web3 synergy is real"
        score, flags = _check_avoid_terms(text, config)
        assert score == pytest.approx(0.1)
        assert len(flags) == 3

    def test_case_insensitive(self):
        config = _make_config()
        text = "BLOCKCHAIN and Web3 are mentioned"
        score, flags = _check_avoid_terms(text, config)
        assert score < 1.0
        assert len(flags) == 2

    def test_no_avoid_terms_configured(self):
        config = _make_config(avoid_terms=[])
        text = "blockchain web3 synergy"
        score, flags = _check_avoid_terms(text, config)
        assert score == 1.0


class TestToneConsistency:
    def test_clean_tone(self):
        text = "A calm, measured announcement about our latest update."
        score, flags = _check_tone_consistency(text)
        assert score == 1.0
        assert flags == []

    def test_all_caps(self):
        text = "THIS IS HUGE NEWS for our community"
        score, flags = _check_tone_consistency(text)
        assert score < 1.0
        assert any("ALL CAPS" in f for f in flags)

    def test_excessive_exclamation(self):
        text = "Amazing news!! We did it!!"
        score, flags = _check_tone_consistency(text)
        assert score < 1.0
        assert any("exclamation" in f for f in flags)

    def test_hashtag_in_body(self):
        text = "Check out our latest #update for the community"
        score, flags = _check_tone_consistency(text)
        assert score < 1.0
        assert any("hashtag" in f for f in flags)

    def test_filler_phrase(self):
        text = "We're excited to announce our new product launch today."
        score, flags = _check_tone_consistency(text)
        assert score < 1.0
        assert any("excited to announce" in f for f in flags)

    def test_multiple_violations(self):
        text = "BUCKLE UP!! We're excited to announce #amazing stuff!!"
        score, flags = _check_tone_consistency(text)
        assert score == 0.0  # Many violations, bottoms out at 0
        assert len(flags) >= 3

    def test_empty_text(self):
        score, flags = _check_tone_consistency("")
        assert score == 1.0
        assert flags == []


class TestLengthCompliance:
    def test_in_range(self):
        text = "x" * 100  # 100 chars, within 50-150
        score, flags = _check_length_compliance(text)
        assert score == 1.0
        assert flags == []

    def test_at_min_boundary(self):
        text = "x" * 50
        score, flags = _check_length_compliance(text)
        assert score == 1.0

    def test_at_max_boundary(self):
        text = "x" * 150
        score, flags = _check_length_compliance(text)
        assert score == 1.0

    def test_too_short(self):
        text = "x" * 25  # 25 chars, below 50
        score, flags = _check_length_compliance(text)
        assert score == pytest.approx(0.5)
        assert any("too short" in f for f in flags)

    def test_too_long(self):
        text = "x" * 225  # 225 chars, 75 over max of 150
        score, flags = _check_length_compliance(text)
        assert score == pytest.approx(0.5)
        assert any("too long" in f for f in flags)

    def test_empty(self):
        score, flags = _check_length_compliance("")
        assert score == 0.0
        assert any("empty" in f for f in flags)

    def test_way_too_long(self):
        text = "x" * 300  # 300 chars, 150 over max
        score, flags = _check_length_compliance(text)
        assert score == 0.0


# ---------------------------------------------------------------------------
# score_brand_alignment tests
# ---------------------------------------------------------------------------

class TestScoreBrandAlignment:
    @patch("agent.brand_alignment.get_config")
    def test_clean_draft_high_score(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config
        draft = _make_draft(
            caption="A bold technical step to build the future with on-chain innovation today.",
        )
        result = score_brand_alignment(draft)

        assert "alignment_score" in result
        assert "checks" in result
        assert "drift_flags" in result
        assert result["alignment_score"] >= 60.0
        assert len(result["checks"]) == 4
        check_names = {c["name"] for c in result["checks"]}
        assert check_names == {"voice_match", "avoid_terms_clean", "tone_consistency", "length_compliance"}

    @patch("agent.brand_alignment.get_config")
    def test_avoid_terms_present_lower_score(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config
        draft = _make_draft(
            caption="The blockchain web3 synergy is driving our ecosystem forward in big ways.",
        )
        result = score_brand_alignment(draft)

        assert result["alignment_score"] < 80.0
        avoid_check = next(c for c in result["checks"] if c["name"] == "avoid_terms_clean")
        assert avoid_check["score"] < 1.0
        assert any("avoid_term" in f for f in result["drift_flags"])

    @patch("agent.brand_alignment.get_config")
    def test_all_caps_and_exclamation(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config
        draft = _make_draft(
            caption="THIS IS HUGE!! Build the future with on-chain innovation!!",
        )
        result = score_brand_alignment(draft)

        tone_check = next(c for c in result["checks"] if c["name"] == "tone_consistency")
        assert tone_check["score"] < 1.0
        assert any("tone_violation" in f for f in result["drift_flags"])

    @patch("agent.brand_alignment.get_config")
    def test_score_is_weighted_average(self, mock_get_config):
        config = _make_config(voice_traits=[], brand_phrases=[], avoid_terms=[])
        mock_get_config.return_value = config
        # With no voice/avoid checks effective, voice=1.0 and avoid=1.0
        # Clean tone, in-range length -> all checks should be 1.0
        draft = _make_draft(
            caption="A perfectly normal sentence about things happening in the world today.",
        )
        result = score_brand_alignment(draft)
        assert result["alignment_score"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# detect_brand_drift tests
# ---------------------------------------------------------------------------

class TestDetectBrandDrift:
    @patch("agent.brand_alignment.get_config")
    def test_empty_history(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config

        with patch("agent.session.load_session") as mock_session, \
             patch("agent.feedback._read_feedback") as mock_feedback:
            mock_sess = MagicMock()
            mock_sess.recent_posts = []
            mock_session.return_value = mock_sess
            mock_feedback.return_value = []

            result = detect_brand_drift(days=7)

        assert result["drift_detected"] is False
        assert result["drift_score"] == 100.0
        assert result["issues"] == []

    @patch("agent.brand_alignment.get_config")
    def test_drift_with_avoid_terms(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config

        now = time.time()
        posts = [
            {"caption": "Great blockchain synergy today", "timestamp": now - 100},
            {"caption": "More blockchain updates coming", "timestamp": now - 200},
            {"caption": "blockchain web3 synergy launch", "timestamp": now - 300},
            {"caption": "Our blockchain approach is unique", "timestamp": now - 400},
        ]

        with patch("agent.session.load_session") as mock_session, \
             patch("agent.feedback._read_feedback") as mock_feedback:
            mock_sess = MagicMock()
            mock_sess.recent_posts = posts
            mock_session.return_value = mock_sess
            mock_feedback.return_value = []

            result = detect_brand_drift(days=7)

        assert result["drift_score"] < 100.0
        assert any("avoid term" in issue.lower() for issue in result["issues"])

    @patch("agent.brand_alignment.get_config")
    def test_drift_with_long_captions(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config

        now = time.time()
        long_caption = "x" * 200  # Exceeds 150 max
        posts = [
            {"caption": long_caption, "timestamp": now - 100},
            {"caption": long_caption, "timestamp": now - 200},
            {"caption": long_caption, "timestamp": now - 300},
        ]

        with patch("agent.session.load_session") as mock_session, \
             patch("agent.feedback._read_feedback") as mock_feedback:
            mock_sess = MagicMock()
            mock_sess.recent_posts = posts
            mock_session.return_value = mock_sess
            mock_feedback.return_value = []

            result = detect_brand_drift(days=7)

        assert any("length" in issue.lower() or "long" in issue.lower() for issue in result["issues"])

    @patch("agent.brand_alignment.get_config")
    def test_drift_with_missing_brand_phrases(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config

        now = time.time()
        posts = [
            {"caption": "Just a normal update without phrases", "timestamp": now - 100},
            {"caption": "Another update that says nothing branded", "timestamp": now - 200},
            {"caption": "Yet another generic post about things", "timestamp": now - 300},
        ]

        with patch("agent.session.load_session") as mock_session, \
             patch("agent.feedback._read_feedback") as mock_feedback:
            mock_sess = MagicMock()
            mock_sess.recent_posts = posts
            mock_session.return_value = mock_sess
            mock_feedback.return_value = []

            result = detect_brand_drift(days=7)

        assert any("brand phrase" in issue.lower() for issue in result["issues"])

    @patch("agent.brand_alignment.get_config")
    def test_old_posts_excluded(self, mock_get_config):
        """Posts older than the specified days window should be excluded."""
        config = _make_config()
        mock_get_config.return_value = config

        now = time.time()
        old_ts = now - (30 * 86400)  # 30 days ago
        posts = [
            {"caption": "blockchain synergy web3", "timestamp": old_ts},
        ]

        with patch("agent.session.load_session") as mock_session, \
             patch("agent.feedback._read_feedback") as mock_feedback:
            mock_sess = MagicMock()
            mock_sess.recent_posts = posts
            mock_session.return_value = mock_sess
            mock_feedback.return_value = []

            result = detect_brand_drift(days=7)

        # Old post should be filtered out, so no data to analyze
        assert result["drift_detected"] is False
        assert result["drift_score"] == 100.0


# ---------------------------------------------------------------------------
# Assertion adapter tests
# ---------------------------------------------------------------------------

class TestAssertionAdapter:
    def test_assertion_metadata(self):
        assert brand_alignment_assertion.name == "brand_alignment"
        assert brand_alignment_assertion.weight == 0.20
        assert brand_alignment_assertion.description == "Brand voice and guideline compliance"

    @patch("agent.brand_alignment.get_config")
    def test_assertion_returns_float(self, mock_get_config):
        config = _make_config(voice_traits=[], brand_phrases=[], avoid_terms=[])
        mock_get_config.return_value = config
        draft = _make_draft(
            caption="A perfectly normal sentence that fits within the length range nicely.",
        )
        score = _score_brand_alignment_for_assertion(draft)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch("agent.brand_alignment.get_config")
    def test_assertion_perfect_score(self, mock_get_config):
        config = _make_config(voice_traits=[], brand_phrases=[], avoid_terms=[])
        mock_get_config.return_value = config
        draft = _make_draft(
            caption="A perfectly normal sentence about things happening in the world today.",
        )
        score = _score_brand_alignment_for_assertion(draft)
        assert score == pytest.approx(1.0)

    @patch("agent.brand_alignment.get_config")
    def test_assertion_low_score(self, mock_get_config):
        config = _make_config()
        mock_get_config.return_value = config
        draft = _make_draft(
            caption="BLOCKCHAIN WEB3 SYNERGY!! #amazing excited to announce!!",
        )
        score = _score_brand_alignment_for_assertion(draft)
        assert score < 0.5

    @patch("agent.brand_alignment.get_config")
    def test_assertion_callable_from_scoring_framework(self, mock_get_config):
        """Verify the assertion can be used with scoring.score_draft."""
        from agent.scoring import score_draft

        config = _make_config(voice_traits=[], brand_phrases=[], avoid_terms=[])
        mock_get_config.return_value = config
        draft = _make_draft(
            caption="A normal statement about technology updates for the community today.",
        )
        result = score_draft(draft, assertions=[brand_alignment_assertion])
        assert "total_score" in result
        assert "grade" in result
        assert result["total_score"] >= 0.0
        assert result["total_score"] <= 100.0


# ---------------------------------------------------------------------------
# get_alignment_context tests
# ---------------------------------------------------------------------------

class TestGetAlignmentContext:
    @patch("agent.brand_alignment.detect_brand_drift")
    @patch("agent.brand_alignment.get_config")
    def test_context_with_no_drift(self, mock_get_config, mock_drift):
        config = _make_config()
        mock_get_config.return_value = config
        mock_drift.return_value = {
            "drift_detected": False,
            "drift_score": 95.0,
            "issues": [],
            "recommendations": [],
        }

        context = get_alignment_context()

        assert "BRAND ALIGNMENT STATUS" in context
        assert "bold" in context  # voice trait
        assert "build the future" in context  # brand phrase
        assert "blockchain" in context  # avoid term
        assert "ON TRACK" in context
        assert "95" in context

    @patch("agent.brand_alignment.detect_brand_drift")
    @patch("agent.brand_alignment.get_config")
    def test_context_with_drift(self, mock_get_config, mock_drift):
        config = _make_config()
        mock_get_config.return_value = config
        mock_drift.return_value = {
            "drift_detected": True,
            "drift_score": 45.0,
            "issues": ["Avoid terms in 40% of posts"],
            "recommendations": ["Reinforce avoid-terms list"],
        }

        context = get_alignment_context()

        assert "BRAND DRIFT ALERT" in context
        assert "45" in context
        assert "Avoid terms" in context
        assert "Reinforce" in context

    @patch("agent.brand_alignment.detect_brand_drift")
    @patch("agent.brand_alignment.get_config")
    def test_context_includes_sweet_spot(self, mock_get_config, mock_drift):
        config = _make_config()
        mock_get_config.return_value = config
        mock_drift.return_value = {
            "drift_detected": False,
            "drift_score": 100.0,
            "issues": [],
            "recommendations": [],
        }

        context = get_alignment_context()
        assert "50-150" in context

    @patch("agent.brand_alignment.detect_brand_drift")
    @patch("agent.brand_alignment.get_config")
    def test_context_with_empty_config(self, mock_get_config, mock_drift):
        config = _make_config(voice_traits=[], brand_phrases=[], avoid_terms=[])
        mock_get_config.return_value = config
        mock_drift.return_value = {
            "drift_detected": False,
            "drift_score": 100.0,
            "issues": [],
            "recommendations": [],
        }

        context = get_alignment_context()
        assert "BRAND ALIGNMENT STATUS" in context
        # Should still have the sweet spot line
        assert "50-150" in context
