"""
DeepEval test suite for BrandMover content quality.

Deterministic metrics run without any API key.
G-Eval metrics (BrandTone, CaptionQuality) require OPENAI_API_KEY and are
skipped automatically when it's not set.

Run:
    python -m pytest eval/deepeval_tests.py -v
"""

import json
import os
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

from eval.deepeval_metrics import (
    HashtagFreeMetric,
    EmojiLimitMetric,
    CaptionLengthMetric,
    ForbiddenPhrasesMetric,
    NoExclamationMetric,
    ContentTypeValidMetric,
    make_brand_tone_metric,
    make_caption_quality_metric,
)


# ---------------------------------------------------------------------------
# Load scenarios
# ---------------------------------------------------------------------------

_SCENARIOS_PATH = Path(__file__).parent / "deepeval_scenarios.json"


def _load_scenarios() -> list[dict]:
    return json.loads(_SCENARIOS_PATH.read_text(encoding="utf-8"))


_SCENARIOS = _load_scenarios()


# Use the default forbidden phrases list (AI-sounding words) for test isolation
_FORBIDDEN = [
    "revolutionizing", "leveraging", "cutting-edge", "seamlessly",
    "dive into", "unlock the power", "game-changer",
]


# ---------------------------------------------------------------------------
# Deterministic metric tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s["id"])
def test_hashtag_free(scenario):
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
    )
    metric = HashtagFreeMetric()
    metric.measure(tc)
    if scenario["id"] == "bad_hashtags":
        assert not metric.is_successful(), f"Should fail: {metric.reason}"
    elif scenario["expect_pass"]:
        assert metric.is_successful(), f"Should pass: {metric.reason}"


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s["id"])
def test_emoji_limit(scenario):
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
    )
    metric = EmojiLimitMetric()
    metric.measure(tc)
    # All test scenarios should pass emoji limit (none have excessive emojis)
    if scenario["expect_pass"]:
        assert metric.is_successful(), f"Should pass: {metric.reason}"


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s["id"])
def test_caption_length(scenario):
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
    )
    metric = CaptionLengthMetric()
    metric.measure(tc)
    if scenario["id"] == "bad_too_short":
        assert not metric.is_successful(), f"Should fail: {metric.reason}"
    elif scenario["expect_pass"]:
        assert metric.is_successful(), f"Should pass: {metric.reason}"


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s["id"])
def test_forbidden_phrases(scenario):
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
    )
    metric = ForbiddenPhrasesMetric(phrases=_FORBIDDEN)
    metric.measure(tc)
    if scenario["id"] == "bad_ai_words":
        assert not metric.is_successful(), f"Should fail: {metric.reason}"
    elif scenario["expect_pass"]:
        assert metric.is_successful(), f"Should pass: {metric.reason}"


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s["id"])
def test_no_exclamation(scenario):
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
    )
    metric = NoExclamationMetric()
    metric.measure(tc)
    if scenario["id"] == "bad_exclamation":
        assert not metric.is_successful(), f"Should fail: {metric.reason}"
    elif scenario["expect_pass"]:
        assert metric.is_successful(), f"Should pass: {metric.reason}"


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s["id"])
def test_content_type_valid(scenario):
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
        additional_metadata={"content_type": scenario["content_type"]},
    )
    metric = ContentTypeValidMetric()
    metric.measure(tc)
    # All test scenarios use valid content types
    assert metric.is_successful(), f"Should pass: {metric.reason}"


# ---------------------------------------------------------------------------
# G-Eval tests (skipped without OPENAI_API_KEY)
# ---------------------------------------------------------------------------

_has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
_skip_geval = pytest.mark.skipif(
    not _has_openai_key,
    reason="OPENAI_API_KEY not set — skipping G-Eval metrics",
)

_PASSING_SCENARIOS = [s for s in _SCENARIOS if s["expect_pass"]]


@_skip_geval
@pytest.mark.parametrize("scenario", _PASSING_SCENARIOS, ids=lambda s: s["id"])
def test_brand_tone(scenario):
    metric = make_brand_tone_metric()
    if metric is None:
        pytest.skip("G-Eval unavailable")
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
    )
    metric.measure(tc)
    assert metric.is_successful(), f"Brand tone check failed: {metric.reason}"


@_skip_geval
@pytest.mark.parametrize("scenario", _PASSING_SCENARIOS, ids=lambda s: s["id"])
def test_caption_quality(scenario):
    metric = make_caption_quality_metric()
    if metric is None:
        pytest.skip("G-Eval unavailable")
    tc = LLMTestCase(
        input=scenario["description"],
        actual_output=scenario["caption"],
    )
    metric.measure(tc)
    assert metric.is_successful(), f"Caption quality check failed: {metric.reason}"
