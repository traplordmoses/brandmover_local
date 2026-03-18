"""Tests for agent.refinement — artifact-scoped refinement mode."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.refinement import (
    build_refinement_prompt,
    extract_artifact_from_pending,
    extract_artifact_from_recent,
    refine_artifact,
    batch_refine,
)


# ---------------------------------------------------------------------------
# build_refinement_prompt
# ---------------------------------------------------------------------------


def test_build_refinement_prompt_includes_artifact_and_instruction():
    artifact = {
        "caption": "Hello world",
        "content_type": "announcement",
        "image_prompt": "a cool image",
        "title": "HELLO",
        "subtitle": "world update",
    }
    instruction = "make it shorter"
    prompt = build_refinement_prompt(artifact, instruction)

    assert "Hello world" in prompt
    assert "announcement" in prompt
    assert "a cool image" in prompt
    assert "HELLO" in prompt
    assert "make it shorter" in prompt
    assert "No hashtags" in prompt


def test_build_refinement_prompt_includes_voice_context():
    artifact = {"caption": "test"}
    instruction = "fix it"

    mock_config = MagicMock()
    mock_config.voice_traits = ["bold", "direct"]
    mock_config.avoid_terms = ["synergy"]
    mock_config.brand_phrases = ["build different"]

    with patch("agent.compositor_config.get_config", return_value=mock_config):
        prompt = build_refinement_prompt(artifact, instruction)

    assert "bold" in prompt
    assert "synergy" in prompt
    assert "build different" in prompt


def test_build_refinement_prompt_handles_thread_format():
    artifact = {
        "caption": "Hook post",
        "format": "thread",
        "thread_posts": [
            {"text": "First post"},
            {"text": "Second post"},
        ],
    }
    prompt = build_refinement_prompt(artifact, "make it casual")

    assert "thread" in prompt
    assert "First post" in prompt
    assert "Second post" in prompt


# ---------------------------------------------------------------------------
# extract_artifact_from_pending
# ---------------------------------------------------------------------------


def test_extract_artifact_from_pending_returns_pending():
    mock_pending = {
        "caption": "Test caption",
        "image_prompt": "test prompt",
        "original_request": "make a post",
        "conversation_history": [{"role": "user", "content": "hi"}],
    }
    with patch("agent.state.get_pending", return_value=mock_pending):
        result = extract_artifact_from_pending(user_id=123)

    assert result is not None
    assert result["caption"] == "Test caption"
    assert result["conversation_history"] == [{"role": "user", "content": "hi"}]


def test_extract_artifact_from_pending_returns_none_when_empty():
    with patch("agent.state.get_pending", return_value=None):
        result = extract_artifact_from_pending(user_id=123)

    assert result is None


# ---------------------------------------------------------------------------
# extract_artifact_from_recent
# ---------------------------------------------------------------------------


def test_extract_artifact_from_recent_returns_most_recent():
    mock_session = MagicMock()
    mock_session.recent_posts = [
        {"caption": "oldest", "slot": "morning", "timestamp": 1000},
        {"caption": "middle", "slot": "noon", "timestamp": 2000},
        {"caption": "newest", "slot": "evening", "timestamp": 3000},
    ]
    with patch("agent.session.load_session", return_value=mock_session):
        result = extract_artifact_from_recent(index=0)

    assert result is not None
    assert result["caption"] == "newest"


def test_extract_artifact_from_recent_index_1():
    mock_session = MagicMock()
    mock_session.recent_posts = [
        {"caption": "oldest", "slot": "morning", "timestamp": 1000},
        {"caption": "newest", "slot": "evening", "timestamp": 3000},
    ]
    with patch("agent.session.load_session", return_value=mock_session):
        result = extract_artifact_from_recent(index=1)

    assert result is not None
    assert result["caption"] == "oldest"


def test_extract_artifact_from_recent_out_of_range():
    mock_session = MagicMock()
    mock_session.recent_posts = [{"caption": "only one"}]
    with patch("agent.session.load_session", return_value=mock_session):
        result = extract_artifact_from_recent(index=5)

    assert result is None


def test_extract_artifact_from_recent_empty_session():
    mock_session = MagicMock()
    mock_session.recent_posts = []
    with patch("agent.session.load_session", return_value=mock_session):
        result = extract_artifact_from_recent(index=0)

    assert result is None


# ---------------------------------------------------------------------------
# refine_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refine_artifact_without_history_calls_run_agent():
    artifact = {
        "caption": "Original caption",
        "content_type": "announcement",
        "image_prompt": "test",
        "original_request": "make a post",
    }
    mock_result = MagicMock()
    mock_result.draft = {"caption": "Refined caption"}
    mock_result.turns_used = 2

    with patch("agent.refinement.run_agent", new_callable=AsyncMock, return_value=mock_result) as mock_run:
        result = await refine_artifact(artifact, "make it shorter")

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    # Should pass instruction as request
    assert call_kwargs.kwargs["request"] == "make it shorter"
    # Should pass revision_context containing the artifact
    assert "Original caption" in call_kwargs.kwargs["revision_context"]
    # Should exclude heavy tools
    excluded = call_kwargs.kwargs["excluded_tools"]
    assert "generate_image" in excluded
    assert "img2img" in excluded
    assert result.draft == {"caption": "Refined caption"}


@pytest.mark.asyncio
async def test_refine_artifact_with_history_calls_run_agent_with_history():
    artifact = {"caption": "Test"}
    history = [
        {"role": "user", "content": "make a post"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]

    mock_result = MagicMock()
    mock_result.draft = {"caption": "Updated"}
    mock_result.turns_used = 1

    with patch("agent.refinement.run_agent_with_history", new_callable=AsyncMock, return_value=mock_result) as mock_run:
        result = await refine_artifact(artifact, "fix the tone", history=history)

    mock_run.assert_called_once()
    # History should have the refinement instruction appended
    passed_history = mock_run.call_args.args[0]
    assert len(passed_history) == 3  # original 2 + new instruction
    assert "fix the tone" in passed_history[-1]["content"]
    # Should exclude heavy tools
    assert "generate_image" in mock_run.call_args.kwargs["excluded_tools"]


# ---------------------------------------------------------------------------
# batch_refine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_refine_processes_all_artifacts():
    artifacts = [
        {"caption": "Post one"},
        {"caption": "Post two"},
        {"caption": "Post three"},
    ]

    call_count = 0

    async def mock_refine(artifact, instruction, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.draft = {"caption": f"Refined {call_count}"}
        result.turns_used = 1
        return result

    with patch("agent.refinement.refine_artifact", side_effect=mock_refine):
        results = await batch_refine(artifacts, "make all casual")

    assert len(results) == 3
    assert results[0].draft["caption"] == "Refined 1"
    assert results[2].draft["caption"] == "Refined 3"
