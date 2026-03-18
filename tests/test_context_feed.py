"""Tests for agent.context_feed — real-time context aggregator."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.context_feed import (
    ContextSignal,
    ContextSnapshot,
    aggregate_context,
    fetch_onchain_signals,
    fetch_x_mentions,
    format_context_for_prompt,
    _MAX_CONTEXT_CHARS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    source="onchain",
    signal_type="proposal",
    title="Test Signal",
    body="Test body text",
    urgency=2,
    timestamp=None,
) -> ContextSignal:
    return ContextSignal(
        source=source,
        signal_type=signal_type,
        title=title,
        body=body,
        urgency=urgency,
        timestamp=timestamp or time.time(),
    )


def _make_loreboard_event(event_type="proposal", title="Test Proposal", description="desc", event_id="evt1"):
    """Create a mock LoreboardEvent."""
    mock = MagicMock()
    mock.event_type = event_type
    mock.title = title
    mock.description = description
    mock.event_id = event_id
    mock.timestamp = time.time()
    mock.metadata = {}
    return mock


# ---------------------------------------------------------------------------
# Tests: onchain signal conversion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_onchain_signals_converts_events():
    """On-chain events should be converted to ContextSignal objects."""
    mock_state = MagicMock()
    mock_state.error = None
    mock_state.is_quiet = False
    mock_state.events = [
        _make_loreboard_event("canonization", "Lore Canonized", "A new entry", "evt1"),
        _make_loreboard_event("proposal", "New Proposal", "Submitted", "evt2"),
    ]

    with patch("agent.context_feed.onchain.fetch_board_state", new_callable=AsyncMock, return_value=mock_state), \
         patch("agent.context_feed.auto_state._read_state", return_value={"posted_event_ids": []}), \
         patch("agent.context_feed._read_feed_state", return_value={"last_onchain_fetch": 0.0}), \
         patch("agent.context_feed._write_feed_state"):

        signals = await fetch_onchain_signals()

    assert len(signals) == 2
    assert signals[0].source == "onchain"
    assert signals[0].signal_type == "canonization"
    assert signals[0].urgency == 1  # canonization is high urgency
    assert signals[1].signal_type == "proposal"
    assert signals[1].urgency == 2  # proposal is medium urgency


@pytest.mark.asyncio
async def test_fetch_onchain_signals_filters_posted():
    """Already-posted event IDs should be filtered out."""
    mock_state = MagicMock()
    mock_state.error = None
    mock_state.is_quiet = False
    mock_state.events = [
        _make_loreboard_event("proposal", "Old Proposal", "Already posted", "evt_old"),
        _make_loreboard_event("proposal", "New Proposal", "Fresh", "evt_new"),
    ]

    with patch("agent.context_feed.onchain.fetch_board_state", new_callable=AsyncMock, return_value=mock_state), \
         patch("agent.context_feed.auto_state._read_state", return_value={"posted_event_ids": ["evt_old"]}), \
         patch("agent.context_feed._read_feed_state", return_value={"last_onchain_fetch": 0.0}), \
         patch("agent.context_feed._write_feed_state"):

        signals = await fetch_onchain_signals()

    assert len(signals) == 1
    assert signals[0].metadata["event_id"] == "evt_new"


@pytest.mark.asyncio
async def test_fetch_onchain_signals_returns_empty_on_error():
    """Should return empty list when on-chain fetch fails."""
    with patch("agent.context_feed.onchain.fetch_board_state", new_callable=AsyncMock, side_effect=Exception("network")):
        signals = await fetch_onchain_signals()

    assert signals == []


@pytest.mark.asyncio
async def test_fetch_onchain_signals_returns_empty_when_quiet():
    """Should return empty list when loreboard is quiet."""
    mock_state = MagicMock()
    mock_state.error = None
    mock_state.is_quiet = True
    mock_state.events = []

    with patch("agent.context_feed.onchain.fetch_board_state", new_callable=AsyncMock, return_value=mock_state):
        signals = await fetch_onchain_signals()

    assert signals == []


# ---------------------------------------------------------------------------
# Tests: aggregate_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_context_no_signals():
    """Empty snapshot when all sources return nothing."""
    with patch("agent.context_feed.fetch_onchain_signals", new_callable=AsyncMock, return_value=[]), \
         patch("agent.context_feed.settings") as mock_settings:
        mock_settings.CONTEXT_FEED_ENABLED = True
        mock_settings.X_MENTIONS_ENABLED = False

        snapshot = await aggregate_context()

    assert snapshot.signals == []
    assert snapshot.summary == ""
    assert snapshot.has_urgent is False
    assert snapshot.fetched_at > 0


@pytest.mark.asyncio
async def test_aggregate_context_disabled():
    """Should return empty snapshot when CONTEXT_FEED_ENABLED is False."""
    with patch("agent.context_feed.settings") as mock_settings:
        mock_settings.CONTEXT_FEED_ENABLED = False

        snapshot = await aggregate_context()

    assert snapshot.signals == []
    assert snapshot.has_urgent is False


@pytest.mark.asyncio
async def test_aggregate_context_sorts_by_urgency():
    """Signals should be sorted by urgency (1 first) then timestamp."""
    from agent.context_feed import invalidate_context_cache
    invalidate_context_cache()

    low = _make_signal(urgency=3, title="Low", timestamp=time.time())
    high = _make_signal(urgency=1, title="High", timestamp=time.time() - 100)
    med = _make_signal(urgency=2, title="Med", timestamp=time.time() - 50)

    with patch("agent.context_feed.fetch_onchain_signals", new_callable=AsyncMock, return_value=[low, high, med]), \
         patch("agent.context_feed.settings") as mock_settings:
        mock_settings.CONTEXT_FEED_ENABLED = True
        mock_settings.X_MENTIONS_ENABLED = False

        snapshot = await aggregate_context()

    assert len(snapshot.signals) == 3
    assert snapshot.signals[0].urgency == 1
    assert snapshot.signals[1].urgency == 2
    assert snapshot.signals[2].urgency == 3
    assert snapshot.has_urgent is True


# ---------------------------------------------------------------------------
# Tests: format_context_for_prompt
# ---------------------------------------------------------------------------


def test_format_empty_snapshot():
    """Empty snapshot should produce empty string."""
    snapshot = ContextSnapshot()
    result = format_context_for_prompt(snapshot)
    assert result == ""


def test_format_single_signal():
    """Single signal should produce a clean one-liner."""
    snapshot = ContextSnapshot(signals=[
        _make_signal(source="onchain", title="New Proposal", body="Submitted by user"),
    ])
    result = format_context_for_prompt(snapshot)
    assert "[BEGIN EXTERNAL DATA" in result
    assert "ONCHAIN" in result
    assert "New Proposal" in result


def test_format_truncates_to_max_chars():
    """Output should not exceed _MAX_CONTEXT_CHARS."""
    signals = [
        _make_signal(title=f"Signal #{i}", body="A" * 200, urgency=(i % 3) + 1)
        for i in range(50)
    ]
    snapshot = ContextSnapshot(signals=signals)
    result = format_context_for_prompt(snapshot)
    assert len(result) <= _MAX_CONTEXT_CHARS


def test_format_shows_urgency_tags():
    """Urgency levels should map to HIGH/MED/LOW tags."""
    snapshot = ContextSnapshot(signals=[
        _make_signal(urgency=1, title="Urgent thing"),
        _make_signal(urgency=2, title="Medium thing"),
        _make_signal(urgency=3, title="Low thing"),
    ])
    result = format_context_for_prompt(snapshot)
    assert "HIGH" in result
    assert "MED" in result
    assert "LOW" in result


# ---------------------------------------------------------------------------
# Tests: X mentions disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x_mentions_disabled_no_token():
    """Should return empty when X_MENTIONS_ENABLED is False."""
    with patch("agent.context_feed.settings") as mock_settings:
        mock_settings.X_MENTIONS_ENABLED = False
        mock_settings.X_BEARER_TOKEN = ""

        signals = await fetch_x_mentions()

    assert signals == []


@pytest.mark.asyncio
async def test_x_mentions_disabled_no_bearer():
    """Should return empty when enabled but no bearer token."""
    with patch("agent.context_feed.settings") as mock_settings:
        mock_settings.X_MENTIONS_ENABLED = True
        mock_settings.X_BEARER_TOKEN = ""

        signals = await fetch_x_mentions()

    assert signals == []
