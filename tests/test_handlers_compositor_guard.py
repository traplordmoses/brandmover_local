"""Tests for _maybe_compose compositor guard in handlers."""

import asyncio
import io
from unittest.mock import patch, AsyncMock

import pytest

from agent.compositor_config import BrandConfig


@pytest.fixture
def cfg_enabled():
    return BrandConfig(compositor_enabled=True)


@pytest.fixture
def cfg_disabled():
    return BrandConfig(compositor_enabled=False)


def test_maybe_compose_skips_when_disabled(cfg_disabled):
    """Compositor is bypassed when compositor_enabled=False."""
    from bot.handlers import _maybe_compose

    async def _run():
        with patch("bot.handlers._cc.get_config", return_value=cfg_disabled):
            return await _maybe_compose(
                {"title": "Test", "subtitle": "Sub"}, "https://example.com/img.png", "default"
            )

    photo, composed = asyncio.run(_run())
    assert photo == "https://example.com/img.png"
    assert composed is None


def test_maybe_compose_calls_compositor_when_enabled(cfg_enabled):
    """Compositor is called when compositor_enabled=True."""
    from bot.handlers import _maybe_compose

    mock_composed = io.BytesIO(b"composed_image_data")

    async def _run():
        with patch("bot.handlers._cc.get_config", return_value=cfg_enabled):
            with patch("bot.handlers.compositor.compose_branded_image", new_callable=AsyncMock, return_value=mock_composed) as mock_compose:
                result = await _maybe_compose(
                    {"title": "Test", "subtitle": "Sub"}, "https://example.com/img.png", "default"
                )
                return result, mock_compose.call_count

    (photo, composed), call_count = asyncio.run(_run())
    assert call_count == 1
    assert photo is mock_composed
    assert composed is mock_composed


def test_maybe_compose_returns_url_on_compositor_failure(cfg_enabled):
    """Falls back to raw URL when compositor returns None."""
    from bot.handlers import _maybe_compose

    async def _run():
        with patch("bot.handlers._cc.get_config", return_value=cfg_enabled):
            with patch("bot.handlers.compositor.compose_branded_image", new_callable=AsyncMock, return_value=None):
                return await _maybe_compose(
                    {"title": "Test"}, "https://example.com/img.png", "default"
                )

    photo, composed = asyncio.run(_run())
    assert photo == "https://example.com/img.png"
    assert composed is None
