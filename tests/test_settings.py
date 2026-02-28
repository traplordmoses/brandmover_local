"""Tests for config.settings — startup validation."""

import os
from unittest.mock import patch

from config import settings


def test_validate_passes_with_env():
    """With test env vars set in conftest, validate should return no errors."""
    errors = settings.validate(exit_on_error=False)
    assert errors == []


def test_validate_missing_token():
    with patch.object(settings, "TELEGRAM_BOT_TOKEN", ""):
        errors = settings.validate(exit_on_error=False)
        assert any("TELEGRAM_BOT_TOKEN" in e for e in errors)


def test_validate_missing_anthropic_key():
    with patch.object(settings, "ANTHROPIC_API_KEY", ""):
        errors = settings.validate(exit_on_error=False)
        assert any("ANTHROPIC_API_KEY" in e for e in errors)


def test_validate_bad_max_turns():
    with patch.object(settings, "AGENT_MAX_TURNS", 0):
        errors = settings.validate(exit_on_error=False)
        assert any("AGENT_MAX_TURNS" in e for e in errors)
