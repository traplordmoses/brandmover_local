"""Tests for brand/config.json loader and override precedence."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.compositor_config import (
    BrandConfig,
    _load_config_json,
    get_config,
    invalidate_cache,
)
from agent.content_types import get_enabled_content_types, ALL_CONTENT_TYPES


class TestLoadConfigJson:
    def test_returns_none_when_missing(self, tmp_path):
        """Returns None when config.json doesn't exist."""
        result = _load_config_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_loads_valid_json(self, tmp_path):
        """Loads and returns valid config.json."""
        p = tmp_path / "config.json"
        data = {"version": "8.0", "pipeline": {"compositor_enabled": False}}
        p.write_text(json.dumps(data), encoding="utf-8")
        result = _load_config_json(p)
        assert result == data

    def test_returns_none_on_malformed_json(self, tmp_path):
        """Returns None for malformed JSON."""
        p = tmp_path / "config.json"
        p.write_text("{broken json", encoding="utf-8")
        result = _load_config_json(p)
        assert result is None

    def test_returns_none_on_non_dict(self, tmp_path):
        """Returns None if JSON is not a dict."""
        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        result = _load_config_json(p)
        assert result is None


class TestConfigJsonOverride:
    def test_compositor_disabled_via_config_json(self, tmp_path):
        """config.json pipeline.compositor_enabled overrides guidelines.md."""
        # Create guidelines with compositor enabled
        guidelines = tmp_path / "guidelines.md"
        guidelines.write_text(
            "**Brand Name:** Test\n\n"
            "## COMPOSITOR\n\n"
            "| Setting | Value |\n"
            "|---------|-------|\n"
            "| Enabled | true  |\n",
            encoding="utf-8",
        )

        # Create config.json that disables it
        config_json = tmp_path / "config.json"
        config_json.write_text(
            json.dumps({"pipeline": {"compositor_enabled": False}}),
            encoding="utf-8",
        )

        invalidate_cache()
        with patch("agent.compositor_config._CONFIG_JSON_PATH", config_json):
            cfg = get_config(guidelines)

        assert cfg.compositor_enabled is False

    def test_badge_text_override(self, tmp_path):
        """config.json can set badge_text to override guidelines.md."""
        guidelines = tmp_path / "guidelines.md"
        guidelines.write_text(
            "**Brand Name:** Test\n\n"
            "## COMPOSITOR\n\n"
            "| Setting | Value |\n"
            "|---------|-------|\n"
            "| Badge text | WEB |\n",
            encoding="utf-8",
        )

        config_json = tmp_path / "config.json"
        config_json.write_text(
            json.dumps({"pipeline": {"badge_text": None}}),
            encoding="utf-8",
        )

        invalidate_cache()
        with patch("agent.compositor_config._CONFIG_JSON_PATH", config_json):
            cfg = get_config(guidelines)

        assert cfg.badge_text is None

    def test_no_config_json_uses_guidelines(self, tmp_path):
        """When config.json is missing, guidelines.md values are used as-is."""
        guidelines = tmp_path / "guidelines.md"
        guidelines.write_text(
            "**Brand Name:** Test\n\n"
            "## COMPOSITOR\n\n"
            "| Setting | Value |\n"
            "|---------|-------|\n"
            "| Enabled | false |\n"
            "| Badge text | APP |\n"
            "| Default mode | text_only |\n",
            encoding="utf-8",
        )

        invalidate_cache()
        with patch("agent.compositor_config._CONFIG_JSON_PATH", tmp_path / "no_such.json"):
            cfg = get_config(guidelines)

        assert cfg.compositor_enabled is False
        assert cfg.badge_text == "APP"
        assert cfg.default_mode == "text_only"


class TestGetEnabledContentTypes:
    def test_returns_all_types_when_no_config_json(self, tmp_path):
        """Legacy mode returns ALL_CONTENT_TYPES."""
        with patch("agent.compositor_config._CONFIG_JSON_PATH", tmp_path / "missing.json"):
            result = get_enabled_content_types()
        assert result == ALL_CONTENT_TYPES

    def test_returns_filtered_types_from_config_json(self, tmp_path):
        """Config.json content_types_enabled filters the list."""
        config_json = tmp_path / "config.json"
        config_json.write_text(
            json.dumps({"content_types_enabled": ["announcement", "meme"]}),
            encoding="utf-8",
        )

        with patch("agent.compositor_config._CONFIG_JSON_PATH", config_json):
            result = get_enabled_content_types()

        assert result == ("announcement", "meme")

    def test_returns_all_types_on_empty_list(self, tmp_path):
        """Empty list falls back to ALL_CONTENT_TYPES."""
        config_json = tmp_path / "config.json"
        config_json.write_text(
            json.dumps({"content_types_enabled": []}),
            encoding="utf-8",
        )

        with patch("agent.compositor_config._CONFIG_JSON_PATH", config_json):
            result = get_enabled_content_types()

        assert result == ALL_CONTENT_TYPES
