"""Tests for agent.asset_audit — asset categorization and archetype detection."""

import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from agent.asset_audit import (
    AssetAuditEntry,
    AssetInventory,
    audit_single_asset,
    audit_batch,
    detect_missing,
    determine_archetype,
    save_inventory,
    load_inventory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entry(category="logo", quality=7, colors=None, style=None):
    return AssetAuditEntry(
        path=f"/tmp/{category}.png",
        category=category,
        dominant_colors=colors or [],
        style_keywords=style or [],
        description=f"A {category} asset",
        quality_score=quality,
    )


# ---------------------------------------------------------------------------
# detect_missing
# ---------------------------------------------------------------------------

class TestDetectMissing:
    def test_all_present(self):
        entries = [
            _entry("logo"), _entry("color_palette"),
            _entry("style_guide"), _entry("font_specimen"),
        ]
        assert detect_missing(entries) == []

    def test_missing_logo(self):
        entries = [_entry("color_palette"), _entry("style_guide")]
        missing = detect_missing(entries)
        assert "logo" in missing
        assert "font_specimen" in missing

    def test_empty_entries(self):
        missing = detect_missing([])
        assert len(missing) == 4


# ---------------------------------------------------------------------------
# determine_archetype
# ---------------------------------------------------------------------------

class TestDetermineArchetype:
    def test_full_brand_with_essentials(self):
        entries = [
            _entry("logo", 8), _entry("color_palette", 7),
            _entry("style_guide", 7),
        ]
        assert determine_archetype(entries) == "full_brand"

    def test_full_brand_with_many_high_quality(self):
        entries = [_entry("photography", 8) for _ in range(5)]
        assert determine_archetype(entries) == "full_brand"

    def test_has_identity_with_logo(self):
        entries = [_entry("logo", 6)]
        assert determine_archetype(entries) == "has_identity"

    def test_has_identity_with_icon_and_colors(self):
        entries = [_entry("icon", 5), _entry("color_palette", 5)]
        assert determine_archetype(entries) == "has_identity"

    def test_starting_fresh_no_entries(self):
        assert determine_archetype([]) == "starting_fresh"

    def test_starting_fresh_only_other(self):
        entries = [_entry("other", 3), _entry("photography", 4)]
        assert determine_archetype(entries) == "starting_fresh"


# ---------------------------------------------------------------------------
# audit_single_asset (mocked Claude Vision)
# ---------------------------------------------------------------------------

class TestAuditSingleAsset:
    def test_parses_claude_response(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "category": "logo",
            "dominant_colors": [{"hex": "#ff0000", "name": "Red"}],
            "style_keywords": ["minimalist", "modern"],
            "description": "A clean logo mark",
            "quality_score": 8,
        }))]

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("base64data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(return_value=mock_response)
                    mock_cls.return_value = mock_client
                    return await audit_single_asset("/tmp/logo.png")

        entry = asyncio.run(_run())
        assert entry.category == "logo"
        assert entry.quality_score == 8
        assert len(entry.dominant_colors) == 1
        assert entry.dominant_colors[0]["hex"] == "#ff0000"
        assert "minimalist" in entry.style_keywords


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_inventory(self, tmp_path):
        entries = [
            _entry("logo", 8, colors=[{"hex": "#0066ff", "name": "Blue"}], style=["bold"]),
            _entry("icon", 6),
        ]
        inventory = AssetInventory(
            entries=entries,
            consolidated_colors=[{"hex": "#0066ff", "name": "Blue"}],
            consolidated_style=["bold"],
            missing_items=["style_guide"],
            archetype="has_identity",
        )

        inv_path = tmp_path / "asset_inventory.json"
        with patch("agent.asset_audit._INVENTORY_PATH", inv_path):
            save_inventory(inventory)
            loaded = load_inventory()

        assert loaded is not None
        assert len(loaded.entries) == 2
        assert loaded.archetype == "has_identity"
        assert loaded.entries[0].category == "logo"
        assert loaded.consolidated_colors[0]["hex"] == "#0066ff"

    def test_load_missing_returns_none(self, tmp_path):
        with patch("agent.asset_audit._INVENTORY_PATH", tmp_path / "missing.json"):
            assert load_inventory() is None
