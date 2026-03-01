"""Tests for agent.asset_audit — asset categorization, deep analysis, and archetype detection."""

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
    _analyze_collection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entry(category="logo", quality=7, colors=None, style=None,
           content_potential=None, brand_signals=None, recommended_formats=None,
           first_impression="", creative_dna=None, never_do=None,
           overall_energy="", what_makes_it_special="", character_system=""):
    return AssetAuditEntry(
        path=f"/tmp/{category}.png",
        category=category,
        dominant_colors=colors or [],
        style_keywords=style or [],
        description=f"A {category} asset",
        quality_score=quality,
        content_potential=content_potential or [],
        brand_signals=brand_signals or [],
        recommended_formats=recommended_formats or [],
        first_impression=first_impression,
        creative_dna=creative_dna or [],
        never_do=never_do or [],
        overall_energy=overall_energy,
        what_makes_it_special=what_makes_it_special,
        character_system=character_system,
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
# audit_single_asset — deep analysis fields (mocked Claude Vision)
# ---------------------------------------------------------------------------

class TestAuditSingleAsset:
    def _mock_response(self, data: dict):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(data))]
        return mock_response

    def test_parses_claude_response(self):
        response = self._mock_response({
            "category": "logo",
            "dominant_colors": [{"hex": "#ff0000", "name": "Red", "role": "primary"}],
            "style_keywords": ["minimalist", "modern"],
            "description": "A clean logo mark",
            "quality_score": 8,
            "content_potential": ["announcement", "community"],
            "brand_signals": ["professional", "tech-forward"],
            "recommended_formats": ["social_post", "avatar"],
        })

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("base64data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(return_value=response)
                    mock_cls.return_value = mock_client
                    return await audit_single_asset("/tmp/logo.png")

        entry = asyncio.run(_run())
        assert entry.category == "logo"
        assert entry.quality_score == 8
        assert len(entry.dominant_colors) == 1
        assert entry.dominant_colors[0]["hex"] == "#ff0000"
        assert entry.dominant_colors[0]["role"] == "primary"
        assert "minimalist" in entry.style_keywords

    def test_parses_content_potential(self):
        response = self._mock_response({
            "category": "illustration",
            "content_potential": ["meme", "engagement", "community"],
            "brand_signals": ["playful", "approachable"],
            "recommended_formats": ["social_post", "story"],
        })

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("base64data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(return_value=response)
                    mock_cls.return_value = mock_client
                    return await audit_single_asset("/tmp/illustration.png")

        entry = asyncio.run(_run())
        assert "meme" in entry.content_potential
        assert "playful" in entry.brand_signals
        assert "social_post" in entry.recommended_formats

    def test_missing_new_fields_default_empty(self):
        """Older Claude responses without new fields get empty defaults."""
        response = self._mock_response({
            "category": "logo",
            "dominant_colors": [{"hex": "#000000", "name": "Black"}],
            "style_keywords": ["bold"],
            "description": "A logo",
            "quality_score": 6,
        })

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("base64data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(return_value=response)
                    mock_cls.return_value = mock_client
                    return await audit_single_asset("/tmp/logo.png")

        entry = asyncio.run(_run())
        assert entry.content_potential == []
        assert entry.brand_signals == []
        assert entry.recommended_formats == []
        # Creative fields also default empty
        assert entry.first_impression == ""
        assert entry.what_makes_it_special == ""
        assert entry.creative_dna == []
        assert entry.content_directions == []
        assert entry.never_do == []
        assert entry.overall_energy == ""
        assert entry.character_system == ""
        assert entry.presentation_formats == []

    def test_parses_creative_fields(self):
        """New creative fields are parsed from Claude response."""
        response = self._mock_response({
            "category": "logo",
            "dominant_colors": [{"hex": "#ff0000", "name": "Red", "role": "primary"}],
            "style_keywords": ["hand-drawn"],
            "description": "A crayon-drawn logo on butcher paper",
            "quality_score": 7,
            "first_impression": "Instant warmth — feels like someone's passion project",
            "what_makes_it_special": "The hand-drawn imperfection is the brand",
            "creative_dna": ["hand-drawn warmth", "deliberate imperfection", "origin story energy"],
            "content_directions": ["behind-the-scenes process", "raw sketches"],
            "never_do": ["Don't pair with stock photography", "Never auto-smooth the edges"],
            "overall_energy": "garage startup confidence",
            "character_system": "The artisan who makes everything by hand",
            "presentation_formats": ["torn paper edge", "notebook sketch"],
        })

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("base64data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(return_value=response)
                    mock_cls.return_value = mock_client
                    return await audit_single_asset("/tmp/logo.png")

        entry = asyncio.run(_run())
        assert entry.first_impression == "Instant warmth — feels like someone's passion project"
        assert entry.what_makes_it_special == "The hand-drawn imperfection is the brand"
        assert "hand-drawn warmth" in entry.creative_dna
        assert "behind-the-scenes process" in entry.content_directions
        assert "Don't pair with stock photography" in entry.never_do
        assert entry.overall_energy == "garage startup confidence"
        assert entry.character_system == "The artisan who makes everything by hand"
        assert "torn paper edge" in entry.presentation_formats


# ---------------------------------------------------------------------------
# _analyze_collection (mocked Claude)
# ---------------------------------------------------------------------------

class TestAnalyzeCollection:
    def test_returns_collection_analysis_and_insights(self):
        entries = [
            _entry("logo", 8, colors=[{"hex": "#0066ff"}], style=["modern"],
                   brand_signals=["professional"]),
            _entry("photography", 7, colors=[{"hex": "#333333"}], style=["minimal"],
                   brand_signals=["trustworthy"]),
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "collection_analysis": {
                "visual_coherence": "high",
                "coherence_notes": "Consistent minimal style",
                "style_diversity": "low",
                "color_harmony": "harmonious",
                "strongest_asset_types": ["logo"],
                "gaps": ["No typography assets"],
            },
            "brand_insights": {
                "personality_traits": ["professional", "reliable"],
                "likely_audience": "B2B tech professionals",
                "suggested_tone": "authoritative",
                "visual_maturity": "polished",
            },
        }))]

        async def _run():
            with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await _analyze_collection(entries)

        analysis, insights = asyncio.run(_run())
        assert analysis["visual_coherence"] == "high"
        assert "logo" in analysis["strongest_asset_types"]
        assert "professional" in insights["personality_traits"]
        assert insights["suggested_tone"] == "authoritative"

    def test_handles_non_json_response(self):
        entries = [_entry("logo", 8), _entry("icon", 7)]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not JSON")]

        async def _run():
            with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await _analyze_collection(entries)

        analysis, insights = asyncio.run(_run())
        assert analysis == {}
        assert insights == {}


# ---------------------------------------------------------------------------
# audit_batch — collection analysis integration
# ---------------------------------------------------------------------------

class TestAuditBatch:
    def _mock_single_response(self, category, quality=7):
        return MagicMock(content=[MagicMock(text=json.dumps({
            "category": category,
            "dominant_colors": [{"hex": "#aabbcc", "name": "Light", "role": "primary"}],
            "style_keywords": ["clean"],
            "description": f"A {category}",
            "quality_score": quality,
            "content_potential": ["announcement"],
            "brand_signals": ["modern"],
            "recommended_formats": ["social_post"],
        }))])

    def _mock_collection_response(self):
        return MagicMock(content=[MagicMock(text=json.dumps({
            "collection_analysis": {
                "visual_coherence": "medium",
                "color_harmony": "harmonious",
                "strongest_asset_types": ["logo"],
                "gaps": [],
            },
            "brand_insights": {
                "personality_traits": ["innovative"],
                "likely_audience": "Gen Z consumers",
                "suggested_tone": "playful",
                "visual_maturity": "developing",
            },
        }))])

    def test_batch_includes_collection_analysis(self):
        """With 2+ assets, collection analysis runs."""
        single_resp_1 = self._mock_single_response("logo")
        single_resp_2 = self._mock_single_response("photography")
        collection_resp = self._mock_collection_response()

        call_count = 0

        async def _create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            # First 2 calls = individual audits, 3rd = collection
            if call_count == 1:
                return single_resp_1
            elif call_count == 2:
                return single_resp_2
            else:
                return collection_resp

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(side_effect=_create_side_effect)
                    mock_cls.return_value = mock_client
                    return await audit_batch(["/tmp/logo.png", "/tmp/photo.png"])

        inventory = asyncio.run(_run())
        assert len(inventory.entries) == 2
        assert inventory.collection_analysis.get("visual_coherence") == "medium"
        assert "innovative" in inventory.brand_insights.get("personality_traits", [])

    def test_batch_skips_collection_for_single_asset(self):
        """With only 1 asset, no collection analysis."""
        single_resp = self._mock_single_response("logo")

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(return_value=single_resp)
                    mock_cls.return_value = mock_client
                    return await audit_batch(["/tmp/logo.png"])

        inventory = asyncio.run(_run())
        assert len(inventory.entries) == 1
        assert inventory.collection_analysis == {}
        assert inventory.brand_insights == {}

    def test_batch_handles_failed_audit(self):
        """Failed individual audits still allow collection analysis on successes."""
        single_resp = self._mock_single_response("logo")
        collection_resp = self._mock_collection_response()

        call_count = 0

        async def _create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return single_resp
            elif call_count == 2:
                raise Exception("API error")
            elif call_count == 3:
                return single_resp
            else:
                return collection_resp

        async def _run():
            with patch("agent.asset_audit._encode_image", return_value=("data", "image/png")):
                with patch("agent.asset_audit.anthropic.AsyncAnthropic") as mock_cls:
                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(side_effect=_create_side_effect)
                    mock_cls.return_value = mock_client
                    return await audit_batch(["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"])

        inventory = asyncio.run(_run())
        assert len(inventory.entries) == 3
        # 2 successful entries → collection analysis runs
        assert inventory.collection_analysis.get("visual_coherence") == "medium"


# ---------------------------------------------------------------------------
# Persistence — new fields
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_inventory(self, tmp_path):
        entries = [
            _entry("logo", 8,
                   colors=[{"hex": "#0066ff", "name": "Blue", "role": "primary"}],
                   style=["bold"],
                   content_potential=["announcement"],
                   brand_signals=["professional"],
                   recommended_formats=["social_post", "avatar"],
                   first_impression="Bold and confident",
                   creative_dna=["tech-forward precision"],
                   never_do=["Never use comic sans"],
                   overall_energy="quiet confidence"),
            _entry("icon", 6),
        ]
        inventory = AssetInventory(
            entries=entries,
            consolidated_colors=[{"hex": "#0066ff", "name": "Blue"}],
            consolidated_style=["bold"],
            missing_items=["style_guide"],
            archetype="has_identity",
            collection_analysis={"visual_coherence": "high", "gaps": []},
            brand_insights={"personality_traits": ["bold"], "suggested_tone": "authoritative"},
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
        # New fields
        assert loaded.entries[0].content_potential == ["announcement"]
        assert loaded.entries[0].brand_signals == ["professional"]
        assert loaded.entries[0].recommended_formats == ["social_post", "avatar"]
        # Creative fields round-trip
        assert loaded.entries[0].first_impression == "Bold and confident"
        assert loaded.entries[0].creative_dna == ["tech-forward precision"]
        assert loaded.entries[0].never_do == ["Never use comic sans"]
        assert loaded.entries[0].overall_energy == "quiet confidence"
        # Second entry has empty creative defaults
        assert loaded.entries[1].first_impression == ""
        assert loaded.entries[1].creative_dna == []
        assert loaded.collection_analysis["visual_coherence"] == "high"
        assert "bold" in loaded.brand_insights["personality_traits"]

    def test_load_legacy_inventory_without_new_fields(self, tmp_path):
        """Old inventories without new fields load with empty defaults."""
        legacy_data = {
            "entries": [{
                "path": "/tmp/logo.png",
                "category": "logo",
                "dominant_colors": [],
                "style_keywords": ["clean"],
                "description": "A logo",
                "quality_score": 7,
            }],
            "consolidated_colors": [],
            "consolidated_style": ["clean"],
            "missing_items": ["style_guide"],
            "archetype": "has_identity",
        }

        inv_path = tmp_path / "asset_inventory.json"
        inv_path.write_text(json.dumps(legacy_data))
        with patch("agent.asset_audit._INVENTORY_PATH", inv_path):
            loaded = load_inventory()

        assert loaded is not None
        assert loaded.entries[0].content_potential == []
        assert loaded.entries[0].brand_signals == []
        assert loaded.entries[0].recommended_formats == []
        assert loaded.collection_analysis == {}
        assert loaded.brand_insights == {}

    def test_load_missing_returns_none(self, tmp_path):
        with patch("agent.asset_audit._INVENTORY_PATH", tmp_path / "missing.json"):
            assert load_inventory() is None


# ---------------------------------------------------------------------------
# Prompt string — creative fields present
# ---------------------------------------------------------------------------

class TestAuditPrompt:
    def test_prompt_contains_creative_field_names(self):
        from agent.asset_audit import _AUDIT_PROMPT
        for field_name in [
            "first_impression", "what_makes_it_special", "creative_dna",
            "content_directions", "never_do", "overall_energy",
            "character_system", "presentation_formats",
        ]:
            assert field_name in _AUDIT_PROMPT, f"Missing {field_name} in _AUDIT_PROMPT"

    def test_prompt_retains_classification_fields(self):
        from agent.asset_audit import _AUDIT_PROMPT
        for field_name in [
            "category", "dominant_colors", "style_keywords",
            "description", "quality_score", "content_potential",
            "brand_signals", "recommended_formats",
        ]:
            assert field_name in _AUDIT_PROMPT, f"Missing {field_name} in _AUDIT_PROMPT"
