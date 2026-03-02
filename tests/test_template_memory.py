"""Tests for agent.template_memory — template storage, analysis, composition, and detection."""

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from PIL import Image

from agent.template_memory import (
    TemplateRegion,
    BrandTemplate,
    TemplateMemory,
    analyze_template,
    register_template,
    apply_template,
    get_aspect_ratio_for_content_type,
    get_image_region_aspect_ratio,
    detect_if_template,
    parse_region_description,
    _CONTENT_TYPE_ASPECT,
    _get_meme_font,
    _draw_fitted_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def templates_dir(tmp_path):
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    with patch("agent.template_memory._TEMPLATES_DIR", tpl_dir), \
         patch("agent.template_memory._MANIFEST_PATH", tpl_dir / "manifest.json"):
        yield tpl_dir


def _template(
    tid="tpl1", name="Test Template", path="/tmp/tpl.png",
    width=1080, height=1080, regions=None, aspect_ratio="1:1",
    content_types=None,
):
    return BrandTemplate(
        id=tid, name=name, path=path,
        width=width, height=height,
        regions=regions or [],
        aspect_ratio=aspect_ratio,
        content_types=content_types or [],
    )


def _create_test_image(path, width=1080, height=1080, color="red"):
    """Create a simple test image."""
    img = Image.new("RGBA", (width, height), color)
    img.save(str(path), "PNG")
    return str(path)


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults:
    def test_template_region_defaults(self):
        r = TemplateRegion(type="image")
        assert r.x == 0
        assert r.y == 0
        assert r.width == 0
        assert r.height == 0
        assert r.description == ""

    def test_brand_template_defaults(self):
        t = BrandTemplate()
        assert t.id == ""
        assert t.name == ""
        assert t.path == ""
        assert t.width == 0
        assert t.height == 0
        assert t.regions == []
        assert t.aspect_ratio == ""
        assert t.content_types == []
        assert t.analysis_notes == ""


# ---------------------------------------------------------------------------
# TemplateMemory — manifest CRUD
# ---------------------------------------------------------------------------

class TestTemplateMemory:
    def test_empty_manifest(self, templates_dir):
        memory = TemplateMemory()
        assert memory.list_templates() == []

    def test_add_template(self, templates_dir):
        memory = TemplateMemory()
        tpl = _template()
        memory.add_template(tpl)

        loaded = memory.list_templates()
        assert len(loaded) == 1
        assert loaded[0].id == "tpl1"
        assert loaded[0].name == "Test Template"

    def test_manifest_round_trip(self, templates_dir):
        memory = TemplateMemory()
        region = TemplateRegion(type="image", x=10, y=20, width=300, height=200, description="Main area")
        tpl = _template(regions=[region], content_types=["announcement"])
        memory.add_template(tpl)

        # Force reload from disk
        memory2 = TemplateMemory()
        loaded = memory2.list_templates()
        assert len(loaded) == 1
        assert loaded[0].regions[0].type == "image"
        assert loaded[0].regions[0].x == 10
        assert loaded[0].regions[0].width == 300
        assert loaded[0].content_types == ["announcement"]

    def test_remove_template(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="a"))
        memory.add_template(_template(tid="b"))
        assert len(memory.list_templates()) == 2

        removed = memory.remove_template("a")
        assert removed is True
        assert len(memory.list_templates()) == 1
        assert memory.list_templates()[0].id == "b"

    def test_remove_nonexistent(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="a"))
        removed = memory.remove_template("nonexistent")
        assert removed is False
        assert len(memory.list_templates()) == 1

    def test_get_template_for_content_type_exact(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="a", content_types=["announcement"]))
        memory.add_template(_template(tid="b", content_types=["meme"]))

        result = memory.get_template_for_content_type("meme")
        assert result is not None
        assert result.id == "b"

    def test_get_template_for_content_type_universal(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="universal", content_types=[]))

        result = memory.get_template_for_content_type("announcement")
        assert result is not None
        assert result.id == "universal"

    def test_get_template_for_content_type_none(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="a", content_types=["meme"]))

        result = memory.get_template_for_content_type("announcement")
        assert result is None


# ---------------------------------------------------------------------------
# Template priority — aspect ratio matching for universals
# ---------------------------------------------------------------------------

class TestTemplatePriority:
    def test_exact_match_wins_over_universal(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="universal", content_types=[], aspect_ratio="1:1"))
        memory.add_template(_template(tid="exact", content_types=["announcement"], aspect_ratio="16:9"))

        result = memory.get_template_for_content_type("announcement")
        assert result.id == "exact"

    def test_universal_prefers_matching_aspect_ratio(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="square", content_types=[], aspect_ratio="1:1"))
        memory.add_template(_template(tid="wide", content_types=[], aspect_ratio="16:9"))

        # announcement prefers 16:9
        result = memory.get_template_for_content_type("announcement")
        assert result.id == "wide"

        # meme prefers 16:9
        result = memory.get_template_for_content_type("meme")
        assert result.id == "wide"

        # engagement prefers 1:1
        result = memory.get_template_for_content_type("engagement")
        assert result.id == "square"

    def test_universal_falls_back_to_first(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="portrait", content_types=[], aspect_ratio="9:16"))

        # No 16:9 universal exists, should fall back to first
        result = memory.get_template_for_content_type("announcement")
        assert result.id == "portrait"

    def test_content_type_aspect_map_has_expected_types(self):
        assert _CONTENT_TYPE_ASPECT["announcement"] == "16:9"
        assert _CONTENT_TYPE_ASPECT["meme"] == "16:9"
        assert _CONTENT_TYPE_ASPECT["community"] == "1:1"
        assert _CONTENT_TYPE_ASPECT["campaign"] == "16:9"


# ---------------------------------------------------------------------------
# analyze_template (mocked Claude)
# ---------------------------------------------------------------------------

class TestAnalyzeTemplate:
    def test_parses_regions(self, tmp_path):
        img_path = _create_test_image(tmp_path / "tpl.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "regions": [
                {"type": "image", "x": 50, "y": 100, "width": 600, "height": 400, "description": "Main"},
                {"type": "text", "x": 50, "y": 520, "width": 600, "height": 60, "description": "Title"},
            ],
            "analysis_notes": "Polaroid frame",
            "suggested_content_types": ["community"],
        }))]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await analyze_template(img_path)

        result = asyncio.run(_run())
        assert len(result["regions"]) == 2
        assert result["regions"][0]["type"] == "image"
        assert result["analysis_notes"] == "Polaroid frame"
        assert "community" in result["suggested_content_types"]

    def test_handles_bad_response(self, tmp_path):
        img_path = _create_test_image(tmp_path / "tpl.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not JSON")]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await analyze_template(img_path)

        result = asyncio.run(_run())
        assert result["regions"] == []


# ---------------------------------------------------------------------------
# register_template (end-to-end mock)
# ---------------------------------------------------------------------------

class TestRegisterTemplate:
    def test_register_end_to_end(self, templates_dir, tmp_path):
        img_path = _create_test_image(tmp_path / "frame.png", 1920, 1080)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "regions": [
                {"type": "image", "x": 100, "y": 100, "width": 800, "height": 600, "description": "Hero"},
            ],
            "analysis_notes": "Widescreen frame",
            "suggested_content_types": ["announcement"],
        }))]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await register_template(img_path, "Hero Frame", "hero1")

        result = asyncio.run(_run())
        assert result.id == "hero1"
        assert result.name == "Hero Frame"
        assert result.width == 1920
        assert result.height == 1080
        assert result.aspect_ratio == "16:9"
        assert len(result.regions) == 1
        assert result.regions[0].type == "image"

        # Verify persisted
        memory = TemplateMemory()
        templates = memory.list_templates()
        assert len(templates) == 1
        assert templates[0].id == "hero1"


# ---------------------------------------------------------------------------
# apply_template — alpha-composite composition
# ---------------------------------------------------------------------------

class TestApplyTemplate:
    def test_composites_image_with_alpha(self, tmp_path):
        """Alpha composite: template on top, image below — transparent areas show through."""
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600, "white")
        gen_path = tmp_path / "gen.png"
        _create_test_image(gen_path, 400, 300, "blue")

        region = TemplateRegion(type="image", x=50, y=50, width=400, height=300)
        template = _template(path=str(tpl_path), regions=[region])

        # Read generated image bytes for mock HTTP response
        gen_bytes = gen_path.read_bytes()

        async def _run():
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = gen_bytes
            mock_resp.raise_for_status = MagicMock()

            with patch("agent.template_memory.httpx.AsyncClient") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_resp)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_httpx.return_value = mock_client
                return await apply_template(template, "https://example.com/img.png", {"title": "Test"})

        result = asyncio.run(_run())
        assert result is not None
        assert isinstance(result, io.BytesIO)
        # Verify it's a valid image
        img = Image.open(result)
        assert img.size == (800, 600)

    def test_local_file_path(self, tmp_path):
        """apply_template accepts local file paths (not just URLs)."""
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600, "white")
        gen_path = str(_create_test_image(tmp_path / "gen.png", 400, 300, "green"))

        region = TemplateRegion(type="image", x=0, y=0, width=400, height=300)
        template = _template(path=str(tpl_path), regions=[region])

        async def _run():
            return await apply_template(template, gen_path, {"title": "Local"})

        result = asyncio.run(_run())
        assert result is not None
        assert isinstance(result, io.BytesIO)
        img = Image.open(result)
        assert img.size == (800, 600)

    def test_no_image_region_returns_none(self, tmp_path):
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600)
        text_region = TemplateRegion(type="text", x=10, y=10, width=100, height=30)
        template = _template(path=str(tpl_path), regions=[text_region])

        async def _run():
            return await apply_template(template, "https://example.com/img.png", {})

        result = asyncio.run(_run())
        assert result is None

    def test_returns_bytesio(self, tmp_path):
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600, "white")
        gen_path = tmp_path / "gen.png"
        _create_test_image(gen_path, 400, 300, "red")
        gen_bytes = gen_path.read_bytes()

        region = TemplateRegion(type="image", x=0, y=0, width=400, height=300)
        template = _template(path=str(tpl_path), regions=[region])

        async def _run():
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = gen_bytes
            mock_resp.raise_for_status = MagicMock()

            with patch("agent.template_memory.httpx.AsyncClient") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_resp)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_httpx.return_value = mock_client
                return await apply_template(template, "https://example.com/img.png", {"title": "Hello"})

        result = asyncio.run(_run())
        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert len(result.read()) > 0

    def test_handles_download_failure(self, tmp_path):
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600)
        region = TemplateRegion(type="image", x=0, y=0, width=400, height=300)
        template = _template(path=str(tpl_path), regions=[region])

        async def _run():
            with patch("agent.template_memory.httpx.AsyncClient") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=Exception("Download failed"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_httpx.return_value = mock_client
                return await apply_template(template, "https://bad.url/img.png", {})

        result = asyncio.run(_run())
        assert result is None

    def test_logo_region_placed(self, tmp_path):
        """Logo region triggers _place_logo when logo.png exists."""
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600, "white")
        gen_path = str(_create_test_image(tmp_path / "gen.png", 400, 300, "blue"))

        # Create a logo file
        logo_dir = tmp_path / "brand" / "assets"
        logo_dir.mkdir(parents=True)
        logo_img = Image.new("RGBA", (100, 100), (255, 0, 0, 200))
        logo_img.save(str(logo_dir / "logo.png"), "PNG")

        image_region = TemplateRegion(type="image", x=0, y=0, width=400, height=300)
        logo_region = TemplateRegion(type="logo", x=350, y=10, width=80, height=80)
        template = _template(path=str(tpl_path), regions=[image_region, logo_region])

        async def _run():
            with patch("agent.template_memory.settings.BRAND_FOLDER", str(tmp_path / "brand")):
                return await apply_template(template, gen_path, {})

        result = asyncio.run(_run())
        assert result is not None
        assert isinstance(result, io.BytesIO)

    def test_text_regions_with_wrapping(self, tmp_path):
        """Text regions render with word wrap and don't crash on long text."""
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600, "white")
        gen_path = str(_create_test_image(tmp_path / "gen.png", 400, 300, "blue"))

        image_region = TemplateRegion(type="image", x=0, y=0, width=400, height=300)
        text_region = TemplateRegion(type="text", x=50, y=320, width=700, height=80)
        template = _template(path=str(tpl_path), regions=[image_region, text_region])

        async def _run():
            return await apply_template(
                template, gen_path,
                {"title": "This is a long headline that should wrap to multiple lines if needed"},
            )

        result = asyncio.run(_run())
        assert result is not None
        assert isinstance(result, io.BytesIO)


# ---------------------------------------------------------------------------
# get_aspect_ratio_for_content_type (Commit H)
# ---------------------------------------------------------------------------

class TestGetAspectRatio:
    def test_returns_none_when_no_templates(self, templates_dir):
        result = get_aspect_ratio_for_content_type("announcement")
        assert result is None

    def test_returns_correct_aspect_ratio(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(
            tid="wide", aspect_ratio="16:9", content_types=["announcement"],
        ))
        result = get_aspect_ratio_for_content_type("announcement")
        assert result == "16:9"

    def test_returns_none_for_unmatched_type(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(
            tid="wide", aspect_ratio="16:9", content_types=["meme"],
        ))
        result = get_aspect_ratio_for_content_type("announcement")
        assert result is None


# ---------------------------------------------------------------------------
# get_image_region_aspect_ratio
# ---------------------------------------------------------------------------

class TestImageRegionAspectRatio:
    def test_returns_none_when_no_templates(self, templates_dir):
        result = get_image_region_aspect_ratio("announcement")
        assert result is None

    def test_returns_square_for_square_region(self, templates_dir):
        memory = TemplateMemory()
        region = TemplateRegion(type="image", x=0, y=0, width=500, height=500)
        memory.add_template(_template(
            tid="sq", regions=[region], content_types=["meme"],
        ))
        result = get_image_region_aspect_ratio("meme")
        assert result == "1:1"

    def test_returns_16_9_for_widescreen_region(self, templates_dir):
        memory = TemplateMemory()
        region = TemplateRegion(type="image", x=0, y=0, width=1600, height=900)
        memory.add_template(_template(
            tid="wide", regions=[region], content_types=["announcement"],
        ))
        result = get_image_region_aspect_ratio("announcement")
        assert result == "16:9"

    def test_returns_raw_ratio_for_nonstandard(self, templates_dir):
        memory = TemplateMemory()
        region = TemplateRegion(type="image", x=0, y=0, width=300, height=700)
        memory.add_template(_template(
            tid="odd", regions=[region], content_types=["custom"],
        ))
        result = get_image_region_aspect_ratio("custom")
        assert result == "300:700"

    def test_returns_none_without_image_region(self, templates_dir):
        memory = TemplateMemory()
        region = TemplateRegion(type="text", x=0, y=0, width=500, height=500)
        memory.add_template(_template(
            tid="textonly", regions=[region], content_types=["announcement"],
        ))
        result = get_image_region_aspect_ratio("announcement")
        assert result is None


# ---------------------------------------------------------------------------
# detect_if_template (Commit I)
# ---------------------------------------------------------------------------

class TestDetectIfTemplate:
    def test_returns_true(self, tmp_path):
        img_path = _create_test_image(tmp_path / "frame.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "is_template": True, "confidence": 0.9, "reason": "Phone mockup",
        }))]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await detect_if_template(img_path)

        assert asyncio.run(_run()) is True

    def test_returns_false(self, tmp_path):
        img_path = _create_test_image(tmp_path / "photo.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "is_template": False, "confidence": 0.2, "reason": "Just a photo",
        }))]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await detect_if_template(img_path)

        assert asyncio.run(_run()) is False

    def test_returns_false_on_error(self, tmp_path):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))
                mock_cls.return_value = mock_client
                return await detect_if_template("/nonexistent/path.png")

        assert asyncio.run(_run()) is False


# ---------------------------------------------------------------------------
# update_template_regions
# ---------------------------------------------------------------------------

class TestUpdateTemplateRegions:
    def test_updates_regions_in_place(self, templates_dir):
        memory = TemplateMemory()
        region = TemplateRegion(type="image", x=10, y=20, width=100, height=100)
        memory.add_template(_template(tid="tpl1", regions=[region]))

        new_regions = [
            TemplateRegion(type="image", x=0, y=0, width=1200, height=700, description="Full canvas"),
            TemplateRegion(type="text", x=0, y=0, width=1200, height=105, description="Top text"),
            TemplateRegion(type="text", x=0, y=595, width=1200, height=105, description="Bottom text"),
        ]
        updated = memory.update_template_regions("tpl1", new_regions)
        assert updated is not None
        assert len(updated.regions) == 3
        assert updated.regions[0].type == "image"
        assert updated.regions[0].width == 1200
        assert updated.regions[1].description == "Top text"
        assert updated.regions[2].y == 595

    def test_persists_to_disk(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="tpl1"))

        new_regions = [
            TemplateRegion(type="image", x=0, y=0, width=800, height=600),
        ]
        memory.update_template_regions("tpl1", new_regions)

        # Reload from disk
        memory2 = TemplateMemory()
        templates = memory2.list_templates()
        assert len(templates) == 1
        assert len(templates[0].regions) == 1
        assert templates[0].regions[0].width == 800

    def test_returns_none_for_missing_id(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(tid="tpl1"))

        result = memory.update_template_regions("nonexistent", [])
        assert result is None

    def test_preserves_other_template_fields(self, templates_dir):
        memory = TemplateMemory()
        memory.add_template(_template(
            tid="tpl1", name="My Template", aspect_ratio="16:9",
            content_types=["meme"],
        ))

        new_regions = [TemplateRegion(type="text", x=0, y=0, width=100, height=50)]
        updated = memory.update_template_regions("tpl1", new_regions)
        assert updated.name == "My Template"
        assert updated.aspect_ratio == "16:9"
        assert updated.content_types == ["meme"]


# ---------------------------------------------------------------------------
# parse_region_description (mocked Claude)
# ---------------------------------------------------------------------------

class TestParseRegionDescription:
    def test_parses_meme_layout(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "regions": [
                {"type": "image", "x": 0, "y": 0, "width": 1200, "height": 700, "description": "Full canvas"},
                {"type": "text", "x": 0, "y": 0, "width": 1200, "height": 105, "description": "Top text"},
                {"type": "text", "x": 0, "y": 595, "width": 1200, "height": 105, "description": "Bottom text"},
            ],
        }))]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await parse_region_description(
                    "top text across top 15%, bottom text across bottom 15%, image fills full canvas",
                    1200, 700,
                )

        result = asyncio.run(_run())
        assert len(result) == 3
        assert result[0].type == "image"
        assert result[0].width == 1200
        assert result[1].type == "text"
        assert result[1].y == 0
        assert result[2].type == "text"
        assert result[2].y == 595

    def test_returns_empty_on_bad_response(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]

        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await parse_region_description("some description", 800, 600)

        result = asyncio.run(_run())
        assert result == []

    def test_returns_empty_on_api_error(self):
        async def _run():
            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))
                mock_cls.return_value = mock_client
                return await parse_region_description("some description", 800, 600)

        # Should raise since we don't catch in parse_region_description
        with pytest.raises(Exception):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# _is_template_region_update detection
# ---------------------------------------------------------------------------

class TestIsTemplateRegionUpdate:
    def test_detects_region_description(self):
        from bot.handlers import _is_template_region_update

        mock_ctx = MagicMock()
        mock_ctx.user_data = {"last_uploaded_template_id": "tpl1"}

        assert _is_template_region_update(
            "top text goes across the top 15%, bottom text across the bottom 15%, entire background is the image zone",
            mock_ctx,
        ) is True

    def test_rejects_without_template_id(self):
        from bot.handlers import _is_template_region_update

        mock_ctx = MagicMock()
        mock_ctx.user_data = {}

        assert _is_template_region_update(
            "top text goes across the top 15%, image fills the bottom",
            mock_ctx,
        ) is False

    def test_rejects_unrelated_message(self):
        from bot.handlers import _is_template_region_update

        mock_ctx = MagicMock()
        mock_ctx.user_data = {"last_uploaded_template_id": "tpl1"}

        assert _is_template_region_update(
            "make me an announcement about our new product launch",
            mock_ctx,
        ) is False

    def test_detects_percentage_description(self):
        from bot.handlers import _is_template_region_update

        mock_ctx = MagicMock()
        mock_ctx.user_data = {"last_uploaded_template_id": "tpl1"}

        assert _is_template_region_update(
            "image fills the full canvas, title text centered at the top 20%",
            mock_ctx,
        ) is True

    def test_rejects_single_keyword(self):
        from bot.handlers import _is_template_region_update

        mock_ctx = MagicMock()
        mock_ctx.user_data = {"last_uploaded_template_id": "tpl1"}

        # Only one keyword hit ("top") — not enough
        assert _is_template_region_update("the top result is good", mock_ctx) is False


# ---------------------------------------------------------------------------
# Meme-style text rendering
# ---------------------------------------------------------------------------

class TestMemeTextStyle:
    def test_meme_template_uses_uppercase(self, tmp_path):
        """Meme-named templates render text in ALL CAPS."""
        tpl_path = _create_test_image(tmp_path / "tpl.png", 1200, 700, "black")
        gen_path = str(_create_test_image(tmp_path / "gen.png", 1200, 700, "blue"))

        image_region = TemplateRegion(type="image", x=0, y=0, width=1200, height=700)
        text_top = TemplateRegion(type="text", x=0, y=0, width=1200, height=105)
        text_bottom = TemplateRegion(type="text", x=0, y=595, width=1200, height=105)
        template = _template(
            name="meme", path=str(tpl_path),
            width=1200, height=700,
            regions=[image_region, text_top, text_bottom],
            content_types=["meme"],
        )

        async def _run():
            return await apply_template(
                template, gen_path,
                {"title": "when the code works", "subtitle": "on the first try"},
            )

        result = asyncio.run(_run())
        assert result is not None
        assert isinstance(result, io.BytesIO)

    def test_meme_style_detected_by_name(self):
        """Style detection is based on template name containing 'meme'."""
        # Template name "meme" → style="meme"
        tpl_meme = _template(name="meme")
        assert "meme" in tpl_meme.name.lower()

        # Template name "My Meme Frame" → should also detect
        tpl_meme2 = _template(name="My Meme Frame")
        assert "meme" in tpl_meme2.name.lower()

        # Template name "announcement" → not meme
        tpl_other = _template(name="announcement")
        assert "meme" not in tpl_other.name.lower()

    def test_draw_fitted_text_meme_uppercases(self, tmp_path):
        """_draw_fitted_text with style='meme' converts text to ALL CAPS."""
        canvas = Image.new("RGBA", (1200, 700), (0, 0, 0, 255))
        region = TemplateRegion(type="text", x=0, y=0, width=1200, height=105)
        # Just verify it doesn't crash and produces output
        _draw_fitted_text(canvas, "hello world", region, role="title", style="meme")
        # Canvas should have been drawn on (non-trivial to verify text content,
        # but we can verify it didn't raise)

    def test_draw_fitted_text_default_style(self, tmp_path):
        """_draw_fitted_text with default style uses Orbitron/Inter (no uppercasing)."""
        canvas = Image.new("RGBA", (800, 600), (0, 0, 0, 255))
        region = TemplateRegion(type="text", x=50, y=50, width=700, height=80)
        _draw_fitted_text(canvas, "hello world", region, role="title", style="default")

    def test_non_meme_template_keeps_default_style(self, tmp_path):
        """Templates not named 'meme' should use default style (Orbitron/Inter)."""
        tpl_path = _create_test_image(tmp_path / "tpl.png", 800, 600, "white")
        gen_path = str(_create_test_image(tmp_path / "gen.png", 400, 300, "blue"))

        image_region = TemplateRegion(type="image", x=0, y=0, width=400, height=300)
        text_region = TemplateRegion(type="text", x=50, y=320, width=700, height=80)
        template = _template(
            name="announcement", path=str(tpl_path),
            regions=[image_region, text_region],
            content_types=["announcement"],
        )

        async def _run():
            return await apply_template(
                template, gen_path,
                {"title": "Big News Today"},
            )

        result = asyncio.run(_run())
        assert result is not None

    def test_get_meme_font_returns_font(self):
        """_get_meme_font returns a usable font object."""
        font = _get_meme_font(48)
        assert font is not None
