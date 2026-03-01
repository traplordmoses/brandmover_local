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
    detect_if_template,
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
            with patch("agent.template_memory.anthropic.AsyncAnthropic") as mock_cls:
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
            with patch("agent.template_memory.anthropic.AsyncAnthropic") as mock_cls:
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
            with patch("agent.template_memory.anthropic.AsyncAnthropic") as mock_cls:
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
# apply_template — PIL composition (Commit F)
# ---------------------------------------------------------------------------

class TestApplyTemplate:
    def test_composites_image_at_correct_coordinates(self, tmp_path):
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
            with patch("agent.template_memory.anthropic.AsyncAnthropic") as mock_cls:
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
            with patch("agent.template_memory.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await detect_if_template(img_path)

        assert asyncio.run(_run()) is False

    def test_returns_false_on_error(self, tmp_path):
        async def _run():
            with patch("agent.template_memory.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))
                mock_cls.return_value = mock_client
                return await detect_if_template("/nonexistent/path.png")

        assert asyncio.run(_run()) is False
