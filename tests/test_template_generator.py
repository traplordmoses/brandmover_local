"""Tests for agent.template_generator — layout analysis, rendering, and registration."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from PIL import Image

from agent.template_generator import (
    LayoutZone,
    LayoutAnalysis,
    _parse_json_response,
    analyze_layout,
    render_template,
    generate_template_from_reference,
)
from bot.handlers import _is_template_from_ref_intent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def templates_dir(tmp_path):
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    with patch("agent.template_generator.settings.BRAND_FOLDER", str(tmp_path)), \
         patch("agent.template_memory._TEMPLATES_DIR", tpl_dir), \
         patch("agent.template_memory._MANIFEST_PATH", tpl_dir / "manifest.json"):
        yield tpl_dir


def _create_test_image(path, width=1280, height=720, color="navy"):
    img = Image.new("RGBA", (width, height), color)
    img.save(str(path), "PNG")
    return str(path)


def _mock_layout_response():
    return json.dumps({
        "canvas_width": 1280,
        "canvas_height": 720,
        "layout_pattern": "split-panel with image left and text right",
        "zones": [
            {
                "type": "background",
                "x": 0, "y": 0, "width": 1280, "height": 720,
                "description": "Dark background",
                "style_notes": "solid dark color",
            },
            {
                "type": "image",
                "x": 40, "y": 80, "width": 580, "height": 560,
                "description": "Main image area",
                "style_notes": "rounded corners",
            },
            {
                "type": "text_primary",
                "x": 660, "y": 200, "width": 560, "height": 80,
                "description": "Headline text",
                "style_notes": "white uppercase bold",
            },
            {
                "type": "text_secondary",
                "x": 660, "y": 300, "width": 560, "height": 50,
                "description": "Subtitle",
                "style_notes": "gray smaller text",
            },
            {
                "type": "logo",
                "x": 40, "y": 20, "width": 80, "height": 50,
                "description": "Brand logo top left",
                "style_notes": "small logo",
            },
        ],
    })


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        result = _parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_fenced_no_lang(self):
        result = _parse_json_response('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_raises_on_invalid(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json")


# ---------------------------------------------------------------------------
# analyze_layout (mocked Claude)
# ---------------------------------------------------------------------------

class TestAnalyzeLayout:
    def test_parses_layout(self, tmp_path):
        img_path = _create_test_image(tmp_path / "ref.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=_mock_layout_response())]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await analyze_layout(img_path)

        result = asyncio.run(_run())
        assert result.canvas_width == 1280
        assert result.canvas_height == 720
        assert len(result.zones) == 5
        assert result.layout_pattern == "split-panel with image left and text right"

        zone_types = [z.type for z in result.zones]
        assert "background" in zone_types
        assert "image" in zone_types
        assert "text_primary" in zone_types

    def test_handles_bad_response(self, tmp_path):
        img_path = _create_test_image(tmp_path / "ref.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON at all")]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await analyze_layout(img_path)

        result = asyncio.run(_run())
        assert result.zones == []
        assert result.canvas_width == 1280  # defaults


# ---------------------------------------------------------------------------
# render_template — PIL rendering
# ---------------------------------------------------------------------------

class TestRenderTemplate:
    def test_renders_correct_size(self):
        layout = LayoutAnalysis(
            canvas_width=1280,
            canvas_height=720,
            zones=[
                LayoutZone(type="background", x=0, y=0, width=1280, height=720),
            ],
        )
        with patch("agent.template_generator._cc.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                brand_name="TestBrand",
                tagline="test tagline",
                colors={},
                fonts={},
            )
            with patch("agent.template_generator._brand_color", return_value=(10, 10, 26)):
                img = render_template(layout)
        assert img.size == (1280, 720)

    def test_renders_all_zone_types(self):
        from PIL import ImageFont
        default_font = ImageFont.load_default()

        layout = LayoutAnalysis(
            canvas_width=800,
            canvas_height=600,
            zones=[
                LayoutZone(type="background", x=0, y=0, width=800, height=600),
                LayoutZone(type="image", x=20, y=60, width=300, height=400),
                LayoutZone(type="header", x=0, y=0, width=800, height=50),
                LayoutZone(type="footer", x=0, y=550, width=800, height=50),
                LayoutZone(type="text_primary", x=350, y=100, width=400, height=60),
                LayoutZone(type="text_secondary", x=350, y=180, width=400, height=40),
                LayoutZone(type="logo", x=20, y=10, width=60, height=40),
                LayoutZone(type="badge", x=350, y=60, width=100, height=30),
                LayoutZone(type="accent", x=340, y=160, width=420, height=4, style_notes="line"),
            ],
        )
        with patch("agent.template_generator._cc.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                brand_name="TestBrand",
                tagline="test tagline",
                colors={},
                fonts={},
            )
            with patch("agent.template_generator._brand_color", return_value=(100, 100, 200)):
                with patch("agent.template_generator._brand_font", return_value=default_font):
                    with patch("agent.template_generator._load_logo", return_value=None):
                        img = render_template(layout)
        assert img.size == (800, 600)
        assert img.mode == "RGBA"

    def test_empty_zones(self):
        layout = LayoutAnalysis(canvas_width=400, canvas_height=300, zones=[])
        with patch("agent.template_generator._cc.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                brand_name="", tagline="", colors={}, fonts={},
            )
            with patch("agent.template_generator._brand_color", return_value=(10, 10, 26)):
                img = render_template(layout)
        assert img.size == (400, 300)


# ---------------------------------------------------------------------------
# generate_template_from_reference (end-to-end mock)
# ---------------------------------------------------------------------------

class TestGenerateFromReference:
    def test_end_to_end(self, templates_dir, tmp_path):
        from PIL import ImageFont
        default_font = ImageFont.load_default()

        img_path = _create_test_image(tmp_path / "screenshot.png", 1280, 720)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=_mock_layout_response())]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.template_generator._brand_color", return_value=(10, 10, 26)), \
                 patch("agent.template_generator._brand_font", return_value=default_font), \
                 patch("agent.template_generator._load_logo", return_value=None):
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await generate_template_from_reference(img_path, name="Test Layout")

        template, img = asyncio.run(_run())
        assert template.name == "Test Layout"
        assert template.width == 1280
        assert template.height == 720
        assert template.aspect_ratio == "16:9"
        assert len(template.regions) >= 1  # At least the image region
        assert img.size == (1280, 720)

        # Check region types mapped correctly
        region_types = [r.type for r in template.regions]
        assert "image" in region_types
        assert "text" in region_types  # text_primary → text
        assert "logo" in region_types

    def test_raises_on_empty_zones(self, templates_dir, tmp_path):
        img_path = _create_test_image(tmp_path / "blank.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"zones": [], "canvas_width": 800, "canvas_height": 600}')]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await generate_template_from_reference(img_path)

        with pytest.raises(ValueError, match="Could not detect"):
            asyncio.run(_run())

    def test_auto_name(self, templates_dir, tmp_path):
        img_path = _create_test_image(tmp_path / "ref.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "canvas_width": 400, "canvas_height": 400,
            "layout_pattern": "simple",
            "zones": [
                {"type": "background", "x": 0, "y": 0, "width": 400, "height": 400,
                 "description": "bg", "style_notes": "dark"},
                {"type": "image", "x": 20, "y": 20, "width": 360, "height": 360,
                 "description": "img", "style_notes": ""},
            ],
        }))]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.template_generator._brand_color", return_value=(10, 10, 26)), \
                 patch("agent.template_generator._load_logo", return_value=None):
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await generate_template_from_reference(img_path)

        template, _ = asyncio.run(_run())
        assert template.name.startswith("Generated ")


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

class TestTemplateFromRefIntent:
    def test_matching_intents(self):
        assert _is_template_from_ref_intent("make a template from this")
        assert _is_template_from_ref_intent("Make A Template from this image")
        assert _is_template_from_ref_intent("use this layout for our posts")
        assert _is_template_from_ref_intent("turn this into a template please")
        assert _is_template_from_ref_intent("can you create template from this?")
        assert _is_template_from_ref_intent("template from this screenshot")
        assert _is_template_from_ref_intent("copy this layout")
        assert _is_template_from_ref_intent("use this format for future posts")

    def test_non_matching_intents(self):
        assert not _is_template_from_ref_intent("write a tweet about crypto")
        assert not _is_template_from_ref_intent("make this post better")
        assert not _is_template_from_ref_intent("generate an image")
        assert not _is_template_from_ref_intent("hello")
        assert not _is_template_from_ref_intent("")


# ---------------------------------------------------------------------------
# LayoutZone and LayoutAnalysis dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_layout_zone_defaults(self):
        z = LayoutZone(type="image")
        assert z.x == 0
        assert z.y == 0
        assert z.width == 0
        assert z.height == 0
        assert z.description == ""
        assert z.style_notes == ""

    def test_layout_analysis_defaults(self):
        a = LayoutAnalysis()
        assert a.canvas_width == 1280
        assert a.canvas_height == 720
        assert a.layout_pattern == ""
        assert a.zones == []
