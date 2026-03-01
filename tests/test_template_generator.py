"""Tests for agent.template_generator — AI-based template generation via flux-kontext-pro."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from PIL import Image

from agent.template_generator import (
    TemplateDesign,
    _parse_json_response,
    _compute_aspect_ratio,
    analyze_reference,
    build_generation_prompt,
    generate_template_image,
    download_image,
    design_to_dict,
    design_from_dict,
    adjust_design,
    register_design,
    analyze_and_generate,
    save_generated_image,
    generate_template_from_reference,
)
from agent.template_memory import TemplateRegion
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
    img = Image.new("RGB", (width, height), color)
    img.save(str(path), "PNG")
    return str(path)


def _mock_analysis_response():
    return json.dumps({
        "layout_description": "Split-panel layout with image on left, text on right, header bar across top",
        "visual_style": "glass-morphism with neon accents on dark background",
        "canvas_width": 1280,
        "canvas_height": 720,
        "regions": [
            {"type": "image", "x": 40, "y": 80, "width": 580, "height": 560, "description": "Main image area"},
            {"type": "text", "x": 660, "y": 200, "width": 560, "height": 80, "description": "Headline text"},
            {"type": "text", "x": 660, "y": 300, "width": 560, "height": 50, "description": "Subtitle"},
            {"type": "logo", "x": 40, "y": 20, "width": 80, "height": 50, "description": "Brand logo top left"},
        ],
    })


def _sample_design(image_path: str = "/tmp/ref.png") -> TemplateDesign:
    return TemplateDesign(
        layout_description="Split-panel with image left and text right",
        visual_style="glass-morphism dark",
        generation_prompt="Recreate this layout as a branded template...",
        reference_image_path=image_path,
        generated_image_url="https://example.com/generated.png",
        canvas_width=1280,
        canvas_height=720,
        regions=[
            TemplateRegion(type="image", x=40, y=80, width=580, height=560, description="Main image"),
            TemplateRegion(type="text", x=660, y=200, width=560, height=80, description="Headline"),
            TemplateRegion(type="logo", x=40, y=20, width=80, height=50, description="Logo"),
        ],
    )


def _mock_brand_config():
    """Return a mock BrandConfig for prompt building."""
    color_entry = MagicMock()
    color_entry.hex = "#FF69B4"
    font_entry = MagicMock()
    font_entry.family = "Orbitron"
    return MagicMock(
        brand_name="FOID",
        tagline="The future of identity",
        colors={"primary": color_entry},
        fonts={"display": font_entry},
        style_keywords=["futuristic", "neon"],
    )


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
# design_to_dict / design_from_dict — serialization round-trip
# ---------------------------------------------------------------------------

class TestDesignSerialization:
    def test_round_trip(self):
        design = _sample_design()
        d = design_to_dict(design)
        restored = design_from_dict(d)

        assert restored.layout_description == "Split-panel with image left and text right"
        assert restored.visual_style == "glass-morphism dark"
        assert restored.generation_prompt == "Recreate this layout as a branded template..."
        assert restored.reference_image_path == "/tmp/ref.png"
        assert restored.generated_image_url == "https://example.com/generated.png"
        assert restored.canvas_width == 1280
        assert restored.canvas_height == 720
        assert len(restored.regions) == 3
        assert restored.regions[0].type == "image"
        assert restored.regions[0].x == 40

    def test_dict_is_json_safe(self):
        design = _sample_design()
        d = design_to_dict(design)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_from_empty_dict(self):
        design = design_from_dict({})
        assert design.canvas_width == 1280
        assert design.layout_description == ""
        assert design.regions == []


# ---------------------------------------------------------------------------
# _compute_aspect_ratio
# ---------------------------------------------------------------------------

class TestComputeAspectRatio:
    def test_standard_ratios(self):
        assert _compute_aspect_ratio(1920, 1080) == "16:9"
        assert _compute_aspect_ratio(1080, 1920) == "9:16"
        assert _compute_aspect_ratio(1080, 1080) == "1:1"
        assert _compute_aspect_ratio(800, 600) == "4:3"

    def test_custom_ratio(self):
        assert _compute_aspect_ratio(500, 300) == "500:300"

    def test_zero_dimensions(self):
        assert _compute_aspect_ratio(0, 0) == ""


# ---------------------------------------------------------------------------
# analyze_reference (mocked Claude)
# ---------------------------------------------------------------------------

class TestAnalyzeReference:
    def test_parses_analysis(self, tmp_path):
        img_path = _create_test_image(tmp_path / "ref.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=_mock_analysis_response())]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await analyze_reference(img_path)

        result = asyncio.run(_run())
        assert result.canvas_width == 1280
        assert result.canvas_height == 720
        assert "Split-panel" in result.layout_description
        assert "glass-morphism" in result.visual_style
        assert len(result.regions) == 4
        assert result.reference_image_path == img_path

    def test_handles_bad_response(self, tmp_path):
        img_path = _create_test_image(tmp_path / "ref.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON at all")]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await analyze_reference(img_path)

        result = asyncio.run(_run())
        assert result.layout_description == ""
        assert result.regions == []
        assert result.reference_image_path == img_path


# ---------------------------------------------------------------------------
# build_generation_prompt
# ---------------------------------------------------------------------------

class TestBuildGenerationPrompt:
    def test_includes_brand_identity(self):
        design = _sample_design()
        with patch("agent.template_generator._cc.get_config", return_value=_mock_brand_config()):
            prompt = build_generation_prompt(design)

        assert "FOID" in prompt
        assert "#FF69B4" in prompt
        assert "Orbitron" in prompt
        assert "futuristic" in prompt
        assert "Split-panel" in prompt
        assert "TEMPLATE" in prompt
        assert design.generation_prompt == prompt

    def test_handles_empty_brand_config(self):
        design = _sample_design()
        empty_cfg = MagicMock(
            brand_name="",
            tagline="",
            colors={},
            fonts={},
            style_keywords=[],
        )
        with patch("agent.template_generator._cc.get_config", return_value=empty_cfg):
            prompt = build_generation_prompt(design)

        assert "the brand" in prompt
        assert "modern palette" in prompt.lower() or "dark modern palette" in prompt


# ---------------------------------------------------------------------------
# generate_template_image (mocked Replicate)
# ---------------------------------------------------------------------------

class TestGenerateTemplateImage:
    def test_calls_img2img(self):
        design = _sample_design()

        async def _run():
            with patch("agent.template_generator.generate_img2img", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "https://replicate.com/output/abc.png"
                url = await generate_template_image(design)
                mock_gen.assert_called_once_with(
                    prompt=design.generation_prompt,
                    input_image_path=design.reference_image_path,
                    strength=0.75,
                )
                return url

        result = asyncio.run(_run())
        assert result == "https://replicate.com/output/abc.png"

    def test_returns_none_on_failure(self):
        design = _sample_design()

        async def _run():
            with patch("agent.template_generator.generate_img2img", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = None
                return await generate_template_image(design)

        result = asyncio.run(_run())
        assert result is None


# ---------------------------------------------------------------------------
# download_image
# ---------------------------------------------------------------------------

class TestDownloadImage:
    def test_downloads_successfully(self, tmp_path):
        # Create a test image to serve as response content
        img = Image.new("RGB", (100, 100), "red")
        import io
        buf = io.BytesIO()
        img.save(buf, "PNG")
        image_bytes = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = image_bytes
        mock_resp.raise_for_status = MagicMock()

        async def _run():
            with patch("agent.template_generator.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_resp)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client
                return await download_image("https://example.com/img.png")

        result = asyncio.run(_run())
        assert result is not None
        assert result.size == (100, 100)

    def test_returns_none_on_error(self):
        async def _run():
            with patch("agent.template_generator.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=Exception("network error"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client
                return await download_image("https://example.com/fail.png")

        result = asyncio.run(_run())
        assert result is None


# ---------------------------------------------------------------------------
# adjust_design (mocked Claude + Replicate)
# ---------------------------------------------------------------------------

class TestAdjustDesign:
    def test_modifies_prompt_and_regenerates(self):
        design = _sample_design()

        adjusted_response = json.dumps({
            "generation_prompt": "Updated prompt with more vibrant colors...",
            "layout_description": "Updated layout description",
        })
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=adjusted_response)]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.template_generator.generate_template_image", new_callable=AsyncMock) as mock_gen:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                mock_gen.return_value = "https://replicate.com/output/new.png"
                return await adjust_design(design, "make the colors more vibrant")

        result = asyncio.run(_run())
        assert result.generation_prompt == "Updated prompt with more vibrant colors..."
        assert result.layout_description == "Updated layout description"

    def test_returns_original_on_bad_response(self):
        design = _sample_design()
        original_prompt = design.generation_prompt

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="invalid json")]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await adjust_design(design, "something")

        result = asyncio.run(_run())
        assert result.generation_prompt == original_prompt

    def test_returns_original_on_empty_prompt(self):
        design = _sample_design()
        original_prompt = design.generation_prompt

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"generation_prompt": "", "layout_description": ""}')]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await adjust_design(design, "remove everything")

        result = asyncio.run(_run())
        assert result.generation_prompt == original_prompt


# ---------------------------------------------------------------------------
# register_design
# ---------------------------------------------------------------------------

class TestRegisterDesign:
    def test_registers_in_memory(self, templates_dir, tmp_path):
        design = _sample_design()
        img_path = _create_test_image(tmp_path / "tpl.png")

        template = register_design(design, img_path, name="My Template")
        assert template.name == "My Template"
        assert template.width == 1280
        assert template.height == 720
        assert template.aspect_ratio == "16:9"
        assert len(template.regions) == 3

        from agent.template_memory import TemplateMemory
        memory = TemplateMemory()
        templates = memory.list_templates()
        assert len(templates) == 1
        assert templates[0].name == "My Template"

    def test_auto_name(self, templates_dir, tmp_path):
        design = _sample_design()
        img_path = _create_test_image(tmp_path / "tpl.png")

        template = register_design(design, img_path)
        assert template.name.startswith("Generated ")


# ---------------------------------------------------------------------------
# analyze_and_generate (full phase 1)
# ---------------------------------------------------------------------------

class TestAnalyzeAndGenerate:
    def test_returns_design_and_image(self, tmp_path):
        import io as _io
        img_path = _create_test_image(tmp_path / "ref.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=_mock_analysis_response())]

        # Create a fake downloaded image
        fake_img = Image.new("RGB", (1280, 720), "blue")
        buf = _io.BytesIO()
        fake_img.save(buf, "PNG")
        image_bytes = buf.getvalue()

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.template_generator._cc.get_config", return_value=_mock_brand_config()), \
                 patch("agent.template_generator.generate_img2img", new_callable=AsyncMock) as mock_gen, \
                 patch("agent.template_generator.download_image", new_callable=AsyncMock) as mock_dl:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                mock_gen.return_value = "https://replicate.com/output/tpl.png"
                mock_dl.return_value = fake_img
                return await analyze_and_generate(img_path)

        design, img = asyncio.run(_run())
        assert isinstance(design, TemplateDesign)
        assert "Split-panel" in design.layout_description
        assert design.generation_prompt  # Should be built
        assert img.size == (1280, 720)

    def test_raises_on_empty_description(self, tmp_path):
        img_path = _create_test_image(tmp_path / "blank.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"layout_description": "", "regions": [], "canvas_width": 800, "canvas_height": 600}')]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await analyze_and_generate(img_path)

        with pytest.raises(ValueError, match="Could not analyze"):
            asyncio.run(_run())

    def test_raises_on_generation_failure(self, tmp_path):
        img_path = _create_test_image(tmp_path / "ref.png")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=_mock_analysis_response())]

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.template_generator._cc.get_config", return_value=_mock_brand_config()), \
                 patch("agent.template_generator.generate_img2img", new_callable=AsyncMock) as mock_gen:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                mock_gen.return_value = None
                return await analyze_and_generate(img_path)

        with pytest.raises(ValueError, match="generation failed"):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# save_generated_image
# ---------------------------------------------------------------------------

class TestSaveGeneratedImage:
    def test_saves_to_disk(self, templates_dir):
        import io as _io
        design = _sample_design()

        fake_img = Image.new("RGB", (1280, 720), "green")

        async def _run():
            with patch("agent.template_generator.download_image", new_callable=AsyncMock) as mock_dl:
                mock_dl.return_value = fake_img
                return await save_generated_image(design)

        path = asyncio.run(_run())
        assert Path(path).exists()
        assert Path(path).suffix == ".png"

    def test_raises_on_download_failure(self, templates_dir):
        design = _sample_design()

        async def _run():
            with patch("agent.template_generator.download_image", new_callable=AsyncMock) as mock_dl:
                mock_dl.return_value = None
                return await save_generated_image(design)

        with pytest.raises(ValueError, match="Failed to download"):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# generate_template_from_reference (full pipeline)
# ---------------------------------------------------------------------------

class TestGenerateFromReference:
    def test_end_to_end(self, templates_dir, tmp_path):
        import io as _io
        img_path = _create_test_image(tmp_path / "screenshot.png", 1280, 720)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=_mock_analysis_response())]

        fake_img = Image.new("RGB", (1280, 720), "purple")

        async def _run():
            with patch("agent.template_generator.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.template_generator._cc.get_config", return_value=_mock_brand_config()), \
                 patch("agent.template_generator.generate_img2img", new_callable=AsyncMock) as mock_gen, \
                 patch("agent.template_generator.download_image", new_callable=AsyncMock) as mock_dl:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                mock_gen.return_value = "https://replicate.com/output/tpl.png"
                mock_dl.return_value = fake_img
                return await generate_template_from_reference(img_path, name="Test Layout")

        template, img = asyncio.run(_run())
        assert template.name == "Test Layout"
        assert template.width == 1280
        assert template.height == 720
        assert template.aspect_ratio == "16:9"
        assert len(template.regions) >= 1
        assert img.size == (1280, 720)

        region_types = [r.type for r in template.regions]
        assert "image" in region_types
        assert "text" in region_types
        assert "logo" in region_types


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
# TemplateDesign dataclass
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_template_design_defaults(self):
        d = TemplateDesign()
        assert d.layout_description == ""
        assert d.visual_style == ""
        assert d.generation_prompt == ""
        assert d.reference_image_path == ""
        assert d.generated_image_url == ""
        assert d.canvas_width == 1280
        assert d.canvas_height == 720
        assert d.regions == []
