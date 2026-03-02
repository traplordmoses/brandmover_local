"""Tests for agent.template_renderer — deterministic PIL template rendering."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from agent.template_spec import (
    Border,
    Fill,
    GradientStop,
    ImageZoneSpec,
    LogoZoneSpec,
    ShapeElement,
    TemplateSpec,
    TextZoneSpec,
    spec_to_dict,
    spec_from_dict,
)
from agent.template_renderer import (
    _parse_color,
    _interpolate_stops,
    render_template_frame,
    render_preview,
    render_to_bytes,
    save_frame,
)


# ---------------------------------------------------------------------------
# _parse_color
# ---------------------------------------------------------------------------

class TestParseColor:
    def test_six_digit_hex(self):
        assert _parse_color("#FF0000") == (255, 0, 0, 255)
        assert _parse_color("#00ff00") == (0, 255, 0, 255)
        assert _parse_color("0000FF") == (0, 0, 255, 255)

    def test_three_digit_hex(self):
        assert _parse_color("#F00") == (255, 0, 0, 255)

    def test_eight_digit_hex_with_alpha(self):
        assert _parse_color("#FF000080") == (255, 0, 0, 128)

    def test_invalid_fallback(self):
        assert _parse_color("xyz") == (0, 0, 0, 255)
        assert _parse_color("") == (0, 0, 0, 255)


# ---------------------------------------------------------------------------
# TemplateSpec serialization round-trip
# ---------------------------------------------------------------------------

class TestSpecSerialization:
    def test_round_trip_solid_bg(self):
        spec = TemplateSpec(
            canvas_width=800,
            canvas_height=600,
            background=Fill(type="solid", color="#1A1A2E"),
            shapes=[
                ShapeElement(
                    shape="rounded_rect", x=20, y=20, width=760, height=560,
                    fill=Fill(type="solid", color="#16213E"),
                    corner_radius=16, z_order=1,
                ),
            ],
            text_zones=[
                TextZoneSpec(x=100, y=50, width=600, height=80, label="title",
                             font_family="Orbitron", font_size=48, color="#FFFFFF"),
            ],
            image_zones=[
                ImageZoneSpec(x=50, y=150, width=700, height=400, corner_radius=12),
            ],
            logo_zones=[
                LogoZoneSpec(x=30, y=20, width=80, height=40),
            ],
        )
        d = spec_to_dict(spec)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        restored = spec_from_dict(json.loads(serialized))
        assert restored.canvas_width == 800
        assert restored.canvas_height == 600
        assert restored.background.color == "#1A1A2E"
        assert len(restored.shapes) == 1
        assert restored.shapes[0].corner_radius == 16
        assert len(restored.text_zones) == 1
        assert restored.text_zones[0].font_family == "Orbitron"
        assert len(restored.image_zones) == 1
        assert restored.image_zones[0].corner_radius == 12
        assert len(restored.logo_zones) == 1

    def test_round_trip_gradient_bg(self):
        spec = TemplateSpec(
            background=Fill(
                type="linear_gradient",
                stops=[
                    GradientStop(offset=0.0, color="#0A0A23"),
                    GradientStop(offset=1.0, color="#1A1A4E"),
                ],
                angle=180.0,
            ),
        )
        d = spec_to_dict(spec)
        restored = spec_from_dict(d)
        assert restored.background.type == "linear_gradient"
        assert len(restored.background.stops) == 2
        assert restored.background.angle == 180.0

    def test_from_empty_dict(self):
        spec = spec_from_dict({})
        assert spec.canvas_width == 1280
        assert spec.canvas_height == 720
        assert spec.background.type == "solid"
        assert spec.shapes == []

    def test_dict_is_json_safe(self):
        spec = TemplateSpec()
        d = spec_to_dict(spec)
        assert isinstance(json.dumps(d), str)


# ---------------------------------------------------------------------------
# render_template_frame — basic rendering
# ---------------------------------------------------------------------------

class TestRenderTemplateFrame:
    def test_solid_background(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=100,
            background=Fill(type="solid", color="#FF0000"),
        )
        img = render_template_frame(spec)
        assert img.size == (200, 100)
        assert img.mode == "RGBA"
        # Center pixel should be red
        r, g, b, a = img.getpixel((100, 50))
        assert r == 255 and g == 0 and b == 0

    def test_shapes_rendered(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#000000"),
            shapes=[
                ShapeElement(
                    shape="rect", x=50, y=50, width=100, height=100,
                    fill=Fill(type="solid", color="#00FF00"),
                    z_order=1,
                ),
            ],
        )
        img = render_template_frame(spec)
        # Center of the green rect
        r, g, b, a = img.getpixel((100, 100))
        assert g == 255

    def test_image_zone_transparency(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#FFFFFF"),
            image_zones=[
                ImageZoneSpec(x=25, y=25, width=150, height=150),
            ],
        )
        img = render_template_frame(spec)
        # Center of the image zone should be transparent
        _, _, _, a = img.getpixel((100, 100))
        assert a == 0
        # Corner outside zone should be opaque white
        _, _, _, a = img.getpixel((10, 10))
        assert a == 255

    def test_rounded_image_zone_cutout(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#FFFFFF"),
            image_zones=[
                ImageZoneSpec(x=20, y=20, width=160, height=160, corner_radius=20),
            ],
        )
        img = render_template_frame(spec)
        # Center should be transparent
        _, _, _, a = img.getpixel((100, 100))
        assert a == 0

    def test_z_order_respected(self):
        """Higher z_order shapes are drawn on top."""
        spec = TemplateSpec(
            canvas_width=100, canvas_height=100,
            background=Fill(type="solid", color="#000000"),
            shapes=[
                ShapeElement(
                    shape="rect", x=0, y=0, width=100, height=100,
                    fill=Fill(type="solid", color="#FF0000"),
                    z_order=2,  # drawn second (on top)
                ),
                ShapeElement(
                    shape="rect", x=0, y=0, width=100, height=100,
                    fill=Fill(type="solid", color="#00FF00"),
                    z_order=1,  # drawn first
                ),
            ],
        )
        img = render_template_frame(spec)
        r, g, b, _ = img.getpixel((50, 50))
        assert r == 255 and g == 0  # Red on top

    def test_ellipse_shape(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#000000"),
            shapes=[
                ShapeElement(
                    shape="ellipse", x=25, y=25, width=150, height=150,
                    fill=Fill(type="solid", color="#0000FF"),
                ),
            ],
        )
        img = render_template_frame(spec)
        # Center of ellipse should be blue
        r, g, b, _ = img.getpixel((100, 100))
        assert b == 255

    def test_shape_with_border(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#000000"),
            shapes=[
                ShapeElement(
                    shape="rect", x=20, y=20, width=160, height=160,
                    fill=Fill(type="solid", color="#333333"),
                    border=Border(color="#FF0000", width=3),
                ),
            ],
        )
        img = render_template_frame(spec)
        assert img.size == (200, 200)

    def test_shape_opacity(self):
        spec = TemplateSpec(
            canvas_width=100, canvas_height=100,
            background=Fill(type="solid", color="#000000"),
            shapes=[
                ShapeElement(
                    shape="rect", x=0, y=0, width=100, height=100,
                    fill=Fill(type="solid", color="#FFFFFF"),
                    opacity=0.5,
                ),
            ],
        )
        img = render_template_frame(spec)
        r, g, b, _ = img.getpixel((50, 50))
        # Should be roughly half-white (around 128)
        assert 100 < r < 160

    def test_line_shape(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#000000"),
            shapes=[
                ShapeElement(
                    shape="line", x=0, y=100, x2=200, y2=100,
                    fill=Fill(type="solid", color="#FFFFFF"),
                    line_width=3,
                ),
            ],
        )
        img = render_template_frame(spec)
        # Line at y=100 should have white pixels
        r, g, b, _ = img.getpixel((100, 100))
        assert r == 255


# ---------------------------------------------------------------------------
# render_preview — labeled overlays
# ---------------------------------------------------------------------------

class TestRenderPreview:
    def test_preview_has_zone_labels(self):
        spec = TemplateSpec(
            canvas_width=400, canvas_height=300,
            background=Fill(type="solid", color="#1A1A2E"),
            image_zones=[ImageZoneSpec(x=20, y=20, width=360, height=200)],
            text_zones=[TextZoneSpec(x=20, y=230, width=360, height=50, label="title")],
            logo_zones=[LogoZoneSpec(x=20, y=10, width=60, height=30)],
        )
        img = render_preview(spec)
        assert img.size == (400, 300)
        assert img.mode == "RGBA"

    def test_preview_has_checkerboard(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#000000"),
            image_zones=[ImageZoneSpec(x=0, y=0, width=200, height=200)],
        )
        img = render_preview(spec)
        # Checkerboard should make alternating pixels different
        p1 = img.getpixel((0, 0))
        p2 = img.getpixel((16, 0))
        assert p1 != p2


# ---------------------------------------------------------------------------
# render_to_bytes
# ---------------------------------------------------------------------------

class TestRenderToBytes:
    def test_returns_bytesio(self):
        spec = TemplateSpec(canvas_width=100, canvas_height=100)
        buf = render_to_bytes(spec)
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        img = Image.open(buf)
        assert img.size == (100, 100)

    def test_preview_mode(self):
        spec = TemplateSpec(
            canvas_width=100, canvas_height=100,
            image_zones=[ImageZoneSpec(x=10, y=10, width=80, height=80)],
        )
        buf = render_to_bytes(spec, preview=True)
        assert isinstance(buf, io.BytesIO)


# ---------------------------------------------------------------------------
# save_frame
# ---------------------------------------------------------------------------

class TestSaveFrame:
    def test_saves_to_disk(self, tmp_path):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=100,
            background=Fill(type="solid", color="#FF0000"),
        )
        out_path = str(tmp_path / "frame.png")
        result = save_frame(spec, out_path)
        assert result == out_path
        assert Path(out_path).exists()
        img = Image.open(out_path)
        assert img.size == (200, 100)


# ---------------------------------------------------------------------------
# Gradient rendering
# ---------------------------------------------------------------------------

class TestGradientRendering:
    def test_linear_gradient_background(self):
        spec = TemplateSpec(
            canvas_width=100, canvas_height=100,
            background=Fill(
                type="linear_gradient",
                stops=[
                    GradientStop(offset=0.0, color="#000000"),
                    GradientStop(offset=1.0, color="#FFFFFF"),
                ],
                angle=0.0,  # top-to-bottom
            ),
        )
        img = render_template_frame(spec)
        # Top should be dark, bottom should be light
        top_pixel = img.getpixel((50, 5))
        bottom_pixel = img.getpixel((50, 95))
        assert top_pixel[0] < bottom_pixel[0]  # R channel increases

    def test_gradient_shape(self):
        spec = TemplateSpec(
            canvas_width=200, canvas_height=200,
            background=Fill(type="solid", color="#000000"),
            shapes=[
                ShapeElement(
                    shape="rect", x=0, y=0, width=200, height=200,
                    fill=Fill(
                        type="linear_gradient",
                        stops=[
                            GradientStop(offset=0.0, color="#FF0000"),
                            GradientStop(offset=1.0, color="#0000FF"),
                        ],
                        angle=0.0,
                    ),
                ),
            ],
        )
        img = render_template_frame(spec)
        assert img.size == (200, 200)


# ---------------------------------------------------------------------------
# Complex spec — integration test
# ---------------------------------------------------------------------------

class TestComplexSpec:
    def test_full_template_spec(self):
        """End-to-end: build a complex spec, render frame + preview, verify output."""
        spec = TemplateSpec(
            canvas_width=1280,
            canvas_height=720,
            background=Fill(type="solid", color="#0E0F2B"),
            shapes=[
                # Glass panel
                ShapeElement(
                    shape="rounded_rect", x=30, y=30, width=1220, height=660,
                    fill=Fill(type="solid", color="#FFFFFF10"),
                    corner_radius=20, opacity=0.3, z_order=1,
                ),
                # Accent line
                ShapeElement(
                    shape="line", x=640, y=80, x2=640, y2=640,
                    fill=Fill(type="solid", color="#72E1FF"),
                    line_width=2, z_order=2,
                ),
            ],
            image_zones=[
                ImageZoneSpec(x=50, y=80, width=560, height=560, corner_radius=16),
            ],
            text_zones=[
                TextZoneSpec(x=680, y=200, width=540, height=80, label="title",
                             font_family="Orbitron", font_size=48, color="#FFFFFF"),
                TextZoneSpec(x=680, y=300, width=540, height=50, label="subtitle",
                             font_family="Inter", font_size=20, color="#AAAAAA"),
            ],
            logo_zones=[
                LogoZoneSpec(x=50, y=20, width=100, height=50),
            ],
        )

        # Render frame
        frame = render_template_frame(spec)
        assert frame.size == (1280, 720)
        assert frame.mode == "RGBA"
        # Image zone center should be transparent
        _, _, _, a = frame.getpixel((330, 360))
        assert a == 0

        # Render preview
        preview = render_preview(spec)
        assert preview.size == (1280, 720)
        # Preview image zone center should NOT be transparent (has checkerboard)
        _, _, _, a = preview.getpixel((330, 360))
        assert a > 0

        # Serialize round-trip
        d = spec_to_dict(spec)
        restored = spec_from_dict(d)
        assert restored.canvas_width == 1280
        assert len(restored.shapes) == 2
        assert len(restored.image_zones) == 1
        assert len(restored.text_zones) == 2
        assert len(restored.logo_zones) == 1
