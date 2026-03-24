"""
Deterministic PIL renderer for TemplateSpec.

Renders template frames as PNG images with exact colors, shapes, and positions.
No AI image generation — pure PIL operations.

Public API:
    render_template_frame(spec) -> Image    — Final frame with transparent cutouts
    render_preview(spec) -> Image           — Preview with labeled zone overlays
    render_to_bytes(spec, preview) -> BytesIO
    save_frame(spec, path) -> str
"""

import io
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from agent.template_spec import (
    Border,
    Fill,
    ImageZoneSpec,
    LogoZoneSpec,
    ShapeElement,
    TemplateSpec,
    TextZoneSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------

def _parse_color(hex_color: str) -> tuple[int, int, int, int]:
    """Parse a hex color string to (R, G, B, A) tuple.

    Supports #RGB, #RRGGBB, #RRGGBBAA formats.
    """
    c = hex_color.strip().lstrip("#")
    try:
        if len(c) == 3:
            r, g, b = int(c[0] * 2, 16), int(c[1] * 2, 16), int(c[2] * 2, 16)
            return (r, g, b, 255)
        if len(c) == 6:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return (r, g, b, 255)
        if len(c) == 8:
            r, g, b, a = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(c[6:8], 16)
            return (r, g, b, a)
    except ValueError:
        pass
    return (0, 0, 0, 255)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def _interpolate_stops_array(stops: list, t_array: np.ndarray) -> np.ndarray:
    """Vectorized interpolation of gradient stops over an array of t values.

    Returns an (H, W, 4) uint8 array of RGBA colors.
    """
    # Parse all stop colors once
    colors = np.array([_parse_color(s.color) for s in stops], dtype=np.float64)
    offsets = np.array([s.offset for s in stops], dtype=np.float64)

    # Initialize output
    result = np.zeros(t_array.shape + (4,), dtype=np.float64)

    # Below first stop
    result[t_array <= offsets[0]] = colors[0]
    # Above last stop
    result[t_array >= offsets[-1]] = colors[-1]

    # Interpolate between each pair of stops
    for i in range(len(stops) - 1):
        mask = (t_array > offsets[i]) & (t_array < offsets[i + 1])
        if not np.any(mask):
            continue
        span = offsets[i + 1] - offsets[i]
        if span <= 0:
            result[mask] = colors[i]
            continue
        local_t = ((t_array[mask] - offsets[i]) / span).reshape(-1, 1)
        result[mask] = colors[i] + (colors[i + 1] - colors[i]) * local_t

    return np.clip(result, 0, 255).astype(np.uint8)


def _draw_linear_gradient(
    img: Image.Image,
    x: int, y: int, w: int, h: int,
    fill: Fill,
) -> None:
    """Draw a linear gradient within the given rectangle (NumPy-vectorized)."""
    if not fill.stops or w <= 0 or h <= 0:
        return
    angle_rad = math.radians(fill.angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Create coordinate grids (normalized 0-1)
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    nx, ny = np.meshgrid(xs, ys)
    t = np.clip(nx * sin_a + ny * cos_a, 0.0, 1.0)

    rgba = _interpolate_stops_array(fill.stops, t)
    sub = Image.fromarray(rgba, "RGBA")
    img.paste(sub, (x, y), sub)


def _draw_radial_gradient(
    img: Image.Image,
    x: int, y: int, w: int, h: int,
    fill: Fill,
) -> None:
    """Draw a radial gradient within the given rectangle (NumPy-vectorized)."""
    if not fill.stops or w <= 0 or h <= 0:
        return
    cx = fill.center_x * w
    cy = fill.center_y * h
    max_dist = math.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2)
    if max_dist == 0:
        max_dist = 1

    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    px, py = np.meshgrid(xs, ys)
    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    t = np.clip(dist / max_dist, 0.0, 1.0)

    rgba = _interpolate_stops_array(fill.stops, t)
    sub = Image.fromarray(rgba, "RGBA")
    img.paste(sub, (x, y), sub)


def _interpolate_stops(
    stops: list, t: float,
) -> tuple[int, int, int, int]:
    """Interpolate color between gradient stops at position t (0.0–1.0)."""
    if not stops:
        return (0, 0, 0, 255)
    if len(stops) == 1:
        return _parse_color(stops[0].color)
    if t <= stops[0].offset:
        return _parse_color(stops[0].color)
    if t >= stops[-1].offset:
        return _parse_color(stops[-1].color)

    # Find the two surrounding stops
    for i in range(len(stops) - 1):
        if stops[i].offset <= t <= stops[i + 1].offset:
            span = stops[i + 1].offset - stops[i].offset
            local_t = (t - stops[i].offset) / span if span > 0 else 0
            c1 = _parse_color(stops[i].color)
            c2 = _parse_color(stops[i + 1].color)
            return (
                int(c1[0] + (c2[0] - c1[0]) * local_t),
                int(c1[1] + (c2[1] - c1[1]) * local_t),
                int(c1[2] + (c2[2] - c1[2]) * local_t),
                int(c1[3] + (c2[3] - c1[3]) * local_t),
            )
    return _parse_color(stops[-1].color)


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def _render_background(spec: TemplateSpec) -> Image.Image:
    """Render the background fill onto a new RGBA canvas."""
    w, h = spec.canvas_width, spec.canvas_height
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    bg = spec.background
    if bg.type == "solid":
        color = _parse_color(bg.color)
        canvas = Image.new("RGBA", (w, h), color)
    elif bg.type == "linear_gradient":
        _draw_linear_gradient(canvas, 0, 0, w, h, bg)
    elif bg.type == "radial_gradient":
        _draw_radial_gradient(canvas, 0, 0, w, h, bg)

    return canvas


# ---------------------------------------------------------------------------
# Shape rendering
# ---------------------------------------------------------------------------

def _render_shape(canvas: Image.Image, shape: ShapeElement) -> None:
    """Draw a single shape element onto the canvas."""
    if shape.opacity < 1.0:
        # Render on a temp layer and alpha-composite
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        _render_shape_direct(layer, shape)
        # Apply opacity
        alpha = layer.split()[3]
        alpha = alpha.point(lambda p: int(p * shape.opacity))
        layer.putalpha(alpha)
        canvas.alpha_composite(layer)
    else:
        _render_shape_direct(canvas, shape)


def _render_shape_direct(canvas: Image.Image, shape: ShapeElement) -> None:
    """Draw a shape directly onto canvas (no opacity handling)."""
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = shape.x, shape.y, shape.width, shape.height
    fill_color = _parse_color(shape.fill.color) if shape.fill.type == "solid" else None

    if shape.shape == "line":
        line_color = _parse_color(shape.fill.color)
        draw.line([(x, y), (shape.x2, shape.y2)], fill=line_color, width=shape.line_width)
        return

    # For gradient fills, render on a sub-image first
    if shape.fill.type in ("linear_gradient", "radial_gradient") and w > 0 and h > 0:
        sub = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        if shape.fill.type == "linear_gradient":
            _draw_linear_gradient(sub, 0, 0, w, h, shape.fill)
        else:
            _draw_radial_gradient(sub, 0, 0, w, h, shape.fill)
        # Mask to shape
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        if shape.shape == "ellipse":
            mask_draw.ellipse([0, 0, w - 1, h - 1], fill=255)
        elif shape.shape == "rounded_rect" and shape.corner_radius > 0:
            mask_draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=shape.corner_radius, fill=255)
        else:
            mask_draw.rectangle([0, 0, w - 1, h - 1], fill=255)
        sub.putalpha(mask)
        canvas.paste(sub, (x, y), sub)
    else:
        # Solid fill
        if shape.shape == "ellipse":
            draw.ellipse([x, y, x + w - 1, y + h - 1], fill=fill_color)
        elif shape.shape == "rounded_rect" and shape.corner_radius > 0:
            draw.rounded_rectangle(
                [x, y, x + w - 1, y + h - 1],
                radius=shape.corner_radius,
                fill=fill_color,
            )
        else:
            draw.rectangle([x, y, x + w - 1, y + h - 1], fill=fill_color)

    # Border
    if shape.border and shape.border.width > 0:
        border_color = _parse_color(shape.border.color)
        r = shape.border.radius or shape.corner_radius
        if shape.shape == "ellipse":
            draw.ellipse(
                [x, y, x + w - 1, y + h - 1],
                outline=border_color, width=shape.border.width,
            )
        elif r > 0:
            draw.rounded_rectangle(
                [x, y, x + w - 1, y + h - 1],
                radius=r, outline=border_color, width=shape.border.width,
            )
        else:
            draw.rectangle(
                [x, y, x + w - 1, y + h - 1],
                outline=border_color, width=shape.border.width,
            )


# ---------------------------------------------------------------------------
# Image zone cutout
# ---------------------------------------------------------------------------

def _cut_image_zone(canvas: Image.Image, zone: ImageZoneSpec) -> None:
    """Cut a transparent hole in the canvas for an image zone.

    At apply_template time, the generated image sits beneath the template
    frame and shows through these transparent areas.
    """
    x, y, w, h = zone.x, zone.y, zone.width, zone.height
    if w <= 0 or h <= 0:
        return

    # Create an alpha mask — 0 = transparent (cutout), 255 = keep
    alpha = canvas.split()[3].copy()
    mask_draw = ImageDraw.Draw(alpha)
    if zone.corner_radius > 0:
        mask_draw.rounded_rectangle(
            [x, y, x + w - 1, y + h - 1],
            radius=zone.corner_radius, fill=0,
        )
    else:
        mask_draw.rectangle([x, y, x + w - 1, y + h - 1], fill=0)
    canvas.putalpha(alpha)


# ---------------------------------------------------------------------------
# Preview overlays — labeled zone indicators
# ---------------------------------------------------------------------------

_CHECKERBOARD_SIZE = 16


def _draw_checkerboard(canvas: Image.Image, zone) -> None:
    """Draw a checkerboard pattern behind a transparent zone for preview."""
    x, y, w, h = zone.x, zone.y, zone.width, zone.height
    draw = ImageDraw.Draw(canvas)
    for cy in range(y, y + h, _CHECKERBOARD_SIZE):
        for cx in range(x, x + w, _CHECKERBOARD_SIZE):
            is_light = ((cx - x) // _CHECKERBOARD_SIZE + (cy - y) // _CHECKERBOARD_SIZE) % 2 == 0
            color = (200, 200, 200, 180) if is_light else (150, 150, 150, 180)
            bx = min(cx + _CHECKERBOARD_SIZE, x + w)
            by = min(cy + _CHECKERBOARD_SIZE, y + h)
            draw.rectangle([cx, cy, bx - 1, by - 1], fill=color)


def _draw_zone_label(canvas: Image.Image, x: int, y: int, w: int, h: int, label: str) -> None:
    """Draw a centered label inside a zone for preview."""
    draw = ImageDraw.Draw(canvas)
    # Use a reasonable font size based on zone dimensions
    font_size = max(12, min(w // len(label) if label else 20, h // 2, 32))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x + (w - tw) // 2
    ty = y + (h - th) // 2

    # Background pill
    pad = 4
    draw.rounded_rectangle(
        [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
        radius=4, fill=(0, 0, 0, 160),
    )
    draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)


# ---------------------------------------------------------------------------
# Main render functions
# ---------------------------------------------------------------------------

def render_template_frame(spec: TemplateSpec) -> Image.Image:
    """Render the final template frame PNG.

    Pipeline: background → shapes (sorted by z_order) → transparent cutouts
    for image zones. No text is rendered — text is drawn at apply_template time.
    """
    canvas = _render_background(spec)

    # Collect all shapes, sorted by z_order
    shapes = sorted(spec.shapes, key=lambda s: s.z_order)
    for shape in shapes:
        _render_shape(canvas, shape)

    # Cut transparent holes for image zones
    for iz in spec.image_zones:
        _cut_image_zone(canvas, iz)

    return canvas


def render_preview(spec: TemplateSpec) -> Image.Image:
    """Render a preview with checkerboard behind transparent zones and labeled overlays.

    Used during the conversational refinement loop so users can see zone placement.
    """
    canvas = _render_background(spec)

    # Draw shapes
    shapes = sorted(spec.shapes, key=lambda s: s.z_order)
    for shape in shapes:
        _render_shape(canvas, shape)

    # Draw checkerboard + labels for image zones
    for iz in spec.image_zones:
        _draw_checkerboard(canvas, iz)
        label = iz.description or "IMAGE"
        _draw_zone_label(canvas, iz.x, iz.y, iz.width, iz.height, label)

    # Draw labeled overlays for text zones
    for tz in spec.text_zones:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [tz.x, tz.y, tz.x + tz.width - 1, tz.y + tz.height - 1],
            fill=(255, 255, 255, 40), outline=(255, 255, 255, 120), width=1,
        )
        label = f"TEXT: {tz.label}" if tz.label else "TEXT"
        _draw_zone_label(canvas, tz.x, tz.y, tz.width, tz.height, label)

    # Draw labeled overlays for logo zones
    for lz in spec.logo_zones:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [lz.x, lz.y, lz.x + lz.width - 1, lz.y + lz.height - 1],
            fill=(255, 215, 0, 40), outline=(255, 215, 0, 120), width=1,
        )
        _draw_zone_label(canvas, lz.x, lz.y, lz.width, lz.height, "LOGO")

    return canvas


def render_to_bytes(spec: TemplateSpec, preview: bool = False) -> io.BytesIO:
    """Render and return as BytesIO PNG."""
    img = render_preview(spec) if preview else render_template_frame(spec)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


def save_frame(spec: TemplateSpec, path: str) -> str:
    """Render the template frame and save to disk. Returns the file path."""
    img = render_template_frame(spec)
    img.save(path, "PNG")
    logger.info("Saved template frame to %s (%dx%d)", path, spec.canvas_width, spec.canvas_height)
    return path
