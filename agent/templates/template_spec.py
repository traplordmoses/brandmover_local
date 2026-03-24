"""
TemplateSpec — Rich data structures for deterministic template rendering.

Instead of using AI image generation (flux-kontext-pro) to recreate templates,
we extract a precise visual spec from reference images via Claude Vision and
render the template frame deterministically with PIL.

Key types:
    TemplateSpec     — Full template specification (canvas, background, shapes, zones)
    Fill             — Solid color or gradient fill
    ShapeElement     — Decorative shape (rect, rounded_rect, ellipse, line)
    TextZoneSpec     — Text placeholder zone with font/style info
    ImageZoneSpec    — Image placeholder zone (rendered as transparent cutout)
    LogoZoneSpec     — Logo placeholder zone
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Fill types
# ---------------------------------------------------------------------------

@dataclass
class GradientStop:
    offset: float  # 0.0–1.0
    color: str     # hex color e.g. "#FF69B4"


@dataclass
class Fill:
    type: Literal["solid", "linear_gradient", "radial_gradient"] = "solid"
    color: str = "#000000"  # used when type == "solid"
    stops: list[GradientStop] = field(default_factory=list)
    angle: float = 0.0  # degrees, for linear_gradient (0 = top-to-bottom)
    center_x: float = 0.5  # for radial_gradient (0.0–1.0)
    center_y: float = 0.5


# ---------------------------------------------------------------------------
# Border
# ---------------------------------------------------------------------------

@dataclass
class Border:
    color: str = "#FFFFFF"
    width: int = 1
    radius: int = 0  # corner radius (0 = sharp corners)


# ---------------------------------------------------------------------------
# Shape elements (decorative)
# ---------------------------------------------------------------------------

@dataclass
class ShapeElement:
    shape: Literal["rect", "rounded_rect", "ellipse", "line"] = "rect"
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    fill: Fill = field(default_factory=Fill)
    border: Border | None = None
    corner_radius: int = 0  # for rounded_rect
    opacity: float = 1.0  # 0.0–1.0
    z_order: int = 0  # higher = drawn later (on top)
    # Line-specific
    x2: int = 0
    y2: int = 0
    line_width: int = 2


# ---------------------------------------------------------------------------
# Zone specs — placeholders for content
# ---------------------------------------------------------------------------

@dataclass
class TextZoneSpec:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    label: str = "title"  # "title", "subtitle", "body", etc.
    font_family: str = ""
    font_size: int = 0
    font_weight: str = ""  # "bold", "regular", "semibold", etc.
    color: str = "#FFFFFF"
    alignment: Literal["left", "center", "right"] = "center"
    uppercase: bool = False
    outline_color: str = ""
    outline_width: int = 0
    description: str = ""
    z_order: int = 10


@dataclass
class ImageZoneSpec:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    corner_radius: int = 0
    description: str = ""
    z_order: int = 5


@dataclass
class LogoZoneSpec:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    description: str = ""
    z_order: int = 15


# ---------------------------------------------------------------------------
# TemplateSpec — the full specification
# ---------------------------------------------------------------------------

@dataclass
class TemplateSpec:
    canvas_width: int = 1280
    canvas_height: int = 720
    background: Fill = field(default_factory=Fill)
    shapes: list[ShapeElement] = field(default_factory=list)
    text_zones: list[TextZoneSpec] = field(default_factory=list)
    image_zones: list[ImageZoneSpec] = field(default_factory=list)
    logo_zones: list[LogoZoneSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialization — spec ↔ dict (JSON-safe)
# ---------------------------------------------------------------------------

def _fill_to_dict(fill: Fill) -> dict:
    d: dict = {"type": fill.type}
    if fill.type == "solid":
        d["color"] = fill.color
    else:
        d["stops"] = [{"offset": s.offset, "color": s.color} for s in fill.stops]
        if fill.type == "linear_gradient":
            d["angle"] = fill.angle
        elif fill.type == "radial_gradient":
            d["center_x"] = fill.center_x
            d["center_y"] = fill.center_y
    return d


def _fill_from_dict(d: dict) -> Fill:
    fill_type = d.get("type", "solid")
    if fill_type == "solid":
        return Fill(type="solid", color=d.get("color", "#000000"))
    stops = [GradientStop(offset=s["offset"], color=s["color"]) for s in d.get("stops", [])]
    return Fill(
        type=fill_type,
        stops=stops,
        angle=d.get("angle", 0.0),
        center_x=d.get("center_x", 0.5),
        center_y=d.get("center_y", 0.5),
    )


def _border_to_dict(border: Border | None) -> dict | None:
    if border is None:
        return None
    return {"color": border.color, "width": border.width, "radius": border.radius}


def _border_from_dict(d: dict | None) -> Border | None:
    if d is None:
        return None
    return Border(color=d.get("color", "#FFFFFF"), width=d.get("width", 1), radius=d.get("radius", 0))


def spec_to_dict(spec: TemplateSpec) -> dict:
    """Serialize a TemplateSpec to a JSON-safe dict."""
    return {
        "canvas_width": spec.canvas_width,
        "canvas_height": spec.canvas_height,
        "background": _fill_to_dict(spec.background),
        "shapes": [
            {
                "shape": s.shape,
                "x": s.x, "y": s.y, "width": s.width, "height": s.height,
                "fill": _fill_to_dict(s.fill),
                "border": _border_to_dict(s.border),
                "corner_radius": s.corner_radius,
                "opacity": s.opacity,
                "z_order": s.z_order,
                "x2": s.x2, "y2": s.y2, "line_width": s.line_width,
            }
            for s in spec.shapes
        ],
        "text_zones": [
            {
                "x": tz.x, "y": tz.y, "width": tz.width, "height": tz.height,
                "label": tz.label,
                "font_family": tz.font_family, "font_size": tz.font_size,
                "font_weight": tz.font_weight, "color": tz.color,
                "alignment": tz.alignment, "uppercase": tz.uppercase,
                "outline_color": tz.outline_color, "outline_width": tz.outline_width,
                "description": tz.description, "z_order": tz.z_order,
            }
            for tz in spec.text_zones
        ],
        "image_zones": [
            {
                "x": iz.x, "y": iz.y, "width": iz.width, "height": iz.height,
                "corner_radius": iz.corner_radius,
                "description": iz.description, "z_order": iz.z_order,
            }
            for iz in spec.image_zones
        ],
        "logo_zones": [
            {
                "x": lz.x, "y": lz.y, "width": lz.width, "height": lz.height,
                "description": lz.description, "z_order": lz.z_order,
            }
            for lz in spec.logo_zones
        ],
    }


def spec_from_dict(d: dict) -> TemplateSpec:
    """Deserialize a dict into a TemplateSpec."""
    return TemplateSpec(
        canvas_width=d.get("canvas_width", 1280),
        canvas_height=d.get("canvas_height", 720),
        background=_fill_from_dict(d.get("background", {})),
        shapes=[
            ShapeElement(
                shape=s.get("shape", "rect"),
                x=s.get("x", 0), y=s.get("y", 0),
                width=s.get("width", 0), height=s.get("height", 0),
                fill=_fill_from_dict(s.get("fill", {})),
                border=_border_from_dict(s.get("border")),
                corner_radius=s.get("corner_radius", 0),
                opacity=s.get("opacity", 1.0),
                z_order=s.get("z_order", 0),
                x2=s.get("x2", 0), y2=s.get("y2", 0),
                line_width=s.get("line_width", 2),
            )
            for s in d.get("shapes", [])
        ],
        text_zones=[
            TextZoneSpec(
                x=tz.get("x", 0), y=tz.get("y", 0),
                width=tz.get("width", 0), height=tz.get("height", 0),
                label=tz.get("label", "title"),
                font_family=tz.get("font_family", ""),
                font_size=tz.get("font_size", 0),
                font_weight=tz.get("font_weight", ""),
                color=tz.get("color", "#FFFFFF"),
                alignment=tz.get("alignment", "center"),
                uppercase=tz.get("uppercase", False),
                outline_color=tz.get("outline_color", ""),
                outline_width=tz.get("outline_width", 0),
                description=tz.get("description", ""),
                z_order=tz.get("z_order", 10),
            )
            for tz in d.get("text_zones", [])
        ],
        image_zones=[
            ImageZoneSpec(
                x=iz.get("x", 0), y=iz.get("y", 0),
                width=iz.get("width", 0), height=iz.get("height", 0),
                corner_radius=iz.get("corner_radius", 0),
                description=iz.get("description", ""),
                z_order=iz.get("z_order", 5),
            )
            for iz in d.get("image_zones", [])
        ],
        logo_zones=[
            LogoZoneSpec(
                x=lz.get("x", 0), y=lz.get("y", 0),
                width=lz.get("width", 0), height=lz.get("height", 0),
                description=lz.get("description", ""),
                z_order=lz.get("z_order", 15),
            )
            for lz in d.get("logo_zones", [])
        ],
    )
