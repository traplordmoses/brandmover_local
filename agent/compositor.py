"""
Smart Image Compositor v2.0 — BloFin BrandMover

Adaptive compositor that reacts to content type:
- Layout: SPLIT / FULL_BLEED / CENTERED
- Glow: intensity, color temperature, position
- Typography: weight, size, spacing per content type
- Logo: vector-drawn BloFin wordmark (no PNG dependency)

Public API:
composed = await compose_branded_image(draft, image_url, content_type)
# Returns io.BytesIO (PNG) or None on failure
"""

import io
import math
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 1280, 720

# ---------------------------------------------------------------------------
# Brand colors (exact from brand guidelines)
# ---------------------------------------------------------------------------
BLOFIN_ORANGE  = (255, 136, 0)  # #FF8800
BLOFIN_BLACK = (0, 0, 0)  # #000000
BLOFIN_WHITE = (255, 255, 255)  # #FFFFFF
BLOFIN_GREEN = (168, 255, 0)  # #A8FF00
BLOFIN_DARKGREY = (76, 76, 76) # #4C4C4C
BADGE_BG = BLOFIN_ORANGE
BADGE_TEXT = BLOFIN_BLACK

# ---------------------------------------------------------------------------
# Content type profiles — the "smart" part
# ---------------------------------------------------------------------------
ContentType = Literal[
    "announcement", "campaign", "market", "meme", "engagement", "advice", "default"
]

@dataclass
class CompositorProfile:
    layout: Literal["SPLIT", "FULL_BLEED", "CENTERED"]
    # Glow
    glow_color: tuple  # RGB
    glow_intensity_1: int  # alpha of outer glow layer (0-255)
    glow_intensity_2: int  # alpha of mid glow
    glow_intensity_3: int  # alpha of hot core
    glow_x_factor: float   # 0.0=left, 1.0=right
    glow_y_factor: float   # 0.0=top, 1.0=bottom
    # Typography
    title_size: int
    subtitle_size: int
    title_color: tuple
    subtitle_color: tuple
    title_uppercase: bool
    # Card
    card_border_color: tuple
    card_inner_pad: int
    # Overlay opacity for FULL_BLEED (0=transparent, 255=opaque)
    scrim_opacity: int


PROFILES: dict[ContentType, CompositorProfile] = {
    "announcement": CompositorProfile(
        layout="SPLIT",
        glow_color=BLOFIN_ORANGE,
        glow_intensity_1=30, glow_intensity_2=60, glow_intensity_3=80,
        glow_x_factor=1.0, glow_y_factor=1.0,
        title_size=62, subtitle_size=22,
        title_color=BLOFIN_ORANGE, subtitle_color=BLOFIN_WHITE,
        title_uppercase=True,
        card_border_color=(50, 50, 50), card_inner_pad=6,
        scrim_opacity=0,
    ),
    "campaign": CompositorProfile(
        layout="FULL_BLEED",
        glow_color=BLOFIN_ORANGE,
        glow_intensity_1=45, glow_intensity_2=80, glow_intensity_3=110,
        glow_x_factor=0.85, glow_y_factor=0.85,
        title_size=72, subtitle_size=24,
        title_color=BLOFIN_ORANGE, subtitle_color=BLOFIN_WHITE,
        title_uppercase=True,
        card_border_color=(60, 60, 60), card_inner_pad=0,
        scrim_opacity=160,
    ),
    "market": CompositorProfile(
        layout="SPLIT",
        glow_color=(40, 80, 140),  # cool blue tint
        glow_intensity_1=20, glow_intensity_2=40, glow_intensity_3=55,
        glow_x_factor=1.0, glow_y_factor=0.5,
        title_size=58, subtitle_size=21,
        title_color=BLOFIN_WHITE, subtitle_color=(180, 180, 180),
        title_uppercase=True,
        card_border_color=(30, 60, 100), card_inner_pad=6,
        scrim_opacity=0,
    ),
    "meme": CompositorProfile(
        layout="FULL_BLEED",
        glow_color=BLOFIN_ORANGE,
        glow_intensity_1=35, glow_intensity_2=65, glow_intensity_3=85,
        glow_x_factor=0.5, glow_y_factor=0.7,
        title_size=64, subtitle_size=26,
        title_color=BLOFIN_WHITE, subtitle_color=BLOFIN_ORANGE,
        title_uppercase=False,
        card_border_color=(50, 50, 50), card_inner_pad=0,
        scrim_opacity=140,
    ),
    "engagement": CompositorProfile(
        layout="CENTERED",
        glow_color=BLOFIN_ORANGE,
        glow_intensity_1=20, glow_intensity_2=45, glow_intensity_3=60,
        glow_x_factor=0.9, glow_y_factor=0.9,
        title_size=58, subtitle_size=22,
        title_color=BLOFIN_ORANGE, subtitle_color=BLOFIN_WHITE,
        title_uppercase=True,
        card_border_color=(45, 45, 45), card_inner_pad=8,
        scrim_opacity=0,
    ),
    "advice": CompositorProfile(
        layout="CENTERED",
        glow_color=BLOFIN_ORANGE,
        glow_intensity_1=15, glow_intensity_2=35, glow_intensity_3=50,
        glow_x_factor=0.95, glow_y_factor=0.95,
        title_size=70, subtitle_size=24,
        title_color=BLOFIN_WHITE, subtitle_color=(160, 160, 160),
        title_uppercase=True,
        card_border_color=(40, 40, 40), card_inner_pad=8,
        scrim_opacity=0,
    ),
    "default": CompositorProfile(
        layout="SPLIT",
        glow_color=BLOFIN_ORANGE,
        glow_intensity_1=25, glow_intensity_2=50, glow_intensity_3=70,
        glow_x_factor=1.0, glow_y_factor=1.0,
        title_size=62, subtitle_size=22,
        title_color=BLOFIN_ORANGE, subtitle_color=BLOFIN_WHITE,
        title_uppercase=True,
        card_border_color=(50, 50, 50), card_inner_pad=6,
        scrim_opacity=0,
    ),
}

# ---------------------------------------------------------------------------
# Font management (Poppins from Google Fonts, Aeonik fallback → Poppins Black)
# ---------------------------------------------------------------------------
_FONT_DIR = Path(__file__).resolve().parent.parent / "brand" / "assets" / "fonts"
_GOOGLE_FONTS_BASE = "https://github.com/google/fonts/raw/main/ofl/poppins/"
_FONT_FILES = {
    "black": "Poppins-Black.ttf",
    "extrabold": "Poppins-ExtraBold.ttf",
    "bold":  "Poppins-Bold.ttf",
    "semibold": "Poppins-SemiBold.ttf",
    "regular": "Poppins-Regular.ttf",
}
_SYSTEM_FONT = "/System/Library/Fonts/Avenir Next.ttc"
_font_cache: dict[str, ImageFont.FreeTypeFont] = {}


def _ensure_fonts() -> None:
    _FONT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in _FONT_FILES.items():
        path = _FONT_DIR / filename
        if path.exists():
            continue
        url = _GOOGLE_FONTS_BASE + filename
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=15)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            logger.info("Downloaded font %s", filename)
        except Exception as e:
            logger.warning("Font download failed for %s: %s", filename, e)


def _load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    key = f"{style}_{size}"
    if key in _font_cache:
        return _font_cache[key]
    font = None
    poppins_path = _FONT_DIR / _FONT_FILES.get(style, _FONT_FILES["regular"])
    if poppins_path.exists():
        try:
            font = ImageFont.truetype(str(poppins_path), size)
        except Exception:
            pass
    if font is None and Path(_SYSTEM_FONT).exists():
        try:
            idx = 8 if "bold" in style or style == "black" else 5
            font = ImageFont.truetype(_SYSTEM_FONT, size, index=idx)
        except Exception:
            pass
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ---------------------------------------------------------------------------
# Vector BloFin logo — drawn via Pillow primitives, no PNG needed
# Matches brand spec: bold white wordmark, orange diagonal slash on the B
# ---------------------------------------------------------------------------

def _draw_blofin_logo(canvas: Image.Image, x: int, y: int, height: int = 36) -> int:
    """
    Draw the BloFin wordmark at position (x, y).
    Returns the total width drawn so callers can position the badge.

    The logo: white "BloFin" text with an orange slash cut into the B.
    We approximate the slash with an orange polygon overlaid on a white B.
    """
    draw = ImageDraw.Draw(canvas)

    logo_font = _load_font("black", height)
    logo_font_rest = _load_font("bold", height)

    # Measure "B"
    b_bbox = logo_font.getbbox("B")
    b_w = b_bbox[2] - b_bbox[0]
    b_h = b_bbox[3] - b_bbox[1]
    b_top = b_bbox[1]

    # Draw "B" in white
    draw.text((x, y - b_top), "B", fill=BLOFIN_WHITE, font=logo_font)

    # Orange diagonal slash on the B — upper-right quadrant
    # Slash goes from top-right of B down-left through the upper bump
    sx = x + int(b_w * 0.30)
    sy = y - b_top + int(b_h * 0.05)
    slash_poly = [
        (sx + int(b_w * 0.55), sy),                    # top-right
        (sx + int(b_w * 0.68), sy),                    # top-right outer
        (sx + int(b_w * 0.18), sy + int(b_h * 0.50)), # bottom-left outer
        (sx + int(b_w * 0.05), sy + int(b_h * 0.50)), # bottom-left inner
    ]
    draw.polygon(slash_poly, fill=BLOFIN_ORANGE)

    # Draw "loFin" in white next to the B
    rest_x = x + b_w + 2
    rest_bbox = logo_font_rest.getbbox("loFin")
    rest_top = rest_bbox[1]
    draw.text((rest_x, y - rest_top), "loFin", fill=BLOFIN_WHITE, font=logo_font_rest)

    rest_w = rest_bbox[2] - rest_bbox[0]
    total_w = b_w + 2 + rest_w
    return total_w


def _draw_platform_badge(
    draw: ImageDraw.ImageDraw,
    x: int, y: int,
    platform: str,
    logo_height: int = 36,
) -> None:
    """Draw the orange rounded pill badge (WEB / APP / PRO)."""
    badge_font = _load_font("bold", 14)
    text = platform.upper() if platform else "WEB"
    tb = badge_font.getbbox(text)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    pad_x, pad_y = 14, 6
    badge_h = th + pad_y * 2
    # Vertically center with logo
    badge_y = y + (logo_height - badge_h) // 2
    rect = [x, badge_y, x + tw + pad_x * 2, badge_y + badge_h]
    draw.rounded_rectangle(rect, radius=6, fill=BADGE_BG)
    draw.text((x + pad_x, badge_y + pad_y - tb[1]), text, fill=BADGE_TEXT, font=badge_font)


# ---------------------------------------------------------------------------
# Background gradient — adaptive glow position & color
# ---------------------------------------------------------------------------

def _create_background(profile: CompositorProfile) -> Image.Image:
    """Pure black canvas with adaptive radial glow."""
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), BLOFIN_BLACK + (255,))

    r, g, b = profile.glow_color
    cx = int(CANVAS_W * profile.glow_x_factor)
    cy = int(CANVAS_H * profile.glow_y_factor)

    # Layer 1: large ambient
    g1 = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    d1 = ImageDraw.Draw(g1)
    d1.ellipse([cx - 700, cy - 500, cx + 700, cy + 500],
               fill=(r, g, b, profile.glow_intensity_1))
    g1 = g1.filter(ImageFilter.GaussianBlur(radius=130))
    canvas = Image.alpha_composite(canvas, g1)

    # Layer 2: mid glow
    g2 = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(g2)
    d2.ellipse([cx - 400, cy - 300, cx + 400, cy + 300],
               fill=(r, g, b, profile.glow_intensity_2))
    g2 = g2.filter(ImageFilter.GaussianBlur(radius=90))
    canvas = Image.alpha_composite(canvas, g2)

    # Layer 3: hot core
    g3 = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    d3 = ImageDraw.Draw(g3)
    d3.ellipse([cx - 180, cy - 150, cx + 180, cy + 150],
               fill=(min(r + 30, 255), g, b, profile.glow_intensity_3))
    g3 = g3.filter(ImageFilter.GaussianBlur(radius=55))
    canvas = Image.alpha_composite(canvas, g3)

    return canvas


# ---------------------------------------------------------------------------
# Rounded rectangle mask helper
# ---------------------------------------------------------------------------

def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255
    )
    return mask


# ---------------------------------------------------------------------------
# Image scaling helpers
# ---------------------------------------------------------------------------

def _crop_fill(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale image to fill target dimensions, center crop — no grey bars."""
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if src_ratio > tgt_ratio:
        new_h = target_h
        new_w = int(new_h * src_ratio)
    else:
        new_w = target_w
        new_h = int(new_w / src_ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    ox = (new_w - target_w) // 2
    oy = (new_h - target_h) // 2
    return img.crop((ox, oy, ox + target_w, oy + target_h))


# ---------------------------------------------------------------------------
# Word wrap
# ---------------------------------------------------------------------------

def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines, current = [], words[0]
    for word in words[1:]:
        test = current + " " + word
        bb = font.getbbox(test)
        if (bb[2] - bb[0]) <= max_w:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _text_block_height(
    title: str, subtitle: str, profile: CompositorProfile, max_w: int
) -> int:
    tf = _load_font("black", profile.title_size)
    sf = _load_font("regular", profile.subtitle_size)
    h = 0
    if title:
        for line in _wrap(title, tf, max_w):
            bb = tf.getbbox(line)
            h += (bb[3] - bb[1]) + 10
    if title and subtitle:
        h += 24
    if subtitle:
        for line in _wrap(subtitle, sf, max_w):
            bb = sf.getbbox(line)
            h += (bb[3] - bb[1]) + 8
    return h


# ---------------------------------------------------------------------------
# Layout renderers
# ---------------------------------------------------------------------------

LOGO_H = 36
LOGO_X, LOGO_Y = 50, 32
CARD_RADIUS = 16


def _render_split(
    canvas: Image.Image,
    feature_img: Image.Image | None,
    title: str,
    subtitle: str,
    platform: str,
    profile: CompositorProfile,
) -> None:
    """Classic left-card / right-text layout."""
    draw = ImageDraw.Draw(canvas)

    # Logo
    logo_w = _draw_blofin_logo(canvas, LOGO_X, LOGO_Y, LOGO_H)
    _draw_platform_badge(draw, LOGO_X + logo_w + 16, LOGO_Y, platform, LOGO_H)

    # Card geometry — tight margins, image fills the card completely
    card_x = 44
    card_y = 88
    card_w = 570
    card_h = CANVAS_H - card_y - 40  # 592px

    # Draw border (1px, as separate slightly-larger rect)
    bx, by = card_x - 1, card_y - 1
    border_img = Image.new("RGBA", (card_w + 2, card_h + 2), profile.card_border_color + (200,))
    border_mask = _rounded_mask(border_img.size, CARD_RADIUS + 1)
    canvas.paste(border_img, (bx, by), border_mask)

    # Card background
    card_bg = Image.new("RGBA", (card_w, card_h), (14, 14, 14, 255))
    card_mask = _rounded_mask((card_w, card_h), CARD_RADIUS)

    if feature_img is not None:
        pad = profile.card_inner_pad
        inner_w = card_w - pad * 2
        inner_h = card_h - pad * 2
        filled = _crop_fill(feature_img.convert("RGBA"), inner_w, inner_h)
        img_mask = _rounded_mask((inner_w, inner_h), max(CARD_RADIUS - 4, 8))
        card_bg.paste(filled, (pad, pad), img_mask)

    canvas.paste(card_bg, (card_x, card_y), card_mask)

    # Text — right panel
    text_x = card_x + card_w + 48
    text_max_w = CANVAS_W - text_x - 44
    tf = _load_font("black", profile.title_size)
    sf = _load_font("regular", profile.subtitle_size)
    block_h = _text_block_height(title, subtitle, profile, text_max_w)

    text_y = card_y + (card_h - block_h) // 2
    t = title.upper() if profile.title_uppercase else title

    if t:
        for line in _wrap(t, tf, text_max_w):
            bb = tf.getbbox(line)
            draw.text((text_x, text_y - bb[1]), line,
                       fill=profile.title_color, font=tf)
            text_y += (bb[3] - bb[1]) + 10
        text_y += 24

    if subtitle:
        for line in _wrap(subtitle, sf, text_max_w):
            bb = sf.getbbox(line)
            draw.text((text_x, text_y - bb[1]), line,
                       fill=profile.subtitle_color, font=sf)
            text_y += (bb[3] - bb[1]) + 8


def _render_full_bleed(
    canvas: Image.Image,
    feature_img: Image.Image | None,
    title: str,
    subtitle: str,
    platform: str,
    profile: CompositorProfile,
) -> None:
    """Full-canvas image with dark gradient scrim and text overlay."""
    draw = ImageDraw.Draw(canvas)

    # Paste feature image as full background
    if feature_img is not None:
        filled = _crop_fill(feature_img.convert("RGBA"), CANVAS_W, CANVAS_H)
        canvas.paste(filled, (0, 0))

    # Gradient scrim — left-heavy dark overlay so text is readable
    scrim = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)

    # Vertical bands from left (dark) to right (transparent)
    steps = 80
    for i in range(steps):
        alpha = int(profile.scrim_opacity * (1 - i / steps) ** 0.6)
        x0 = int(CANVAS_W * i / steps)
        x1 = int(CANVAS_W * (i + 1) / steps)
        sd.rectangle([x0, 0, x1, CANVAS_H], fill=(0, 0, 0, alpha))

    # Bottom scrim always for logo readability
    for i in range(30):
        alpha = int(180 * (i / 30))
        y0 = CANVAS_H - int(CANVAS_H * 0.35 * i / 30)
        sd.rectangle([0, y0, CANVAS_W, CANVAS_H], fill=(0, 0, 0, 12))

    canvas.alpha_composite(scrim)

    # Logo top-left
    logo_w = _draw_blofin_logo(canvas, LOGO_X, LOGO_Y, LOGO_H)
    _draw_platform_badge(draw, LOGO_X + logo_w + 16, LOGO_Y, platform, LOGO_H)

    # Text — left side, vertically centered-ish
    text_x = 56
    text_max_w = int(CANVAS_W * 0.52)
    tf = _load_font("black", profile.title_size)
    sf = _load_font("regular", profile.subtitle_size)
    block_h = _text_block_height(title, subtitle, profile, text_max_w)
    text_y = (CANVAS_H - block_h) // 2

    t = title.upper() if profile.title_uppercase else title
    if t:
        for line in _wrap(t, tf, text_max_w):
            bb = tf.getbbox(line)
            draw.text((text_x, text_y - bb[1]), line,
                       fill=profile.title_color, font=tf)
            text_y += (bb[3] - bb[1]) + 10
        text_y += 24

    if subtitle:
        for line in _wrap(subtitle, sf, text_max_w):
            bb = sf.getbbox(line)
            draw.text((text_x, text_y - bb[1]), line,
                       fill=profile.subtitle_color, font=sf)
            text_y += (bb[3] - bb[1]) + 8


def _render_centered(
    canvas: Image.Image,
    feature_img: Image.Image | None,
    title: str,
    subtitle: str,
    platform: str,
    profile: CompositorProfile,
) -> None:
    """Editorial: image centered top half, text below — clean and minimal."""
    draw = ImageDraw.Draw(canvas)

    # Logo top-left
    logo_w = _draw_blofin_logo(canvas, LOGO_X, LOGO_Y, LOGO_H)
    _draw_platform_badge(draw, LOGO_X + logo_w + 16, LOGO_Y, platform, LOGO_H)

    # Image card — centered, upper portion
    img_w = 560
    img_h = 360
    img_x = (CANVAS_W - img_w) // 2
    img_y = 90

    border_img = Image.new("RGBA", (img_w + 2, img_h + 2), profile.card_border_color + (180,))
    bm = _rounded_mask(border_img.size, CARD_RADIUS + 1)
    canvas.paste(border_img, (img_x - 1, img_y - 1), bm)

    card = Image.new("RGBA", (img_w, img_h), (14, 14, 14, 255))
    cm = _rounded_mask((img_w, img_h), CARD_RADIUS)

    if feature_img is not None:
        pad = profile.card_inner_pad
        iw, ih = img_w - pad * 2, img_h - pad * 2
        filled = _crop_fill(feature_img.convert("RGBA"), iw, ih)
        im = _rounded_mask((iw, ih), max(CARD_RADIUS - 4, 8))
        card.paste(filled, (pad, pad), im)

    canvas.paste(card, (img_x, img_y), cm)

    # Text below image, centered
    text_y = img_y + img_h + 32
    text_max_w = CANVAS_W - 120
    tf = _load_font("black", profile.title_size)
    sf = _load_font("regular", profile.subtitle_size)

    t = title.upper() if profile.title_uppercase else title
    if t:
        for line in _wrap(t, tf, text_max_w):
            bb = tf.getbbox(line)
            lw = bb[2] - bb[0]
            tx = (CANVAS_W - lw) // 2
            draw.text((tx, text_y - bb[1]), line,
                       fill=profile.title_color, font=tf)
            text_y += (bb[3] - bb[1]) + 8
        text_y += 16

    if subtitle:
        for line in _wrap(subtitle, sf, text_max_w):
            bb = sf.getbbox(line)
            lw = bb[2] - bb[0]
            tx = (CANVAS_W - lw) // 2
            draw.text((tx, text_y - bb[1]), line,
                       fill=profile.subtitle_color, font=sf)
            text_y += (bb[3] - bb[1]) + 6


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

async def _download_image(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        logger.warning("Image download failed (%s): %s", url[:80], e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def compose_branded_image(
    draft: dict,
    image_url: str | None,
    content_type: str = "default",
) -> io.BytesIO | None:
    """
    Compose a 1280x720 branded image.

    Args:
        draft: Dict with title, subtitle, platform keys.
        image_url: URL of AI-generated image.
        content_type: One of announcement/campaign/market/meme/engagement/advice/default.

    Returns:
        io.BytesIO PNG or None (caller falls back to raw image).
    """
    title = (draft.get("title") or "").strip()
    subtitle = (draft.get("subtitle") or "").strip()
    platform = (draft.get("platform") or "WEB").strip()

    if not title and not subtitle:
        return None

    profile = PROFILES.get(content_type, PROFILES["default"])

    try:
        _ensure_fonts()

        feature_img = None
        if image_url:
            raw = await _download_image(image_url)
            if raw:
                feature_img = Image.open(io.BytesIO(raw)).convert("RGBA")

        # Build adaptive background
        canvas = _create_background(profile)

        # Route to layout renderer
        if profile.layout == "SPLIT":
            _render_split(canvas, feature_img, title, subtitle, platform, profile)
        elif profile.layout == "FULL_BLEED":
            _render_full_bleed(canvas, feature_img, title, subtitle, platform, profile)
        elif profile.layout == "CENTERED":
            _render_centered(canvas, feature_img, title, subtitle, platform, profile)

        out = io.BytesIO()
        canvas.convert("RGB").save(out, format="PNG", optimize=True)
        out.seek(0)
        return out

    except Exception as e:
        logger.error("Composition failed: %s", e, exc_info=True)
        return None
