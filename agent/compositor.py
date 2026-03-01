"""
Smart Image Compositor — BrandMover

Public API:
    composed = await compose_branded_image(draft, image_url, content_type)
"""

import io
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Brand colors and layout — resolved at runtime from brand/guidelines.md
from agent import compositor_config as _brand_cfg
from agent.content_types import COMPOSITOR_PROFILE_MAP as _PROFILE_MAP


def _layout():
    """Return layout dimensions from brand config."""
    return _brand_cfg.get_config()

def _c(role: str, fallback: tuple) -> tuple:
    """Shorthand for brand color lookup with hardcoded fallback."""
    return _brand_cfg.get_color_rgb(role, fallback)

ContentType = Literal[
    "announcement", "campaign", "market", "meme", "engagement", "advice", "default"
]

@dataclass
class CompositorProfile:
    layout: Literal["SPLIT", "FULL_BLEED", "CENTERED"]
    glow_color: tuple
    glow_intensity_1: int
    glow_intensity_2: int
    glow_intensity_3: int
    glow_x_factor: float
    glow_y_factor: float
    title_size: int
    subtitle_size: int
    title_color: tuple
    subtitle_color: tuple
    title_uppercase: bool
    card_inner_pad: int
    scrim_opacity: int


_profiles_cache: dict[str, CompositorProfile] | None = None
_profiles_hash: str = ""


def _get_profiles() -> dict[str, CompositorProfile]:
    """Build PROFILES dict from brand config, cached until config changes."""
    global _profiles_cache, _profiles_hash
    cfg_hash = _brand_cfg.get_config().raw_hash
    if _profiles_cache is not None and _profiles_hash == cfg_hash:
        return _profiles_cache

    PRIMARY = _c("primary", (114, 225, 255))
    TEXT = _c("text", (255, 255, 255))
    BG_ALT = _c("background_alt", (11, 46, 78))

    _profiles_cache = {
        "announcement": CompositorProfile(
            layout="SPLIT",
            glow_color=PRIMARY,
            glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
            glow_x_factor=1.0, glow_y_factor=1.0,
            title_size=68, subtitle_size=21,
            title_color=TEXT, subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        ),
        "campaign": CompositorProfile(
            layout="FULL_BLEED",
            glow_color=PRIMARY,
            glow_intensity_1=45, glow_intensity_2=80, glow_intensity_3=110,
            glow_x_factor=0.85, glow_y_factor=0.85,
            title_size=72, subtitle_size=22,
            title_color=TEXT, subtitle_color=TEXT,
            title_uppercase=True, card_inner_pad=0, scrim_opacity=160,
        ),
        "market": CompositorProfile(
            layout="SPLIT",
            glow_color=BG_ALT,
            glow_intensity_1=20, glow_intensity_2=40, glow_intensity_3=55,
            glow_x_factor=1.0, glow_y_factor=0.5,
            title_size=62, subtitle_size=20,
            title_color=TEXT, subtitle_color=(160, 160, 160),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        ),
        "meme": CompositorProfile(
            layout="FULL_BLEED",
            glow_color=PRIMARY,
            glow_intensity_1=35, glow_intensity_2=65, glow_intensity_3=85,
            glow_x_factor=0.5, glow_y_factor=0.7,
            title_size=64, subtitle_size=24,
            title_color=TEXT, subtitle_color=PRIMARY,
            title_uppercase=False, card_inner_pad=0, scrim_opacity=140,
        ),
        "engagement": CompositorProfile(
            layout="CENTERED",
            glow_color=PRIMARY,
            glow_intensity_1=20, glow_intensity_2=45, glow_intensity_3=60,
            glow_x_factor=0.9, glow_y_factor=0.9,
            title_size=62, subtitle_size=21,
            title_color=TEXT, subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        ),
        "advice": CompositorProfile(
            layout="CENTERED",
            glow_color=PRIMARY,
            glow_intensity_1=15, glow_intensity_2=35, glow_intensity_3=50,
            glow_x_factor=0.95, glow_y_factor=0.95,
            title_size=70, subtitle_size=22,
            title_color=TEXT, subtitle_color=(150, 150, 150),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        ),
        "default": CompositorProfile(
            layout="SPLIT",
            glow_color=PRIMARY,
            glow_intensity_1=28, glow_intensity_2=55, glow_intensity_3=75,
            glow_x_factor=1.0, glow_y_factor=1.0,
            title_size=68, subtitle_size=21,
            title_color=TEXT, subtitle_color=(170, 170, 170),
            title_uppercase=True, card_inner_pad=0, scrim_opacity=0,
        ),
    }
    _profiles_hash = cfg_hash
    return _profiles_cache

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
_FONT_DIR = Path(__file__).resolve().parent.parent / "brand" / "assets" / "fonts"
_SYSTEM_FONT = "/System/Library/Fonts/Avenir Next.ttc"
_font_cache: dict[str, ImageFont.FreeTypeFont] = {}


def clear_font_cache() -> None:
    """Clear cached font objects. Called after brand change via /setup."""
    _font_cache.clear()


def _ensure_fonts() -> None:
    _FONT_DIR.mkdir(parents=True, exist_ok=True)
    font_map = _brand_cfg.get_font_map()
    seen: set[str] = set()
    for style, info in font_map.items():
        filename = info["filename"]
        if filename in seen:
            continue
        seen.add(filename)
        path = _FONT_DIR / filename
        if path.exists():
            continue
        try:
            resp = httpx.get(info["url"], follow_redirects=True, timeout=15)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        except Exception as e:
            logger.warning("Font download failed %s: %s", filename, e)


def _load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    key = f"{style}_{size}"
    if key in _font_cache:
        return _font_cache[key]
    font = None
    font_map = _brand_cfg.get_font_map()
    info = font_map.get(style, font_map["regular"])
    p = _FONT_DIR / info["filename"]
    if p.exists():
        try:
            font = ImageFont.truetype(str(p), size)
        except Exception as e:
            logger.debug("Font load failed for %s: %s", p, e)
    if font is None and Path(_SYSTEM_FONT).exists():
        try:
            idx = 8 if ("bold" in style or style == "black") else 5
            font = ImageFont.truetype(_SYSTEM_FONT, size, index=idx)
        except Exception as e:
            logger.debug("System font load failed: %s", e)
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def _fit_font_to_width(text: str, style: str, start_size: int, max_w: int, tracking: int = 1) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > 28:
        font = _load_font(style, size)
        bb = font.getbbox(text)
        total_w = (bb[2] - bb[0]) + (len(text) * tracking)
        if total_w <= max_w:
            return font
        size -= 2
    return _load_font(style, 28)


# ---------------------------------------------------------------------------
# Logo — load brand logo PNG
# ---------------------------------------------------------------------------
_LOGO_PNG = Path(__file__).resolve().parent.parent / "brand" / "assets" / "logo.png"
logger.info("Logo PNG path resolved to: %s (exists=%s)", _LOGO_PNG, _LOGO_PNG.exists())
_logo_cache: dict[int, tuple] = {}


def _load_logo_png(height: int):
    if height in _logo_cache:
        return _logo_cache[height]
    if not _LOGO_PNG.exists():
        logger.warning("Logo PNG not found at %s — place your logo as brand/assets/logo.png", _LOGO_PNG)
        return None
    try:
        logo = Image.open(_LOGO_PNG).convert("RGBA")
        w = int(height * logo.width / logo.height)
        logo = logo.resize((w, height), Image.LANCZOS)
        _logo_cache[height] = (logo, w)
        return logo, w
    except Exception as e:
        logger.warning("Logo PNG load failed: %s", e)
        return None


def _draw_brand_logo(canvas: Image.Image, x: int, y: int, height: int = 44) -> int:
    """Draw brand logo on canvas. Returns width, or 0 if logo not found."""
    result = _load_logo_png(height)
    if result:
        logo_img, logo_w = result
        canvas.paste(logo_img, (x, y), logo_img)
        return logo_w
    return 0


def _draw_platform_badge(
    draw: ImageDraw.ImageDraw, x: int, y: int,
    platform: str, logo_height: int = 44,
) -> None:
    """Outline-style badge — brand color border, transparent fill, brand color text."""
    cfg = _brand_cfg.get_config()
    if cfg.badge_text is None:
        return
    bf   = _load_font("bold", 15)
    text = cfg.badge_text.upper()
    tb   = bf.getbbox(text)
    tw, th = tb[2]-tb[0], tb[3]-tb[1]
    px, py = 16, 7
    bh     = th + py * 2
    by_    = y + (logo_height - bh) // 2
    rect   = [x, by_, x + tw + px*2, by_ + bh]
    badge_color = _c("primary", (114, 225, 255))
    draw.rounded_rectangle(rect, radius=7, outline=badge_color, width=2)
    draw.text((x + px, by_ + py - tb[1]), text, fill=badge_color, font=bf)


# ---------------------------------------------------------------------------
# Tracked text (letter spacing)
# ---------------------------------------------------------------------------

def _draw_tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    tracking: int = 3,
) -> None:
    """Draw text char-by-char with extra letter spacing."""
    x, y = xy
    for char in text:
        draw.text((x, y), char, fill=fill, font=font)
        bb = font.getbbox(char)
        x += (bb[2] - bb[0]) + tracking


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def _create_background(profile: CompositorProfile) -> Image.Image:
    """Create a branded background with glass-morphism orbs.

    Colors and effects read from brand config. Glow geometry is relative
    to canvas dimensions so layouts scale with config changes.
    """
    cfg = _brand_cfg.get_config()
    cw, ch = cfg.canvas_width, cfg.canvas_height
    canvas = Image.new("RGBA", (cw, ch), _c("background", (14, 15, 43)) + (255,))
    r, g, b = profile.glow_color

    # Scale factors relative to the reference 1280x720 canvas
    sx, sy = cw / 1280, ch / 720

    # --- Primary glow (from original profile) ---
    cx = int(cw * profile.glow_x_factor + 140 * sx)
    cy = int(ch * profile.glow_y_factor + 90 * sy)

    def _glow(x, y, size, color, alpha, blur):
        lay = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        cr, cg, cb = color
        ImageDraw.Draw(lay).ellipse(
            [x - size, y - int(size * 0.75), x + size, y + int(size * 0.75)],
            fill=(cr, cg, cb, alpha),
        )
        return lay.filter(ImageFilter.GaussianBlur(radius=blur))

    # Base glow — softer spread (sizes scale with canvas)
    canvas = Image.alpha_composite(canvas, _glow(cx, cy, int(500 * sx), (r, g, b), profile.glow_intensity_1, int(110 * sx)))
    canvas = Image.alpha_composite(canvas, _glow(cx, cy, int(280 * sx), (r, g, b), profile.glow_intensity_2, int(70 * sx)))

    # --- Soft glow orbs — colors from brand palette ---
    # Positions and radii are specified as fractions of the reference 1280x720 canvas
    # and scaled to the actual canvas dimensions.
    _ab = cfg.orb_alpha_base
    _ORB_SPECS = [
        # (x_frac, y_frac, ref_radius, color, center_alpha, ref_blur)
        (0.141, 0.194, 110, _c("primary",  (114, 225, 255)), _ab + 4, 55),   # primary orb
        (0.820, 0.722,  90, _c("accent_2", (205, 183, 255)), _ab,     45),   # accent_2 orb
        (0.703, 0.139,  70, _c("accent_3", (143, 170, 242)), _ab + 2, 40),   # accent_3 orb
        (0.273, 0.778,  85, _c("accent_1", (255, 179, 217)), max(_ab - 6, 2), 50),   # accent_1 orb
        (0.500, 0.500, 130, _c("primary",  (114, 225, 255)), max(_ab - 8, 2), 80),   # primary wash
        (0.063, 0.556,  60, _c("accent_3", (143, 170, 242)), _ab - 2, 35),   # accent_3 edge
        (0.922, 0.361,  55, _c("accent_2", (205, 183, 255)), _ab - 4, 30),   # accent_2 edge
    ]

    for xf, yf, ref_rad, color, alpha, ref_blur in _ORB_SPECS[:cfg.orb_count]:
        ox, oy = int(xf * cw), int(yf * ch)
        rad = int(ref_rad * sx)
        blur = int(ref_blur * sx)
        # Outer glow
        canvas = Image.alpha_composite(canvas, _glow(ox, oy, rad, color, alpha, blur))
        # Inner bright core (smaller, brighter)
        canvas = Image.alpha_composite(canvas, _glow(ox, oy, rad // 3, color, alpha * 2, blur // 3))

    # --- Glass morphism tint overlay — faint frosted panel ---
    gi = cfg.glass_inset  # (left, top, right, bottom)
    frost = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    fd = ImageDraw.Draw(frost)
    fd.rounded_rectangle(
        [gi[0], gi[1], cw - gi[2], ch - gi[3]],
        radius=cfg.glass_radius,
        fill=(255, 255, 255, cfg.glass_opacity),
    )
    frost = frost.filter(ImageFilter.GaussianBlur(radius=cfg.glass_blur))
    canvas = Image.alpha_composite(canvas, frost)

    return canvas


# ---------------------------------------------------------------------------
# Image blend — elliptical fade into black, no card/border
# ---------------------------------------------------------------------------

def _blend_image_into_canvas(
    canvas: Image.Image,
    feature_img: Image.Image,
    x: int, y: int, w: int, h: int,
    fade_strength: float = 1.4,
) -> None:
    filled = _crop_fill(feature_img.convert("RGB"), w, h)

    mask = Image.new("L", (w, h), 0)
    for i in range(100):
        t     = i / 100
        alpha = int(255 * (t ** fade_strength))
        pad_x = int((1 - t) * w * 0.52)
        pad_y = int((1 - t) * h * 0.38)
        if pad_x < w // 2 and pad_y < h // 2:
            ImageDraw.Draw(mask).ellipse(
                [pad_x, pad_y, w - pad_x, h - pad_y], fill=alpha
            )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=w // 10))

    black_base = Image.new("RGB", (w, h), (0, 0, 0))
    black_base.paste(filled, (0, 0), mask)

    rgb = canvas.convert("RGB")
    rgb.paste(black_base, (x, y))
    canvas.paste(rgb.convert("RGBA"), (0, 0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_fill(img: Image.Image, tw: int, th: int) -> Image.Image:
    sr = img.width / img.height
    tr = tw / th
    if sr > tr:
        nw, nh = int(th * sr), th
    else:
        nw, nh = tw, int(tw / sr)
    img = img.resize((nw, nh), Image.LANCZOS)
    ox, oy = (nw - tw) // 2, (nh - th) // 2
    return img.crop((ox, oy, ox + tw, oy + th))


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines, cur = [], words[0]
    for w in words[1:]:
        test = cur + " " + w
        if (font.getbbox(test)[2] - font.getbbox(test)[0]) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _block_h(title: str, subtitle: str, profile: CompositorProfile, max_w: int) -> int:
    tf = _load_font("black",    profile.title_size)
    sf = _load_font("semibold", profile.subtitle_size)
    h  = 0
    if title:
        bb = tf.getbbox(title)
        h += (bb[3] - bb[1]) + 28
    if subtitle:
        for line in _wrap(subtitle, sf, max_w):
            bb = sf.getbbox(line)
            h += (bb[3] - bb[1]) + 8
    return h


def _logo_xy():
    """Return (logo_x, logo_y, logo_h) from config."""
    cfg = _layout()
    return cfg.logo_padding[0], cfg.logo_padding[1], cfg.logo_height


def _img_area():
    """Return (img_x, img_y, img_w, img_h) from config."""
    cfg = _layout()
    return cfg.image_x, cfg.image_y, cfg.image_width, cfg.canvas_height - cfg.image_y - cfg.image_bottom_margin


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_split(
    canvas, feature_img, title, subtitle, platform, profile
):
    draw = ImageDraw.Draw(canvas)
    cfg = _layout()
    cw = cfg.canvas_width
    lx, ly, lh = _logo_xy()
    ix, iy, iw, ih = _img_area()

    logo_w = _draw_brand_logo(canvas, lx, ly, lh)
    _draw_platform_badge(draw, lx + logo_w + 10, ly, platform, lh)

    if feature_img:
        _blend_image_into_canvas(canvas, feature_img, ix, iy, iw, ih,
                                 fade_strength=1.4)

    text_x    = ix + iw + 48
    text_max_w = cw - text_x - 44
    t  = title.upper() if profile.title_uppercase else title
    tf = _fit_font_to_width(t, "black", profile.title_size, text_max_w) if t \
         else _load_font("black", profile.title_size)
    sf = _load_font("semibold", profile.subtitle_size)

    bh     = _block_h(t, subtitle, profile, text_max_w)
    text_y = iy + (ih - bh) // 2

    if t:
        bb = tf.getbbox(t)
        _draw_tracked(draw, (text_x, text_y - bb[1]), t, tf, profile.title_color, tracking=1)
        text_y += (bb[3] - bb[1]) + 30

    if subtitle:
        for line in _wrap(subtitle, sf, text_max_w):
            bb = sf.getbbox(line)
            draw.text((text_x, text_y - bb[1]), line, fill=profile.subtitle_color, font=sf)
            text_y += (bb[3] - bb[1]) + 8


def _render_full_bleed(
    canvas, feature_img, title, subtitle, platform, profile
):
    draw = ImageDraw.Draw(canvas)
    cfg = _layout()
    cw, ch = cfg.canvas_width, cfg.canvas_height
    lx, ly, lh = _logo_xy()

    if feature_img:
        filled = _crop_fill(feature_img.convert("RGBA"), cw, ch)
        canvas.paste(filled, (0, 0))

    scrim = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    sd    = ImageDraw.Draw(scrim)
    for i in range(80):
        a  = int(profile.scrim_opacity * (1 - i / 80) ** 0.6)
        x0 = int(cw * i / 80)
        x1 = int(cw * (i+1) / 80)
        sd.rectangle([x0, 0, x1, ch], fill=(0, 0, 0, a))
    canvas.alpha_composite(scrim)

    logo_w = _draw_brand_logo(canvas, lx, ly, lh)
    _draw_platform_badge(draw, lx + logo_w + 10, ly, platform, lh)

    text_x    = 56
    text_max_w = int(cw * 0.52)
    t  = title.upper() if profile.title_uppercase else title
    tf = _fit_font_to_width(t, "black", profile.title_size, text_max_w) if t \
         else _load_font("black", profile.title_size)
    sf = _load_font("semibold", profile.subtitle_size)

    bh     = _block_h(t, subtitle, profile, text_max_w)
    text_y = (ch - bh) // 2

    if t:
        bb = tf.getbbox(t)
        _draw_tracked(draw, (text_x, text_y - bb[1]), t, tf, profile.title_color, tracking=1)
        text_y += (bb[3] - bb[1]) + 30

    if subtitle:
        for line in _wrap(subtitle, sf, text_max_w):
            bb = sf.getbbox(line)
            draw.text((text_x, text_y - bb[1]), line, fill=profile.subtitle_color, font=sf)
            text_y += (bb[3] - bb[1]) + 8


def _render_centered(
    canvas, feature_img, title, subtitle, platform, profile
):
    draw = ImageDraw.Draw(canvas)
    cfg = _layout()
    cw, ch = cfg.canvas_width, cfg.canvas_height
    lx, ly, lh = _logo_xy()

    logo_w = _draw_brand_logo(canvas, lx, ly, lh)
    _draw_platform_badge(draw, lx + logo_w + 10, ly, platform, lh)

    # Centered image area — proportional to canvas
    iw = int(cw * 600 / 1280)
    ih = int(ch * 380 / 720)
    ix = (cw - iw) // 2
    iy = int(ch * 85 / 720)

    if feature_img:
        _blend_image_into_canvas(canvas, feature_img, ix, iy, iw, ih, fade_strength=1.3)

    text_y    = iy + ih + 28
    text_max_w = cw - 120
    t  = title.upper() if profile.title_uppercase else title
    tf = _fit_font_to_width(t, "black", profile.title_size, text_max_w) if t \
         else _load_font("black", profile.title_size)
    sf = _load_font("semibold", profile.subtitle_size)

    if t:
        bb = tf.getbbox(t)
        lw = bb[2] - bb[0]
        _draw_tracked(draw, ((cw - lw) // 2, text_y - bb[1]), t, tf, profile.title_color, tracking=1)
        text_y += (bb[3] - bb[1]) + 24

    if subtitle:
        for line in _wrap(subtitle, sf, text_max_w):
            bb = sf.getbbox(line)
            lw = bb[2] - bb[0]
            draw.text(((cw - lw) // 2, text_y - bb[1]), line,
                      fill=profile.subtitle_color, font=sf)
            text_y += (bb[3] - bb[1]) + 6


# ---------------------------------------------------------------------------
# Download + public API
# ---------------------------------------------------------------------------

async def _download_image(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.content
    except Exception as e:
        logger.warning("Image download failed: %s", e)
        return None


async def compose_branded_image(
    draft: dict,
    image_url: str | None,
    content_type: str = "default",
) -> io.BytesIO | None:
    title    = (draft.get("title")    or "").strip()
    subtitle = (draft.get("subtitle") or "").strip()
    platform = (draft.get("platform") or "").strip()

    if not title and not subtitle:
        return None

    profiles = _get_profiles()
    cfg = _layout()
    # Config layout_mappings override hardcoded COMPOSITOR_PROFILE_MAP
    profile_key = cfg.layout_mappings.get(content_type) or _PROFILE_MAP.get(content_type, "default")
    profile = profiles.get(profile_key, profiles["default"])

    try:
        _ensure_fonts()

        feature_img = None
        if image_url:
            raw = await _download_image(image_url)
            if raw:
                feature_img = Image.open(io.BytesIO(raw)).convert("RGBA")

        canvas = _create_background(profile)

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