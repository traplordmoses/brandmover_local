"""
Text Card Renderer for video promos.

Renders a sequence of PNG frames showing a glassmorphism card
with typewriter-animated text. Each frame is a transparent PNG
that gets composited over the background.

The render flow:
1. Pre-calculate all text layout (line wrapping, positions)
2. Build a timeline: which characters are visible at each frame
3. Render each frame as a transparent PNG with:
   - Frosted glass card (semi-transparent rounded rect with border)
   - Title text (always visible after fade-in)
   - Subtitle text (always visible after fade-in)
   - Conversation lines (typewriter effect, line by line)
   - Brand logo + prefix (always visible)
   - Blinking cursor at current typing position
"""

import os
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from modules.video_promo.config_schema import VideoPromoConfig, TextCardConfig, ConversationLine

logger = logging.getLogger(__name__)


# ── Color parsing ───────────────────────────────────────────────────────────

def parse_color(color_str: str) -> tuple[int, int, int, int]:
    """Parse color string to RGBA tuple. Supports hex (#FFF, #FFFFFF) and rgba()."""
    color_str = color_str.strip()

    if color_str.startswith("rgba("):
        parts = color_str[5:-1].split(",")
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        a = int(float(parts[3].strip()) * 255)
        return (r, g, b, a)
    elif color_str.startswith("rgb("):
        parts = color_str[4:-1].split(",")
        return (int(parts[0]), int(parts[1]), int(parts[2]), 255)
    elif color_str.startswith("#"):
        hex_str = color_str[1:]
        if len(hex_str) == 3:
            hex_str = "".join(c * 2 for c in hex_str)
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        a = int(hex_str[6:8], 16) if len(hex_str) == 8 else 255
        return (r, g, b, a)
    else:
        raise ValueError(f"Cannot parse color: {color_str}")


# ── Font loading ────────────────────────────────────────────────────────────

def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font, falling back to a default if not found."""
    if os.path.exists(path):
        return ImageFont.truetype(path, size)

    # Fallback: try common system font locations
    fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFPro.ttf",
    ]
    for fb in fallbacks:
        if os.path.exists(fb):
            logger.warning("Font not found: %s, using fallback: %s", path, fb)
            return ImageFont.truetype(fb, size)

    logger.warning("No fonts found, using PIL default")
    return ImageFont.load_default()


# ── Text layout engine ──────────────────────────────────────────────────────

@dataclass
class TextSpan:
    """A span of text with position, font, and color info."""
    text: str
    x: int
    y: int
    font: ImageFont.FreeTypeFont
    color: tuple[int, int, int, int]
    # For typewriter: global char index range this span covers
    char_start: int = 0
    char_end: int = 0
    # Whether this is always visible (title, subtitle, logo) or typed
    always_visible: bool = False


@dataclass
class TextLayout:
    """Pre-calculated layout of all text elements."""
    spans: list[TextSpan] = field(default_factory=list)
    total_typed_chars: int = 0
    cursor_positions: list[tuple[int, int]] = field(default_factory=list)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue

        # Preserve leading whitespace (for indented lines)
        leading_space = len(raw_line) - len(raw_line.lstrip())
        prefix = raw_line[:leading_space]
        words = raw_line.strip().split(" ")

        current_line = prefix
        for word in words:
            test_line = current_line + (" " if current_line.strip() else "") + word
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] > max_width and current_line.strip():
                lines.append(current_line)
                current_line = prefix + "  " + word  # Continuation indent
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)

    return lines


def calculate_layout(config: VideoPromoConfig) -> TextLayout:
    """
    Pre-calculate the full text layout for all elements.
    Returns a TextLayout with all spans positioned and indexed.
    """
    tc = config.text_card
    layout = TextLayout()

    # Load fonts
    font_bold = load_font(config.font_bold, tc.title_font_size)
    font_subtitle = load_font(config.font_regular, tc.subtitle_font_size)
    font_body = load_font(config.font_regular, tc.body_font_size)

    # Card interior padding
    pad_x = 30
    pad_y = 30
    content_width = tc.card_width - (pad_x * 2)

    # ── Title (always visible) ──
    layout.spans.append(TextSpan(
        text=tc.title,
        x=tc.card_x + pad_x,
        y=tc.card_y + pad_y,
        font=font_bold,
        color=parse_color(tc.title_color),
        always_visible=True,
    ))

    # Calculate title height for positioning below it
    title_bbox = font_bold.getbbox(tc.title)
    title_lines = tc.title.split("\n")
    title_height = len(title_lines) * (title_bbox[3] - title_bbox[1] + 10)
    current_y = tc.card_y + pad_y + title_height + 20

    # ── Subtitle (always visible) ──
    if tc.subtitle:
        layout.spans.append(TextSpan(
            text=tc.subtitle,
            x=tc.card_x + pad_x + 200,
            y=current_y,
            font=font_subtitle,
            color=parse_color(tc.subtitle_color),
            always_visible=True,
        ))
        sub_bbox = font_subtitle.getbbox(tc.subtitle)
        current_y += (sub_bbox[3] - sub_bbox[1]) + 30

    # ── Conversation lines (typewriter animated) ──
    typed_char_index = 0
    line_spacing = 6
    message_spacing = 24

    for msg in tc.conversation:
        full_text = f"{msg.role}: {msg.text}"
        wrapped_lines = wrap_text(full_text, font_body, content_width)

        for i, line in enumerate(wrapped_lines):
            y_pos = current_y + (i * (tc.body_font_size + line_spacing))

            span = TextSpan(
                text=line,
                x=tc.card_x + pad_x + 200,
                y=y_pos,
                font=font_body,
                color=parse_color(tc.body_color),
                char_start=typed_char_index,
                char_end=typed_char_index + len(line),
            )
            layout.spans.append(span)

            # Track cursor positions for each character
            for ci, char in enumerate(line):
                partial = line[:ci + 1]
                bbox = font_body.getbbox(partial)
                cursor_x = span.x + (bbox[2] - bbox[0])
                cursor_y = y_pos
                layout.cursor_positions.append((cursor_x, cursor_y))

            typed_char_index += len(line)

        current_y += len(wrapped_lines) * (tc.body_font_size + line_spacing) + message_spacing

    layout.total_typed_chars = typed_char_index

    return layout


# ── Frame rendering ─────────────────────────────────────────────────────────

def draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    outline_width: int = 1,
):
    """Draw a rounded rectangle with optional border."""
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=outline_width)


def render_frame(
    config: VideoPromoConfig,
    layout: TextLayout,
    frame_number: int,
    total_frames: int,
) -> Image.Image:
    """
    Render a single frame of the text card overlay.
    Returns a transparent RGBA PIL Image.
    """
    tc = config.text_card

    # Create transparent canvas
    img = Image.new("RGBA", (config.width, config.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    # ── Calculate timing ──
    title_hold_frames = int(config.title_hold_seconds * config.fps)
    end_hold_frames = int(config.end_hold_seconds * config.fps)
    typing_frames = total_frames - title_hold_frames - end_hold_frames

    # How many typed characters should be visible at this frame
    if frame_number < title_hold_frames:
        visible_chars = 0
    elif frame_number >= total_frames - end_hold_frames:
        visible_chars = layout.total_typed_chars
    else:
        typing_progress = (frame_number - title_hold_frames) / max(typing_frames, 1)
        visible_chars = int(typing_progress * layout.total_typed_chars)

    # ── Card fade-in (first 10 frames) ──
    fade_frames = 10
    card_alpha_mult = min(1.0, frame_number / fade_frames) if fade_frames > 0 else 1.0

    # ── Draw glassmorphism card ──
    card_bg = parse_color(tc.card_bg_color)
    card_bg_faded = (card_bg[0], card_bg[1], card_bg[2], int(card_bg[3] * card_alpha_mult))
    card_border = parse_color(tc.card_border_color)
    card_border_faded = (card_border[0], card_border[1], card_border[2], int(card_border[3] * card_alpha_mult))

    draw_rounded_rect(
        draw,
        (tc.card_x, tc.card_y, tc.card_x + tc.card_width, tc.card_y + tc.card_height),
        radius=tc.card_corner_radius,
        fill=card_bg_faded,
        outline=card_border_faded,
        outline_width=1,
    )

    # ── Draw text spans ──
    for span in layout.spans:
        if span.always_visible:
            alpha = int(span.color[3] * card_alpha_mult)
            color = (span.color[0], span.color[1], span.color[2], alpha)

            for i, line in enumerate(span.text.split("\n")):
                line_y = span.y + i * (span.font.size + 10)
                draw.text((span.x, line_y), line, font=span.font, fill=color)
        else:
            if visible_chars <= span.char_start:
                continue

            chars_to_show = min(visible_chars - span.char_start, len(span.text))
            if chars_to_show <= 0:
                continue

            visible_text = span.text[:chars_to_show]
            draw.text((span.x, span.y), visible_text, font=span.font, fill=span.color)

    # ── Draw blinking cursor ──
    if 0 < visible_chars < layout.total_typed_chars and layout.cursor_positions:
        cursor_idx = min(visible_chars - 1, len(layout.cursor_positions) - 1)
        if cursor_idx >= 0:
            cx, cy = layout.cursor_positions[cursor_idx]
            blink_cycle = frame_number % 25
            if blink_cycle < 15:
                cursor_color = parse_color(tc.title_color)
                cursor_height = tc.body_font_size + 4
                draw.rectangle(
                    (cx + 2, cy, cx + 4, cy + cursor_height),
                    fill=(cursor_color[0], cursor_color[1], cursor_color[2], 200),
                )

    # ── Draw brand logo / prefix ──
    if config.brand:
        brand = config.brand
        if brand.prefix_text:
            prefix_font = load_font(config.font_regular, 22)
            prefix_color = parse_color("rgba(255,255,255,0.5)")
            draw.text((brand.prefix_x, brand.prefix_y), brand.prefix_text, font=prefix_font, fill=prefix_color)

        if os.path.exists(brand.logo_path):
            logo = Image.open(brand.logo_path).convert("RGBA")
            logo.load()
            scale = brand.logo_height / logo.height
            new_size = (int(logo.width * scale), brand.logo_height)
            logo = logo.resize(new_size, Image.Resampling.LANCZOS)
            img.paste(logo, (brand.logo_x, brand.logo_y), logo)

    return img


# ── Main render pipeline ───────────────────────────────────────────────────

def render_text_card_frames(config: VideoPromoConfig, output_dir: str) -> tuple[str, int]:
    """
    Render all frames of the text card overlay as PNGs.

    Returns:
        (frame_pattern, total_frames) — e.g. ("frame_%04d.png", 375)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_frames = int(config.total_duration_seconds * config.fps)
    layout = calculate_layout(config)

    logger.info("Rendering %d text card frames...", total_frames)
    logger.info("Total typed characters: %d", layout.total_typed_chars)

    for i in range(total_frames):
        frame = render_frame(config, layout, i, total_frames)
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        frame.save(frame_path, "PNG")

        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Rendered frame %d/%d", i + 1, total_frames)

    logger.info("All %d frames rendered.", total_frames)
    return os.path.join(output_dir, "frame_%04d.png"), total_frames
