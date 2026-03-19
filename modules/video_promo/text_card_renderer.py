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
class ChatBubble:
    """A chat message bubble with its own background."""
    role: str
    lines: list[str]  # pre-wrapped text lines
    x: int
    y: int
    width: int
    height: int
    is_user: bool  # True = right-aligned user bubble, False = left-aligned AI
    bg_color: tuple[int, int, int, int]
    text_color: tuple[int, int, int, int]
    role_color: tuple[int, int, int, int]
    font: ImageFont.FreeTypeFont
    role_font: ImageFont.FreeTypeFont
    char_start: int = 0
    char_end: int = 0
    corner_radius: int = 16


@dataclass
class TextLayout:
    """Pre-calculated layout of all text elements."""
    spans: list[TextSpan] = field(default_factory=list)
    bubbles: list[ChatBubble] = field(default_factory=list)
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
    Uses centered card with chat-bubble style conversation.
    """
    tc = config.text_card
    layout = TextLayout()

    # Load fonts
    font_bold = load_font(config.font_bold, tc.title_font_size)
    font_subtitle = load_font(config.font_regular, tc.subtitle_font_size)
    font_body = load_font(config.font_regular, tc.body_font_size)
    font_role = load_font(config.font_bold, tc.role_font_size)

    # ── Center the card on screen ──
    card_x = (config.width - tc.card_width) // 2
    card_y = tc.card_y  # Will be adjusted after measuring content

    pad_x = 36
    pad_y = 36
    content_width = tc.card_width - (pad_x * 2)
    bubble_max_width = int(content_width * 0.82)
    bubble_pad = 14
    bubble_spacing = 14
    line_spacing = 5

    # ── Measure total content height first for vertical centering ──
    # Title height
    title_text = tc.title.replace("\\n", "\n")  # Handle literal \n from JSON
    title_lines = title_text.split("\n")
    single_bbox = font_bold.getbbox("Ag")
    title_line_h = single_bbox[3] - single_bbox[1] + 12
    title_height = len(title_lines) * title_line_h

    # Subtitle height
    sub_height = 0
    if tc.subtitle:
        sub_bbox = font_subtitle.getbbox(tc.subtitle)
        sub_height = (sub_bbox[3] - sub_bbox[1]) + 24

    # Bubble heights
    total_bubble_h = 0
    for msg in tc.conversation:
        wrapped = wrap_text(msg.text, font_body, bubble_max_width - bubble_pad * 2)
        role_h = tc.role_font_size + 6
        text_h = len(wrapped) * (tc.body_font_size + line_spacing)
        total_bubble_h += role_h + text_h + bubble_pad * 2 + bubble_spacing

    total_content_h = pad_y + title_height + 20 + sub_height + total_bubble_h + pad_y
    # Auto-size card height
    actual_card_h = max(tc.card_height, total_content_h)
    # Vertically center the card
    card_y = max(80, (config.height - actual_card_h) // 2)

    # ── Title (always visible, centered in card) ──
    layout.spans.append(TextSpan(
        text=title_text,
        x=card_x + pad_x,
        y=card_y + pad_y,
        font=font_bold,
        color=parse_color(tc.title_color),
        always_visible=True,
    ))

    current_y = card_y + pad_y + title_height + 20

    # ── Subtitle (always visible) ──
    if tc.subtitle:
        layout.spans.append(TextSpan(
            text=tc.subtitle,
            x=card_x + pad_x,
            y=current_y,
            font=font_subtitle,
            color=parse_color(tc.subtitle_color),
            always_visible=True,
        ))
        current_y += sub_height

    # ── Chat bubbles ──
    typed_char_index = 0

    for msg in tc.conversation:
        is_user = msg.role.lower() in ("you", "user", "me")
        wrapped = wrap_text(msg.text, font_body, bubble_max_width - bubble_pad * 2)
        total_text = "\n".join(wrapped)

        # Measure bubble dimensions
        role_h = tc.role_font_size + 6
        text_h = len(wrapped) * (tc.body_font_size + line_spacing)
        bub_h = role_h + text_h + bubble_pad * 2
        bub_w = min(bubble_max_width, content_width)

        # Position: user bubbles right-aligned, AI left-aligned
        if is_user:
            bub_x = card_x + tc.card_width - pad_x - bub_w
            bg = (255, 255, 255, 18)  # subtle white tint
            text_col = parse_color("#FFFFFF")
            role_col = parse_color("#3EEEC4")  # accent green for user
        else:
            bub_x = card_x + pad_x
            bg = (107, 159, 212, 25)  # subtle blue tint
            text_col = parse_color("rgba(255,255,255,0.9)")
            role_col = parse_color("#6B9FD4")  # blue for AI

        bubble = ChatBubble(
            role=msg.role,
            lines=wrapped,
            x=bub_x,
            y=current_y,
            width=bub_w,
            height=bub_h,
            is_user=is_user,
            bg_color=bg,
            text_color=text_col,
            role_color=role_col,
            font=font_body,
            role_font=font_role,
            char_start=typed_char_index,
            char_end=typed_char_index + len(total_text),
        )
        layout.bubbles.append(bubble)

        # Track cursor positions for each character
        text_x = bub_x + bubble_pad
        text_y_start = current_y + bubble_pad + role_h
        for li, line in enumerate(wrapped):
            ly = text_y_start + li * (tc.body_font_size + line_spacing)
            for ci in range(len(line)):
                partial = line[:ci + 1]
                bbox = font_body.getbbox(partial)
                layout.cursor_positions.append((text_x + (bbox[2] - bbox[0]), ly))
            typed_char_index += len(line)

        current_y += bub_h + bubble_spacing

    layout.total_typed_chars = typed_char_index

    # Store computed card geometry for render_frame
    layout._card_x = card_x
    layout._card_y = card_y
    layout._card_h = actual_card_h

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

    # Use computed card geometry from layout
    card_x = getattr(layout, "_card_x", tc.card_x)
    card_y = getattr(layout, "_card_y", tc.card_y)
    card_h = getattr(layout, "_card_h", tc.card_height)

    # ── Calculate timing ──
    title_hold_frames = int(config.title_hold_seconds * config.fps)
    end_hold_frames = int(config.end_hold_seconds * config.fps)
    typing_frames = total_frames - title_hold_frames - end_hold_frames

    if frame_number < title_hold_frames:
        visible_chars = 0
    elif frame_number >= total_frames - end_hold_frames:
        visible_chars = layout.total_typed_chars
    else:
        typing_progress = (frame_number - title_hold_frames) / max(typing_frames, 1)
        visible_chars = int(typing_progress * layout.total_typed_chars)

    # ── Card fade-in (first 12 frames) ──
    fade_frames = 12
    card_alpha_mult = min(1.0, frame_number / fade_frames) if fade_frames > 0 else 1.0

    # ── Draw main glassmorphism card ──
    card_bg = parse_color(tc.card_bg_color)
    card_bg_faded = (card_bg[0], card_bg[1], card_bg[2], int(card_bg[3] * card_alpha_mult))
    card_border = parse_color(tc.card_border_color)
    card_border_faded = (card_border[0], card_border[1], card_border[2], int(card_border[3] * card_alpha_mult))

    draw_rounded_rect(
        draw,
        (card_x, card_y, card_x + tc.card_width, card_y + card_h),
        radius=tc.card_corner_radius,
        fill=card_bg_faded,
        outline=card_border_faded,
        outline_width=1,
    )

    # ── Draw title + subtitle (always visible, fade in with card) ──
    for span in layout.spans:
        if not span.always_visible:
            continue
        alpha = int(span.color[3] * card_alpha_mult)
        color = (span.color[0], span.color[1], span.color[2], alpha)
        for i, line in enumerate(span.text.split("\n")):
            line_y = span.y + i * (span.font.size + 12)
            draw.text((span.x, line_y), line, font=span.font, fill=color)

    # ── Draw chat bubbles ──
    line_spacing = 5
    for bubble in layout.bubbles:
        # Skip if no chars from this bubble are visible yet
        if visible_chars <= bubble.char_start:
            continue

        # How many chars of this bubble are visible
        bub_visible = min(visible_chars - bubble.char_start, bubble.char_end - bubble.char_start)
        if bub_visible <= 0:
            continue

        # Bubble slide-in animation (first 6 frames after appearing)
        bub_age = visible_chars - bubble.char_start
        slide_progress = min(1.0, bub_age / 30)  # ~30 chars to fully slide in
        slide_offset = int((1.0 - slide_progress) * 20)
        bub_alpha = min(1.0, bub_age / 15) * card_alpha_mult

        bub_y = bubble.y + slide_offset

        # Draw bubble background
        bg = bubble.bg_color
        bg_faded = (bg[0], bg[1], bg[2], int(bg[3] * bub_alpha))
        border_a = int(40 * bub_alpha)
        border_col = (255, 255, 255, border_a)

        draw_rounded_rect(
            draw,
            (bubble.x, bub_y, bubble.x + bubble.width, bub_y + bubble.height),
            radius=bubble.corner_radius,
            fill=bg_faded,
            outline=border_col,
            outline_width=1,
        )

        # Draw role label
        role_alpha = int(bubble.role_color[3] * bub_alpha)
        role_col = (*bubble.role_color[:3], role_alpha)
        draw.text(
            (bubble.x + 14, bub_y + 10),
            bubble.role,
            font=bubble.role_font,
            fill=role_col,
        )

        # Draw visible text lines
        text_x = bubble.x + 14
        text_y = bub_y + 10 + bubble.role_font.size + 6
        chars_drawn = 0
        text_alpha = int(bubble.text_color[3] * bub_alpha)
        text_col = (*bubble.text_color[:3], text_alpha)

        for li, line in enumerate(bubble.lines):
            ly = text_y + li * (bubble.font.size + line_spacing)
            remaining = bub_visible - chars_drawn
            if remaining <= 0:
                break
            visible_line = line[:remaining]
            draw.text((text_x, ly), visible_line, font=bubble.font, fill=text_col)
            chars_drawn += len(line)

    # ── Draw blinking cursor ──
    if 0 < visible_chars < layout.total_typed_chars and layout.cursor_positions:
        cursor_idx = min(visible_chars - 1, len(layout.cursor_positions) - 1)
        if cursor_idx >= 0:
            cx, cy = layout.cursor_positions[cursor_idx]
            blink_cycle = frame_number % 25
            if blink_cycle < 15:
                cursor_height = tc.body_font_size + 4
                draw.rectangle(
                    (cx + 2, cy, cx + 4, cy + cursor_height),
                    fill=(255, 255, 255, int(200 * card_alpha_mult)),
                )

    # ── Draw brand logo / prefix (bottom center of card) ──
    if config.brand:
        brand = config.brand
        logo_y = card_y + card_h - 50

        if brand.prefix_text:
            prefix_font = load_font(config.font_regular, 18)
            prefix_color = (255, 255, 255, int(100 * card_alpha_mult))
            draw.text((card_x + 36, logo_y), brand.prefix_text, font=prefix_font, fill=prefix_color)

        if os.path.exists(brand.logo_path):
            logo = Image.open(brand.logo_path).convert("RGBA")
            logo.load()
            target_h = max(brand.logo_height, 24)
            scale = target_h / logo.height
            new_size = (int(logo.width * scale), target_h)
            logo = logo.resize(new_size, Image.Resampling.LANCZOS)
            # Position after prefix text
            logo_x = card_x + 70
            img.paste(logo, (logo_x, logo_y), logo)

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
