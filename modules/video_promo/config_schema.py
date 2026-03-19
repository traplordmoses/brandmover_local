"""
Config schema for video promo generation.
Defines the full structure for a branded promo video.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class BackgroundStyle(str, Enum):
    LIQUID_METAL = "liquid_metal"       # Metallic blob, good for fintech/crypto
    AURORA = "aurora"                   # Soft aurora gradients, good for consumer apps
    PARTICLE_FIELD = "particle_field"   # Floating particles, good for tech/AI
    SMOKE = "smoke"                     # Flowing smoke/ink, good for luxury/dark brands
    CUSTOM = "custom"                   # User provides their own background video


class ConversationLine(BaseModel):
    """Single line in the demo conversation."""
    role: str = Field(description="Speaker label, e.g. 'You', 'AI', 'User'")
    text: str = Field(description="The message text. Use \\n for line breaks within a message.")


class TextCardConfig(BaseModel):
    """Configuration for the glassmorphism text card overlay."""
    title: str = Field(description="Big bold title, e.g. 'BLOFIN MCP'")
    subtitle: Optional[str] = Field(default=None, description="Smaller subtitle below title, e.g. '// Example conversation with AI'")
    conversation: list[ConversationLine] = Field(description="The demo conversation to type out")
    typing_speed_cps: int = Field(default=35, description="Characters per second for typewriter effect")
    # Card positioning (relative to 1080x1920 canvas)
    card_x: int = Field(default=60, description="Card left edge X position")
    card_y: int = Field(default=380, description="Card top edge Y position")
    card_width: int = Field(default=960, description="Card width in pixels")
    card_height: int = Field(default=700, description="Card height in pixels — auto-expands if needed")
    card_corner_radius: int = Field(default=24, description="Border radius of the card")
    # Typography
    title_font_size: int = Field(default=96, description="Title font size")
    subtitle_font_size: int = Field(default=22, description="Subtitle font size")
    body_font_size: int = Field(default=22, description="Conversation body font size")
    role_font_size: int = Field(default=22, description="Role label font size")
    # Colors (RGBA tuples as strings for JSON compat)
    card_bg_color: str = Field(default="rgba(30,30,30,0.55)", description="Card background with alpha")
    card_border_color: str = Field(default="rgba(255,255,255,0.12)", description="Card border color")
    title_color: str = Field(default="#FFFFFF", description="Title text color")
    subtitle_color: str = Field(default="rgba(255,255,255,0.5)", description="Subtitle text color")
    role_color_you: str = Field(default="#FFFFFF", description="Color for user role text")
    role_color_ai: str = Field(default="#FFFFFF", description="Color for AI role text")
    body_color: str = Field(default="rgba(255,255,255,0.85)", description="Body text color")
    indent_color: str = Field(default="rgba(255,255,255,0.75)", description="Indented response text color")


class BackgroundConfig(BaseModel):
    """Configuration for AI-generated background."""
    style: BackgroundStyle = Field(default=BackgroundStyle.LIQUID_METAL)
    # AI generation prompt components
    primary_color: str = Field(default="amber gold", description="Primary color description for the blob/effect")
    secondary_color: Optional[str] = Field(default=None, description="Optional secondary color")
    mood: str = Field(default="cinematic, dark, luxurious", description="Mood keywords for generation")
    # If style is CUSTOM, provide a path to a video file
    custom_video_path: Optional[str] = Field(default=None, description="Path to custom background video")
    # Generation settings
    duration_seconds: float = Field(default=5.0, description="Duration of base loop (will be extended to match total)")
    provider: str = Field(default="fal", description="AI provider: 'fal', 'replicate', or 'local'")
    model: Optional[str] = Field(default=None, description="Specific model override. Defaults: fal=minimax-video, replicate=kling")


class BrandOverlay(BaseModel):
    """Brand logo and watermark config."""
    logo_path: str = Field(description="Path to brand logo PNG (white/transparent recommended)")
    logo_x: int = Field(default=130, description="Logo X position")
    logo_y: int = Field(default=1000, description="Logo Y position")
    logo_height: int = Field(default=28, description="Logo render height (width auto-scaled)")
    prefix_text: Optional[str] = Field(default="//", description="Text before logo, e.g. '//'")
    prefix_x: int = Field(default=88, description="Prefix text X position")
    prefix_y: int = Field(default=1000, description="Prefix text Y position")


class AudioConfig(BaseModel):
    """Audio track config."""
    audio_path: Optional[str] = Field(default=None, description="Path to audio file. None = silent video.")
    volume: float = Field(default=0.8, description="Audio volume 0.0-1.0")
    fade_in_seconds: float = Field(default=0.5, description="Audio fade-in duration")
    fade_out_seconds: float = Field(default=1.0, description="Audio fade-out duration")


class VideoPromoConfig(BaseModel):
    """
    Top-level config for generating a branded promo video.
    This is the main input to generate_promo_video().
    """
    # Output settings
    output_path: str = Field(description="Where to save the final .mp4")
    width: int = Field(default=1080, description="Video width")
    height: int = Field(default=1920, description="Video height")
    fps: int = Field(default=25, description="Frames per second")
    total_duration_seconds: float = Field(default=15.0, description="Total video duration")

    # Sub-configs
    text_card: TextCardConfig
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    brand: Optional[BrandOverlay] = Field(default=None, description="Brand logo overlay. None = no logo.")
    audio: AudioConfig = Field(default_factory=AudioConfig)

    # Font paths (loaded from BrandConfig or defaults)
    font_bold: str = Field(default="fonts/Inter-Bold.ttf", description="Path to bold font for titles")
    font_regular: str = Field(default="fonts/Inter-Regular.ttf", description="Path to regular font for body")
    font_mono: Optional[str] = Field(default=None, description="Path to mono font for code-like text")

    # Brand identity (used for background caching)
    brand_name: str = Field(default="default", description="Brand name for background cache directory")

    # Timing
    title_hold_seconds: float = Field(default=1.5, description="How long to show title before conversation starts typing")
    end_hold_seconds: float = Field(default=2.0, description="How long to hold final frame after typing completes")
