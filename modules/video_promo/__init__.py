"""
Video Promo Module — Short-form branded promo video generation.

Composites three layers: AI-generated background + glassmorphism text card + audio.
"""

from modules.video_promo.video_promo_module import generate_promo_video, generate_promo_video_sync
from modules.video_promo.config_schema import (
    VideoPromoConfig,
    TextCardConfig,
    BackgroundConfig,
    BackgroundStyle,
    BrandOverlay,
    AudioConfig,
    ConversationLine,
)

__all__ = [
    "generate_promo_video",
    "generate_promo_video_sync",
    "VideoPromoConfig",
    "TextCardConfig",
    "BackgroundConfig",
    "BackgroundStyle",
    "BrandOverlay",
    "AudioConfig",
    "ConversationLine",
]
