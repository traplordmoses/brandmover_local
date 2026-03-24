"""
Channel abstraction layer — normalized message envelope for multi-channel publishing.

Inspired by OpenClaw's plugin-sdk. Decouples content generation from platform-specific
publishing. Each channel implements the Channel interface.

Currently supported: X/Twitter, Discord, LinkedIn, Instagram
Future: Telegram channels
"""

from agent.channels.base import Channel, MessageEnvelope, PublishResult
from agent.channels.registry import get_channel, register_channel, list_channels

__all__ = [
    "Channel",
    "MessageEnvelope",
    "PublishResult",
    "get_channel",
    "register_channel",
    "list_channels",
]
