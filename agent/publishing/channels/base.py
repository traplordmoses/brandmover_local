"""
Base channel interface and message envelope.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MessageEnvelope:
    """Normalized message format for any publishing channel.

    Platform-agnostic representation of a post. Each channel adapter
    translates this into platform-specific API calls.
    """
    text: str
    image_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    alt_text: str = ""
    title: str = ""
    subtitle: str = ""
    content_type: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Text with hashtags appended."""
        if not self.hashtags:
            return self.text
        tag_str = " ".join(self.hashtags)
        return f"{self.text}\n\n{tag_str}".strip()

    def truncate_text(self, max_length: int) -> str:
        """Return text truncated to max_length with ellipsis."""
        full = self.full_text
        if len(full) <= max_length:
            return full
        return full[:max_length - 3] + "..."


@dataclass
class PublishResult:
    """Result of publishing to a channel."""
    success: bool
    url: str = ""
    platform: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)


class Channel(ABC):
    """Abstract base class for publishing channels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel identifier (e.g., 'twitter', 'discord', 'linkedin')."""
        ...

    @property
    @abstractmethod
    def max_text_length(self) -> int:
        """Maximum text length for this platform."""
        ...

    @abstractmethod
    async def publish(self, envelope: MessageEnvelope) -> PublishResult:
        """Publish a message to this channel."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this channel has valid credentials."""
        ...
