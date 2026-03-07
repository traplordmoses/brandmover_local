"""
DeepEval metrics for BrandMover content evaluation.

6 deterministic metrics (no LLM, no API key):
- HashtagFreeMetric
- EmojiLimitMetric
- CaptionLengthMetric
- ForbiddenPhrasesMetric
- NoExclamationMetric
- ContentTypeValidMetric

2 G-Eval factories (LLM-graded, needs OPENAI_API_KEY):
- BrandToneMetric
- CaptionQualityMetric
"""

import re
import unicodedata
from pathlib import Path

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HASHTAG_RE = re.compile(r"#[A-Za-z]\w*")
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-a
    "\U00002600-\U000026ff"  # misc symbols
    "]+",
    flags=re.UNICODE,
)


def _count_emojis(text: str) -> int:
    """Count individual emoji characters in text."""
    return sum(1 for ch in text if unicodedata.category(ch) in ("So", "Sk") or _EMOJI_RE.match(ch))


def _load_voice_section() -> str:
    """Extract VOICE & TONE section from brand/guidelines.md.

    Returns the raw text of the section, or empty string if not found.
    """
    # Try common brand folder locations
    for base in (Path("brand"), Path("./brand")):
        guidelines = base / "guidelines.md"
        if guidelines.exists():
            text = guidelines.read_text(encoding="utf-8")
            sections = []
            for section_name in ("VOICE & TONE", "BRAND PHRASES & SLANG", "NEVER DO"):
                start = text.find(f"## {section_name}")
                if start == -1:
                    continue
                end = text.find("\n## ", start + 1)
                chunk = text[start:end] if end != -1 else text[start:]
                sections.append(chunk.strip())
            return "\n\n".join(sections)
    return ""


def _load_forbidden_phrases() -> list[str]:
    """Load forbidden phrases from guidelines.md 'Never use' / 'NEVER DO' sections."""
    for base in (Path("brand"), Path("./brand")):
        guidelines = base / "guidelines.md"
        if guidelines.exists():
            text = guidelines.read_text(encoding="utf-8")
            phrases = []
            # Look for "Never use" or "NEVER DO" lists
            for marker in ("Never use", "NEVER DO", "never use"):
                idx = text.find(marker)
                if idx == -1:
                    continue
                # Read lines after the marker until next section
                block = text[idx:]
                end = block.find("\n## ", 1)
                block = block[:end] if end != -1 else block
                for line in block.split("\n"):
                    line = line.strip().lstrip("-•* ")
                    if line and not line.startswith("#") and len(line) < 100:
                        # Skip section headers and the marker line itself
                        if line.lower().startswith("never"):
                            continue
                        phrases.append(line.lower())
            return phrases
    # Fallback: common AI-sounding phrases
    return [
        "revolutionizing", "leveraging", "cutting-edge", "seamlessly",
        "dive into", "unlock the power", "game-changer",
    ]


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

class HashtagFreeMetric(BaseMetric):
    """Fails if the caption contains any hashtags."""

    def __init__(self):
        self.threshold = 1.0
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "HashtagFree"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        hashtags = _HASHTAG_RE.findall(text)
        if hashtags:
            self.score = 0.0
            self.reason = f"Found {len(hashtags)} hashtag(s): {', '.join(hashtags[:5])}"
        else:
            self.score = 1.0
            self.reason = "No hashtags found"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class EmojiLimitMetric(BaseMetric):
    """Fails if caption has >1 emoji or starts with an emoji."""

    def __init__(self):
        self.threshold = 1.0
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "EmojiLimit"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        count = _count_emojis(text)
        starts_with_emoji = bool(text) and _count_emojis(text[0]) > 0

        if count > 1:
            self.score = 0.0
            self.reason = f"Too many emojis: {count} (max 1)"
        elif starts_with_emoji:
            self.score = 0.0
            self.reason = "Caption starts with emoji"
        else:
            self.score = 1.0
            self.reason = f"Emoji count OK ({count})"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class CaptionLengthMetric(BaseMetric):
    """Fails if caption is outside 50-150 characters."""

    def __init__(self, min_chars: int = 50, max_chars: int = 150):
        self.threshold = 1.0
        self.score = 0.0
        self.reason = ""
        self.min_chars = min_chars
        self.max_chars = max_chars

    @property
    def __name__(self):
        return "CaptionLength"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        length = len(text)
        if length < self.min_chars:
            self.score = 0.0
            self.reason = f"Too short: {length} chars (min {self.min_chars})"
        elif length > self.max_chars:
            self.score = 0.0
            self.reason = f"Too long: {length} chars (max {self.max_chars})"
        else:
            self.score = 1.0
            self.reason = f"Length OK: {length} chars"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class ForbiddenPhrasesMetric(BaseMetric):
    """Fails if caption contains any forbidden phrases from guidelines."""

    def __init__(self, phrases: list[str] | None = None):
        self.threshold = 1.0
        self.score = 0.0
        self.reason = ""
        self._phrases = phrases

    @property
    def _forbidden(self) -> list[str]:
        if self._phrases is None:
            self._phrases = _load_forbidden_phrases()
        return self._phrases

    @property
    def __name__(self):
        return "ForbiddenPhrases"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = (test_case.actual_output or "").lower()
        found = [p for p in self._forbidden if p in text]
        if found:
            self.score = 0.0
            self.reason = f"Forbidden phrase(s): {', '.join(found[:5])}"
        else:
            self.score = 1.0
            self.reason = "No forbidden phrases"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class NoExclamationMetric(BaseMetric):
    """Fails if caption contains exclamation marks."""

    def __init__(self):
        self.threshold = 1.0
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "NoExclamation"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        count = text.count("!")
        if count > 0:
            self.score = 0.0
            self.reason = f"Found {count} exclamation mark(s)"
        else:
            self.score = 1.0
            self.reason = "No exclamation marks"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class ContentTypeValidMetric(BaseMetric):
    """Fails if content_type is not in ALL_CONTENT_TYPES."""

    def __init__(self):
        self.threshold = 1.0
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "ContentTypeValid"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        from agent.content_types import ALL_CONTENT_TYPES

        # content_type passed via additional_metadata or extracted from context
        ct = ""
        if test_case.additional_metadata:
            ct = test_case.additional_metadata.get("content_type", "")
        if not ct:
            ct = test_case.context[0] if test_case.context else ""

        if ct in ALL_CONTENT_TYPES:
            self.score = 1.0
            self.reason = f"Valid content type: {ct}"
        else:
            self.score = 0.0
            self.reason = f"Invalid content type: '{ct}' (not in ALL_CONTENT_TYPES)"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


# ---------------------------------------------------------------------------
# G-Eval factories (LLM-graded)
# ---------------------------------------------------------------------------

def _build_geval_criteria(voice_section: str, metric_type: str) -> str:
    """Build G-Eval criteria string from brand voice rules."""
    if metric_type == "tone":
        if voice_section:
            return (
                "Evaluate whether the caption matches the brand's voice and tone rules:\n\n"
                f"{voice_section}\n\n"
                "Score 1.0 if the caption sounds natural and matches these rules. "
                "Score 0.0 if it sounds corporate, robotic, or violates the voice rules."
            )
        return (
            "Evaluate whether the caption sounds natural and human-written. "
            "It should NOT sound like corporate marketing or AI-generated content. "
            "Score 1.0 for natural, conversational tone. Score 0.0 for robotic or overly formal."
        )
    else:  # quality
        if voice_section:
            return (
                "Evaluate the overall quality of this social media caption based on "
                "the brand's personality:\n\n"
                f"{voice_section}\n\n"
                "Score on: information density (says something meaningful), "
                "natural voice (matches the brand personality above), "
                "shareability (would someone actually engage with this). "
                "Score 1.0 for high quality. Score 0.0 for generic filler."
            )
        return (
            "Evaluate the overall quality of this social media caption. "
            "Score on: information density (says something meaningful), "
            "natural voice (sounds human, not AI), "
            "shareability (would someone actually engage with this). "
            "Score 1.0 for high quality. Score 0.0 for generic filler."
        )


def make_brand_tone_metric():
    """Create a G-Eval metric for brand tone compliance.

    Requires OPENAI_API_KEY. Returns None if deepeval G-Eval is unavailable.
    """
    try:
        from deepeval.metrics import GEval

        voice = _load_voice_section()
        criteria = _build_geval_criteria(voice, "tone")
        return GEval(
            name="BrandTone",
            criteria=criteria,
            evaluation_params=[],
            threshold=0.6,
        )
    except Exception:
        return None


def make_caption_quality_metric():
    """Create a G-Eval metric for caption quality.

    Requires OPENAI_API_KEY. Returns None if deepeval G-Eval is unavailable.
    """
    try:
        from deepeval.metrics import GEval

        voice = _load_voice_section()
        criteria = _build_geval_criteria(voice, "quality")
        return GEval(
            name="CaptionQuality",
            criteria=criteria,
            evaluation_params=[],
            threshold=0.6,
        )
    except Exception:
        return None
