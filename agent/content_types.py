"""
Canonical content type definitions for the BrandMover pipeline.

Single source of truth used by image generation routing, compositor profiles,
tool definitions, and the agent skill prompt.
"""

import re

# AI buzzword regex — shared across engine.py and self_review.py
AI_WORDS_PATTERN = re.compile(
    r"\b(?:"
    r"revolutionizing|leveraging|cutting-edge|seamlessly|dive into|unlock|"
    r"reimagining|redefining|supercharging|turbocharging|"
    r"game-?changing|groundbreaking|trailblazing|pioneering|"
    r"next-?gen(?:eration)?|best-in-class|world-class|state-of-the-art|"
    r"harness(?:ing)?|empower(?:ing)?|elevat(?:e|ing)|"
    r"robust|scalable|synerg(?:y|ies|istic)|holistic|"
    r"ecosystem|paradigm|disruptive|innovative|"
    r"transformative|comprehensive|streamlin(?:e|ed|ing)|"
    r"architected|architecting|architecturally|"
    r"self-sustaining|human-driven|autonomous(?:ly)?|"
    r"delve|unpack|navigate the|landscape|"
    r"at the forefront|at the intersection|on the cutting edge|"
    r"excited to announce|thrilled to share|proud to|"
    r"double down|move the needle|low-hanging fruit|"
    r"north star|deep dive|circle back|"
    r"the result is|it's worth noting|importantly"
    r")\b",
    re.IGNORECASE,
)

# --- Content types that the agent/pipeline can produce ---
# These are the values used in draft["content_type"], image_gen routing,
# and compositor profile selection.

ALL_CONTENT_TYPES = (
    "announcement",
    "campaign",
    "market",
    "meme",
    "engagement",
    "advice",
    "lifestyle",
    "event",
    "educational",
    "brand_asset",
    "community",
    "market_commentary",
    "brand_3d",
    "default",
)

# Types eligible for LoRA training data collection
LORA_ELIGIBLE_TYPES = {"brand_asset", "community", "brand_3d", "lifestyle"}

# Types the compositor has dedicated visual profiles for
COMPOSITOR_PROFILE_TYPES = {
    "announcement", "campaign", "market", "meme",
    "engagement", "advice", "default",
}

# Types the agent can select in its JSON output
AGENT_SELECTABLE_TYPES = (
    "announcement",
    "campaign",
    "meme",
    "engagement",
    "advice",
    "lifestyle",
    "event",
    "educational",
    "brand_asset",
    "community",
    "market_commentary",
    "brand_3d",
)

# Mapping from agent content_type → compositor profile key
# Types not listed here fall through to "default"
COMPOSITOR_PROFILE_MAP = {
    "announcement": "announcement",
    "campaign": "campaign",
    "market": "market",
    "market_commentary": "market",
    "meme": "meme",
    "engagement": "engagement",
    "advice": "advice",
    "lifestyle": "default",
    "event": "default",
    "educational": "default",
    "brand_asset": "default",
    "community": "default",
    "brand_3d": "default",
    "default": "default",
}


def get_enabled_content_types() -> tuple[str, ...]:
    """Return types from config.json, or ALL_CONTENT_TYPES in legacy mode."""
    from agent.compositor_config import _load_config_json
    config_json = _load_config_json()
    if config_json and "content_types_enabled" in config_json:
        enabled = config_json["content_types_enabled"]
        if isinstance(enabled, list) and enabled:
            return tuple(enabled)
    return ALL_CONTENT_TYPES
