"""
Canonical content type definitions for the BrandMover pipeline.

Single source of truth used by image generation routing, compositor profiles,
tool definitions, and the agent skill prompt.
"""

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
