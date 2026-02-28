"""
Schedule logic and prompt builders for automated X posting.
Reads config/schedule.json, determines due slots, and builds type-specific prompts.
"""

import json
import logging
import random
import time
from pathlib import Path

from agent import auto_state, compositor_config, onchain
from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule loading (hot-reloadable)
# ---------------------------------------------------------------------------


def load_schedule() -> dict:
    """Parse config/schedule.json. Hot-reloadable — read on each cycle."""
    path = Path(settings.AUTO_POST_SCHEDULE_FILE)
    if not path.exists():
        logger.error("Schedule file not found: %s", path)
        return {"slots": {}, "global": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("slots", {})
        data.setdefault("global", {})
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to read schedule.json: %s", e)
        return {"slots": {}, "global": {}}


# ---------------------------------------------------------------------------
# Slot scheduling
# ---------------------------------------------------------------------------


def get_due_slots(schedule: dict) -> list[str]:
    """Determine which slots are due based on current UTC time + state.

    A slot is due when:
    1. It's enabled
    2. Current hour matches slot hour (within jitter window)
    3. It hasn't been posted today
    """
    now = time.gmtime()
    current_hour = now.tm_hour
    current_min = now.tm_min
    current_minutes_of_day = current_hour * 60 + current_min

    due = []
    for slot_name, slot_config in schedule.get("slots", {}).items():
        if not slot_config.get("enabled", True):
            continue

        if auto_state.is_slot_posted(slot_name):
            continue

        slot_hour = slot_config.get("hour_utc", 0)
        jitter = slot_config.get("jitter_minutes", 30)
        slot_center = slot_hour * 60
        window_start = slot_center - jitter
        window_end = slot_center + jitter

        if window_start <= current_minutes_of_day <= window_end:
            due.append(slot_name)

    return due


def is_slot_due(slot_name: str, schedule: dict) -> bool:
    """Check if a specific slot is within its time window (for --force)."""
    return slot_name in get_due_slots(schedule)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dynamic prompt helpers — all brand data from BrandConfig
# ---------------------------------------------------------------------------


def _visual_style() -> str:
    """Build image style directive from brand config."""
    cfg = compositor_config.get_config()
    if cfg.visual_style_prompt:
        return f"Image style: {cfg.visual_style_prompt}"
    kw = ", ".join(cfg.style_keywords[:6]) if cfg.style_keywords else "high quality, professional"
    return f"Image style: {kw}"


def _voice_summary() -> str:
    """Build a short voice description from brand config."""
    cfg = compositor_config.get_config()
    name = cfg.brand_name or settings.BRAND_NAME
    if cfg.voice_traits:
        return f"{name} voice: {', '.join(cfg.voice_traits[:3])}"
    return f"{name} voice"


def _brand_context() -> str:
    """Build product/tagline/phrases context string."""
    cfg = compositor_config.get_config()
    parts = []
    if cfg.tagline:
        parts.append(f"Brand perspective: {cfg.tagline}")
    if cfg.product_description:
        parts.append(f"Products: {cfg.product_description[:200]}")
    if cfg.brand_phrases:
        parts.append(f"Key phrases: {'; '.join(cfg.brand_phrases[:3])}")
    return ". ".join(parts) if parts else ""


def _brand_name() -> str:
    cfg = compositor_config.get_config()
    return cfg.brand_name or settings.BRAND_NAME


def _themes_hint() -> str:
    cfg = compositor_config.get_config()
    if cfg.content_themes:
        return ", ".join(cfg.content_themes[:5])
    return "culture, community, identity"


def _engagement_templates() -> list[str]:
    """Build engagement prompt templates from brand config."""
    name = _brand_name()
    voice = _voice_summary()
    ctx = _brand_context()
    vs = _visual_style()
    themes = _themes_hint()

    ctx_line = f" {ctx}" if ctx else ""

    return [
        # Culture commentary
        (
            "Write a short, punchy X post about internet culture and identity. "
            "Comment on how memes become culture, how online communities form identity, "
            "or how digital rituals shape belonging. Make it feel observational and wise, "
            f"not preachy. {name} perspective: {themes}.{ctx_line} "
            f"Generate an accompanying image. {vs}"
        ),
        # Community question
        (
            "Write a short X post that asks the community a thought-provoking question. "
            f"Topics: {themes}. "
            f"Keep it open-ended and inviting. {voice}. "
            f"Generate an accompanying image. {vs}"
        ),
        # Philosophical musing
        (
            "Write a short, philosophical X post about lore, memory, or identity in the digital age. "
            "Think about what it means to canonize something on-chain, how collective memory works, "
            f"or why people build culture in decentralized spaces. {name} tone: "
            "contemplative but not pretentious. "
            f"Generate an accompanying image. {vs}"
        ),
        # Ritual
        (
            "Write a short X post about rituals — digital or otherwise. "
            "How do communities create shared rituals? What makes a gesture sacred when it's just pixels? "
            f"{name} angle: {themes}. Keep it atmospheric and brief. "
            f"Generate an accompanying image. {vs}"
        ),
    ]


def _brand_meme_templates() -> list[str]:
    """Build brand meme prompt templates from brand config."""
    name = _brand_name()
    voice = _voice_summary()
    vs = _visual_style()
    cfg = compositor_config.get_config()
    product_hint = cfg.product_description[:120] if cfg.product_description else "the brand's products and culture"

    return [
        # Meme
        (
            f"Create a branded meme post for {name}. Generate a funny, culturally relevant image "
            "with a witty caption. Topics: crypto culture, internet identity, or "
            f"{product_hint}. The image should be eye-catching and shareable. "
            f"Keep the caption under 100 characters — punchy, not try-hard. {vs}"
        ),
        # Announcement-style
        (
            f"Create a branded announcement-style post for {name}. Something related to "
            f"{product_hint}. Generate a striking image "
            "and write a caption that feels like a dispatch from the future. "
            f"{voice}: declarative, slightly mysterious. {vs}"
        ),
        # Visual celebration
        (
            f"Create a visual post celebrating {name} culture. Generate an image that captures the "
            f"brand's visual aesthetic. "
            "Write a caption about culture being built on-chain, one creation at a time. "
            f"Keep it short and confident. {vs}"
        ),
        # Community curation
        (
            "Create a branded post about digital curation or collective memory. "
            "Generate an image showing the concept of community-driven cultural preservation. "
            "Caption should be about how the community decides "
            f"what becomes canonical. {voice}: declarative and certain. {vs}"
        ),
    ]


async def build_prompt_for_slot(slot_name: str, slot_config: dict) -> tuple[str, list[str]]:
    """Build an agent prompt for a specific slot type.

    Returns (prompt, event_ids) where event_ids is a list of on-chain event IDs
    referenced in the prompt (empty for non-onchain slots).
    """
    slot_type = slot_config.get("type", "engagement")

    if slot_type == "engagement":
        return _build_engagement_prompt(), []

    elif slot_type == "onchain_review":
        return await _build_onchain_prompt()

    elif slot_type == "brand_meme":
        return _build_brand_meme_prompt(), []

    else:
        logger.warning("Unknown slot type: %s, falling back to engagement", slot_type)
        return _build_engagement_prompt(), []


def _build_engagement_prompt() -> str:
    """Build an engagement prompt, rotating through templates."""
    templates = _engagement_templates()
    idx = auto_state.get_rotation_index("engagement")
    prompt = templates[idx % len(templates)]
    auto_state.advance_rotation("engagement", len(templates))
    return prompt


async def _build_onchain_prompt() -> tuple[str, list[str]]:
    """Build an on-chain review prompt with live loreboard data.

    Falls back to philosophical thought-mode if the board is quiet.
    """
    name = _brand_name()
    voice = _voice_summary()
    vs = _visual_style()

    state = await onchain.fetch_board_state()

    # Get already-posted event IDs to avoid duplication
    auto_state_data = auto_state._read_state()
    posted_ids = auto_state_data.get("posted_event_ids", [])

    summary = onchain.format_onchain_summary(state, posted_event_ids=posted_ids)
    new_event_ids = onchain.get_new_event_ids(state, posted_event_ids=posted_ids)

    if state.is_quiet or not new_event_ids:
        # Thought-mode fallback
        return (
            f"The {name} loreboard is quiet right now. Write a reflective X post about: "
            "the nature of collective memory on-chain, why lore matters, or what happens "
            f"in the spaces between canonizations. {voice}: contemplative, "
            "in a quiet moment. Include an image that captures "
            f"this mood — serene digital landscapes, quiet interfaces, or glowing terminals. {vs}"
        ), []

    prompt = (
        f"Write an X post commenting on recent {name} loreboard activity. "
        f"React to what's happening on-chain with {name}'s voice: "
        f"observational, knowing, sometimes amused.\n\n"
        f"CURRENT LOREBOARD STATE:\n{summary}\n\n"
        f"Pick the most interesting event(s) to comment on. Be specific — "
        f"reference actual proposals or votes. Generate an image that captures "
        f"the energy of what's happening (activity, voting, canonization). "
        f"Keep the caption punchy, under 150 characters. {vs}"
    )

    return prompt, new_event_ids


def _build_brand_meme_prompt() -> str:
    """Build a brand meme prompt, rotating through templates."""
    templates = _brand_meme_templates()
    idx = auto_state.get_rotation_index("brand_meme")
    prompt = templates[idx % len(templates)]
    auto_state.advance_rotation("brand_meme", len(templates))
    return prompt


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------


def apply_jitter(base_delay: float, jitter_minutes: int) -> float:
    """Add random jitter to a delay to avoid posting at exact intervals."""
    jitter_seconds = random.randint(0, jitter_minutes * 60)
    return base_delay + jitter_seconds
