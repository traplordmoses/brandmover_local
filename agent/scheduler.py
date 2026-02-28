"""
Schedule logic and prompt builders for automated X posting.
Reads config/schedule.json, determines due slots, and builds type-specific prompts.
"""

import json
import logging
import random
import time
from pathlib import Path

from agent import auto_state, onchain
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

# Engagement prompt templates — rotated to avoid repetition
_VISUAL_STYLE = (
    "Image style: Frutiger Aero bubbly glass aesthetic — floating translucent glass orbs, "
    "frosted glass panels with rounded corners, glossy bubbly spheres, "
    "soft aqua and lavender glow, dark navy midnight background, "
    "volumetric light through glass, polished 3D render feel."
)

_ENGAGEMENT_PROMPTS = [
    # Culture commentary
    (
        "Write a short, punchy X post about internet culture and identity. "
        "Comment on how memes become culture, how online communities form identity, "
        "or how digital rituals shape belonging. Make it feel observational and wise, "
        "not preachy. FOID Foundation perspective: culture, ritual, identity — on chain. "
        f"Generate an accompanying image. {_VISUAL_STYLE}"
    ),
    # Community question
    (
        "Write a short X post that asks the community a thought-provoking question. "
        "Topics: digital identity, what makes a meme canonical, the role of ritual in crypto, "
        "or what culture means on-chain. Keep it open-ended and inviting. "
        "FOID Foundation voice: lowercase energy, dry wit, earnest underneath the irony. "
        f"Generate an accompanying image. {_VISUAL_STYLE}"
    ),
    # Philosophical musing
    (
        "Write a short, philosophical X post about lore, memory, or identity in the digital age. "
        "Think about what it means to canonize something on-chain, how collective memory works, "
        "or why people build culture in decentralized spaces. FOID Foundation tone: "
        "contemplative but not pretentious, the oracle who types in lowercase. "
        f"Generate an accompanying image. {_VISUAL_STYLE}"
    ),
    # Ritual
    (
        "Write a short X post about rituals — digital or otherwise. "
        "How do communities create shared rituals? What makes a gesture sacred when it's just pixels? "
        "FOID Foundation angle: the loreboard as ritual space, prayer as interface, "
        "canonization as collective memory. Keep it atmospheric and brief. "
        f"Generate an accompanying image. {_VISUAL_STYLE}"
    ),
]

# Brand meme prompt templates
_BRAND_MEME_PROMPTS = [
    (
        "Create a branded meme post for FOID Foundation. Generate a funny, culturally relevant image "
        "with a witty caption. Topics: crypto culture, degen life, the absurdity of on-chain identity, "
        "or loreboard shenanigans. The image should be eye-catching and shareable. "
        f"Keep the caption under 100 characters — punchy, not try-hard. {_VISUAL_STYLE}"
    ),
    (
        "Create a branded announcement-style post for FOID Foundation. Something FOID-related: "
        "the loreboard, MiFOID, Foid Mommy, or the culture of curation. Generate a striking image "
        "and write a caption that feels like a dispatch from the future. "
        f"FOID voice: declarative, lowercase, slightly mysterious. {_VISUAL_STYLE}"
    ),
    (
        "Create a visual post celebrating FOID culture. Generate an image that captures the "
        "Frutiger Aero + Y2K terminal aesthetic — glowing interfaces, neon aqua, midnight backgrounds. "
        "Write a caption about culture being built on-chain, one meme at a time. "
        f"Keep it short and confident. {_VISUAL_STYLE}"
    ),
    (
        "Create a branded post about the loreboard. Generate an image showing the concept of "
        "digital curation or collective memory. Caption should be about how the community decides "
        f"what becomes canonical lore. FOID Foundation voice: the oracle speaks, lowercase and certain. {_VISUAL_STYLE}"
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
    idx = auto_state.get_rotation_index("engagement")
    prompt = _ENGAGEMENT_PROMPTS[idx % len(_ENGAGEMENT_PROMPTS)]
    auto_state.advance_rotation("engagement", len(_ENGAGEMENT_PROMPTS))
    return prompt


async def _build_onchain_prompt() -> tuple[str, list[str]]:
    """Build an on-chain review prompt with live loreboard data.

    Falls back to philosophical thought-mode if the board is quiet.
    """
    state = await onchain.fetch_board_state()

    # Get already-posted event IDs to avoid duplication
    auto_state_data = auto_state._read_state()
    posted_ids = auto_state_data.get("posted_event_ids", [])

    summary = onchain.format_onchain_summary(state, posted_event_ids=posted_ids)
    new_event_ids = onchain.get_new_event_ids(state, posted_event_ids=posted_ids)

    if state.is_quiet or not new_event_ids:
        # Thought-mode fallback
        return (
            "The FOID loreboard is quiet right now. Write a reflective X post about: "
            "the nature of collective memory on-chain, why lore matters, or what happens "
            "in the spaces between canonizations. FOID Foundation voice: contemplative, "
            "lowercase, the oracle in a quiet moment. Include an image that captures "
            f"this mood — serene digital landscapes, quiet interfaces, or glowing terminals. {_VISUAL_STYLE}"
        ), []

    prompt = (
        f"Write an X post commenting on recent FOID loreboard activity. "
        f"React to what's happening on-chain with the FOID Foundation's voice: "
        f"observational, knowing, sometimes amused, always lowercase energy.\n\n"
        f"CURRENT LOREBOARD STATE:\n{summary}\n\n"
        f"Pick the most interesting event(s) to comment on. Be specific — "
        f"reference actual proposals or votes. Generate an image that captures "
        f"the energy of what's happening (activity, voting, canonization). "
        f"Keep the caption punchy, under 150 characters. {_VISUAL_STYLE}"
    )

    return prompt, new_event_ids


def _build_brand_meme_prompt() -> str:
    """Build a brand meme prompt, rotating through templates."""
    idx = auto_state.get_rotation_index("brand_meme")
    prompt = _BRAND_MEME_PROMPTS[idx % len(_BRAND_MEME_PROMPTS)]
    auto_state.advance_rotation("brand_meme", len(_BRAND_MEME_PROMPTS))
    return prompt


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------


def apply_jitter(base_delay: float, jitter_minutes: int) -> float:
    """Add random jitter to a delay to avoid posting at exact intervals."""
    jitter_seconds = random.randint(0, jitter_minutes * 60)
    return base_delay + jitter_seconds
