"""
Topic bank — a rotating menu of content angles the agent can draw from.
Used primarily by the heartbeat's proactive trigger, but also available
to enrich scheduled slot prompts.

The bank is a JSON file with categories and angles. Each angle tracks
when it was last used so the agent avoids repetition.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPIC_BANK_PATH = _PROJECT_ROOT / "state" / "topic_bank.json"


@dataclass
class TopicAngle:
    id: str                        # unique slug: "product_highlight_fees"
    category: str                  # "product", "education", "community", "culture", "engagement"
    angle: str                     # "Highlight low trading fees vs competitors"
    example_hooks: list[str] = field(default_factory=list)
    last_used: float | None = None # timestamp of last use, None if never used
    times_used: int = 0
    retired: bool = False


@dataclass
class TopicBank:
    categories: list[str] = field(default_factory=list)
    angles: list[dict] = field(default_factory=list)  # serialized TopicAngles
    last_refreshed: float | None = None
    version: int = 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_bank() -> TopicBank:
    """Read topic bank from disk. Returns empty bank if doesn't exist."""
    try:
        if TOPIC_BANK_PATH.exists():
            data = json.loads(TOPIC_BANK_PATH.read_text())
            return TopicBank(
                categories=data.get("categories", []),
                angles=data.get("angles", []),
                last_refreshed=data.get("last_refreshed"),
                version=data.get("version", 1),
            )
    except Exception as e:
        logger.warning("Failed to load topic bank: %s", e)
    return TopicBank()


def save_bank(bank: TopicBank) -> None:
    """Write topic bank to disk."""
    TOPIC_BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        TOPIC_BANK_PATH.write_text(json.dumps(asdict(bank), indent=2, default=str))
    except Exception as e:
        logger.error("Failed to save topic bank: %s", e)


# ---------------------------------------------------------------------------
# Query & mutation
# ---------------------------------------------------------------------------

def _get_angle_performance(angle: dict) -> float:
    """Return a performance boost multiplier for an angle based on analytics.

    Checks ``state/performance_data.json`` to see if the content types
    associated with this angle's category have historically performed well.
    Returns a multiplier >= 1.0 (higher = better performing category).
    """
    # Lazy mapping from topic-bank categories to content types they most
    # commonly feed into.  An angle in "engagement" category typically
    # produces meme / engagement content, etc.
    _CATEGORY_TO_CONTENT_TYPES: dict[str, list[str]] = {
        "product": ["announcement", "brand_asset"],
        "education": ["educational", "advice"],
        "community": ["community", "engagement"],
        "culture": ["meme", "lifestyle"],
        "engagement": ["engagement", "meme"],
    }

    category = angle.get("category", "")
    related_types = _CATEGORY_TO_CONTENT_TYPES.get(category, [])
    if not related_types:
        return 1.0

    try:
        from agent.scheduling.content_planner import _load_performance_weights
        weights = _load_performance_weights()
    except Exception:
        return 1.0

    if not weights:
        return 1.0

    # Average the engagement multiplier of related content types
    relevant = [weights[ct] for ct in related_types if ct in weights]
    if not relevant:
        return 1.0

    avg_weight = sum(relevant) / len(relevant)
    # Clamp to [0.5, 2.0] — we don't want performance to completely dominate freshness
    return max(0.5, min(2.0, avg_weight))


def get_fresh_angles(
    n: int = 5,
    category: str | None = None,
    use_performance: bool = True,
) -> list[dict]:
    """Return the N least-recently-used non-retired angles.

    Sort: never-used first, then oldest last_used.  When *use_performance*
    is True (default), angles whose categories map to high-engagement content
    types are boosted in the ranking via ``_get_angle_performance()``.
    """
    bank = load_bank()
    candidates = [
        a for a in bank.angles
        if not a.get("retired", False)
        and (category is None or a.get("category") == category)
    ]

    def sort_key(a: dict) -> tuple:
        lu = a.get("last_used")
        # Base freshness score: never-used = 0, then ascending by last_used
        freshness = (0 if lu is None else 1, lu or 0)

        if not use_performance:
            return freshness

        # Invert performance multiplier so higher performing sorts earlier
        # (lower sort key = picked sooner).  A multiplier of 2.0 becomes 0.5.
        perf_multiplier = _get_angle_performance(a)
        perf_sort = 1.0 / perf_multiplier if perf_multiplier > 0 else 1.0

        return (freshness[0], freshness[1] * perf_sort)

    candidates.sort(key=sort_key)
    return candidates[:n]


def mark_angle_used(angle_id: str) -> None:
    """Update last_used and increment times_used for an angle."""
    bank = load_bank()
    for angle in bank.angles:
        if angle.get("id") == angle_id:
            angle["last_used"] = time.time()
            angle["times_used"] = angle.get("times_used", 0) + 1
            save_bank(bank)
            logger.info("Topic bank: marked angle %s as used (%d total)", angle_id, angle["times_used"])
            return
    logger.warning("Topic bank: angle %s not found", angle_id)


def retire_angle(angle_id: str) -> bool:
    """Set an angle as retired. Returns True if found."""
    bank = load_bank()
    for angle in bank.angles:
        if angle.get("id") == angle_id:
            angle["retired"] = True
            save_bank(bank)
            logger.info("Topic bank: retired angle %s", angle_id)
            return True
    return False


def _slugify(category: str, angle: str) -> str:
    """Generate an ID slug from category + angle."""
    text = f"{category}_{angle}"
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    # Truncate to reasonable length
    return text[:60]


def add_angle(
    category: str, angle: str, example_hooks: list[str] | None = None
) -> str:
    """Add a new angle to the bank. Returns the generated ID."""
    bank = load_bank()
    angle_id = _slugify(category, angle)

    # Deduplicate: skip if same ID already exists
    existing_ids = {a.get("id") for a in bank.angles}
    if angle_id in existing_ids:
        # Append a counter
        i = 2
        while f"{angle_id}_{i}" in existing_ids:
            i += 1
        angle_id = f"{angle_id}_{i}"

    entry = asdict(TopicAngle(
        id=angle_id,
        category=category,
        angle=angle,
        example_hooks=example_hooks or [],
    ))
    bank.angles.append(entry)

    # Ensure category is tracked
    if category not in bank.categories:
        bank.categories.append(category)

    save_bank(bank)
    logger.info("Topic bank: added angle %s (category=%s)", angle_id, category)
    return angle_id


# ---------------------------------------------------------------------------
# Seeding — starter bank for fresh installs
# ---------------------------------------------------------------------------

STARTER_CATEGORIES = ["product", "education", "community", "culture", "engagement"]

STARTER_ANGLES = [
    {
        "category": "product",
        "angle": "Highlight a specific product feature or capability",
        "example_hooks": ["Did you know you can...", "Most people miss this feature"],
    },
    {
        "category": "product",
        "angle": "Compare an advantage vs competitors (without naming them)",
        "example_hooks": ["While others charge...", "Not all platforms are built the same"],
    },
    {
        "category": "education",
        "angle": "Explain a concept relevant to your audience",
        "example_hooks": ["Here's how X actually works", "Most people get X wrong"],
    },
    {
        "category": "education",
        "angle": "Share a practical tip or workflow",
        "example_hooks": ["Pro tip:", "The fastest way to..."],
    },
    {
        "category": "community",
        "angle": "Celebrate a community milestone or user story",
        "example_hooks": ["Shoutout to...", "Our community just hit..."],
    },
    {
        "category": "community",
        "angle": "Ask a question to spark discussion",
        "example_hooks": ["What's your take on...", "Hot take:"],
    },
    {
        "category": "culture",
        "angle": "React to a trending topic in your industry",
        "example_hooks": ["Everyone's talking about X, here's what matters", "The real story behind..."],
    },
    {
        "category": "culture",
        "angle": "Behind-the-scenes or team personality",
        "example_hooks": ["What building X actually looks like", "POV:"],
    },
    {
        "category": "engagement",
        "angle": "Poll or this-or-that",
        "example_hooks": ["X or Y?", "Rate your..."],
    },
    {
        "category": "engagement",
        "angle": "Meme or humor post relevant to your niche",
        "example_hooks": ["When you...", "Nobody: ... Us:"],
    },
]


def seed_bank_if_empty() -> bool:
    """Create starter bank if empty. Returns True if seeded."""
    bank = load_bank()
    if bank.angles:
        return False

    bank.categories = list(STARTER_CATEGORIES)
    for starter in STARTER_ANGLES:
        entry = asdict(TopicAngle(
            id=_slugify(starter["category"], starter["angle"]),
            category=starter["category"],
            angle=starter["angle"],
            example_hooks=starter.get("example_hooks", []),
        ))
        bank.angles.append(entry)

    bank.last_refreshed = time.time()
    save_bank(bank)
    logger.info("Topic bank: seeded with %d starter angles", len(bank.angles))
    return True
