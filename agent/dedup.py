"""
Semantic similarity-based deduplication for generated content.

Before posting content, check if it's too similar to recent posts. This
prevents the bot from generating repetitive captions. Uses TF-IDF cosine
similarity (inspired by promptfoo's similarity assertions) — no external
dependencies beyond the standard library.

Usage:
    result = check_duplicate("Exciting news about our latest feature!")
    if result["is_duplicate"]:
        print(f"Too similar to: {result['similar_to']}")

    # Or check against an arbitrary list (e.g., campaign slot dedup):
    result = check_duplicate_in_list(caption, existing_captions, threshold=0.75)
"""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from agent.paths import STATE_DIR as _STATE_DIR

logger = logging.getLogger(__name__)

_HISTORY_FILE = _STATE_DIR / "generation_history.json"

# Stopwords — common English words that carry little semantic signal.
# Kept deliberately small (~30) to avoid over-filtering short social captions.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "be",
    "was", "has", "have", "do", "not", "so", "if", "as", "we", "our",
    "you", "your", "its", "can", "will",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha characters, remove stopwords.

    Returns a list of tokens (not deduplicated — frequency matters for TF).
    Short tokens (length <= 2) are dropped to reduce noise from fragments.
    """
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using term frequency vectors.

    Builds a shared vocabulary from both texts, computes TF vectors, then
    returns cosine similarity in the range 0.0 (no overlap) to 1.0
    (identical after tokenization). Uses raw TF rather than TF-IDF because
    IDF is degenerate for a two-document corpus (shared terms get IDF=0).

    Uses manual dot product — no numpy needed for small vectors.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Build term frequency vectors over the shared vocabulary.
    # We use raw TF (not TF-IDF) because with only two documents, IDF
    # gives log(2/2)=0 for every shared term — exactly the terms we care
    # about. TF cosine is the right tool for pairwise similarity.
    count_a = Counter(tokens_a)
    count_b = Counter(tokens_b)
    all_vocab = set(count_a) | set(count_b)

    if not all_vocab:
        return 0.0

    # Cosine similarity: dot(a,b) / (|a| * |b|)
    dot = sum(count_a.get(t, 0) * count_b.get(t, 0) for t in all_vocab)
    mag_a = math.sqrt(sum(v * v for v in count_a.values()))
    mag_b = math.sqrt(sum(v * v for v in count_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _load_recent_captions(lookback: int) -> list[str]:
    """Load the last `lookback` captions from generation_history.json.

    Falls back gracefully if the file is missing or malformed.
    """
    if not _HISTORY_FILE.exists():
        return []
    try:
        entries = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("dedup: failed to read generation_history.json: %s", e)
        return []

    # Take the last `lookback` entries, extract the prompt/caption text
    recent = entries[-lookback:] if lookback < len(entries) else entries
    captions = []
    for entry in recent:
        # Prefer 'prompt' (the generated caption), fall back to 'original_request'
        text = entry.get("prompt") or entry.get("original_request") or ""
        if text.strip():
            captions.append(text)
    return captions


def check_duplicate(
    caption: str,
    lookback: int = 20,
    threshold: float = 0.75,
) -> dict:
    """Check if a caption is too similar to recent generations.

    Args:
        caption: The new caption to check.
        lookback: How many recent history entries to compare against.
        threshold: Cosine similarity threshold above which content is
            considered a duplicate (0.0-1.0). Default 0.75.

    Returns:
        {
            "is_duplicate": bool,
            "max_similarity": float,
            "similar_to": str | None,      # most similar caption if above threshold
            "similarity_scores": list[float]
        }
    """
    recent_captions = _load_recent_captions(lookback)
    return check_duplicate_in_list(caption, recent_captions, threshold)


def check_duplicate_in_list(
    caption: str,
    existing_captions: list[str],
    threshold: float = 0.75,
) -> dict:
    """Check if a caption is too similar to any caption in a list.

    Useful for campaign slot dedup where you have an arbitrary list of
    already-generated captions to compare against.

    Args:
        caption: The new caption to check.
        existing_captions: List of existing captions to compare against.
        threshold: Cosine similarity threshold (0.0-1.0). Default 0.75.

    Returns:
        {
            "is_duplicate": bool,
            "max_similarity": float,
            "similar_to": str | None,
            "similarity_scores": list[float]
        }
    """
    if not caption.strip() or not existing_captions:
        return {
            "is_duplicate": False,
            "max_similarity": 0.0,
            "similar_to": None,
            "similarity_scores": [],
        }

    scores: list[float] = []
    max_score = 0.0
    most_similar: str | None = None

    for existing in existing_captions:
        score = _tfidf_cosine(caption, existing)
        scores.append(round(score, 4))
        if score > max_score:
            max_score = score
            most_similar = existing

    is_dup = max_score >= threshold

    return {
        "is_duplicate": is_dup,
        "max_similarity": round(max_score, 4),
        "similar_to": most_similar if is_dup else None,
        "similarity_scores": scores,
    }
