"""
Semantic memory — search past generations by relevance.

Provides keyword-based similarity search over generation_history.json without
requiring external embedding models. Uses TF-IDF-inspired term frequency scoring
for fast, dependency-free matching.

For the future: can be upgraded to use embeddings (OpenAI, Voyage, or local)
for true semantic search. The interface stays the same.

Usage:
    results = search_past_generations("announcement about new partnership", top_k=5)
    # Returns list of {score, entry} sorted by relevance
"""

import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from agent.paths import STATE_DIR as _STATE_DIR
_HISTORY_FILE = _STATE_DIR / "generation_history.json"

# Stopwords for English — filtered out during tokenization
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "was", "are",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "so", "if", "as", "up", "out", "about", "into", "over",
    "after", "under", "between", "through", "during", "before", "just",
    "than", "then", "also", "very", "too", "here", "there", "when", "where",
    "how", "what", "which", "who", "whom", "why", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "your", "my", "its", "our", "their", "we", "you", "he", "she",
    "they", "me", "him", "her", "us", "them", "i",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _build_entry_text(entry: dict) -> str:
    """Build a searchable text representation of a generation entry."""
    parts = [
        entry.get("original_request", ""),
        entry.get("prompt", ""),
        entry.get("content_type", ""),
        entry.get("asset_type", ""),
    ]
    return " ".join(parts)


def _tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    """Compute TF-IDF similarity between query and document."""
    if not doc_tokens:
        return 0.0

    doc_counts = Counter(doc_tokens)
    doc_len = len(doc_tokens)

    score = 0.0
    for token in set(query_tokens):
        tf = doc_counts.get(token, 0) / doc_len
        score += tf * idf.get(token, 0)

    return score


# Temporal decay: entries lose relevance over time
# Half-life of 30 days — entries from 30 days ago score 50% of a fresh entry
_HALF_LIFE_SECONDS = 30 * 86400


def _temporal_weight(timestamp: float) -> float:
    """Apply temporal decay — recent entries score higher."""
    age = time.time() - timestamp
    if age <= 0:
        return 1.0
    return math.exp(-0.693 * age / _HALF_LIFE_SECONDS)  # 0.693 = ln(2)


def _load_history() -> list[dict]:
    """Load generation history from disk."""
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read generation_history.json: %s", e)
        return []


def search_past_generations(
    query: str,
    top_k: int = 5,
    status_filter: str | None = None,
    min_score: float = 0.01,
) -> list[dict]:
    """Search past generations by relevance to a query.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return.
        status_filter: Only include entries with this status (e.g., "approved").
        min_score: Minimum relevance score to include.

    Returns:
        List of {score, entry} dicts sorted by descending relevance.
    """
    entries = _load_history()
    if not entries:
        return []

    # Filter by status if specified
    if status_filter:
        entries = [e for e in entries if e.get("status") == status_filter]

    if not entries:
        return []

    # Tokenize query
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Tokenize all documents
    docs = []
    for entry in entries:
        text = _build_entry_text(entry)
        tokens = _tokenize(text)
        docs.append(tokens)

    # Compute IDF scores
    n_docs = len(docs)
    doc_freq: Counter = Counter()
    for doc_tokens in docs:
        for token in set(doc_tokens):
            doc_freq[token] += 1

    idf = {
        token: math.log(n_docs / (1 + freq))
        for token, freq in doc_freq.items()
    }

    # Score each entry
    scored = []
    for entry, doc_tokens in zip(entries, docs):
        relevance = _tfidf_score(query_tokens, doc_tokens, idf)
        temporal = _temporal_weight(entry.get("timestamp", 0))

        # Boost approved entries (they represent "what worked")
        status_boost = 1.5 if entry.get("status") == "approved" else 1.0

        final_score = relevance * temporal * status_boost

        if final_score >= min_score:
            scored.append({
                "score": round(final_score, 4),
                "entry": {
                    "original_request": entry.get("original_request", "")[:200],
                    "prompt": entry.get("prompt", "")[:300],
                    "content_type": entry.get("content_type", ""),
                    "model_id": entry.get("model_id", ""),
                    "status": entry.get("status", ""),
                    "timestamp": entry.get("timestamp", 0),
                },
            })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def get_approved_examples(content_type: str | None = None, limit: int = 3) -> list[dict]:
    """Get recent approved generations, optionally filtered by content type.

    Useful for giving the agent concrete examples of past successes.
    """
    entries = _load_history()
    approved = [e for e in entries if e.get("status") == "approved"]

    if content_type:
        approved = [e for e in approved if e.get("content_type") == content_type]

    # Return most recent
    results = []
    for entry in approved[-limit:]:
        results.append({
            "original_request": entry.get("original_request", "")[:200],
            "prompt": entry.get("prompt", "")[:300],
            "content_type": entry.get("content_type", ""),
            "model_id": entry.get("model_id", ""),
            "timestamp": entry.get("timestamp", 0),
        })

    return results
