"""
Asset library — indexed storage for generated, uploaded, and approved brand images.

Images are stored in brand/assets/library/.
Index is maintained in brand/asset_library.json.
"""

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_LIBRARY_DIR = Path(settings.BRAND_FOLDER) / "assets" / "library"
_INDEX_PATH = Path(settings.BRAND_FOLDER) / "asset_library.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LibraryEntry:
    id: str = ""
    path: str = ""             # relative to brand/assets/library/
    source: str = "generated"  # "generated" | "uploaded" | "approved"
    content_type: str = ""
    prompt: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    used_count: int = 0
    last_used: float = 0.0


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

def _load_index() -> list[dict]:
    if not _INDEX_PATH.exists():
        return []
    try:
        data = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return data.get("entries", [])
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(entries: list[dict]) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_PATH.write_text(
        json.dumps({"entries": entries}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add(image_path: str, source: str, content_type: str, prompt: str = "", tags: list[str] | None = None) -> LibraryEntry:
    """Add an image to the library. Copies the file to brand/assets/library/."""
    _LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    entry_id = uuid.uuid4().hex[:12]
    src = Path(image_path)
    ext = src.suffix or ".png"
    dest_name = f"{entry_id}{ext}"
    dest = _LIBRARY_DIR / dest_name

    # Copy file (handle both local files and URLs later)
    if src.exists():
        shutil.copy2(str(src), str(dest))
    else:
        # For URLs, store the path as-is (download happens at use time)
        dest_name = image_path

    entry = LibraryEntry(
        id=entry_id,
        path=dest_name,
        source=source,
        content_type=content_type,
        prompt=prompt,
        tags=tags or [],
        created_at=time.time(),
    )

    entries = _load_index()
    entries.append(asdict(entry))
    _save_index(entries)

    logger.info("Asset library: added %s (%s, %s)", entry_id, source, content_type)
    return entry


def find(query: str = "", content_type: str = "", limit: int = 5) -> list[LibraryEntry]:
    """Search the library by keyword or content type."""
    entries = _load_index()
    results = []

    query_lower = query.lower()
    for e in entries:
        # Filter by content_type
        if content_type and e.get("content_type", "") != content_type:
            continue
        # Filter by query (search prompt and tags)
        if query_lower:
            prompt_lower = e.get("prompt", "").lower()
            tags_lower = " ".join(e.get("tags", [])).lower()
            if query_lower not in prompt_lower and query_lower not in tags_lower:
                continue
        results.append(LibraryEntry(**e))

    # Sort by recency (newest first)
    results.sort(key=lambda x: x.created_at, reverse=True)
    return results[:limit]


def suggest(prompt: str, content_type: str) -> LibraryEntry | None:
    """Find a library entry that matches a prompt by keyword overlap.

    Returns the best match or None if no good match exists.
    """
    entries = _load_index()
    if not entries:
        return None

    prompt_words = set(prompt.lower().split())
    best_match = None
    best_score = 0

    for e in entries:
        # Must match content type
        if content_type and e.get("content_type", "") != content_type:
            continue

        entry_prompt = e.get("prompt", "")
        entry_words = set(entry_prompt.lower().split())
        tags_words = set(w.lower() for w in e.get("tags", []))
        all_words = entry_words | tags_words

        overlap = len(prompt_words & all_words)
        if overlap > best_score and overlap >= 3:  # minimum 3 words overlap
            best_score = overlap
            best_match = e

    if best_match:
        return LibraryEntry(**best_match)
    return None


def mark_used(entry_id: str) -> None:
    """Increment the used_count and update last_used for an entry."""
    entries = _load_index()
    for e in entries:
        if e.get("id") == entry_id:
            e["used_count"] = e.get("used_count", 0) + 1
            e["last_used"] = time.time()
            break
    _save_index(entries)


def list_all(content_type: str = "", limit: int = 20) -> list[LibraryEntry]:
    """List all library entries, optionally filtered by content type."""
    return find(query="", content_type=content_type, limit=limit)


def get_library_path(entry: LibraryEntry) -> Path | None:
    """Get the full path for a library entry's image."""
    if entry.path.startswith("http"):
        return None  # URL-based entry
    p = _LIBRARY_DIR / entry.path
    return p if p.exists() else None
