"""
Simple JSON-file state management for pending approvals.
One pending draft at a time. Stored in state.json at project root.
"""

import json
import logging
import shutil
import time
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).resolve().parent.parent / "state.json"


def _read_state() -> dict:
    """Read state.json, return empty dict if missing or corrupt."""
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read state.json: %s", e)
        return {}


def _write_state(data: dict) -> None:
    """Write state dict to state.json."""
    _STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def has_pending() -> bool:
    """Check if there is a draft pending approval."""
    return "pending" in _read_state()


def get_pending() -> dict | None:
    """Return the pending draft dict, or None if nothing is pending."""
    state = _read_state()
    return state.get("pending")


def save_pending(
    caption: str,
    hashtags: list[str],
    image_url: str | None,
    alt_text: str,
    image_prompt: str,
    original_request: str,
) -> None:
    """
    Save a draft as pending approval.

    Args:
        caption: Generated post caption.
        hashtags: List of hashtags.
        image_url: Generated image URL or None.
        alt_text: Image alt text.
        image_prompt: The prompt used for image generation.
        original_request: The user's original message.
    """
    state = {
        "pending": {
            "caption": caption,
            "hashtags": hashtags,
            "image_url": image_url,
            "alt_text": alt_text,
            "image_prompt": image_prompt,
            "original_request": original_request,
            "timestamp": time.time(),
        }
    }
    _write_state(state)
    logger.info("Saved pending draft for: %s", original_request[:80])


def clear_pending() -> None:
    """Clear any pending draft."""
    _write_state({})
    logger.info("Cleared pending state")


def set_reference_image(path: str) -> None:
    """Store a reference image path for use in the next img2img generation."""
    s = _read_state()
    s["reference_image_path"] = path
    _write_state(s)
    logger.info("Reference image saved: %s", path)


def get_reference_image() -> str | None:
    """Return the stored reference image path, or None."""
    return _read_state().get("reference_image_path")


def clear_reference_image() -> None:
    """Remove the stored reference image path."""
    s = _read_state()
    s.pop("reference_image_path", None)
    _write_state(s)
    logger.info("Reference image cleared")


def set_last_composed(path: str, content_type: str) -> None:
    """Store the path to the last composed image and its content type."""
    s = _read_state()
    s["last_composed_path"] = path
    s["last_composed_content_type"] = content_type
    _write_state(s)


def get_last_composed() -> tuple[str | None, str]:
    """Return (composed_image_path, content_type) or (None, 'default')."""
    s = _read_state()
    return s.get("last_composed_path"), s.get("last_composed_content_type", "default")


def clear_last_composed() -> None:
    """Remove the last composed image metadata."""
    s = _read_state()
    s.pop("last_composed_path", None)
    s.pop("last_composed_content_type", None)
    _write_state(s)


# ---------------------------------------------------------------------------
# Style profiles — managed via brand/styles.json
# ---------------------------------------------------------------------------

_STYLES_FILE = Path(settings.BRAND_FOLDER) / "styles.json"
_STYLES_DIR = Path(settings.BRAND_FOLDER) / "references" / "styles"


def _read_styles() -> dict:
    """Read styles.json, return empty structure if missing or corrupt."""
    if not _STYLES_FILE.exists():
        return {"profiles": {}, "active": {}}
    try:
        data = json.loads(_STYLES_FILE.read_text(encoding="utf-8"))
        data.setdefault("profiles", {})
        data.setdefault("active", {})
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read styles.json: %s", e)
        return {"profiles": {}, "active": {}}


def _write_styles(data: dict) -> None:
    """Write styles dict to styles.json."""
    _STYLES_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_style_profiles() -> dict:
    """Return all profile definitions (name -> {description, strength, prompt_prefix})."""
    return _read_styles().get("profiles", {})


def get_active_profile(content_type: str) -> str | None:
    """Return the active profile name for a content_type, or None."""
    active = _read_styles().get("active", {})
    return active.get(content_type)


def set_active_profile(content_type: str, profile_name: str) -> None:
    """Set the active style profile for a content_type."""
    data = _read_styles()
    if profile_name not in data["profiles"]:
        raise ValueError(f"Profile '{profile_name}' does not exist")
    data["active"][content_type] = profile_name
    _write_styles(data)
    logger.info("Set active profile for %s → %s", content_type, profile_name)


def add_style_profile(
    name: str,
    description: str = "",
    strength: float = 0.3,
    prompt_prefix: str = "",
) -> None:
    """Create a new style profile entry and its directory."""
    data = _read_styles()
    if name in data["profiles"]:
        raise ValueError(f"Profile '{name}' already exists")
    data["profiles"][name] = {
        "description": description,
        "strength": strength,
        "prompt_prefix": prompt_prefix,
    }
    _write_styles(data)
    profile_dir = _STYLES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created style profile: %s (%s)", name, profile_dir)


def get_profile_refs(name: str) -> list[str]:
    """Return sorted list of image paths in a profile folder."""
    profile_dir = _STYLES_DIR / name
    if not profile_dir.is_dir():
        return []
    return sorted(str(p) for p in profile_dir.glob("*.png"))


def add_profile_image(name: str, image_path: str) -> int:
    """Copy image into profile folder as ref_{ts}.png. Caps at 10, prunes oldest.

    Returns the new total image count.
    """
    profile_dir = _STYLES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    dest = profile_dir / f"ref_{ts}.png"
    shutil.copy2(image_path, dest)
    logger.info("Added image to profile %s: %s", name, dest)

    # Cap at 10 images — prune oldest
    existing = sorted(profile_dir.glob("*.png"))
    if len(existing) > 10:
        for old in existing[:-10]:
            old.unlink(missing_ok=True)
            logger.info("Pruned old profile image: %s", old.name)

    return len(list(profile_dir.glob("*.png")))


def remove_active_profile(profile_name: str) -> None:
    """Remove a profile from all active content_type mappings (keep images)."""
    data = _read_styles()
    data["active"] = {
        ct: p for ct, p in data["active"].items() if p != profile_name
    }
    _write_styles(data)
    logger.info("Removed profile %s from all active mappings", profile_name)


def list_profiles() -> list[dict]:
    """Return list of profiles with name, description, image count, active-for list."""
    data = _read_styles()
    result = []
    for name, info in data["profiles"].items():
        active_for = [ct for ct, p in data["active"].items() if p == name]
        result.append({
            "name": name,
            "description": info.get("description", ""),
            "strength": info.get("strength", 0.3),
            "prompt_prefix": info.get("prompt_prefix", ""),
            "image_count": len(get_profile_refs(name)),
            "active_for": active_for,
        })
    return result
