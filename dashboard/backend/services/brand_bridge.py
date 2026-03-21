"""
Brand bridge — aggregates brand data from agent modules for the Mini App.

All agent imports are lazy (inside function bodies) so the dashboard can
start even when agent dependencies are unavailable.
"""

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BRAND_DIR = _PROJECT_ROOT / "brand"
_STATE_DIR = _PROJECT_ROOT / "state"


def _read_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}


# ---------------------------------------------------------------------------
# Brand board
# ---------------------------------------------------------------------------

def get_brand_board() -> dict:
    """Full brand data as a JSON-ready dict."""
    summary: dict = {}
    try:
        from agent.compositor_config import get_brand_summary
        summary = get_brand_summary()
    except Exception as exc:
        logger.warning("Could not load brand summary: %s", exc)
        summary = {
            "brand_name": "Unknown",
            "tagline": "",
            "colors": {},
            "fonts": {},
            "style_keywords": [],
            "avoid_terms": [],
        }

    # Available fonts
    available_fonts: list[str] = []
    try:
        from agent.font_manager import list_available_fonts
        available_fonts = list_available_fonts()
    except Exception as exc:
        logger.debug("Could not list fonts: %s", exc)

    # Logo path
    logo_url: str | None = None
    for candidate in (
        _BRAND_DIR / "assets" / "logo.png",
        _BRAND_DIR / "logo.png",
    ):
        if candidate.exists():
            logo_url = f"/static/brand/{candidate.relative_to(_BRAND_DIR)}"
            break

    # Content types
    content_types = get_content_types_list()

    # Templates
    templates = get_templates_list()

    return {
        **summary,
        "available_fonts": available_fonts,
        "logo_url": logo_url,
        "content_types": content_types,
        "templates": templates,
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def get_templates_list() -> list[dict]:
    """All templates with metadata."""
    try:
        from agent.template_memory import TemplateMemory
        mem = TemplateMemory()
        templates = mem.list_templates()
        return [
            {
                "id": t.id,
                "name": t.name,
                "aspect_ratio": t.aspect_ratio,
                "width": t.width,
                "height": t.height,
                "regions": [
                    {
                        "type": r.type,
                        "x": r.x,
                        "y": r.y,
                        "width": r.width,
                        "height": r.height,
                        "description": r.description,
                    }
                    for r in getattr(t, "regions", [])
                ],
            }
            for t in templates
        ]
    except Exception as exc:
        logger.debug("Could not load templates from TemplateMemory: %s", exc)

    # Fallback: read manifest directly
    manifest_path = _BRAND_DIR / "templates" / "manifest.json"
    if manifest_path.exists():
        data = _read_json(manifest_path, [])
        if isinstance(data, list):
            return data
    return []


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------

def get_content_types_list() -> list[dict]:
    """Content types with metadata."""
    try:
        from agent.content_types import AGENT_SELECTABLE_TYPES, COMPOSITOR_PROFILE_MAP
        return [
            {
                "id": ct,
                "label": ct.replace("_", " ").title(),
                "profile": COMPOSITOR_PROFILE_MAP.get(ct, "default"),
            }
            for ct in AGENT_SELECTABLE_TYPES
        ]
    except Exception as exc:
        logger.warning("Could not load content types: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Generation history
# ---------------------------------------------------------------------------

def get_generation_history(limit: int = 20) -> list[dict]:
    """Recent generations with image URLs mapped to static paths."""
    data = _read_json(_STATE_DIR / "generation_history.json", [])
    if not isinstance(data, list):
        return []

    entries = data[-limit:]

    for entry in entries:
        # Map image paths to static URLs
        image_url = entry.get("image_url") or entry.get("image_path")
        if image_url and not image_url.startswith(("http://", "https://", "/static/")):
            # Assume it is a local path relative to state/images
            path = Path(image_url)
            if path.exists():
                try:
                    rel = path.relative_to(_STATE_DIR / "images")
                    entry["image_url"] = f"/static/images/{rel}"
                except ValueError:
                    pass

    return entries


# ---------------------------------------------------------------------------
# Reference image analysis
# ---------------------------------------------------------------------------

async def analyze_reference_image(image_path: str) -> dict:
    """Run Claude Vision analysis on a reference image."""
    from agent.asset_ingest import analyze_for_library
    return await analyze_for_library(image_path)
