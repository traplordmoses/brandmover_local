"""
Figma REST API integration.
Fetches design tokens, styles, metadata, and screenshots from Figma files.
Supports template import via parse_figma_url + get_node_children_detailed.
"""

import logging
import re

import httpx

from config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.figma.com/v1"

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

_FIGMA_URL_RE = re.compile(
    r"figma\.com/(?:design|file)/(?P<file_key>[A-Za-z0-9]+)"
    r"(?:/branch/(?P<branch_key>[A-Za-z0-9]+))?"
    r"(?:/[^?]*)?"
    r"(?:\?.*node-id=(?P<node_id>[0-9]+-[0-9]+))?"
)


def parse_figma_url(url: str) -> tuple[str, str]:
    """Parse a Figma URL into (file_key, node_id).

    Supports:
      figma.com/design/:fileKey/:fileName?node-id=1-2
      figma.com/design/:fileKey/branch/:branchKey/:fileName?node-id=1-2
      figma.com/file/:fileKey/...

    Returns (file_key, node_id) with node_id using ":" separator.
    Branch URLs use branchKey as file_key.
    """
    m = _FIGMA_URL_RE.search(url)
    if not m:
        return ("", "")
    file_key = m.group("branch_key") or m.group("file_key")
    node_id_raw = m.group("node_id") or ""
    # Convert "1-2" → "1:2" for API calls
    node_id = node_id_raw.replace("-", ":") if node_id_raw else ""
    return (file_key, node_id)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _is_configured() -> bool:
    """Check if Figma access token is set."""
    return bool(settings.FIGMA_ACCESS_TOKEN)


def _headers() -> dict:
    return {"X-Figma-Token": settings.FIGMA_ACCESS_TOKEN}


def _file_key() -> str:
    return settings.FIGMA_FILE_KEY


def _client() -> httpx.AsyncClient:
    """Return the shared httpx client."""
    from agent._client import get_httpx
    return get_httpx()


async def get_file_styles() -> dict:
    """Fetch all published styles from the Figma file."""
    if not _is_configured():
        return {"error": "Figma not configured. Set FIGMA_ACCESS_TOKEN in .env to enable design integration."}

    try:
        resp = await _client().get(
            f"{_BASE_URL}/files/{_file_key()}/styles",
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        styles = []
        for meta in data.get("meta", {}).get("styles", []):
            styles.append({
                "key": meta.get("key"),
                "name": meta.get("name"),
                "style_type": meta.get("style_type"),
                "description": meta.get("description", ""),
            })

        logger.info("Fetched %d styles from Figma file %s", len(styles), _file_key())
        return {"styles": styles, "count": len(styles)}

    except httpx.HTTPStatusError as e:
        logger.error("Figma API error: %s %s", e.response.status_code, e.response.text[:200])
        return {"error": f"Figma API error: {e.response.status_code}"}
    except Exception as e:
        logger.error("Figma request failed: %s", e)
        return {"error": str(e)}


async def get_design_tokens(node_id: str | None = None) -> dict:
    """Fetch node properties that represent design tokens (colors, typography, spacing)."""
    if not _is_configured():
        return {"error": "Figma not configured. Set FIGMA_ACCESS_TOKEN in .env to enable design integration."}

    node_id = node_id or settings.FIGMA_NODE_ID

    try:
        resp = await _client().get(
            f"{_BASE_URL}/files/{_file_key()}/nodes",
            params={"ids": node_id},
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        nodes = data.get("nodes", {})
        node_data = nodes.get(node_id, {}).get("document", {})

        tokens = {
            "node_id": node_id,
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
            "children_count": len(node_data.get("children", [])),
        }

        # Extract fills/colors if present
        fills = node_data.get("fills", [])
        if fills:
            tokens["colors"] = [
                {
                    "r": round(f["color"]["r"] * 255),
                    "g": round(f["color"]["g"] * 255),
                    "b": round(f["color"]["b"] * 255),
                    "a": f["color"].get("a", 1),
                }
                for f in fills
                if f.get("type") == "SOLID" and "color" in f
            ]

        # Extract text style if present
        style = node_data.get("style", {})
        if style:
            tokens["text_style"] = {
                "font_family": style.get("fontFamily"),
                "font_size": style.get("fontSize"),
                "font_weight": style.get("fontWeight"),
                "line_height": style.get("lineHeightPx"),
            }

        logger.info("Fetched design tokens for node %s", node_id)
        return tokens

    except httpx.HTTPStatusError as e:
        logger.error("Figma API error: %s", e.response.status_code)
        return {"error": f"Figma API error: {e.response.status_code}"}
    except Exception as e:
        logger.error("Figma request failed: %s", e)
        return {"error": str(e)}


async def get_node_screenshot(node_id: str | None = None) -> dict:
    """Get a rendered screenshot URL for a Figma node."""
    if not _is_configured():
        return {"error": "Figma not configured. Set FIGMA_ACCESS_TOKEN in .env to enable design integration."}

    node_id = node_id or settings.FIGMA_NODE_ID

    try:
        resp = await _client().get(
            f"{_BASE_URL}/images/{_file_key()}",
            params={"ids": node_id, "format": "png", "scale": 2},
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        images = data.get("images", {})
        url = images.get(node_id)

        if url:
            logger.info("Got screenshot for node %s", node_id)
            return {"node_id": node_id, "image_url": url}
        else:
            return {"error": f"No image returned for node {node_id}"}

    except httpx.HTTPStatusError as e:
        logger.error("Figma image API error: %s", e.response.status_code)
        return {"error": f"Figma API error: {e.response.status_code}"}
    except Exception as e:
        logger.error("Figma screenshot failed: %s", e)
        return {"error": str(e)}


async def get_node_metadata(node_id: str | None = None) -> dict:
    """Get metadata (name, type, bounding box, children) for a Figma node."""
    if not _is_configured():
        return {"error": "Figma not configured. Set FIGMA_ACCESS_TOKEN in .env to enable design integration."}

    node_id = node_id or settings.FIGMA_NODE_ID

    try:
        resp = await _client().get(
            f"{_BASE_URL}/files/{_file_key()}/nodes",
            params={"ids": node_id},
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        nodes = data.get("nodes", {})
        node_data = nodes.get(node_id, {}).get("document", {})

        metadata = {
            "node_id": node_id,
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
        }

        bbox = node_data.get("absoluteBoundingBox")
        if bbox:
            metadata["bounding_box"] = bbox

        children = node_data.get("children", [])
        if children:
            metadata["children"] = [
                {"name": c.get("name", ""), "type": c.get("type", ""), "id": c.get("id", "")}
                for c in children[:20]  # limit to first 20
            ]
            metadata["total_children"] = len(children)

        logger.info("Fetched metadata for node %s: %s (%s)", node_id, metadata["name"], metadata["type"])
        return metadata

    except httpx.HTTPStatusError as e:
        logger.error("Figma API error: %s", e.response.status_code)
        return {"error": f"Figma API error: {e.response.status_code}"}
    except Exception as e:
        logger.error("Figma metadata failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Template import helpers
# ---------------------------------------------------------------------------

async def get_node_screenshot_with_key(file_key: str, node_id: str) -> dict:
    """Get a rendered screenshot URL for a Figma node with explicit file key.

    Unlike get_node_screenshot() which uses the default FIGMA_FILE_KEY setting,
    this accepts an explicit file key (e.g. parsed from a URL).
    """
    if not _is_configured():
        return {"error": "Figma not configured. Set FIGMA_ACCESS_TOKEN in .env."}

    try:
        resp = await _client().get(
            f"{_BASE_URL}/images/{file_key}",
            params={"ids": node_id, "format": "png", "scale": 2},
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        images = data.get("images", {})
        url = images.get(node_id)

        if url:
            logger.info("Got screenshot for %s/%s", file_key, node_id)
            return {"node_id": node_id, "image_url": url}
        else:
            return {"error": f"No image returned for node {node_id}"}

    except httpx.HTTPStatusError as e:
        logger.error("Figma image API error: %s", e.response.status_code)
        return {"error": f"Figma API error: {e.response.status_code}"}
    except Exception as e:
        logger.error("Figma screenshot failed: %s", e)
        return {"error": str(e)}


async def get_node_children_detailed(file_key: str, node_id: str) -> list[dict]:
    """Return all children of a node with relative bounding boxes, fills, and text styles.

    Used for template import — classifying which layers are content placeholders.
    """
    if not _is_configured():
        return []

    try:
        resp = await _client().get(
            f"{_BASE_URL}/files/{file_key}/nodes",
            params={"ids": node_id},
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        nodes = data.get("nodes", {})
        node_data = nodes.get(node_id, {}).get("document", {})
        parent_bbox = node_data.get("absoluteBoundingBox", {})
        parent_x = parent_bbox.get("x", 0)
        parent_y = parent_bbox.get("y", 0)

        children = []
        for child in node_data.get("children", [])[:30]:
            child_bbox = child.get("absoluteBoundingBox", {})
            # Compute relative positions
            rel_x = round(child_bbox.get("x", 0) - parent_x)
            rel_y = round(child_bbox.get("y", 0) - parent_y)
            width = round(child_bbox.get("width", 0))
            height = round(child_bbox.get("height", 0))

            entry = {
                "name": child.get("name", ""),
                "type": child.get("type", ""),
                "id": child.get("id", ""),
                "x": rel_x, "y": rel_y,
                "width": width, "height": height,
            }

            # Extract fills
            fills = child.get("fills", [])
            if fills:
                entry["fills"] = [
                    {
                        "type": f.get("type", ""),
                        "color": {
                            "r": round(f.get("color", {}).get("r", 0) * 255),
                            "g": round(f.get("color", {}).get("g", 0) * 255),
                            "b": round(f.get("color", {}).get("b", 0) * 255),
                        },
                    }
                    for f in fills
                    if f.get("type") == "SOLID" and "color" in f
                ]

            # Extract text style
            style = child.get("style", {})
            if style:
                entry["text_style"] = {
                    "font_family": style.get("fontFamily"),
                    "font_size": style.get("fontSize"),
                    "font_weight": style.get("fontWeight"),
                }

            # Extract corner radius if present
            cr = child.get("cornerRadius")
            if cr:
                entry["corner_radius"] = round(cr)

            children.append(entry)

        logger.info("Fetched %d children for %s/%s", len(children), file_key, node_id)
        return children

    except Exception as e:
        logger.error("Figma children fetch failed: %s", e)
        return []
