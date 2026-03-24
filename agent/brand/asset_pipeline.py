"""
Asset Pipeline — downloads stock footage and resolves asset references
before Remotion rendering.

Uses Pexels API for free stock video/images.
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path

import httpx

from agent.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

ASSETS_DIR = PROJECT_ROOT / "video" / "remotion" / "public" / "assets"

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")


async def search_pexels_video(query: str) -> str | None:
    """Search Pexels for a video, return download URL of smallest HD file."""
    if not PEXELS_API_KEY:
        logger.warning("PEXELS_API_KEY not set, skipping stock footage")
        return None

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": PEXELS_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("videos"):
            return None

        video = data["videos"][0]
        # Find smallest HD file
        files = sorted(
            video.get("video_files", []), key=lambda f: f.get("width", 9999)
        )
        for f in files:
            if f.get("width", 0) >= 720:
                return f["link"]
        return files[0]["link"] if files else None


async def download_asset(url: str, filename: str) -> str:
    """Download a URL to the assets directory, return local path."""
    from agent.net_guard import validate_url
    validate_url(url)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ASSETS_DIR / filename

    if output_path.exists():
        return str(output_path)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)

    logger.info("Downloaded asset: %s -> %s", url[:60], output_path)
    return str(output_path)


async def resolve_storyboard_assets(scene_data: dict) -> dict:
    """Walk storyboard scenes, download any needed assets.

    Modifies scene_data in place, adding assetPath to stock_footage scenes.
    Returns the modified scene_data.
    """
    for i, scene in enumerate(scene_data.get("scenes", [])):
        if scene.get("type") == "stock_footage" and scene.get("query"):
            query = scene["query"]
            try:
                url = await search_pexels_video(query)
                if url:
                    ext = ".mp4"
                    if ".jpg" in url or ".jpeg" in url:
                        ext = ".jpg"
                    elif ".png" in url:
                        ext = ".png"

                    filename = f"stock_{i}_{uuid.uuid4().hex[:6]}{ext}"
                    local_path = await download_asset(url, filename)
                    scene["assetPath"] = local_path
                    logger.info(
                        "Resolved stock footage for scene %d: %s", i, query
                    )
                else:
                    logger.warning("No stock footage found for: %s", query)
            except Exception as e:
                logger.warning(
                    "Failed to download stock footage for '%s': %s", query, e
                )

    return scene_data
