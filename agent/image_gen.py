"""
Image generation via Replicate with smart model routing.
Routes to different models based on content_type and prompt keywords.
"""

import logging
import asyncio
import re

import httpx

from config import settings

logger = logging.getLogger(__name__)

_REPLICATE_BASE_URL = "https://api.replicate.com/v1/models"

# Model routing table
_MODELS = {
    "flux": "black-forest-labs/flux-1.1-pro",
    "nano-banana": "google/nano-banana-pro",
    "recraft-svg": "recraft-ai/recraft-v3-svg",
    "seedream": "bytedance/seedream-4.5",
}

# Keywords that suggest text overlay in the prompt
_TEXT_OVERLAY_KEYWORDS = re.compile(
    r"text reads|headline|title overlay|bold text|text says|typography|lettering|words?.*overlay",
    re.IGNORECASE,
)


def select_model(content_type: str, prompt: str) -> tuple[str, str]:
    """
    Select the best image model based on content type and prompt.

    Returns:
        (model_id, reason) tuple.
    """
    # Manual override
    if settings.IMAGE_MODEL != "auto":
        return settings.IMAGE_MODEL, "manual override"

    ct = content_type.lower()

    # Text overlay detection
    if ct == "announcement" or _TEXT_OVERLAY_KEYWORDS.search(prompt):
        return _MODELS["nano-banana"], "announcement + text overlay"

    # Brand assets
    if ct == "brand_asset" or any(kw in prompt.lower() for kw in ("icon", "logo", "svg")):
        return _MODELS["recraft-svg"], "brand asset / icon"

    # Lifestyle / event photography
    if ct in ("lifestyle", "event") or "photography" in prompt.lower():
        return _MODELS["seedream"], "lifestyle / photography"

    # Default
    return _MODELS["flux"], "default (general purpose)"


async def generate_image(prompt: str, content_type: str = "announcement") -> str | None:
    """
    Generate an image using Replicate with smart model routing.

    Args:
        prompt: Text prompt describing the desired image.
        content_type: Content type for model selection routing.

    Returns:
        URL of the generated image, or None if generation fails.
    """
    if not settings.REPLICATE_API_TOKEN:
        logger.warning("REPLICATE_API_TOKEN not set — skipping image generation")
        return None

    model_id, reason = select_model(content_type, prompt)
    logger.info("\U0001F3A8 Image model: %s (reason: %s)", model_id, reason)

    api_url = f"{_REPLICATE_BASE_URL}/{model_id}/predictions"

    headers = {
        "Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    payload = {
        "input": {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "output_format": "webp",
        },
    }

    try:
        logger.info("Generating image with prompt: %s", prompt[:120])

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            # If Prefer: wait returned a completed prediction
            if data.get("status") == "succeeded" and data.get("output"):
                output = data["output"]
                image_url = output if isinstance(output, str) else str(output)
                logger.info("Image generated: %s", image_url[:120])
                return image_url

            # Otherwise poll for completion
            poll_url = data.get("urls", {}).get("get")
            if not poll_url:
                logger.error("No poll URL in Replicate response")
                return None

            for _ in range(60):  # Poll up to 60 times (2 min)
                await asyncio.sleep(2)
                poll_resp = await client.get(poll_url, headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                status = poll_data.get("status")
                if status == "succeeded":
                    output = poll_data.get("output")
                    image_url = output if isinstance(output, str) else str(output)
                    logger.info("Image generated: %s", image_url[:120])
                    return image_url
                elif status in ("failed", "canceled"):
                    logger.error("Image generation %s: %s", status, poll_data.get("error"))
                    return None

            logger.error("Image generation timed out after polling")
            return None

    except Exception as e:
        logger.error("Image generation failed: %s", e)
        return None
