"""
Image generation via Replicate Flux 1.1 Pro.
Uses the Replicate HTTP API directly (no SDK dependency).
"""

import logging
import asyncio

import httpx

from config import settings

logger = logging.getLogger(__name__)

_REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"


async def generate_image(prompt: str) -> str | None:
    """
    Generate an image using Replicate Flux 1.1 Pro.

    Args:
        prompt: Text prompt describing the desired image.

    Returns:
        URL of the generated image, or None if generation fails.
    """
    if not settings.REPLICATE_API_TOKEN:
        logger.warning("REPLICATE_API_TOKEN not set — skipping image generation")
        return None

    headers = {
        "Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    payload = {
        "version": "2e8de10f217bc6be1dece1a7bc0e33e3af29e648a0b89bba2e1b0d0eb3db2b73",
        "input": {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "output_format": "webp",
            "safety_tolerance": 2,
        },
    }

    try:
        logger.info("Generating image with prompt: %s", prompt[:120])

        async with httpx.AsyncClient(timeout=120) as client:
            # Create prediction
            resp = await client.post(_REPLICATE_API_URL, json=payload, headers=headers)
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
