"""
Image generation via Replicate with smart model routing and prompt enhancement.

Routes to different models based on content_type and prompt keywords.
Automatically enhances prompts with SPLICE-based structure, quality boosters,
brand-specific terms, and negative prompts for higher-quality output.
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

# ---------------------------------------------------------------------------
# Prompt enhancement — SPLICE-based enrichment
# ---------------------------------------------------------------------------

# Quality boosters by content type (Lighting + Image Type + Enhancers)
_QUALITY_PROFILES = {
    "announcement": (
        "dramatic rim lighting, volumetric light rays, "
        "product visualization render, sharp focus, "
        "8K resolution, ultra-detailed, professional"
    ),
    "lifestyle": (
        "cinematic golden hour lighting, shallow depth of field, "
        "photorealistic photography, bokeh background, "
        "high detail, natural skin tones"
    ),
    "event": (
        "cinematic wide-angle shot, atmospheric stage lighting, "
        "photojournalistic style, vivid colors, "
        "high detail, dynamic composition"
    ),
    "educational": (
        "clean studio lighting, soft diffused shadows, "
        "technical illustration, sharp focus, "
        "minimalist composition, high contrast"
    ),
    "brand_asset": (
        "flat studio lighting, clean edges, "
        "vector-ready design, scalable, "
        "professional graphic design, minimal"
    ),
    "community": (
        "vibrant studio lighting, playful atmosphere, "
        "3D CGI render, smooth shading, "
        "high detail, polished finish"
    ),
    "market_commentary": (
        "futuristic neon glow, data-driven aesthetic, "
        "HUD visualization style, high contrast, "
        "sharp focus, cinematic color grading"
    ),
}

# BloFin brand enforcement terms
_BRAND_TERMS = (
    "black background, orange (#FF8800) accent lighting and glow effects, "
    "bold futuristic crypto exchange aesthetic, high contrast premium feel, "
    "dark moody atmosphere with dramatic orange rim highlights"
)

# Terms to check — if present, skip adding brand enforcement
_BRAND_INDICATORS = re.compile(
    r"black background|orange.*(glow|accent|highlight)|#FF8800|dark.*background|crypto.*aesthetic",
    re.IGNORECASE,
)

# Negative prompt — what to exclude from generation
_NEGATIVE_PROMPT = (
    "pastel colors, soft gradients, light backgrounds, watercolor, "
    "blurry, low quality, low resolution, grainy, washed out, "
    "text artifacts, distorted, deformed, extra limbs, "
    "generic stock photo, clip art, childish, amateur, "
    "rainbow colors, pink, purple tones, soft aesthetic"
)

# Finny-specific negative prompt
_FINNY_NEGATIVE = (
    "realistic human, photorealistic, uncanny valley, "
    "low quality, blurry, deformed, extra limbs, "
    "scary, horror, dark mood"
)

# Models that accept negative_prompt parameter
_MODELS_WITH_NEGATIVE = {_MODELS["seedream"]}


def enhance_prompt(raw_prompt: str, content_type: str) -> tuple[str, str]:
    """
    Enhance a raw image prompt with quality boosters and brand terms.

    Uses SPLICE-inspired enrichment:
    - Subject: preserved from raw prompt (agent-written)
    - Parameters + Lighting + Image Type: added from quality profile
    - Composition: preserved from raw prompt if present
    - Enhancers: quality boosters and brand terms

    Args:
        raw_prompt: The agent's original image prompt.
        content_type: Content type for profile selection.

    Returns:
        (enhanced_prompt, negative_prompt) tuple.
    """
    ct = content_type.lower()
    prompt = raw_prompt.strip().rstrip(",. ")

    # --- Finny has its own aesthetic ---
    if "finny" in prompt.lower():
        quality = (
            "3D CGI render, Pixar-quality smooth shading, "
            "soft studio lighting, high detail, polished finish, "
            "simple clean background"
        )
        enhanced = f"{prompt}, {quality}"
        logger.info("Prompt enhanced (Finny): +%d chars", len(enhanced) - len(raw_prompt))
        return enhanced, _FINNY_NEGATIVE

    # --- Build enhanced prompt ---
    parts = [prompt]

    # Add content-type quality profile
    quality = _QUALITY_PROFILES.get(ct, _QUALITY_PROFILES["announcement"])
    parts.append(quality)

    # Add brand terms if not already present
    if not _BRAND_INDICATORS.search(prompt):
        parts.append(_BRAND_TERMS)

    enhanced = ", ".join(parts)

    # Log the enhancement
    added = len(enhanced) - len(raw_prompt)
    logger.info(
        "Prompt enhanced: +%d chars (%s profile, brand=%s)",
        added,
        ct,
        "skipped" if _BRAND_INDICATORS.search(prompt) else "added",
    )

    return enhanced, _NEGATIVE_PROMPT


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


def _extract_url(output) -> str | None:
    """Extract image URL from Replicate output (str, list, or dict)."""
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return str(output[0])
    if isinstance(output, dict):
        # Some models return {"url": "..."} or {"image": "..."}
        for key in ("url", "image", "svg"):
            if key in output:
                return str(output[key])
    return str(output) if output else None


def _build_input(model_id: str, prompt: str, negative_prompt: str = "") -> dict:
    """Build model-specific input payload for Replicate."""
    base: dict = {}

    if model_id == _MODELS["flux"]:
        base = {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "output_format": "webp",
        }
    elif model_id == _MODELS["nano-banana"]:
        base = {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "output_format": "jpg",
            "resolution": "2K",
        }
    elif model_id == _MODELS["recraft-svg"]:
        base = {
            "prompt": prompt,
            "size": "1820x1024",
        }
    elif model_id == _MODELS["seedream"]:
        base = {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "size": "2K",
        }
    else:
        base = {"prompt": prompt}

    # Add negative prompt for models that support it
    if negative_prompt and model_id in _MODELS_WITH_NEGATIVE:
        base["negative_prompt"] = negative_prompt
        logger.info("Added negative prompt for %s", model_id.split("/")[-1])

    return base


async def generate_image(prompt: str, content_type: str = "announcement") -> str | None:
    """
    Generate an image using Replicate with smart model routing and prompt enhancement.

    The raw prompt from the agent is automatically enhanced with quality boosters,
    brand terms, and negative prompts before being sent to the model.

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

    # Enhance the prompt with quality boosters and brand terms
    enhanced_prompt, negative_prompt = enhance_prompt(prompt, content_type)
    logger.info("Original prompt (%d chars): %s", len(prompt), prompt[:120])
    logger.info("Enhanced prompt (%d chars): %s", len(enhanced_prompt), enhanced_prompt[:200])

    api_url = f"{_REPLICATE_BASE_URL}/{model_id}/predictions"

    headers = {
        "Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    payload = {
        "input": _build_input(model_id, enhanced_prompt, negative_prompt),
    }

    try:
        logger.info("Generating image with enhanced prompt: %s", enhanced_prompt[:150])

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            # If Prefer: wait returned a completed prediction
            if data.get("status") == "succeeded" and data.get("output"):
                image_url = _extract_url(data["output"])
                if image_url:
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
                    image_url = _extract_url(poll_data.get("output"))
                    if image_url:
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
