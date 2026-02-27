"""
Image generation via Replicate with smart model routing and prompt enhancement.

Routes to different models based on content_type and prompt keywords.
Automatically enhances prompts with SPLICE-based structure, quality boosters,
brand-specific terms, and negative prompts for higher-quality output.
"""

import base64
import logging
import asyncio
import re
from pathlib import Path

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

# Brand enforcement terms — loaded from guidelines context at runtime.
# This fallback is used when the agent doesn't specify brand aesthetics in the prompt.
_BRAND_TERMS = (
    "on-brand color scheme, high contrast, professional aesthetic, "
    "clean composition, premium feel"
)

# Terms to check — if present, skip adding brand enforcement
_BRAND_INDICATORS = re.compile(
    r"brand.*(color|style|aesthetic)|on.?brand|color scheme|#[0-9A-Fa-f]{6}",
    re.IGNORECASE,
)

# Negative prompt — what to exclude from generation
_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, grainy, washed out, "
    "text artifacts, distorted, deformed, extra limbs, "
    "generic stock photo, clip art, childish, amateur"
)

# Mascot-specific negative prompt
_MASCOT_NEGATIVE = (
    "realistic human, photorealistic, uncanny valley, "
    "low quality, blurry, deformed, extra limbs, "
    "scary, horror, dark mood"
)

# Models that accept negative_prompt parameter
_MODELS_WITH_NEGATIVE = {_MODELS["seedream"]}

# ---------------------------------------------------------------------------
# Locked directives — terms that must survive enhancement verbatim.
# If the raw prompt contains any of these, the enhancer preserves them and
# strips any added phrases that would contradict them.
# ---------------------------------------------------------------------------
_LOCKED_DIRECTIVES = [
    "upright",
    "portrait",
    "70 degree",
    "80 degree",
    "matte black background",
    "rim light",
]

# Map locked directives to contradictory phrases that must be stripped from
# quality profiles and brand terms when the directive is present.
_CONTRADICTION_MAP = {
    "matte black background": re.compile(
        r"black background|dark.*background|dark moody atmosphere", re.IGNORECASE
    ),
    "rim light": re.compile(
        r"rim lighting|rim highlights", re.IGNORECASE
    ),
}


def _extract_locked(prompt: str) -> list[str]:
    """Return locked directives found in the prompt (case-insensitive)."""
    lower = prompt.lower()
    return [d for d in _LOCKED_DIRECTIVES if d.lower() in lower]


def _strip_contradictions(text: str, locked: list[str]) -> str:
    """Remove phrases from enhancement text that contradict locked directives."""
    for directive in locked:
        pattern = _CONTRADICTION_MAP.get(directive)
        if pattern:
            text = pattern.sub("", text)
    # Clean up leftover double commas / leading commas
    text = re.sub(r",\s*,", ",", text)
    text = text.strip(", ")
    return text


def enhance_prompt(raw_prompt: str, content_type: str) -> tuple[str, str]:
    """
    Enhance a raw image prompt with quality boosters and brand terms.

    Uses SPLICE-inspired enrichment:
    - Subject: preserved from raw prompt (agent-written)
    - Parameters + Lighting + Image Type: added from quality profile
    - Composition: preserved from raw prompt if present
    - Enhancers: quality boosters and brand terms

    Locked directives (e.g. "upright", "70 degree", "matte black background")
    are detected in the raw prompt and preserved verbatim.  Enhancement phrases
    that would contradict them are stripped automatically.

    Args:
        raw_prompt: The agent's original image prompt.
        content_type: Content type for profile selection.

    Returns:
        (enhanced_prompt, negative_prompt) tuple.
    """
    ct = content_type.lower()
    prompt = raw_prompt.strip().rstrip(",. ")

    # --- Detect locked directives ---
    locked = _extract_locked(prompt)
    if locked:
        logger.info("Locked directives detected: %s", locked)

    # --- Mascot content has its own aesthetic ---
    if ct == "community" and any(kw in prompt.lower() for kw in ("mascot", "character", "cartoon")):
        quality = (
            "3D CGI render, Pixar-quality smooth shading, "
            "soft studio lighting, high detail, polished finish, "
            "simple clean background"
        )
        enhanced = f"{prompt}, {quality}"
        logger.info("Prompt enhanced (mascot): +%d chars", len(enhanced) - len(raw_prompt))
        return enhanced, _MASCOT_NEGATIVE

    # --- Build enhanced prompt ---
    # Add content-type quality profile (strip contradictions if locked)
    quality = _QUALITY_PROFILES.get(ct, _QUALITY_PROFILES["announcement"])
    if locked:
        quality = _strip_contradictions(quality, locked)

    parts = [prompt]
    if quality:
        parts.append(quality)

    # Add brand terms if not already present (strip contradictions if locked)
    if not _BRAND_INDICATORS.search(prompt):
        brand = _BRAND_TERMS
        if locked:
            brand = _strip_contradictions(brand, locked)
        if brand:
            parts.append(brand)

    enhanced = ", ".join(parts)

    # Log the enhancement
    added = len(enhanced) - len(raw_prompt)
    logger.info(
        "Prompt enhanced: +%d chars (%s profile, brand=%s, locked=%s)",
        added,
        ct,
        "skipped" if _BRAND_INDICATORS.search(prompt) else "added",
        locked or "none",
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


async def generate_img2img(
    prompt: str,
    input_image_path: str,
    strength: float = 0.8,
) -> str | None:
    """
    Generate an image from a reference image + prompt using flux-kontext-pro.

    Args:
        prompt: Text prompt describing the desired output.
        input_image_path: Absolute path to the input image on disk.
        strength: How much to deviate from the input (0.0-1.0).

    Returns:
        URL of the generated image, or None on failure.
    """
    if not settings.REPLICATE_API_TOKEN:
        logger.warning("REPLICATE_API_TOKEN not set — skipping img2img generation")
        return None

    model_id = "black-forest-labs/flux-kontext-pro"
    logger.info("img2img model: %s | input: %s | prompt: %s", model_id, input_image_path, prompt[:120])

    try:
        image_bytes = Path(input_image_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        input_image_data_url = f"data:image/png;base64,{b64}"
    except OSError as e:
        logger.error("Failed to read input image %s: %s", input_image_path, e)
        return None

    api_url = f"{_REPLICATE_BASE_URL}/{model_id}/predictions"

    headers = {
        "Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    payload = {
        "input": {
            "prompt": prompt,
            "input_image": input_image_data_url,
            "strength": strength,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "succeeded" and data.get("output"):
                image_url = _extract_url(data["output"])
                if image_url:
                    logger.info("img2img generated: %s", image_url[:120])
                    return image_url

            poll_url = data.get("urls", {}).get("get")
            if not poll_url:
                logger.error("No poll URL in Replicate img2img response")
                return None

            for _ in range(60):
                await asyncio.sleep(2)
                poll_resp = await client.get(poll_url, headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                status = poll_data.get("status")
                if status == "succeeded":
                    image_url = _extract_url(poll_data.get("output"))
                    if image_url:
                        logger.info("img2img generated: %s", image_url[:120])
                        return image_url
                elif status in ("failed", "canceled"):
                    logger.error("img2img %s: %s", status, poll_data.get("error"))
                    return None

            logger.error("img2img timed out after polling")
            return None

    except Exception as e:
        logger.error("img2img generation failed: %s", e)
        return None
