"""
Brand ingestion via Claude Vision — extract colors, fonts, and style from images,
then compare against the active brand guidelines.
"""

import base64
import json
import logging
from pathlib import Path

import anthropic

from agent import compositor_config
from config import settings

logger = logging.getLogger(__name__)


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


async def extract_brand_from_image(image_path: str) -> dict:
    """Use Claude Vision to extract brand elements from an image.

    Returns a dict with keys:
        colors: list of {"hex": str, "name": str, "role": str}
        fonts: list of {"family": str, "weight": str, "use": str}
        style_keywords: list[str]
        logo_description: str
    """
    image_data, media_type = _encode_image(image_path)

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image and extract brand identity elements. "
                            "Return ONLY valid JSON with this exact structure:\n"
                            "{\n"
                            '  "colors": [{"hex": "#rrggbb", "name": "Color Name", "role": "primary/accent/background"}],\n'
                            '  "fonts": [{"family": "Font Name", "weight": "Bold/Regular", "use": "display/body"}],\n'
                            '  "style_keywords": ["keyword1", "keyword2"],\n'
                            '  "logo_description": "Brief description of any logo/mark visible"\n'
                            "}\n\n"
                            "For colors, identify the dominant palette (3-6 colors). "
                            "For fonts, identify visible typefaces if recognizable. "
                            "For style_keywords, describe the visual aesthetic (e.g. 'minimalist', 'glass morphism', 'neon'). "
                            "Return ONLY the JSON, no markdown formatting."
                        ),
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude Vision returned non-JSON: %s", raw[:200])
        return {
            "colors": [],
            "fonts": [],
            "style_keywords": [],
            "logo_description": "",
            "_raw_response": raw,
        }


async def diff_against_guidelines(extracted: dict) -> str:
    """Compare extracted brand elements against the active brand config.

    Returns a human-readable compliance report.
    """
    cfg = compositor_config.get_config()
    summary = compositor_config.get_brand_summary()

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": (
                    "Compare these extracted brand elements against the established brand guidelines "
                    "and write a brief compliance report.\n\n"
                    f"EXTRACTED FROM IMAGE:\n{json.dumps(extracted, indent=2)}\n\n"
                    f"BRAND GUIDELINES:\n{json.dumps(summary, indent=2)}\n\n"
                    "Report on:\n"
                    "1. Color match — do extracted colors align with brand palette?\n"
                    "2. Font match — are the fonts consistent with brand typography?\n"
                    "3. Style match — do the style keywords align with brand aesthetic?\n"
                    "4. Overall compliance score (1-10)\n\n"
                    "Be concise. Use plain text, no markdown."
                ),
            }
        ],
    )

    return response.content[0].text.strip()


async def apply_extracted_to_guidelines(extracted: dict) -> str:
    """Merge extracted brand elements into the current guidelines.md.

    Reads the current guidelines, sends both to Claude Sonnet with merge
    instructions, and returns the full updated markdown content.
    """
    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
    current_content = ""
    if guidelines_path.exists():
        current_content = guidelines_path.read_text(encoding="utf-8")

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a brand guidelines editor. Merge the extracted brand elements "
                    "into the existing guidelines markdown. Preserve the existing structure "
                    "and sections. Add new information, update colors/fonts if the extracted "
                    "data provides better or additional values. Do NOT remove existing content "
                    "unless it directly conflicts with the extracted data.\n\n"
                    f"CURRENT GUIDELINES:\n```markdown\n{current_content}\n```\n\n"
                    f"EXTRACTED FROM IMAGE:\n{json.dumps(extracted, indent=2)}\n\n"
                    "Return ONLY the complete updated markdown content, no code fences."
                ),
            }
        ],
    )

    result = response.content[0].text.strip()
    # Strip markdown code fences if the model wrapped it
    if result.startswith("```"):
        result = result.split("\n", 1)[1] if "\n" in result else result[3:]
    if result.endswith("```"):
        result = result[:-3].rstrip()

    return result


async def check_asset_compliance(image_path: str) -> str:
    """Convenience: extract brand elements from an image and compare against guidelines."""
    extracted = await extract_brand_from_image(image_path)
    return await diff_against_guidelines(extracted)
