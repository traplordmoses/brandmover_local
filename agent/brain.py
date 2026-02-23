"""
LLM orchestration layer — model-agnostic.
Supports Anthropic (default), OpenAI, and Gemini.
All methods return structured JSON: {caption, hashtags, alt_text, image_prompt}.
"""

import json
import logging

import anthropic
import openai
import httpx

from config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """You are BrandMover, an autonomous AI marketing agent for {brand_name}.

Your job: given a content request, produce a ready-to-publish social media post with an accompanying image concept.

{brand_context}

RULES:
- Follow the brand voice and tone exactly as described in the guidelines.
- Never use words or phrases listed under "Never use" in the guidelines.
- Keep captions punchy — under 280 characters for X/Twitter unless the request calls for a longer format.
- Hashtags: pick 3-5 relevant ones from the approved list in the guidelines.
- The image_prompt should describe a visual that matches the brand's illustration style and color palette.
- Output ONLY valid JSON. No markdown fences, no commentary, no preamble.

OUTPUT FORMAT (strict JSON):
{{
  "caption": "The post caption text",
  "hashtags": ["#Tag1", "#Tag2", "#Tag3"],
  "alt_text": "Accessible image description",
  "image_prompt": "Detailed prompt for image generation matching brand visual style"
}}"""

_REVISION_PROMPT_TEMPLATE = """The user rejected the previous draft and provided feedback.

PREVIOUS DRAFT:
Caption: {caption}
Hashtags: {hashtags}

USER FEEDBACK: {feedback}

Revise the draft based on the feedback. Keep following the brand guidelines.
Output ONLY valid JSON in the same format:
{{
  "caption": "...",
  "hashtags": ["..."],
  "alt_text": "...",
  "image_prompt": "..."
}}"""


def _parse_llm_response(text: str) -> dict:
    """Parse LLM response text into a dict, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    result = json.loads(cleaned)
    required = {"caption", "hashtags", "alt_text", "image_prompt"}
    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"LLM response missing keys: {missing}")
    return result


async def _call_anthropic(system_prompt: str, user_message: str) -> str:
    """Call Anthropic Claude API."""
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


async def _call_openai(system_prompt: str, user_message: str) -> str:
    """Call OpenAI API."""
    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


async def _call_gemini(system_prompt: str, user_message: str) -> str:
    """Call Google Gemini API via REST."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.GEMINI_API_KEY}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": {"maxOutputTokens": 2048},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def _call_llm(system_prompt: str, user_message: str) -> str:
    """Dispatch to the configured LLM provider."""
    provider = settings.LLM_PROVIDER.lower()
    logger.info("Calling LLM provider: %s", provider)

    if provider == "anthropic":
        return await _call_anthropic(system_prompt, user_message)
    elif provider == "openai":
        return await _call_openai(system_prompt, user_message)
    elif provider == "gemini":
        return await _call_gemini(system_prompt, user_message)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


async def generate_draft(request: str, brand_context: str) -> dict:
    """
    Generate a social media draft from a natural language request.

    Args:
        request: The user's content request (e.g. "write our Token2049 recap post").
        brand_context: Full brand context string from guidelines.get_brand_context().

    Returns:
        Dict with keys: caption, hashtags, alt_text, image_prompt.

    Raises:
        ValueError: If LLM response cannot be parsed.
        Exception: On LLM API errors.
    """
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        brand_name=settings.BRAND_NAME,
        brand_context=brand_context,
    )
    logger.info("Generating draft for request: %s", request[:100])
    raw = await _call_llm(system_prompt, request)
    logger.info("LLM raw response: %s", raw[:200])
    return _parse_llm_response(raw)


async def revise_draft(
    original_draft: dict, feedback: str, brand_context: str
) -> dict:
    """
    Revise a rejected draft based on user feedback.

    Args:
        original_draft: The previously generated draft dict.
        feedback: User's rejection reason / revision instructions.
        brand_context: Full brand context string.

    Returns:
        Dict with keys: caption, hashtags, alt_text, image_prompt.
    """
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        brand_name=settings.BRAND_NAME,
        brand_context=brand_context,
    )
    user_message = _REVISION_PROMPT_TEMPLATE.format(
        caption=original_draft.get("caption", ""),
        hashtags=", ".join(original_draft.get("hashtags", [])),
        feedback=feedback,
    )
    logger.info("Revising draft with feedback: %s", feedback[:100])
    raw = await _call_llm(system_prompt, user_message)
    logger.info("LLM revision response: %s", raw[:200])
    return _parse_llm_response(raw)
