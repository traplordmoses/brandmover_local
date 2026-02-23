"""
LLM orchestration layer — model-agnostic.
Supports Anthropic (default), OpenAI, and Gemini.

Two modes:
- Single-shot: generate_draft() / revise_draft() — original behavior
- Pipeline: pipeline_generate() — 4-step (or 3-step "fast") with intermediate callbacks
"""

import json
import logging
import time
from dataclasses import dataclass, field

import anthropic
import openai
import httpx

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-shot prompts (kept as fallback)
# ---------------------------------------------------------------------------

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
  "image_prompt": "Detailed prompt for image generation matching brand visual style",
  "content_type": "announcement"
}}

CONTENT_TYPE values (pick the best fit):
- "announcement" — product launches, updates, news, partnerships
- "lifestyle" — aspirational, day-in-the-life, culture
- "event" — conferences, AMAs, meetups
- "educational" — tutorials, explainers, how-tos
- "brand_asset" — logos, icons, badges, graphics
- "community" — giveaways, polls, engagement posts
- "market_commentary" — market analysis, price action, trends"""

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
  "image_prompt": "...",
  "content_type": "..."
}}"""

# ---------------------------------------------------------------------------
# Pipeline prompts (4 steps)
# ---------------------------------------------------------------------------

_ANALYZE_PROMPT = """You are BrandMover's analysis module for {brand_name}.

Analyze the following content request and identify:
1. Content type (tweet, thread, announcement, educational, meme, etc.)
2. Key themes and topics to cover
3. Target tone (from brand guidelines)
4. Target audience segment
5. Any specific data points, events, or references mentioned

BRAND CONTEXT (summary):
{context_summary}

USER REQUEST:
{request}

Output ONLY valid JSON:
{{
  "content_type": "...",
  "themes": ["...", "..."],
  "tone": "...",
  "audience": "...",
  "key_points": ["...", "..."],
  "references_to_use": ["any relevant reference material names"]
}}"""

_PLAN_PROMPT = """You are BrandMover's creative planning module for {brand_name}.

Based on the analysis below, design a specific content plan:
1. Caption approach — hook, body structure, CTA
2. Hashtag strategy — which tags and why
3. Visual concept — what the image should depict
4. Key differentiator — what makes this post stand out

ANALYSIS:
{analysis}

BRAND CONTEXT (summary):
{context_summary}

USER REQUEST:
{request}

Output ONLY valid JSON:
{{
  "caption_approach": "...",
  "hook_idea": "...",
  "body_structure": "...",
  "cta": "...",
  "hashtag_picks": ["#Tag1", "#Tag2", "#Tag3"],
  "hashtag_reasoning": "...",
  "visual_concept": "...",
  "differentiator": "..."
}}"""

_VERIFY_PROMPT = """You are BrandMover's brand compliance checker for {brand_name}.

Review the creative plan below against the brand guidelines. Check:
1. Tone matches guidelines
2. No forbidden words/phrases used
3. Hashtags are from the approved list (or clearly relevant)
4. Visual concept matches brand style
5. Overall brand alignment score (1-10)

CREATIVE PLAN:
{plan}

BRAND GUIDELINES:
{guidelines}

Output ONLY valid JSON:
{{
  "tone_ok": true,
  "forbidden_words_found": [],
  "hashtags_ok": true,
  "visual_ok": true,
  "alignment_score": 8,
  "issues": ["any issues found"],
  "suggestions": ["any improvements"],
  "approved": true
}}"""

_GENERATE_PROMPT = """You are BrandMover, producing the final social media post for {brand_name}.

You have already analyzed the request, planned the approach, and verified brand compliance.
Now produce the final, publish-ready output.

ANALYSIS:
{analysis}

CREATIVE PLAN:
{plan}

COMPLIANCE CHECK:
{verification}

FULL BRAND CONTEXT:
{brand_context}

USER REQUEST:
{request}

Output ONLY valid JSON — no markdown fences, no commentary:
{{
  "caption": "The post caption text",
  "hashtags": ["#Tag1", "#Tag2", "#Tag3"],
  "alt_text": "Accessible image description",
  "image_prompt": "Detailed prompt for image generation matching brand visual style",
  "content_type": "announcement"
}}"""

_PLAN_AND_VERIFY_PROMPT = """You are BrandMover's creative planning + compliance module for {brand_name}.

Based on the analysis below, design a content plan AND verify it against brand guidelines in one step:
1. Caption approach — hook, body structure, CTA
2. Hashtag strategy
3. Visual concept
4. Brand compliance check

ANALYSIS:
{analysis}

BRAND CONTEXT (summary):
{context_summary}

BRAND GUIDELINES:
{guidelines}

USER REQUEST:
{request}

Output ONLY valid JSON:
{{
  "caption_approach": "...",
  "hook_idea": "...",
  "cta": "...",
  "hashtag_picks": ["#Tag1", "#Tag2", "#Tag3"],
  "visual_concept": "...",
  "alignment_score": 8,
  "issues": [],
  "suggestions": []
}}"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Holds the final draft and all intermediate outputs from the pipeline."""
    draft: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    step_timings: dict = field(default_factory=dict)
    fell_back: bool = False


# ---------------------------------------------------------------------------
# LLM dispatch (shared by single-shot and pipeline)
# ---------------------------------------------------------------------------

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


def _parse_json_response(text: str) -> dict:
    """Parse intermediate JSON from LLM without requiring draft keys."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


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


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _make_context_summary(brand_context: str, max_chars: int = 4000) -> str:
    """Truncate context for intermediate pipeline steps that don't need the full thing."""
    if len(brand_context) <= max_chars:
        return brand_context
    return brand_context[:max_chars] + "\n\n[... truncated for intermediate step ...]"


def _extract_guidelines_section(brand_context: str) -> str:
    """Pull just the guidelines section from the full brand context."""
    marker = "--- BRAND GUIDELINES ---"
    idx = brand_context.find(marker)
    if idx == -1:
        return brand_context[:3000]
    # Find the next section marker
    end_markers = ["--- EXAMPLE POSTS ---", "--- REFERENCE MATERIALS ---"]
    end = len(brand_context)
    for m in end_markers:
        pos = brand_context.find(m, idx + len(marker))
        if pos != -1 and pos < end:
            end = pos
    return brand_context[idx:end].strip()


# ---------------------------------------------------------------------------
# Pipeline mode
# ---------------------------------------------------------------------------

async def pipeline_generate(
    request: str,
    brand_context: str,
    on_step=None,
) -> PipelineResult:
    """
    Run the multi-step generation pipeline.

    Args:
        request: User's content request.
        brand_context: Full brand context from guidelines.get_brand_context().
        on_step: Optional async callback(step_num, total_steps, step_name, summary).
                 Called after each step so the bot can send progress messages.

    Returns:
        PipelineResult with the final draft and intermediate outputs.
    """
    result = PipelineResult()
    mode = settings.PIPELINE_MODE.lower()
    total_steps = 3 if mode == "fast" else 4

    context_summary = _make_context_summary(brand_context)
    guidelines_text = _extract_guidelines_section(brand_context)

    try:
        # --- Step 1: Analyze ---
        t0 = time.time()
        if on_step:
            await on_step(1, total_steps, "Analyze", "Understanding your request...")

        analyze_system = "You are a content analysis assistant. Output only valid JSON."
        analyze_user = _ANALYZE_PROMPT.format(
            brand_name=settings.BRAND_NAME,
            context_summary=context_summary,
            request=request,
        )
        raw_analysis = await _call_llm(analyze_system, analyze_user)
        result.analysis = _parse_json_response(raw_analysis)
        result.step_timings["analyze"] = round(time.time() - t0, 1)
        logger.info("Pipeline step 1 (Analyze) done in %.1fs", result.step_timings["analyze"])

        if mode == "fast":
            # --- Fast mode: merged Plan+Verify ---
            t0 = time.time()
            if on_step:
                await on_step(2, total_steps, "Plan & Verify", "Designing approach and checking brand compliance...")

            pv_system = "You are a creative planning and brand compliance assistant. Output only valid JSON."
            pv_user = _PLAN_AND_VERIFY_PROMPT.format(
                brand_name=settings.BRAND_NAME,
                analysis=json.dumps(result.analysis, indent=2),
                context_summary=context_summary,
                guidelines=guidelines_text,
                request=request,
            )
            raw_pv = await _call_llm(pv_system, pv_user)
            result.plan = _parse_json_response(raw_pv)
            result.verification = result.plan  # merged
            result.step_timings["plan_verify"] = round(time.time() - t0, 1)
            logger.info("Pipeline step 2 (Plan+Verify) done in %.1fs", result.step_timings["plan_verify"])

        else:
            # --- Full mode: separate Plan and Verify ---
            # Step 2: Plan
            t0 = time.time()
            if on_step:
                await on_step(2, total_steps, "Plan", "Designing caption approach and visual concept...")

            plan_system = "You are a creative content planning assistant. Output only valid JSON."
            plan_user = _PLAN_PROMPT.format(
                brand_name=settings.BRAND_NAME,
                analysis=json.dumps(result.analysis, indent=2),
                context_summary=context_summary,
                request=request,
            )
            raw_plan = await _call_llm(plan_system, plan_user)
            result.plan = _parse_json_response(raw_plan)
            result.step_timings["plan"] = round(time.time() - t0, 1)
            logger.info("Pipeline step 2 (Plan) done in %.1fs", result.step_timings["plan"])

            # Step 3: Verify
            t0 = time.time()
            if on_step:
                await on_step(3, total_steps, "Verify", "Checking brand compliance...")

            verify_system = "You are a brand compliance checker. Output only valid JSON."
            verify_user = _VERIFY_PROMPT.format(
                brand_name=settings.BRAND_NAME,
                plan=json.dumps(result.plan, indent=2),
                guidelines=guidelines_text,
            )
            raw_verify = await _call_llm(verify_system, verify_user)
            result.verification = _parse_json_response(raw_verify)
            result.step_timings["verify"] = round(time.time() - t0, 1)
            logger.info("Pipeline step 3 (Verify) done in %.1fs", result.step_timings["verify"])

        # --- Final step: Generate ---
        t0 = time.time()
        if on_step:
            await on_step(total_steps, total_steps, "Generate", "Producing final draft...")

        gen_system = _SYSTEM_PROMPT_TEMPLATE.format(
            brand_name=settings.BRAND_NAME,
            brand_context="",  # context passed in user message
        )
        gen_user = _GENERATE_PROMPT.format(
            brand_name=settings.BRAND_NAME,
            analysis=json.dumps(result.analysis, indent=2),
            plan=json.dumps(result.plan, indent=2),
            verification=json.dumps(result.verification, indent=2),
            brand_context=brand_context,
            request=request,
        )
        raw_draft = await _call_llm(gen_system, gen_user)
        result.draft = _parse_llm_response(raw_draft)
        result.step_timings["generate"] = round(time.time() - t0, 1)
        logger.info("Pipeline step %d (Generate) done in %.1fs", total_steps, result.step_timings["generate"])

        total = sum(result.step_timings.values())
        logger.info("Pipeline complete: %.1fs total (%s)", total, result.step_timings)
        return result

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Pipeline step failed (%s), falling back to single-shot", e)
        result.fell_back = True
        if on_step:
            await on_step(total_steps, total_steps, "Generate", "Falling back to direct generation...")
        result.draft = await generate_draft(request, brand_context)
        return result


# ---------------------------------------------------------------------------
# Single-shot (original behavior, kept as fallback)
# ---------------------------------------------------------------------------

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
