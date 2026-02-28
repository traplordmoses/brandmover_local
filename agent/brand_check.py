"""
Brand compliance checker — single Claude Vision call to analyze an image
against the active brand guidelines and return a structured report.
"""

import base64
import json
import logging
from pathlib import Path

import anthropic

from agent import compositor_config
from config import settings

logger = logging.getLogger(__name__)

# Dimensions checked in the compliance report
DIMENSIONS = ("colors", "typography", "visual_style", "brand_elements", "layout")

VERDICT_PASS = "pass"
VERDICT_PARTIAL = "partial"
VERDICT_FAIL = "fail"

_VERDICT_ICONS = {
    VERDICT_PASS: "\u2705",      # check mark
    VERDICT_PARTIAL: "\u26a0\ufe0f",  # warning
    VERDICT_FAIL: "\u274c",      # cross mark
}

_VERDICT_LABELS = {
    VERDICT_PASS: "On-brand",
    VERDICT_PARTIAL: "Partial match",
    VERDICT_FAIL: "Off-brand",
}


# ---------------------------------------------------------------------------
# Guidelines context builder
# ---------------------------------------------------------------------------

def _build_guidelines_context(cfg: compositor_config.BrandConfig) -> str:
    """Format the active BrandConfig into a text block for the Claude prompt."""
    parts: list[str] = []

    if cfg.brand_name:
        parts.append(f"Brand: {cfg.brand_name}")
    if cfg.tagline:
        parts.append(f"Tagline: {cfg.tagline}")

    # Colors
    if cfg.colors:
        color_lines = []
        for role, entry in cfg.colors.items():
            color_lines.append(f"  {role}: {entry.name} {entry.hex}")
        parts.append("Color palette:\n" + "\n".join(color_lines))

    # Typography
    if cfg.fonts:
        font_lines = []
        for use, entry in cfg.fonts.items():
            font_lines.append(f"  {use}: {entry.family} ({entry.weight})")
        parts.append("Typography:\n" + "\n".join(font_lines))

    # Style
    if cfg.style_keywords:
        parts.append(f"Style keywords: {', '.join(cfg.style_keywords)}")
    if cfg.visual_style_prompt:
        parts.append(f"Visual style: {cfg.visual_style_prompt}")
    if cfg.avoid_terms:
        parts.append(f"Avoid: {', '.join(cfg.avoid_terms)}")

    # Brand phrases
    if cfg.brand_phrases:
        parts.append(f"Brand phrases: {'; '.join(cfg.brand_phrases)}")

    return "\n\n".join(parts)


def _build_check_prompt(guidelines_context: str) -> str:
    """Build the structured prompt for the single Claude Vision compliance call."""
    return (
        "You are a brand compliance auditor. Analyze this image against the brand "
        "guidelines below and return a structured JSON compliance report.\n\n"
        f"BRAND GUIDELINES:\n{guidelines_context}\n\n"
        "Analyze the image and return ONLY valid JSON with this exact structure:\n"
        "{\n"
        '  "colors": {\n'
        '    "verdict": "pass" | "partial" | "fail",\n'
        '    "found": ["#hex1", "#hex2", ...],\n'
        '    "on_palette": ["#hex values that match brand palette"],\n'
        '    "off_palette": ["#hex values NOT in brand palette"],\n'
        '    "findings": "Brief explanation of color compliance"\n'
        "  },\n"
        '  "typography": {\n'
        '    "verdict": "pass" | "partial" | "fail",\n'
        '    "found_fonts": ["Font Name (weight)"],\n'
        '    "expected_fonts": ["Font Name from guidelines"],\n'
        '    "findings": "Brief explanation of typography compliance"\n'
        "  },\n"
        '  "visual_style": {\n'
        '    "verdict": "pass" | "partial" | "fail",\n'
        '    "matched_keywords": ["style keywords that match"],\n'
        '    "conflicting_elements": ["elements that conflict with brand style"],\n'
        '    "findings": "Brief explanation of style compliance"\n'
        "  },\n"
        '  "brand_elements": {\n'
        '    "verdict": "pass" | "partial" | "fail",\n'
        '    "logo_present": true | false,\n'
        '    "logo_correct": true | false | null,\n'
        '    "text_found": ["any visible text strings"],\n'
        '    "brand_phrases_used": ["matching brand phrases if any"],\n'
        '    "findings": "Brief explanation of brand element compliance"\n'
        "  },\n"
        '  "layout": {\n'
        '    "verdict": "pass" | "partial" | "fail",\n'
        '    "composition_notes": "Brief description of layout/composition",\n'
        '    "findings": "Brief explanation of layout compliance"\n'
        "  },\n"
        '  "recommendations": ["Actionable fix 1", "Actionable fix 2"]\n'
        "}\n\n"
        "Rules for verdicts:\n"
        '- "pass": fully on-brand for this dimension\n'
        '- "partial": mostly on-brand with minor deviations\n'
        '- "fail": significant deviation from brand guidelines\n'
        "- For colors, compare found hex values against the brand palette hex values. "
        "A color is on-palette if it's within a close perceptual range of any brand color.\n"
        "- For typography, if no text is visible, verdict is \"pass\" with a note.\n"
        "- For brand_elements, if no logo or text is expected/visible, verdict is \"pass\".\n\n"
        "Return ONLY the JSON, no markdown formatting or code fences."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Strip markdown code fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def parse_compliance_response(raw: str) -> dict:
    """Parse Claude's JSON compliance response into a structured report.

    Returns a normalized dict with per-dimension verdicts. Missing or
    malformed dimensions get a default "partial" verdict with an error note.
    """
    raw = _strip_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Brand check returned non-JSON: %s", raw[:200])
        return _empty_report("Could not parse compliance response from AI.")

    report: dict = {}
    for dim in DIMENSIONS:
        dim_data = data.get(dim)
        if isinstance(dim_data, dict) and "verdict" in dim_data:
            # Normalize verdict value
            v = dim_data["verdict"].lower().strip()
            if v not in (VERDICT_PASS, VERDICT_PARTIAL, VERDICT_FAIL):
                v = VERDICT_PARTIAL
            dim_data["verdict"] = v
            report[dim] = dim_data
        else:
            report[dim] = {
                "verdict": VERDICT_PARTIAL,
                "findings": "Dimension not evaluated by AI.",
            }

    report["recommendations"] = data.get("recommendations", [])
    if not isinstance(report["recommendations"], list):
        report["recommendations"] = []

    return report


def _empty_report(error_msg: str) -> dict:
    """Build an empty report when parsing fails."""
    report: dict = {}
    for dim in DIMENSIONS:
        report[dim] = {
            "verdict": VERDICT_PARTIAL,
            "findings": error_msg,
        }
    report["recommendations"] = []
    return report


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def calculate_score(report: dict) -> tuple[int, int]:
    """Calculate (passed, total) from a compliance report.

    pass = 1 point, partial = 0, fail = 0.
    """
    total = 0
    passed = 0
    for dim in DIMENSIONS:
        dim_data = report.get(dim)
        if not isinstance(dim_data, dict):
            continue
        total += 1
        if dim_data.get("verdict") == VERDICT_PASS:
            passed += 1
    return passed, total


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

_DIM_LABELS = {
    "colors": "Colors",
    "typography": "Typography",
    "visual_style": "Visual Style",
    "brand_elements": "Brand Elements",
    "layout": "Layout",
}


def format_compliance_report(report: dict) -> str:
    """Format a structured compliance report into a readable Telegram message."""
    passed, total = calculate_score(report)
    lines: list[str] = []

    lines.append(f"<b>Brand Compliance Report</b>  ({passed}/{total} checks passed)\n")

    for dim in DIMENSIONS:
        dim_data = report.get(dim, {})
        verdict = dim_data.get("verdict", VERDICT_PARTIAL)
        icon = _VERDICT_ICONS.get(verdict, "?")
        label = _DIM_LABELS.get(dim, dim)
        verdict_label = _VERDICT_LABELS.get(verdict, verdict)

        lines.append(f"{icon} <b>{label}</b>: {verdict_label}")

        # Dimension-specific details
        findings = dim_data.get("findings", "")
        if findings:
            lines.append(f"   {findings}")

        # Colors: show off-palette
        if dim == "colors":
            off = dim_data.get("off_palette", [])
            if off:
                lines.append(f"   Off-palette: {', '.join(off)}")

        # Typography: show found vs expected
        if dim == "typography":
            found = dim_data.get("found_fonts", [])
            if found:
                lines.append(f"   Found: {', '.join(found)}")

        # Visual style: show conflicts
        if dim == "visual_style":
            conflicts = dim_data.get("conflicting_elements", [])
            if conflicts:
                lines.append(f"   Conflicts: {', '.join(conflicts)}")

        lines.append("")  # blank line between dimensions

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        lines.append("<b>Recommendations:</b>")
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

async def check_brand_compliance(image_path: str) -> dict:
    """Analyze an image against brand guidelines in a single Claude Vision call.

    Returns a structured compliance report dict with per-dimension verdicts,
    findings, and recommendations.
    """
    cfg = compositor_config.get_config()
    guidelines_context = _build_guidelines_context(cfg)
    prompt = _build_check_prompt(guidelines_context)

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
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text
    report = parse_compliance_response(raw)
    return report
