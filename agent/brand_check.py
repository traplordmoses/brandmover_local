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


def _build_inventory_context() -> str:
    """Load asset_inventory.json and format as context for compliance checking."""
    inv_path = Path(settings.BRAND_FOLDER) / "asset_inventory.json"
    if not inv_path.exists():
        return ""
    try:
        data = json.loads(inv_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    parts: list[str] = []
    colors = data.get("consolidated_colors", [])
    if colors:
        color_lines = [f"  {c.get('name', '?')} {c.get('hex', '?')} ({c.get('role', '?')})" for c in colors[:8]]
        parts.append("Asset library colors:\n" + "\n".join(color_lines))

    styles = data.get("consolidated_style", [])
    if styles:
        parts.append(f"Asset library style: {', '.join(styles[:10])}")

    entries = data.get("entries", [])
    if entries:
        cats = {}
        for e in entries:
            cat = e.get("category", "other")
            cats[cat] = cats.get(cat, 0) + 1
        cat_str = ", ".join(f"{k}: {v}" for k, v in cats.items())
        parts.append(f"Asset library contents: {cat_str}")

    return "\n\n".join(parts) if parts else ""


def _load_raw_guidelines() -> str:
    """Load the full raw guidelines.md text (includes CREATIVE BRIEF, NEVER DO, MASCOT, etc.)."""
    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
    if not guidelines_path.exists():
        return ""
    try:
        return guidelines_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _build_check_prompt(guidelines_context: str, inventory_context: str = "",
                        raw_guidelines: str = "") -> str:
    """Build the structured prompt for the single Claude Vision compliance call."""
    inventory_block = ""
    if inventory_context:
        inventory_block = (
            f"\n\nASSET LIBRARY CONTEXT:\n{inventory_context}\n\n"
            "The brand guidelines were derived from the brand's own uploaded assets. "
            "If this image shares visual DNA with the brand's established identity "
            "(same character designs, same color relationships, same artistic style), "
            "it IS brand-compliant."
        )

    raw_guidelines_block = ""
    if raw_guidelines:
        raw_guidelines_block = (
            f"\n\nFULL BRAND GUIDELINES DOCUMENT:\n{raw_guidelines}\n\n"
            "Use ALL sections above (including CREATIVE BRIEF, NEVER DO, MASCOT, "
            "CHARACTER SYSTEM) when evaluating brand compliance — not just colors and fonts."
        )

    return (
        "You are a brand compliance auditor. Analyze this image against the brand "
        "guidelines below and return a structured JSON compliance report.\n\n"
        "IMPORTANT: Brand compliance means 'does this look like it belongs with the rest "
        "of the brand's visual identity' — not 'does every pixel match a hex code.' "
        "Evaluate the overall visual DNA: character designs, color relationships, "
        "artistic style, and energy.\n\n"
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
        "- For colors, a color is on-palette if it's within a close perceptual range "
        "of any brand color OR if it's a natural variation of the brand's color family. "
        "Do not fail colors for minor shade differences.\n"
        "- For typography, if no text is visible, verdict is \"pass\" with a note. "
        "If guidelines say \"defined by brand assets\" for typography, verdict is \"pass\".\n"
        "- For brand_elements, if no logo or text is expected/visible, verdict is \"pass\".\n"
        "- For visual_style, evaluate whether the image shares the same artistic DNA "
        "as described in the guidelines (character style, illustration approach, energy).\n\n"
        "Return ONLY the JSON, no markdown formatting or code fences."
        + inventory_block
        + raw_guidelines_block
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
    inventory_context = _build_inventory_context()
    raw_guidelines = _load_raw_guidelines()
    prompt = _build_check_prompt(guidelines_context, inventory_context, raw_guidelines)

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
