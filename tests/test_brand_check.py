"""Tests for agent.brand_check — compliance response parsing, score calculation,
report formatting, and guidelines context building."""

import json

from agent.brand_check import (
    DIMENSIONS,
    VERDICT_FAIL,
    VERDICT_PARTIAL,
    VERDICT_PASS,
    _build_guidelines_context,
    _empty_report,
    calculate_score,
    format_compliance_report,
    parse_compliance_response,
)
from agent.compositor_config import BrandConfig, ColorEntry, FontEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_FULL_RESPONSE = json.dumps({
    "colors": {
        "verdict": "pass",
        "found": ["#0066ff", "#ff0066", "#0a0a1a"],
        "on_palette": ["#0066ff", "#ff0066", "#0a0a1a"],
        "off_palette": [],
        "findings": "All dominant colors match the brand palette.",
    },
    "typography": {
        "verdict": "partial",
        "found_fonts": ["Orbitron (Bold)", "Arial (Regular)"],
        "expected_fonts": ["Orbitron", "Inter"],
        "findings": "Display font matches. Body font Arial instead of Inter.",
    },
    "visual_style": {
        "verdict": "pass",
        "matched_keywords": ["neon glow", "glass morphism", "dark background"],
        "conflicting_elements": [],
        "findings": "Visual aesthetic matches Frutiger Aero / Y2K style.",
    },
    "brand_elements": {
        "verdict": "fail",
        "logo_present": False,
        "logo_correct": None,
        "text_found": ["SALE NOW"],
        "brand_phrases_used": [],
        "findings": "No logo present. Text 'SALE NOW' is off-brand — corporate hype.",
    },
    "layout": {
        "verdict": "pass",
        "composition_notes": "Dark background, centered composition with glow effects.",
        "findings": "Layout follows brand composition guidelines.",
    },
    "recommendations": [
        "Add brand logo to the top-left corner",
        "Replace Arial body text with Inter",
        "Remove 'SALE NOW' text — use understated brand phrasing instead",
    ],
})

MOCK_ALL_PASS = json.dumps({
    "colors": {"verdict": "pass", "found": ["#0066ff"], "on_palette": ["#0066ff"], "off_palette": [], "findings": "Perfect."},
    "typography": {"verdict": "pass", "found_fonts": ["Orbitron"], "expected_fonts": ["Orbitron"], "findings": "Matches."},
    "visual_style": {"verdict": "pass", "matched_keywords": ["neon"], "conflicting_elements": [], "findings": "Great."},
    "brand_elements": {"verdict": "pass", "logo_present": True, "logo_correct": True, "text_found": [], "brand_phrases_used": [], "findings": "Logo correct."},
    "layout": {"verdict": "pass", "composition_notes": "Good", "findings": "Correct layout."},
    "recommendations": [],
})

MOCK_ALL_FAIL = json.dumps({
    "colors": {"verdict": "fail", "found": ["#ff0000"], "on_palette": [], "off_palette": ["#ff0000"], "findings": "Wrong colors."},
    "typography": {"verdict": "fail", "found_fonts": ["Comic Sans"], "expected_fonts": ["Orbitron"], "findings": "Wrong font."},
    "visual_style": {"verdict": "fail", "matched_keywords": [], "conflicting_elements": ["white background"], "findings": "Off-brand."},
    "brand_elements": {"verdict": "fail", "logo_present": False, "logo_correct": None, "text_found": ["BUY NOW"], "brand_phrases_used": [], "findings": "No logo."},
    "layout": {"verdict": "fail", "composition_notes": "Cluttered", "findings": "Bad layout."},
    "recommendations": ["Fix everything"],
})


# ---------------------------------------------------------------------------
# parse_compliance_response
# ---------------------------------------------------------------------------

class TestParseComplianceResponse:
    def test_full_response(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        assert report["colors"]["verdict"] == VERDICT_PASS
        assert report["typography"]["verdict"] == VERDICT_PARTIAL
        assert report["visual_style"]["verdict"] == VERDICT_PASS
        assert report["brand_elements"]["verdict"] == VERDICT_FAIL
        assert report["layout"]["verdict"] == VERDICT_PASS
        assert len(report["recommendations"]) == 3

    def test_all_dimensions_present(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        for dim in DIMENSIONS:
            assert dim in report

    def test_code_fences_stripped(self):
        fenced = f"```json\n{MOCK_ALL_PASS}\n```"
        report = parse_compliance_response(fenced)
        assert report["colors"]["verdict"] == VERDICT_PASS

    def test_invalid_json_returns_empty_report(self):
        report = parse_compliance_response("this is not json at all")
        for dim in DIMENSIONS:
            assert dim in report
            assert report[dim]["verdict"] == VERDICT_PARTIAL
        assert report["recommendations"] == []

    def test_missing_dimension_gets_default(self):
        partial = json.dumps({
            "colors": {"verdict": "pass", "findings": "Good"},
            # typography, visual_style, brand_elements, layout missing
        })
        report = parse_compliance_response(partial)
        assert report["colors"]["verdict"] == VERDICT_PASS
        assert report["typography"]["verdict"] == VERDICT_PARTIAL
        assert "not evaluated" in report["typography"]["findings"].lower()

    def test_invalid_verdict_normalized_to_partial(self):
        bad_verdict = json.dumps({
            "colors": {"verdict": "amazing", "findings": "Great"},
            "typography": {"verdict": "pass", "findings": "OK"},
            "visual_style": {"verdict": "pass", "findings": "OK"},
            "brand_elements": {"verdict": "pass", "findings": "OK"},
            "layout": {"verdict": "pass", "findings": "OK"},
        })
        report = parse_compliance_response(bad_verdict)
        assert report["colors"]["verdict"] == VERDICT_PARTIAL

    def test_recommendations_not_a_list(self):
        resp = json.dumps({
            "colors": {"verdict": "pass", "findings": "OK"},
            "typography": {"verdict": "pass", "findings": "OK"},
            "visual_style": {"verdict": "pass", "findings": "OK"},
            "brand_elements": {"verdict": "pass", "findings": "OK"},
            "layout": {"verdict": "pass", "findings": "OK"},
            "recommendations": "just a string",
        })
        report = parse_compliance_response(resp)
        assert report["recommendations"] == []

    def test_preserves_detail_fields(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        assert "#0066ff" in report["colors"]["on_palette"]
        assert report["brand_elements"]["logo_present"] is False
        assert "Arial (Regular)" in report["typography"]["found_fonts"]


# ---------------------------------------------------------------------------
# calculate_score
# ---------------------------------------------------------------------------

class TestCalculateScore:
    def test_mixed_score(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        passed, total = calculate_score(report)
        assert total == 5
        assert passed == 3  # colors=pass, typography=partial, visual=pass, brand_elements=fail, layout=pass

    def test_all_pass(self):
        report = parse_compliance_response(MOCK_ALL_PASS)
        passed, total = calculate_score(report)
        assert passed == 5
        assert total == 5

    def test_all_fail(self):
        report = parse_compliance_response(MOCK_ALL_FAIL)
        passed, total = calculate_score(report)
        assert passed == 0
        assert total == 5

    def test_empty_report(self):
        report = _empty_report("Error")
        passed, total = calculate_score(report)
        assert passed == 0
        assert total == 5


# ---------------------------------------------------------------------------
# format_compliance_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_contains_score(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        formatted = format_compliance_report(report)
        assert "3/5" in formatted

    def test_contains_all_dimension_labels(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        formatted = format_compliance_report(report)
        assert "Colors" in formatted
        assert "Typography" in formatted
        assert "Visual Style" in formatted
        assert "Brand Elements" in formatted
        assert "Layout" in formatted

    def test_contains_verdict_icons(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        formatted = format_compliance_report(report)
        assert "\u2705" in formatted  # pass
        assert "\u26a0" in formatted  # partial (warning)
        assert "\u274c" in formatted  # fail

    def test_contains_findings(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        formatted = format_compliance_report(report)
        assert "All dominant colors match" in formatted
        assert "Arial instead of Inter" in formatted

    def test_contains_recommendations(self):
        report = parse_compliance_response(MOCK_FULL_RESPONSE)
        formatted = format_compliance_report(report)
        assert "Recommendations" in formatted
        assert "Add brand logo" in formatted

    def test_no_recommendations_when_empty(self):
        report = parse_compliance_response(MOCK_ALL_PASS)
        formatted = format_compliance_report(report)
        assert "Recommendations" not in formatted

    def test_off_palette_shown(self):
        report = parse_compliance_response(MOCK_ALL_FAIL)
        formatted = format_compliance_report(report)
        assert "#ff0000" in formatted

    def test_conflicting_elements_shown(self):
        report = parse_compliance_response(MOCK_ALL_FAIL)
        formatted = format_compliance_report(report)
        assert "white background" in formatted


# ---------------------------------------------------------------------------
# _build_guidelines_context
# ---------------------------------------------------------------------------

class TestGuidelinesContext:
    def test_includes_brand_name(self):
        cfg = BrandConfig(brand_name="TestBrand", tagline="Build things")
        ctx = _build_guidelines_context(cfg)
        assert "TestBrand" in ctx
        assert "Build things" in ctx

    def test_includes_colors(self):
        cfg = BrandConfig(colors={
            "primary": ColorEntry(role="primary", name="Blue", hex="#0000ff", rgb=(0, 0, 255)),
        })
        ctx = _build_guidelines_context(cfg)
        assert "#0000ff" in ctx
        assert "Blue" in ctx

    def test_includes_fonts(self):
        cfg = BrandConfig(fonts={
            "display": FontEntry(use="display", family="Orbitron", weight="Bold"),
        })
        ctx = _build_guidelines_context(cfg)
        assert "Orbitron" in ctx

    def test_includes_style(self):
        cfg = BrandConfig(
            style_keywords=["glass morphism", "neon glow"],
            visual_style_prompt="dark midnight background",
        )
        ctx = _build_guidelines_context(cfg)
        assert "glass morphism" in ctx
        assert "dark midnight background" in ctx

    def test_includes_avoid_terms(self):
        cfg = BrandConfig(avoid_terms=["flat colors", "white backgrounds"])
        ctx = _build_guidelines_context(cfg)
        assert "flat colors" in ctx

    def test_includes_brand_phrases(self):
        cfg = BrandConfig(brand_phrases=["ship it.", "build the future"])
        ctx = _build_guidelines_context(cfg)
        assert "ship it." in ctx

    def test_empty_config_produces_output(self):
        cfg = BrandConfig()
        ctx = _build_guidelines_context(cfg)
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# _empty_report
# ---------------------------------------------------------------------------

class TestEmptyReport:
    def test_all_dimensions_present(self):
        report = _empty_report("something went wrong")
        for dim in DIMENSIONS:
            assert dim in report
            assert report[dim]["verdict"] == VERDICT_PARTIAL

    def test_error_message_propagated(self):
        report = _empty_report("API timeout")
        assert "API timeout" in report["colors"]["findings"]
