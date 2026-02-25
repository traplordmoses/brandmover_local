"""Scoring engine for agent evaluation traces.

Pure functions — no side effects, no I/O.
"""

from __future__ import annotations

import re

# Tools that are interchangeable — treat either as matching the other
TOOL_ALIASES: dict[str, str] = {
    "img2img": "generate_image",
    "generate_image": "img2img",
}

# Utility tools that the agent may call freely — never count as misfires
UTILITY_TOOLS: frozenset[str] = frozenset({
    "log_resource_usage",
    "read_references",
    "check_figma_design",
})


def _expand_with_aliases(tools: set[str]) -> set[str]:
    """Expand a tool set by adding aliases for every tool present."""
    expanded = set(tools)
    for tool in tools:
        if tool in TOOL_ALIASES:
            expanded.add(TOOL_ALIASES[tool])
    return expanded


def score(scenario: dict, trace: dict) -> dict:
    """Score an agent trace against a scenario's expectations.

    Args:
        scenario: Scenario definition from scenarios.json.
        trace: Serialized AgentResult + metadata from runner.

    Returns:
        Dict of score dimensions.
    """
    expected = set(scenario.get("expected_tools", []))
    actual = set(trace.get("tool_calls_made", []))
    total_calls = len(trace.get("tool_calls_made", []))

    # Expand both sets with aliases so img2img ↔ generate_image match
    expected_expanded = _expand_with_aliases(expected)
    actual_expanded = _expand_with_aliases(actual)

    # Tool correctness: intersection / expected
    tool_correctness = (
        len(expected_expanded & actual_expanded) / len(expected)
        if expected else 1.0
    )

    # Misfire rate: tools called that weren't expected / total calls
    # Exclude utility tools — they're always acceptable
    unexpected = actual - expected_expanded - UTILITY_TOOLS
    tool_misfire_rate = (
        len(unexpected) / total_calls if total_calls > 0 else 0.0
    )

    # Rounds check
    max_rounds = scenario.get("max_rounds", 10)
    turns_used = trace.get("turns_used", 0)
    rounds_ok = turns_used <= max_rounds

    # Forbidden term scan — only check user-facing draft fields,
    # NOT final_text (which contains agent reasoning + hex codes like #000000)
    forbidden = scenario.get("forbidden_terms", [])
    draft = trace.get("draft", {})
    text_corpus = " ".join(
        filter(
            None,
            [
                draft.get("caption", ""),
                draft.get("text", ""),
                draft.get("title", ""),
                draft.get("subtitle", ""),
            ],
        )
    )
    violations = []
    for term in forbidden:
        if re.search(re.escape(term), text_corpus, re.IGNORECASE):
            violations.append(term)

    hallucination_detected = len(violations) > 0

    # Planning: did the agent read guidelines?
    planning_present = "read_brand_guidelines" in actual

    # Observation: did the agent check feedback history?
    observation_incorporated = "read_feedback_history" in actual

    # Latency
    latency_seconds = trace.get("total_time", 0.0)

    # Redundant tool calls: consecutive same-tool calls
    calls_list = trace.get("tool_calls_made", [])
    redundant = sum(
        1
        for i in range(1, len(calls_list))
        if calls_list[i] == calls_list[i - 1]
    )

    # Overall success
    success = (
        tool_correctness > 0.7
        and rounds_ok
        and not hallucination_detected
    )

    return {
        "task_id": scenario["task_id"],
        "tool_correctness": round(tool_correctness, 3),
        "tool_misfire_rate": round(tool_misfire_rate, 3),
        "rounds_ok": rounds_ok,
        "turns_used": turns_used,
        "max_rounds": max_rounds,
        "forbidden_term_violations": violations,
        "hallucination_detected": hallucination_detected,
        "planning_present": planning_present,
        "observation_incorporated": observation_incorporated,
        "latency_seconds": round(latency_seconds, 2),
        "total_tool_calls": total_calls,
        "redundant_tool_calls": redundant,
        "success": success,
    }
