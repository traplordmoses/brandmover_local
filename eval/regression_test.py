"""Regression test — detects behavioral drift between agent versions.

Hashes the master prompt, runs a scenario, and compares against a saved baseline.

Usage:
    python -m eval.regression_test                # compare vs baseline
    python eval/regression_test.py --save-baseline # save new baseline
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eval import PROJECT_ROOT, StateBackup
from eval.runner import _load_scenarios, _run_one

PROMPT_PATH = PROJECT_ROOT / "brand" / "prompts" / "master_prompt_3d.txt"
BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "brand_3d_baseline.json"
TARGET_TASK_ID = "brand_3d_001"

# Regression thresholds
LATENCY_TOLERANCE_PCT = 50  # allow 50% slower before flagging
TOOL_CORRECTNESS_TOLERANCE = 0.1
TURNS_TOLERANCE = 2


def _prompt_hash() -> str:
    """SHA-256 of the master 3D prompt file."""
    if not PROMPT_PATH.exists():
        return "MISSING"
    content = PROMPT_PATH.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _find_scenario(task_id: str) -> dict:
    for sc in _load_scenarios():
        if sc["task_id"] == task_id:
            return sc
    raise ValueError(f"Scenario {task_id} not found")


async def _run_target() -> dict:
    scenario = _find_scenario(TARGET_TASK_ID)
    with StateBackup():
        result = await _run_one(scenario)
    result["prompt_hash"] = _prompt_hash()
    return result


def save_baseline():
    print(f"Running {TARGET_TASK_ID} to save baseline...")
    result = asyncio.run(_run_target())
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Baseline saved to {BASELINE_PATH}")
    print(f"  Prompt hash: {result['prompt_hash'][:16]}...")
    print(f"  Tool correctness: {result['scores']['tool_correctness']}")
    print(f"  Turns used: {result['scores']['turns_used']}")
    print(f"  Latency: {result['scores']['latency_seconds']}s")


def compare():
    if not BASELINE_PATH.exists():
        print("No baseline found. Run with --save-baseline first.")
        sys.exit(1)

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    print(f"Running {TARGET_TASK_ID} for regression comparison...")
    current = asyncio.run(_run_target())

    bs = baseline["scores"]
    cs = current["scores"]
    bp_hash = baseline.get("prompt_hash", "UNKNOWN")
    cp_hash = current.get("prompt_hash", "UNKNOWN")

    checks = {}

    # Prompt hash
    checks["prompt_hash_match"] = bp_hash == cp_hash

    # Latency regression
    if bs["latency_seconds"] > 0:
        latency_pct = (
            (cs["latency_seconds"] - bs["latency_seconds"])
            / bs["latency_seconds"]
            * 100
        )
    else:
        latency_pct = 0
    checks["latency_ok"] = latency_pct <= LATENCY_TOLERANCE_PCT

    # Tool correctness delta
    tc_delta = abs(cs["tool_correctness"] - bs["tool_correctness"])
    checks["tool_correctness_ok"] = tc_delta <= TOOL_CORRECTNESS_TOLERANCE

    # Turns delta
    turns_delta = abs(cs["turns_used"] - bs["turns_used"])
    checks["turns_ok"] = turns_delta <= TURNS_TOLERANCE

    # Print report
    print("\n" + "=" * 60)
    print("Regression Test Report")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Baseline':>10} {'Current':>10} {'Status':>8}")
    print("-" * 60)
    print(
        f"  {'Prompt hash':<25} {bp_hash[:10]:>10} {cp_hash[:10]:>10} "
        f"{'PASS' if checks['prompt_hash_match'] else 'WARN':>8}"
    )
    print(
        f"  {'Latency (s)':<25} {bs['latency_seconds']:>10.1f} {cs['latency_seconds']:>10.1f} "
        f"{'PASS' if checks['latency_ok'] else 'FAIL':>8}"
    )
    print(
        f"  {'Tool correctness':<25} {bs['tool_correctness']:>10.3f} {cs['tool_correctness']:>10.3f} "
        f"{'PASS' if checks['tool_correctness_ok'] else 'FAIL':>8}"
    )
    print(
        f"  {'Turns used':<25} {bs['turns_used']:>10} {cs['turns_used']:>10} "
        f"{'PASS' if checks['turns_ok'] else 'FAIL':>8}"
    )
    print("-" * 60)

    all_pass = all(checks.values())
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    if not all_pass:
        sys.exit(1)


def main():
    if "--save-baseline" in sys.argv:
        save_baseline()
    else:
        compare()


if __name__ == "__main__":
    main()
