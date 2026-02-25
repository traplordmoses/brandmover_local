"""Regression test — detects behavioral drift between agent versions.

Hashes the system prompt, runs scenarios, and compares against saved baselines.

Usage:
    python eval/regression_test.py                          # compare all vs baselines
    python eval/regression_test.py --save-baseline          # save baselines for all scenarios
    python eval/regression_test.py --task-id brand_3d_001   # run a single scenario
    python eval/regression_test.py --save-baseline --task-id copy_only_002
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

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
SKILL_PROMPT_PATH = PROJECT_ROOT / "agent" / "skill_prompt.py"

# Regression thresholds
LATENCY_TOLERANCE_PCT = 50  # allow 50% slower before flagging
TOOL_CORRECTNESS_TOLERANCE = 0.1
TURNS_TOLERANCE = 2


def _prompt_hash() -> str:
    """SHA-256 of the system prompt source (skill_prompt.py)."""
    if not SKILL_PROMPT_PATH.exists():
        return "MISSING"
    content = SKILL_PROMPT_PATH.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _baseline_path(task_id: str) -> Path:
    return BASELINES_DIR / f"{task_id}_baseline.json"


def _find_scenarios(task_id: str | None) -> list[dict]:
    """Return matching scenarios — all if task_id is None, else just the one."""
    scenarios = _load_scenarios()
    if task_id is None:
        return scenarios
    for sc in scenarios:
        if sc["task_id"] == task_id:
            return [sc]
    raise ValueError(f"Scenario {task_id} not found")


async def _run_scenario(scenario: dict) -> dict:
    result = await _run_one(scenario)
    result["prompt_hash"] = _prompt_hash()
    return result


def save_baseline(task_id: str | None = None):
    """Save baselines for one or all scenarios."""
    scenarios = _find_scenarios(task_id)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    async def _run_all():
        results = []
        with StateBackup():
            for sc in scenarios:
                tid = sc["task_id"]
                print(f"Running {tid} to save baseline...", end=" ", flush=True)
                result = await _run_scenario(sc)
                bp = _baseline_path(tid)
                with open(bp, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                s = result["scores"]
                print(f"OK ({s['latency_seconds']}s)")
                print(f"  Saved to {bp}")
                print(f"  Prompt hash: {result['prompt_hash'][:16]}...")
                print(f"  Tool correctness: {s['tool_correctness']}")
                print(f"  Turns used: {s['turns_used']}")
                results.append(result)
        return results

    results = asyncio.run(_run_all())
    print(f"\nBaselines saved: {len(results)}/{len(scenarios)}")


def _compare_one(scenario: dict, current: dict) -> dict:
    """Compare a single scenario result against its baseline. Returns check dict."""
    tid = scenario["task_id"]
    bp = _baseline_path(tid)

    if not bp.exists():
        return {"task_id": tid, "status": "SKIP", "reason": "no baseline"}

    with open(bp) as f:
        baseline = json.load(f)

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

    # Success must not regress
    checks["success_ok"] = cs["success"] or not bs["success"]

    all_pass = all(checks.values())

    return {
        "task_id": tid,
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "baseline_scores": bs,
        "current_scores": cs,
        "baseline_hash": bp_hash,
        "current_hash": cp_hash,
    }


def compare(task_id: str | None = None):
    """Run scenarios and compare against baselines."""
    scenarios = _find_scenarios(task_id)

    async def _run_all():
        results = []
        with StateBackup():
            for sc in scenarios:
                tid = sc["task_id"]
                print(f"  Running {tid}...", end=" ", flush=True)
                result = await _run_scenario(sc)
                latency = result["scores"]["latency_seconds"]
                print(f"done ({latency}s)")
                results.append((sc, result))
        return results

    print("BrandMover Regression Test")
    print("=" * 72)
    run_results = asyncio.run(_run_all())

    # Compare each
    comparisons = []
    for sc, current in run_results:
        comp = _compare_one(sc, current)
        comparisons.append(comp)

    # Print report
    print("\n" + "=" * 72)
    print("Regression Report")
    print("=" * 72)
    print(f"  {'Task ID':<24} {'Status':<8} {'Prompt':<8} {'TC':<8} {'Turns':<8} {'Latency':<8} {'Pass':<6}")
    print("-" * 72)

    any_fail = False
    for c in comparisons:
        if c["status"] == "SKIP":
            print(f"  {c['task_id']:<24} {'SKIP':<8} {'—':^8} {'—':^8} {'—':^8} {'—':^8} {'—':^6}")
            continue

        ch = c["checks"]
        status = c["status"]
        if status == "FAIL":
            any_fail = True

        def _mark(ok):
            return "OK" if ok else "FAIL"

        print(
            f"  {c['task_id']:<24} {status:<8} "
            f"{_mark(ch['prompt_hash_match']):<8} "
            f"{_mark(ch['tool_correctness_ok']):<8} "
            f"{_mark(ch['turns_ok']):<8} "
            f"{_mark(ch['latency_ok']):<8} "
            f"{_mark(ch['success_ok']):<6}"
        )

    print("-" * 72)
    passed = sum(1 for c in comparisons if c["status"] == "PASS")
    skipped = sum(1 for c in comparisons if c["status"] == "SKIP")
    total = len(comparisons)
    print(f"  {passed}/{total} passed", end="")
    if skipped:
        print(f" ({skipped} skipped — no baseline)", end="")
    print()
    print("=" * 72)

    if any_fail:
        sys.exit(1)


def main():
    # Parse --task-id
    task_id = None
    if "--task-id" in sys.argv:
        idx = sys.argv.index("--task-id")
        if idx + 1 < len(sys.argv):
            task_id = sys.argv[idx + 1]
        else:
            print("Error: --task-id requires a value")
            sys.exit(1)

    if "--save-baseline" in sys.argv:
        save_baseline(task_id)
    else:
        compare(task_id)


if __name__ == "__main__":
    main()
