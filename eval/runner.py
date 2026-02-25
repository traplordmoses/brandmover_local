"""Evaluation runner — executes scenarios against the live agent.

Usage:
    python -m eval.runner
    python eval/runner.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eval import PROJECT_ROOT, StateBackup
from eval.scorer import score

SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PER_SCENARIO_TIMEOUT = 300  # seconds


def _load_scenarios() -> list[dict]:
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def _serialize_result(result) -> dict:
    """Convert an AgentResult dataclass to a plain dict."""
    return {
        "final_text": result.final_text,
        "draft": result.draft,
        "image_url": result.image_url,
        "image_urls": result.image_urls,
        "tool_calls_made": result.tool_calls_made,
        "turns_used": result.turns_used,
        "total_time": result.total_time,
        "resources": result.resources.to_list(),
    }


async def _run_one(scenario: dict) -> dict:
    """Run a single scenario and return the trace + scores."""
    from agent.engine import run_agent

    tool_trace: list[dict] = []

    async def on_tool_call(tool_name: str, description: str):
        tool_trace.append(
            {
                "tool": tool_name,
                "description": description,
                "timestamp": time.time(),
            }
        )

    request = scenario["request"]
    revision_context = scenario.get("revision_context")

    try:
        result = await asyncio.wait_for(
            run_agent(request, on_tool_call=on_tool_call, revision_context=revision_context),
            timeout=PER_SCENARIO_TIMEOUT,
        )
        trace = _serialize_result(result)
        trace["tool_trace"] = tool_trace
        trace["error"] = None
    except asyncio.TimeoutError:
        trace = {
            "final_text": "",
            "draft": {},
            "image_url": None,
            "image_urls": [],
            "tool_calls_made": [t["tool"] for t in tool_trace],
            "turns_used": 0,
            "total_time": PER_SCENARIO_TIMEOUT,
            "resources": [],
            "tool_trace": tool_trace,
            "error": f"Timeout after {PER_SCENARIO_TIMEOUT}s",
        }
    except Exception as e:
        trace = {
            "final_text": "",
            "draft": {},
            "image_url": None,
            "image_urls": [],
            "tool_calls_made": [t["tool"] for t in tool_trace],
            "turns_used": 0,
            "total_time": 0.0,
            "resources": [],
            "tool_trace": tool_trace,
            "error": f"{type(e).__name__}: {e}",
        }

    scores = score(scenario, trace)
    return {"scenario": scenario, "trace": trace, "scores": scores}


async def run_all() -> list[dict]:
    """Run all scenarios sequentially, return list of results."""
    scenarios = _load_scenarios()
    results = []

    with StateBackup():
        for sc in scenarios:
            task_id = sc["task_id"]
            print(f"  Running {task_id}...", end=" ", flush=True)
            result = await _run_one(sc)
            status = "PASS" if result["scores"]["success"] else "FAIL"
            latency = result["scores"]["latency_seconds"]
            print(f"{status}  ({latency}s)")

            # Save individual result
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            out_path = RESULTS_DIR / f"{task_id}_{ts}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)

    return results


def _print_summary(results: list[dict]):
    """Print a summary table to stdout."""
    print("\n" + "=" * 72)
    print(f"{'Task ID':<22} {'Status':<8} {'Tools':<8} {'Turns':<8} {'Time':>8}")
    print("-" * 72)

    passed = 0
    for r in results:
        s = r["scores"]
        status = "PASS" if s["success"] else "FAIL"
        if s["success"]:
            passed += 1
        print(
            f"{s['task_id']:<22} {status:<8} "
            f"{s['tool_correctness']:<8.1%} "
            f"{s['turns_used']:<8} "
            f"{s['latency_seconds']:>7.1f}s"
        )

    print("-" * 72)
    print(f"Total: {passed}/{len(results)} passed")
    print("=" * 72)


def main():
    print("BrandMover Agent Evaluation")
    print("=" * 72)
    results = asyncio.run(run_all())
    _print_summary(results)
    # Exit 1 if any scenario failed
    if any(not r["scores"]["success"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
