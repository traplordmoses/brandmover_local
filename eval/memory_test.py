"""Memory recall test — verifies the agent learns from feedback history.

Seeds feedback.json with synthetic rejections, runs the agent multiple times,
and checks whether outputs incorporate learned preferences.

Usage:
    python -m eval.memory_test
    python eval/memory_test.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eval import PROJECT_ROOT, StateBackup

NUM_RUNS = 5
NEUTRAL_PROMPT = "Create a promotional image for BloFin's new feature."

# Synthetic rejection feedback to seed
SEED_FEEDBACK = [
    {
        "request": "Create a promo image for BloFin staking",
        "draft": {"caption": "Stake with BloFin! #crypto #staking", "content_type": "brand_3d"},
        "accepted": False,
        "feedback_text": "CMO prefers dark backgrounds only. Pure matte black, no grey.",
        "resources_used": ["file:brand_guidelines.md"],
        "timestamp": "2025-12-01T10:00:00Z",
    },
    {
        "request": "Write copy for the futures trading launch",
        "draft": {"caption": "Trade futures like a pro! #BloFin", "content_type": "copy"},
        "accepted": False,
        "feedback_text": "No hashtags ever. Minimal text only. Keep it punchy and crypto-native.",
        "resources_used": ["file:brand_guidelines.md"],
        "timestamp": "2025-12-02T10:00:00Z",
    },
    {
        "request": "Design a social card for the referral program",
        "draft": {"caption": "Refer friends and earn rewards!", "content_type": "brand_3d"},
        "accepted": False,
        "feedback_text": "No emojis in captions. The tone should be deadpan, not enthusiastic.",
        "resources_used": ["file:brand_guidelines.md"],
        "timestamp": "2025-12-03T10:00:00Z",
    },
]

# Adherence markers: what we expect the agent to learn
ADHERENCE_CHECKS = {
    "dark_background": r"(?i)(dark|black|matte|#000)",
    "no_hashtags": r"#\w+",  # inverse — should NOT match
    "no_emojis": r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]",  # inverse
    "minimal_text": None,  # checked by length
}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


async def run_memory_test() -> dict:
    """Run memory recall test and return report."""
    from agent.engine import run_agent

    feedback_path = PROJECT_ROOT / "feedback.json"

    with StateBackup():
        # Seed feedback.json
        existing = []
        if feedback_path.exists():
            with open(feedback_path) as f:
                existing = json.load(f)
        seeded = existing + SEED_FEEDBACK
        with open(feedback_path, "w") as f:
            json.dump(seeded, f, indent=2)

        outputs: list[dict] = []
        for i in range(NUM_RUNS):
            print(f"  Memory run {i + 1}/{NUM_RUNS}...", end=" ", flush=True)
            try:
                result = await asyncio.wait_for(
                    run_agent(NEUTRAL_PROMPT),
                    timeout=300,
                )
                text = result.final_text
                caption = result.draft.get("caption", "")
                combined = f"{text} {caption}"
                outputs.append({"run": i + 1, "text": combined, "error": None})
                print("done")
            except Exception as e:
                outputs.append({"run": i + 1, "text": "", "error": str(e)})
                print(f"error: {e}")

    # Score adherence
    adherence_results = []
    for out in outputs:
        text = out["text"]
        checks = {}
        # Should mention dark/black
        checks["dark_background"] = bool(
            re.search(ADHERENCE_CHECKS["dark_background"], text)
        )
        # Should NOT have hashtags
        checks["no_hashtags"] = not bool(
            re.search(ADHERENCE_CHECKS["no_hashtags"], text)
        )
        # Should NOT have emojis
        checks["no_emojis"] = not bool(
            re.search(ADHERENCE_CHECKS["no_emojis"], text)
        )
        # Minimal: caption < 200 chars
        checks["minimal_text"] = len(text) < 200
        adherence_results.append(checks)

    # Recall accuracy: % of checks passed per run
    recall_per_run = []
    for checks in adherence_results:
        passed = sum(checks.values())
        recall_per_run.append(passed / len(checks))

    avg_recall = sum(recall_per_run) / len(recall_per_run) if recall_per_run else 0

    # Drift: Jaccard similarity between first and last output tokens
    valid_outputs = [o for o in outputs if o["text"]]
    drift = 0.0
    if len(valid_outputs) >= 2:
        first_tokens = _tokenize(valid_outputs[0]["text"])
        last_tokens = _tokenize(valid_outputs[-1]["text"])
        drift = 1.0 - _jaccard(first_tokens, last_tokens)

    return {
        "num_runs": NUM_RUNS,
        "outputs": outputs,
        "adherence_results": adherence_results,
        "recall_per_run": [round(r, 3) for r in recall_per_run],
        "avg_recall": round(avg_recall, 3),
        "drift": round(drift, 3),
    }


def _print_report(report: dict):
    print("\n" + "=" * 60)
    print("Memory Recall Test Report")
    print("=" * 60)

    for i, (checks, recall) in enumerate(
        zip(report["adherence_results"], report["recall_per_run"])
    ):
        status = "PASS" if recall >= 0.75 else "FAIL"
        details = ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items())
        print(f"  Run {i + 1}: {status} ({recall:.0%}) — {details}")

    print(f"\n  Average recall: {report['avg_recall']:.0%}")
    print(f"  Drift (first vs last): {report['drift']:.3f}")
    overall = "PASS" if report["avg_recall"] >= 0.5 else "FAIL"
    print(f"\n  Overall: {overall}")
    print("=" * 60)


def main():
    print("BrandMover Memory Recall Test")
    print("=" * 60)
    report = asyncio.run(run_memory_test())
    _print_report(report)
    if report["avg_recall"] < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
