# BrandMover Agent Evaluation System

Standalone evaluation module that runs the BrandMover agent through defined scenarios, scores its behavior on multiple dimensions, and surfaces results via a web dashboard.

## Architecture

```
eval/
├── __init__.py          # StateBackup context manager, PROJECT_ROOT
├── scenarios.json       # 5 evaluation scenarios
├── scorer.py            # Pure scoring functions (no I/O)
├── runner.py            # Scenario executor with timeout + error handling
├── memory_test.py       # Feedback recall verification
├── regression_test.py   # Behavioral drift detection
├── dashboard.py         # Flask web UI on port 5001
├── run_all.sh           # Full suite runner
├── results/             # Per-run JSON output (gitignored)
└── baselines/           # Regression baselines
```

## Quick Start

```bash
# Run all scenarios
python eval/runner.py

# Memory recall test
python eval/memory_test.py

# Save regression baseline / compare
python eval/regression_test.py --save-baseline
python eval/regression_test.py

# Web dashboard
python eval/dashboard.py
# → http://localhost:5001

# Full suite
bash eval/run_all.sh
```

## Scoring Dimensions

| Metric | Description |
|--------|-------------|
| `tool_correctness` | Expected tools used / expected tools defined |
| `tool_misfire_rate` | Unexpected tools / total tool calls |
| `rounds_ok` | Agent finished within max_rounds |
| `forbidden_term_violations` | Brand-unsafe terms found in output |
| `hallucination_detected` | Any forbidden terms present |
| `planning_present` | Agent read brand guidelines |
| `observation_incorporated` | Agent checked feedback history |
| `latency_seconds` | Wall-clock time for the run |
| `redundant_tool_calls` | Consecutive duplicate tool calls |
| `success` | Composite: tool_correctness > 0.7 AND rounds_ok AND no violations |

## State Safety

All runs use `StateBackup` — a context manager that snapshots `state.json`, `feedback.json`, and `learned_preferences.md` before the run and restores them afterward. Evaluation never corrupts production state.

## Design Constraint

This module does NOT modify any files in `agent/` or `bot/`. It treats the agent as a black box, calling only its public API (`agent.engine.run_agent`).
