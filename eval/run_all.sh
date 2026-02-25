#!/usr/bin/env bash
# BrandMover Eval — full suite runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT"

echo "========================================"
echo "  BrandMover Agent Evaluation Suite"
echo "========================================"

EXIT_CODE=0

# 1. Scenario runner
echo ""
echo "[1/3] Running scenario evaluation..."
echo "----------------------------------------"
if python3 "$SCRIPT_DIR/runner.py"; then
    echo "  Scenarios: PASS"
else
    echo "  Scenarios: FAIL"
    EXIT_CODE=1
fi

# 2. Memory test
echo ""
echo "[2/3] Running memory recall test..."
echo "----------------------------------------"
if python3 "$SCRIPT_DIR/memory_test.py"; then
    echo "  Memory: PASS"
else
    echo "  Memory: FAIL"
    EXIT_CODE=1
fi

# 3. Regression test
echo ""
echo "[3/3] Running regression test..."
echo "----------------------------------------"
if python3 "$SCRIPT_DIR/regression_test.py" --save-baseline; then
    echo "  Regression baseline saved: PASS"
else
    echo "  Regression: FAIL"
    EXIT_CODE=1
fi

# Summary
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  All tests passed"
else
    echo "  Some tests failed (exit code $EXIT_CODE)"
fi
echo "========================================"

exit $EXIT_CODE
