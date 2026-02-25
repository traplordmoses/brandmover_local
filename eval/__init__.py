"""BrandMover Agent Evaluation System.

Provides tools for running the agent through defined scenarios,
scoring behavior on multiple dimensions, and surfacing results.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_STATE_FILES = ["state.json", "feedback.json", "learned_preferences.md"]


class StateBackup:
    """Context manager that backs up and restores agent state files.

    Ensures evaluation runs don't corrupt production state.
    """

    def __init__(self):
        self._backups: dict[Path, Path] = {}

    def __enter__(self):
        for name in _STATE_FILES:
            src = PROJECT_ROOT / name
            if src.exists():
                dst = src.with_suffix(src.suffix + ".evalbak")
                shutil.copy2(src, dst)
                self._backups[src] = dst
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for src, dst in self._backups.items():
            shutil.copy2(dst, src)
            dst.unlink(missing_ok=True)
        return False
