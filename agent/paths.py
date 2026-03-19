"""
Central path definitions for state and brand directories.

All modules should import paths from here rather than computing them locally.
This enables multi-brand isolation via the STATE_FOLDER and BRAND_FOLDER settings.
"""

from pathlib import Path
from config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(settings.STATE_FOLDER)
BRAND_DIR = Path(settings.BRAND_FOLDER)


def migrate_state_file(old_path: Path, new_path: Path) -> None:
    """Move a state file from old location to new if needed."""
    if old_path.exists() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(old_path), str(new_path))
