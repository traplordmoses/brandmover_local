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
