"""Shared fixtures for BrandMover tests."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path so agent/config imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Set minimal env vars so settings doesn't complain during import
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_ALLOWED_USER_ID", "12345")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("BRAND_FOLDER", str(_project_root / "brand"))


@pytest.fixture
def tmp_state_dir(tmp_path):
    """Provide a temporary directory for state files."""
    return tmp_path


@pytest.fixture
def tmp_json(tmp_path):
    """Return a helper that creates a temp JSON file with given data."""
    def _write(name: str, data):
        p = tmp_path / name
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p
    return _write
