"""Tests for LoRA versioning — manifest management, version switching,
rollback, and formatting in agent.lora_pipeline."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from agent.lora_pipeline import (
    _load_lora_manifest,
    _save_lora_manifest,
    _next_version_number,
    _record_version,
    switch_active_version,
    rollback_version,
    get_lora_manifest,
    format_versions_list,
)


# ---------------------------------------------------------------------------
# Fixtures — redirect lora paths to tmp_path
# ---------------------------------------------------------------------------

def _setup_lora_dir(tmp_path):
    """Patch _LORA_DIR, _LORA_MANIFEST_PATH, _ACTIVE_WEIGHTS to tmp_path."""
    manifest_path = tmp_path / "manifest.json"
    active_path = tmp_path / "brand3d.safetensors"
    return {
        "_LORA_DIR": tmp_path,
        "_LORA_MANIFEST_PATH": manifest_path,
        "_ACTIVE_WEIGHTS": active_path,
    }


def _write_fake_weights(tmp_path, version_num: int) -> Path:
    """Write a fake weights file for a given version number."""
    path = tmp_path / f"brand3d_v{version_num}.safetensors"
    path.write_bytes(f"fake_weights_v{version_num}".encode())
    return path


# ---------------------------------------------------------------------------
# Manifest creation and reading
# ---------------------------------------------------------------------------

class TestManifestBasics:
    def test_empty_manifest(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            manifest = _load_lora_manifest()
            assert manifest["versions"] == []
            assert manifest["active_version"] is None

    def test_save_and_load(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            data = {"versions": [{"version_number": 1}], "active_version": 1}
            _save_lora_manifest(data)
            loaded = _load_lora_manifest()
            assert loaded["active_version"] == 1
            assert len(loaded["versions"]) == 1

    def test_get_lora_manifest_public(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            manifest = get_lora_manifest()
            assert "versions" in manifest
            assert "active_version" in manifest

    def test_corrupt_manifest_returns_empty(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("not json{{{", encoding="utf-8")
        with patch.multiple("agent.lora_pipeline", **patches):
            manifest = _load_lora_manifest()
            assert manifest["versions"] == []


# ---------------------------------------------------------------------------
# Version incrementing
# ---------------------------------------------------------------------------

class TestVersionIncrementing:
    def test_first_version(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            manifest = _load_lora_manifest()
            assert _next_version_number(manifest) == 1

    def test_increment_after_one(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            manifest = {"versions": [{"version_number": 1}], "active_version": 1}
            assert _next_version_number(manifest) == 2

    def test_increment_after_gap(self, tmp_path):
        """If versions are [1, 3], next should be 4."""
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            manifest = {"versions": [
                {"version_number": 1},
                {"version_number": 3},
            ], "active_version": 3}
            assert _next_version_number(manifest) == 4

    def test_record_version_auto_deactivates(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            # Record v1
            path1 = _write_fake_weights(tmp_path, 1)
            _record_version(1, "pred1", 20, ["brand_asset"], "BRAND3D", str(path1))

            # Record v2
            path2 = _write_fake_weights(tmp_path, 2)
            _record_version(2, "pred2", 25, ["brand_asset", "community"], "BRAND3D", str(path2))

            manifest = _load_lora_manifest()
            assert manifest["active_version"] == 2
            assert manifest["versions"][0]["status"] == "inactive"
            assert manifest["versions"][1]["status"] == "active"


# ---------------------------------------------------------------------------
# Switch active version
# ---------------------------------------------------------------------------

class TestSwitchVersion:
    def test_switch_to_existing(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            path1 = _write_fake_weights(tmp_path, 1)
            _record_version(1, "pred1", 20, [], "BRAND3D", str(path1))

            path2 = _write_fake_weights(tmp_path, 2)
            _record_version(2, "pred2", 25, [], "BRAND3D", str(path2))

            result = switch_active_version(1)
            assert isinstance(result, dict)
            assert result["version_number"] == 1
            assert result["status"] == "active"

            # Check active weights file has v1 content
            active = tmp_path / "brand3d.safetensors"
            assert active.read_bytes() == b"fake_weights_v1"

            # Check manifest
            manifest = _load_lora_manifest()
            assert manifest["active_version"] == 1
            assert manifest["versions"][0]["status"] == "active"
            assert manifest["versions"][1]["status"] == "inactive"

    def test_switch_to_nonexistent(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            path1 = _write_fake_weights(tmp_path, 1)
            _record_version(1, "pred1", 20, [], "BRAND3D", str(path1))

            result = switch_active_version(99)
            assert isinstance(result, str)
            assert "not found" in result.lower()

    def test_switch_with_missing_weights_file(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            # Record version but don't create the weights file
            _record_version(1, "pred1", 20, [], "BRAND3D", str(tmp_path / "nonexistent.safetensors"))

            path2 = _write_fake_weights(tmp_path, 2)
            _record_version(2, "pred2", 25, [], "BRAND3D", str(path2))

            result = switch_active_version(1)
            assert isinstance(result, str)
            assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_to_previous(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            path1 = _write_fake_weights(tmp_path, 1)
            _record_version(1, "pred1", 20, [], "BRAND3D", str(path1))

            path2 = _write_fake_weights(tmp_path, 2)
            _record_version(2, "pred2", 25, [], "BRAND3D", str(path2))

            result = rollback_version()
            assert isinstance(result, dict)
            assert result["version_number"] == 1

            active = tmp_path / "brand3d.safetensors"
            assert active.read_bytes() == b"fake_weights_v1"

    def test_rollback_at_v1(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            path1 = _write_fake_weights(tmp_path, 1)
            _record_version(1, "pred1", 20, [], "BRAND3D", str(path1))

            result = rollback_version()
            assert isinstance(result, str)
            assert "version 1" in result.lower()

    def test_rollback_no_versions(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            result = rollback_version()
            assert isinstance(result, str)
            assert "no" in result.lower()

    def test_rollback_no_active(self, tmp_path):
        patches = _setup_lora_dir(tmp_path)
        with patch.multiple("agent.lora_pipeline", **patches):
            # Manually create a manifest with no active version
            _save_lora_manifest({"versions": [{"version_number": 1, "status": "inactive"}], "active_version": None})
            result = rollback_version()
            assert isinstance(result, str)
            assert "no active" in result.lower()


# ---------------------------------------------------------------------------
# Format versions list
# ---------------------------------------------------------------------------

class TestFormatVersionsList:
    def test_empty_manifest(self):
        manifest = {"versions": [], "active_version": None}
        result = format_versions_list(manifest)
        assert "no lora versions" in result.lower()

    def test_single_version(self, tmp_path):
        manifest = {
            "versions": [{
                "version_number": 1,
                "status": "active",
                "image_count": 20,
                "training_date": 1709078400,  # 2024-02-28
            }],
            "active_version": 1,
        }
        result = format_versions_list(manifest)
        assert "v1" in result
        assert "(active)" in result
        assert "20 images" in result

    def test_multiple_versions(self):
        manifest = {
            "versions": [
                {"version_number": 1, "status": "inactive", "image_count": 20, "training_date": 1709078400},
                {"version_number": 2, "status": "inactive", "image_count": 25, "training_date": 1709164800},
                {"version_number": 3, "status": "active", "image_count": 30, "training_date": 1709251200},
            ],
            "active_version": 3,
        }
        result = format_versions_list(manifest)
        assert "v1" in result
        assert "v2" in result
        assert "v3" in result
        assert "(active)" in result
        # Only v3 should have the active marker
        lines = result.split("\n")
        active_lines = [l for l in lines if "(active)" in l]
        assert len(active_lines) == 1
        assert "v3" in active_lines[0]

    def test_contains_status_icons(self):
        manifest = {
            "versions": [
                {"version_number": 1, "status": "active", "image_count": 20, "training_date": 0},
                {"version_number": 2, "status": "inactive", "image_count": 25, "training_date": 0},
            ],
            "active_version": 1,
        }
        result = format_versions_list(manifest)
        assert "\u2705" in result  # active icon
        assert "\u2796" in result  # inactive icon
