"""Tests for agent.generation_history — append-only generation log."""

import json
from unittest.mock import patch
from pathlib import Path

from agent import generation_history


def test_estimate_cost_known_model():
    cost = generation_history._estimate_cost("black-forest-labs/flux-1.1-pro", 1)
    assert cost == 0.04


def test_estimate_cost_multiple_images():
    cost = generation_history._estimate_cost("black-forest-labs/flux-1.1-pro", 3)
    assert cost == 0.12


def test_estimate_cost_unknown_model():
    cost = generation_history._estimate_cost("some/unknown-model", 1)
    assert cost == 0.04  # default fallback


def test_log_and_read(tmp_path):
    history_file = tmp_path / "gen_history.json"

    with patch.object(generation_history, "_HISTORY_FILE", history_file):
        count = generation_history.log_generation(
            asset_type="social_post",
            content_type="announcement",
            prompt="test prompt",
            model_id="black-forest-labs/flux-1.1-pro",
            image_urls=["https://example.com/img.png"],
            original_request="test request",
        )
        assert count == 1

        entries = generation_history._read_history()
        assert len(entries) == 1
        assert entries[0]["status"] == "draft"
        assert entries[0]["estimated_cost_usd"] == 0.04


def test_update_status(tmp_path):
    history_file = tmp_path / "gen_history.json"

    with patch.object(generation_history, "_HISTORY_FILE", history_file):
        generation_history.log_generation(
            asset_type="social_post",
            content_type="brand_3d",
            prompt="test",
            model_id="flux-1.1-pro",
            image_urls=["url"],
            original_request="req",
        )

        entries = generation_history._read_history()
        ts = entries[0]["timestamp"]

        found = generation_history.update_generation_status(ts, "approved")
        assert found is True

        entries = generation_history._read_history()
        assert entries[0]["status"] == "approved"


def test_stats(tmp_path):
    history_file = tmp_path / "gen_history.json"

    with patch.object(generation_history, "_HISTORY_FILE", history_file):
        generation_history.log_generation("social_post", "announcement", "p", "flux-1.1-pro", ["u"], "r")
        generation_history.log_generation("social_post", "brand_3d", "p", "recraft-v3-svg", ["u"], "r")

        stats = generation_history.get_generation_stats()
        assert stats["total"] == 2
        assert stats["estimated_total_cost_usd"] > 0


def test_approval_analytics(tmp_path):
    history_file = tmp_path / "gen_history.json"

    with patch.object(generation_history, "_HISTORY_FILE", history_file):
        # Write entries directly with distinct timestamps to avoid race
        import json
        entries = [
            {
                "asset_type": "social_post", "content_type": "announcement",
                "prompt": "p1", "model_id": "flux-1.1-pro", "image_urls": ["u"],
                "original_request": "r1", "status": "approved",
                "estimated_cost_usd": 0.04, "timestamp": 1000000.0,
            },
            {
                "asset_type": "social_post", "content_type": "announcement",
                "prompt": "p2", "model_id": "flux-1.1-pro", "image_urls": ["u"],
                "original_request": "r2", "status": "rejected",
                "estimated_cost_usd": 0.04, "timestamp": 1000100.0,
            },
        ]
        history_file.write_text(json.dumps(entries), encoding="utf-8")

        analytics = generation_history.get_approval_analytics()
        ct_stats = analytics["by_content_type"]["announcement"]
        assert ct_stats["approved"] == 1
        assert ct_stats["rejected"] == 1
        assert ct_stats["rate"] == 50.0
