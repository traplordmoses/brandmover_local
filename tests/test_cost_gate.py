"""Tests for agent.cost_gate — cost budget tracking."""

import json
import time
from pathlib import Path
from unittest.mock import patch

from agent.cost_gate import check_cost_budget, get_cost_summary


class TestCheckCostBudget:
    def test_allowed_when_no_history(self, tmp_path):
        fake_file = tmp_path / "generation_history.json"
        with patch("agent.cost_gate._HISTORY_FILE", fake_file):
            result = check_cost_budget()
            assert result["allowed"] is True
            assert result["spent_today_usd"] == 0.0
            assert result["remaining_usd"] > 0

    def test_blocked_when_over_budget(self, tmp_path):
        fake_file = tmp_path / "generation_history.json"
        now = time.time()
        entries = [
            {"timestamp": now, "estimated_cost_usd": 3.0},
            {"timestamp": now - 60, "estimated_cost_usd": 3.0},
        ]
        fake_file.write_text(json.dumps(entries))
        with patch("agent.cost_gate._HISTORY_FILE", fake_file), \
             patch("agent.cost_gate.settings") as mock_settings:
            mock_settings.DAILY_COST_BUDGET_USD = 5.0
            result = check_cost_budget()
            assert result["allowed"] is False
            assert result["spent_today_usd"] == 6.0

    def test_estimated_cost_parameter(self, tmp_path):
        fake_file = tmp_path / "generation_history.json"
        now = time.time()
        entries = [{"timestamp": now, "estimated_cost_usd": 4.5}]
        fake_file.write_text(json.dumps(entries))
        with patch("agent.cost_gate._HISTORY_FILE", fake_file), \
             patch("agent.cost_gate.settings") as mock_settings:
            mock_settings.DAILY_COST_BUDGET_USD = 5.0
            result = check_cost_budget(estimated_cost=1.0)
            assert result["allowed"] is False


class TestGetCostSummary:
    def test_empty_history(self, tmp_path):
        fake_file = tmp_path / "generation_history.json"
        with patch("agent.cost_gate._HISTORY_FILE", fake_file):
            result = get_cost_summary(days=7)
            assert result["total_usd"] == 0.0
            assert result["daily_costs"] == []

    def test_summarizes_costs(self, tmp_path):
        fake_file = tmp_path / "generation_history.json"
        now = time.time()
        entries = [
            {"timestamp": now, "estimated_cost_usd": 1.5},
            {"timestamp": now - 60, "estimated_cost_usd": 0.5},
        ]
        fake_file.write_text(json.dumps(entries))
        with patch("agent.cost_gate._HISTORY_FILE", fake_file):
            result = get_cost_summary(days=7)
            assert result["total_usd"] == 2.0
            assert len(result["daily_costs"]) == 1
