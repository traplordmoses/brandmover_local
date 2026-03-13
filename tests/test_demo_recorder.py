"""
Unit tests for demo_recorder and demo_narrator.

No Playwright or ffmpeg needed — tests cover parsing, validation, and timeline math.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.demo_recorder import (
    DemoResult,
    DemoScript,
    DemoStep,
    load_demo_script,
    validate_url,
)
from agent.demo_narrator import (
    _escape_drawtext,
    build_narration_timeline,
    _build_drawtext_filters,
)


# ---------------------------------------------------------------------------
# Script loading
# ---------------------------------------------------------------------------

class TestLoadDemoScript:
    def _write_script(self, tmp_path, data):
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_valid_script(self, tmp_path):
        path = self._write_script(tmp_path, {
            "name": "test-demo",
            "url": "https://example.com",
            "steps": [
                {"action": "goto", "target": "/page", "narration": "Go to page", "wait": 1.5},
                {"action": "click", "target": ".btn"},
            ],
        })
        script = load_demo_script(path)
        assert script.name == "test-demo"
        assert script.url == "https://example.com"
        assert len(script.steps) == 2
        assert script.steps[0].action == "goto"
        assert script.steps[0].wait == 1.5
        assert script.steps[0].narration == "Go to page"
        assert script.steps[1].wait == 2.0  # default

    def test_defaults(self, tmp_path):
        path = self._write_script(tmp_path, {
            "name": "defaults",
            "url": "https://example.com",
            "steps": [{"action": "wait"}],
        })
        script = load_demo_script(path)
        assert script.viewport_width == 1280
        assert script.viewport_height == 720
        assert script.mode == "video"
        assert script.steps[0].target == ""
        assert script.steps[0].value == ""
        assert script.steps[0].narration == ""

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_demo_script("/nonexistent/path.json")

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_demo_script(str(p))

    def test_missing_name(self, tmp_path):
        path = self._write_script(tmp_path, {
            "url": "https://example.com",
            "steps": [{"action": "wait"}],
        })
        with pytest.raises(ValueError, match="name"):
            load_demo_script(path)

    def test_missing_steps(self, tmp_path):
        path = self._write_script(tmp_path, {
            "name": "no-steps",
            "url": "https://example.com",
        })
        with pytest.raises(ValueError, match="steps"):
            load_demo_script(path)

    def test_step_missing_action(self, tmp_path):
        path = self._write_script(tmp_path, {
            "name": "bad-step",
            "url": "https://example.com",
            "steps": [{"target": ".btn"}],
        })
        with pytest.raises(ValueError, match="action"):
            load_demo_script(path)


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

class TestValidateUrl:
    def test_https_allowed(self):
        with patch("agent.demo_recorder.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]):
            validate_url("https://example.com")  # Should not raise

    def test_http_allowed(self):
        with patch("agent.demo_recorder.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 80)),
        ]):
            validate_url("http://example.com")  # Should not raise

    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="http/https"):
            validate_url("file:///etc/passwd")

    def test_ftp_rejected(self):
        with pytest.raises(ValueError, match="http/https"):
            validate_url("ftp://example.com/file")

    def test_private_ip_127(self):
        with patch("agent.demo_recorder.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("127.0.0.1", 80)),
        ]):
            with pytest.raises(ValueError, match="private"):
                validate_url("http://localhost")

    def test_private_ip_10(self):
        with patch("agent.demo_recorder.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("10.0.0.1", 80)),
        ]):
            with pytest.raises(ValueError, match="private"):
                validate_url("http://internal.corp")

    def test_private_ip_192(self):
        with patch("agent.demo_recorder.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("192.168.1.1", 80)),
        ]):
            with pytest.raises(ValueError, match="private"):
                validate_url("http://router.local")

    def test_private_ip_172(self):
        with patch("agent.demo_recorder.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("172.16.0.1", 80)),
        ]):
            with pytest.raises(ValueError, match="private"):
                validate_url("http://docker-host")

    def test_unresolvable_host(self):
        import socket as _socket
        with patch("agent.demo_recorder.socket.getaddrinfo", side_effect=_socket.gaierror("Name resolution failed")):
            with pytest.raises(ValueError, match="resolve"):
                validate_url("http://does-not-exist.invalid")


# ---------------------------------------------------------------------------
# Narration timeline
# ---------------------------------------------------------------------------

class TestBuildNarrationTimeline:
    def test_basic_timeline(self):
        steps = [
            DemoStep(action="goto", narration="Step one", wait=2.0),
            DemoStep(action="click", narration="Step two", wait=3.0),
            DemoStep(action="wait", narration="Step three", wait=1.5),
        ]
        tl = build_narration_timeline(steps)
        assert len(tl) == 3
        assert tl[0] == {"text": "Step one", "start": 0.0, "end": 2.0}
        assert tl[1] == {"text": "Step two", "start": 2.0, "end": 5.0}
        assert tl[2] == {"text": "Step three", "start": 5.0, "end": 6.5}

    def test_empty_narrations_skipped(self):
        steps = [
            DemoStep(action="goto", narration="Visible", wait=2.0),
            DemoStep(action="click", narration="", wait=1.0),
            DemoStep(action="wait", narration="Also visible", wait=3.0),
        ]
        tl = build_narration_timeline(steps)
        assert len(tl) == 2
        assert tl[0]["text"] == "Visible"
        assert tl[0]["start"] == 0.0
        assert tl[1]["text"] == "Also visible"
        assert tl[1]["start"] == 3.0  # 2.0 + 1.0

    def test_empty_steps(self):
        assert build_narration_timeline([]) == []

    def test_dict_steps(self):
        steps = [
            {"narration": "Dict step", "wait": 2.5},
            {"narration": "", "wait": 1.0},
        ]
        tl = build_narration_timeline(steps)
        assert len(tl) == 1
        assert tl[0] == {"text": "Dict step", "start": 0.0, "end": 2.5}


# ---------------------------------------------------------------------------
# ffmpeg filter escaping
# ---------------------------------------------------------------------------

class TestEscapeDrawtext:
    def test_colon_escaped(self):
        assert "\\:" in _escape_drawtext("Time: 12:30")

    def test_apostrophe_replaced(self):
        result = _escape_drawtext("It's working")
        assert "'" not in result
        assert "\u2019" in result  # Unicode right single quote

    def test_percent_doubled(self):
        assert "%%" in _escape_drawtext("100% done")

    def test_backslash_escaped(self):
        result = _escape_drawtext("path\\to\\file")
        assert "\\\\" in result

    def test_plain_text_unchanged(self):
        assert _escape_drawtext("Hello world") == "Hello world"


class TestBuildDrawtextFilters:
    def test_single_entry(self):
        timeline = [{"text": "Hello", "start": 0.0, "end": 2.0}]
        result = _build_drawtext_filters(timeline)
        assert "drawtext=" in result
        assert "Hello" in result
        assert "between(t,0.0,2.0)" in result

    def test_multiple_entries(self):
        timeline = [
            {"text": "First", "start": 0.0, "end": 2.0},
            {"text": "Second", "start": 2.0, "end": 4.0},
        ]
        result = _build_drawtext_filters(timeline)
        assert result.count("drawtext=") == 2
        assert "," in result  # Comma-separated filters

    def test_empty_timeline(self):
        assert _build_drawtext_filters([]) == ""


# ---------------------------------------------------------------------------
# Output path naming
# ---------------------------------------------------------------------------

class TestDemoResult:
    def test_default_fields(self):
        r = DemoResult(script_name="test", mode="video")
        assert r.video_path == ""
        assert r.screenshot_paths == []
        assert r.duration_seconds == 0.0
        assert r.error == ""

    def test_screenshot_paths_isolated(self):
        """Ensure default list factory creates independent instances."""
        r1 = DemoResult(script_name="a", mode="screenshot")
        r2 = DemoResult(script_name="b", mode="screenshot")
        r1.screenshot_paths.append("/path/1.png")
        assert r2.screenshot_paths == []
