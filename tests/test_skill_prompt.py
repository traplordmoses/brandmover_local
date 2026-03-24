"""Tests for agent/skill_prompt.py — system prompt construction."""

import pytest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brand_config(
    badge_text=None,
    default_mode="image_optional",
):
    """Build a minimal BrandConfig-like object for testing."""
    from agent.compositor_config import BrandConfig
    cfg = BrandConfig()
    cfg.badge_text = badge_text
    cfg.default_mode = default_mode
    return cfg


# ---------------------------------------------------------------------------
# _get_platform_block
# ---------------------------------------------------------------------------

class TestGetPlatformBlock:
    """Tests for _get_platform_block()."""

    def test_returns_no_badge_message_when_badge_is_none(self):
        from agent.skill_prompt import _get_platform_block
        cfg = _make_brand_config(badge_text=None)
        result = _get_platform_block(config=cfg)
        assert "Do NOT include" in result
        assert "platform" in result

    def test_returns_fixed_badge_when_badge_text_set(self):
        from agent.skill_prompt import _get_platform_block
        cfg = _make_brand_config(badge_text="PRO")
        result = _get_platform_block(config=cfg)
        assert '"PRO"' in result
        assert "fixed" in result.lower() or "Always" in result

    def test_falls_back_to_get_config_when_no_config_passed(self):
        from agent.skill_prompt import _get_platform_block
        cfg = _make_brand_config(badge_text="APP")
        with patch("agent.compositor_config.get_config", return_value=cfg):
            result = _get_platform_block(config=None)
        assert '"APP"' in result

    def test_returns_fallback_string_on_config_exception(self):
        from agent.skill_prompt import _get_platform_block
        with patch("agent.compositor_config.get_config", side_effect=Exception("boom")):
            result = _get_platform_block(config=None)
        # Should return a sensible fallback, not crash
        assert isinstance(result, str)
        assert "platform" in result.lower()


# ---------------------------------------------------------------------------
# _get_platform_json_line
# ---------------------------------------------------------------------------

class TestGetPlatformJsonLine:
    """Tests for _get_platform_json_line()."""

    def test_returns_empty_when_badge_is_none(self):
        from agent.skill_prompt import _get_platform_json_line
        cfg = _make_brand_config(badge_text=None)
        result = _get_platform_json_line(config=cfg)
        assert result == ""

    def test_returns_json_field_when_badge_set(self):
        from agent.skill_prompt import _get_platform_json_line
        cfg = _make_brand_config(badge_text="WEB")
        result = _get_platform_json_line(config=cfg)
        assert '"platform"' in result
        assert '"WEB"' in result

    def test_returns_fallback_on_exception(self):
        from agent.skill_prompt import _get_platform_json_line
        with patch("agent.compositor_config.get_config", side_effect=Exception("boom")):
            result = _get_platform_json_line(config=None)
        assert '"platform"' in result
        assert '"WEB"' in result


# ---------------------------------------------------------------------------
# _get_image_mode_block
# ---------------------------------------------------------------------------

class TestGetImageModeBlock:
    """Tests for _get_image_mode_block()."""

    def test_text_only_mode(self):
        from agent.skill_prompt import _get_image_mode_block
        cfg = _make_brand_config(default_mode="text_only")
        result = _get_image_mode_block(config=cfg)
        assert "TEXT ONLY" in result
        assert "Do NOT generate images" in result

    def test_image_always_mode(self):
        from agent.skill_prompt import _get_image_mode_block
        cfg = _make_brand_config(default_mode="image_always")
        result = _get_image_mode_block(config=cfg)
        assert "ALWAYS" in result
        assert "image_prompt" in result

    def test_image_optional_returns_empty(self):
        from agent.skill_prompt import _get_image_mode_block
        cfg = _make_brand_config(default_mode="image_optional")
        result = _get_image_mode_block(config=cfg)
        assert result == ""

    def test_unknown_mode_returns_empty(self):
        from agent.skill_prompt import _get_image_mode_block
        cfg = _make_brand_config(default_mode="something_else")
        result = _get_image_mode_block(config=cfg)
        assert result == ""

    def test_returns_empty_on_config_exception(self):
        from agent.skill_prompt import _get_image_mode_block
        with patch("agent.compositor_config.get_config", side_effect=Exception("boom")):
            result = _get_image_mode_block(config=None)
        assert result == ""


# ---------------------------------------------------------------------------
# _get_content_types_block
# ---------------------------------------------------------------------------

class TestGetContentTypesBlock:
    """Tests for _get_content_types_block()."""

    def test_contains_all_agent_selectable_types(self):
        from agent.skill_prompt import _get_content_types_block
        from agent.content_types import AGENT_SELECTABLE_TYPES
        result = _get_content_types_block()
        for ct in AGENT_SELECTABLE_TYPES:
            assert f'"{ct}"' in result, f"Missing content type: {ct}"

    def test_contains_known_type_names(self):
        from agent.skill_prompt import _get_content_types_block
        result = _get_content_types_block()
        for name in ("announcement", "meme", "lifestyle", "brand_3d", "educational"):
            assert name in result

    def test_each_type_has_description(self):
        from agent.skill_prompt import _get_content_types_block
        result = _get_content_types_block()
        lines = [l for l in result.splitlines() if l.strip()]
        for line in lines:
            # Each line should have format: - "type" — description
            assert line.startswith("- "), f"Unexpected line format: {line}"
            assert "\u2014" in line or "—" in line or " — " in line, (
                f"Missing description separator in: {line}"
            )

    def test_returns_nonempty_string(self):
        from agent.skill_prompt import _get_content_types_block
        result = _get_content_types_block()
        assert len(result) > 50


# ---------------------------------------------------------------------------
# _get_workspace_injection
# ---------------------------------------------------------------------------

class TestGetWorkspaceInjection:
    """Tests for _get_workspace_injection()."""

    def test_returns_empty_when_no_personality_dir(self, tmp_path):
        from agent import skill_prompt
        with patch.object(skill_prompt, "_PERSONALITY_DIR", tmp_path / "nonexistent"):
            result = skill_prompt._get_workspace_injection()
        assert result == ""

    def test_loads_system_prompt_file(self, tmp_path):
        from agent import skill_prompt
        personality_dir = tmp_path / "personality"
        personality_dir.mkdir()
        (personality_dir / "system_prompt.md").write_text("You are a snarky brand agent.", encoding="utf-8")
        with patch.object(skill_prompt, "_PERSONALITY_DIR", personality_dir):
            result = skill_prompt._get_workspace_injection()
        assert "PERSONALITY" in result
        assert "snarky brand agent" in result

    def test_loads_memory_file(self, tmp_path):
        from agent import skill_prompt
        personality_dir = tmp_path / "personality"
        personality_dir.mkdir()
        (personality_dir / "memory.md").write_text("Always use dark backgrounds.", encoding="utf-8")
        with patch.object(skill_prompt, "_PERSONALITY_DIR", personality_dir):
            result = skill_prompt._get_workspace_injection()
        assert "OPERATOR NOTES" in result
        assert "dark backgrounds" in result

    def test_loads_both_files(self, tmp_path):
        from agent import skill_prompt
        personality_dir = tmp_path / "personality"
        personality_dir.mkdir()
        (personality_dir / "system_prompt.md").write_text("Be bold.", encoding="utf-8")
        (personality_dir / "memory.md").write_text("Prefer neon.", encoding="utf-8")
        with patch.object(skill_prompt, "_PERSONALITY_DIR", personality_dir):
            result = skill_prompt._get_workspace_injection()
        assert "PERSONALITY" in result
        assert "OPERATOR NOTES" in result
        assert "Be bold." in result
        assert "Prefer neon." in result

    def test_skips_empty_files(self, tmp_path):
        from agent import skill_prompt
        personality_dir = tmp_path / "personality"
        personality_dir.mkdir()
        (personality_dir / "system_prompt.md").write_text("", encoding="utf-8")
        (personality_dir / "memory.md").write_text("   ", encoding="utf-8")
        with patch.object(skill_prompt, "_PERSONALITY_DIR", personality_dir):
            result = skill_prompt._get_workspace_injection()
        assert result == ""

    def test_handles_read_error_gracefully(self, tmp_path):
        from agent import skill_prompt
        personality_dir = tmp_path / "personality"
        personality_dir.mkdir()
        # Create a file then make it a directory to provoke an error
        sp = personality_dir / "system_prompt.md"
        sp.write_text("content", encoding="utf-8")
        with patch.object(skill_prompt, "_PERSONALITY_DIR", personality_dir):
            # Patch read_text to raise
            with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
                result = skill_prompt._get_workspace_injection()
        assert result == ""


# ---------------------------------------------------------------------------
# _get_skills_block
# ---------------------------------------------------------------------------

class TestGetSkillsBlock:
    """Tests for _get_skills_block()."""

    def test_returns_empty_when_no_skills(self):
        from agent.skill_prompt import _get_skills_block
        with patch("agent.skills.get_skills_for_routing", return_value=""):
            result = _get_skills_block()
        assert result == ""

    def test_returns_skills_section_when_skills_exist(self):
        from agent.skill_prompt import _get_skills_block
        skills_text = "- thread_storm: Create viral thread sequences"
        with patch("agent.skills.get_skills_for_routing", return_value=skills_text):
            result = _get_skills_block()
        assert "## SKILLS" in result
        assert "thread_storm" in result
        assert "use_skill" in result
        assert "create_skill" in result

    def test_passes_max_tokens(self):
        from agent.skill_prompt import _get_skills_block
        with patch("agent.skills.get_skills_for_routing", return_value="x") as mock_fn:
            _get_skills_block()
        mock_fn.assert_called_once_with(max_tokens=600)


# ---------------------------------------------------------------------------
# build_system_prompt — integration-level tests
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """Tests for build_system_prompt() — the main entry point."""

    @pytest.fixture(autouse=True)
    def _patch_externals(self, tmp_path):
        """Patch external dependencies so build_system_prompt() works in isolation."""
        cfg = _make_brand_config(badge_text=None, default_mode="image_optional")
        self._patches = [
            patch("agent.compositor_config.get_config", return_value=cfg),
            patch("agent.skills.get_skills_for_routing", return_value=""),
        ]
        for p in self._patches:
            p.start()

        # Ensure personality dir points to empty tmp dir (no files)
        from agent import skill_prompt
        self._personality_patch = patch.object(
            skill_prompt, "_PERSONALITY_DIR", tmp_path / "personality"
        )
        self._personality_patch.start()

        yield

        self._personality_patch.stop()
        for p in self._patches:
            p.stop()

    def test_returns_nonempty_string(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_contains_brand_name(self):
        from agent.skill_prompt import build_system_prompt
        from config import settings
        result = build_system_prompt()
        assert settings.BRAND_NAME.lower() in result.lower()

    def test_contains_content_types_section(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "content types" in result.lower()
        assert "announcement" in result
        assert "meme" in result

    def test_contains_hard_rules(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "hard rules" in result.lower()
        assert "zero hashtags" in result.lower()
        assert "em-dash" in result.lower() or "em\u2014dash" in result.lower()

    def test_contains_no_ai_language_rule(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "no ai language" in result.lower()
        # Check some banned words are listed
        for word in ("revolutionizing", "leveraging", "seamlessly"):
            assert word in result.lower()

    def test_contains_caption_length_guidance(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "280" in result or "caption" in result.lower()

    def test_contains_json_output_via_finish_tool(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "finish" in result.lower()
        assert "caption" in result
        assert "image_prompt" in result
        assert "content_type" in result

    def test_contains_workflow_section(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "## workflow" in result.lower() or "workflow" in result.lower()
        assert "think" in result

    def test_no_platform_field_when_badge_none(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        # When badge_text is None, the platform JSON line is empty
        assert "Do NOT include" in result

    def test_platform_field_included_when_badge_set(self):
        from agent.skill_prompt import build_system_prompt
        cfg = _make_brand_config(badge_text="WEB")
        with patch("agent.compositor_config.get_config", return_value=cfg):
            result = build_system_prompt()
        assert '"WEB"' in result

    def test_image_mode_text_only_injected(self):
        from agent.skill_prompt import build_system_prompt
        cfg = _make_brand_config(default_mode="text_only")
        with patch("agent.compositor_config.get_config", return_value=cfg):
            result = build_system_prompt()
        assert "TEXT ONLY" in result

    def test_image_mode_always_injected(self):
        from agent.skill_prompt import build_system_prompt
        cfg = _make_brand_config(default_mode="image_always")
        with patch("agent.compositor_config.get_config", return_value=cfg):
            result = build_system_prompt()
        assert "IMAGE MODE: ALWAYS" in result

    def test_skills_block_included_when_skills_exist(self):
        from agent.skill_prompt import build_system_prompt
        with patch(
            "agent.skills.get_skills_for_routing",
            return_value="- meme_lord: Generate dank memes",
        ):
            result = build_system_prompt()
        assert "## SKILLS" in result
        assert "meme_lord" in result

    def test_workspace_injection_included(self, tmp_path):
        from agent import skill_prompt
        from agent.skill_prompt import build_system_prompt

        personality_dir = tmp_path / "personality"
        personality_dir.mkdir()
        (personality_dir / "system_prompt.md").write_text(
            "You are a savage crypto degen.", encoding="utf-8"
        )

        with patch.object(skill_prompt, "_PERSONALITY_DIR", personality_dir):
            result = build_system_prompt()
        assert "PERSONALITY" in result
        assert "savage crypto degen" in result

    def test_survives_compositor_config_failure(self):
        from agent.skill_prompt import build_system_prompt
        with patch(
            "agent.compositor_config.get_config",
            side_effect=Exception("guidelines missing"),
        ):
            result = build_system_prompt()
        # Should still produce a valid prompt
        assert isinstance(result, str)
        assert len(result) > 200
        assert "creative director" in result.lower()

    def test_contains_video_workflow_section(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "video workflow" in result.lower()

    def test_contains_revision_mode_section(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "revision mode" in result.lower()

    def test_contains_brand_3d_section(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "brand_3d" in result

    def test_contains_thread_instructions(self):
        from agent.skill_prompt import build_system_prompt
        result = build_system_prompt()
        assert "thread" in result.lower()
        assert "thread_posts" in result
