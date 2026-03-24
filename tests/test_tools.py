"""
Comprehensive tests for agent/tools.py — tool definitions, dispatch, and handlers.
"""

import asyncio
import json
import re
import subprocess

import pytest
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock

from agent.resource_log import ResourceTracker
from agent.tools import TOOL_DEFINITIONS, execute_tool, _OPENCLAW_ALLOWLIST, _HANDLERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously for tests."""
    return asyncio.run(coro)


def _make_tracker() -> ResourceTracker:
    return ResourceTracker()


def _tool_names() -> list[str]:
    return [t["name"] for t in TOOL_DEFINITIONS]


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS structure
# ---------------------------------------------------------------------------

class TestToolDefinitionsStructure:
    """Verify TOOL_DEFINITIONS list has correct shape and all expected tools."""

    EXPECTED_TOOL_NAMES = {
        "read_brand_guidelines",
        "generate_image",
        "img2img",
        "finish",
        "think",
        "execute_openclaw_script",
        "use_skill",
        "create_skill",
        "list_skills",
        "delegate_task",
        "research_trends",
        "search_memory",
        "read_references",
        "check_figma_design",
        "read_feedback_history",
        "log_resource_usage",
        "generate_promo_video",
        "verify_draft",
        "suggest_variations",
        "repurpose_content",
        "plan_growth_thread",
        "create_campaign",
    }

    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == len(self.EXPECTED_TOOL_NAMES)

    def test_all_expected_names_present(self):
        actual = set(_tool_names())
        assert actual == self.EXPECTED_TOOL_NAMES

    @pytest.mark.parametrize("tool_def", TOOL_DEFINITIONS, ids=_tool_names())
    def test_required_fields(self, tool_def):
        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def

    @pytest.mark.parametrize("tool_def", TOOL_DEFINITIONS, ids=_tool_names())
    def test_input_schema_is_object(self, tool_def):
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

    @pytest.mark.parametrize("tool_def", TOOL_DEFINITIONS, ids=_tool_names())
    def test_description_is_nonempty_string(self, tool_def):
        assert isinstance(tool_def["description"], str)
        assert len(tool_def["description"]) > 10

    def test_no_duplicate_names(self):
        names = _tool_names()
        assert len(names) == len(set(names))

    def test_handler_exists_for_every_definition(self):
        """Every defined tool must have a matching handler in _HANDLERS."""
        defined = set(_tool_names())
        handled = set(_HANDLERS.keys())
        assert defined == handled


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------

class TestExecuteToolDispatch:
    """Verify execute_tool dispatches correctly and rejects unknowns."""

    def test_unknown_tool_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown tool"):
            _run(execute_tool("nonexistent_tool", {}, _make_tracker()))

    def test_dispatches_to_think(self):
        result = _run(execute_tool("think", {"thought": "testing"}, _make_tracker()))
        assert result == "ok"

    def test_dispatches_to_finish(self):
        result = _run(execute_tool("finish", {"caption": "hello"}, _make_tracker()))
        parsed = json.loads(result)
        assert parsed["status"] == "complete"

    def test_dispatches_to_log_resource_usage(self):
        result = _run(execute_tool("log_resource_usage", {"summary": "test"}, _make_tracker()))
        assert "test" in result


# ---------------------------------------------------------------------------
# _handle_think
# ---------------------------------------------------------------------------

class TestHandleThink:
    def test_returns_ok(self):
        result = _run(execute_tool("think", {"thought": "some reasoning"}, _make_tracker()))
        assert result == "ok"

    def test_empty_thought_still_ok(self):
        result = _run(execute_tool("think", {"thought": ""}, _make_tracker()))
        assert result == "ok"

    def test_missing_thought_key_still_ok(self):
        result = _run(execute_tool("think", {}, _make_tracker()))
        assert result == "ok"


# ---------------------------------------------------------------------------
# _handle_finish
# ---------------------------------------------------------------------------

class TestHandleFinish:
    def test_basic_caption(self):
        result = _run(execute_tool("finish", {"caption": "Hello world"}, _make_tracker()))
        parsed = json.loads(result)
        assert parsed["status"] == "complete"
        assert parsed["draft"]["caption"] == "Hello world"

    def test_full_draft_fields(self):
        draft = {
            "caption": "Big announcement!",
            "content_type": "announcement",
            "hashtags": ["#crypto", "#defi"],
            "alt_text": "brand image",
            "title": "Title",
            "subtitle": "Sub",
            "platform": "twitter",
        }
        result = _run(execute_tool("finish", draft, _make_tracker()))
        parsed = json.loads(result)
        assert parsed["draft"]["content_type"] == "announcement"
        assert parsed["draft"]["hashtags"] == ["#crypto", "#defi"]

    def test_thread_format(self):
        draft = {
            "caption": "Thread start",
            "format": "thread",
            "thread_posts": [
                {"text": "Post 1"},
                {"text": "Post 2", "image_prompt": "cool image"},
            ],
        }
        result = _run(execute_tool("finish", draft, _make_tracker()))
        parsed = json.loads(result)
        assert parsed["draft"]["format"] == "thread"
        assert len(parsed["draft"]["thread_posts"]) == 2

    def test_empty_caption(self):
        result = _run(execute_tool("finish", {"caption": ""}, _make_tracker()))
        parsed = json.loads(result)
        assert parsed["status"] == "complete"
        assert parsed["draft"]["caption"] == ""


# ---------------------------------------------------------------------------
# _handle_log_resource_usage
# ---------------------------------------------------------------------------

class TestHandleLogResourceUsage:
    def test_returns_summary_string(self):
        tracker = _make_tracker()
        result = _run(execute_tool("log_resource_usage", {"summary": "guidelines.md, feedback"}, tracker))
        assert "guidelines.md, feedback" in result
        assert "Resource usage logged" in result

    def test_includes_tracker_summary(self):
        tracker = _make_tracker()
        tracker.log_file("guidelines.md")
        result = _run(execute_tool("log_resource_usage", {"summary": "test"}, tracker))
        assert "guidelines.md" in result

    def test_empty_summary(self):
        result = _run(execute_tool("log_resource_usage", {"summary": ""}, _make_tracker()))
        assert "Resource usage logged" in result


# ---------------------------------------------------------------------------
# _handle_read_brand_guidelines
# ---------------------------------------------------------------------------

class TestHandleReadBrandGuidelines:
    def test_returns_json_status(self):
        """Handler returns a JSON pointer (guidelines are pre-loaded in system prompt)."""
        tracker = _make_tracker()
        result = _run(execute_tool("read_brand_guidelines", {}, tracker))
        parsed = json.loads(result)
        assert "status" in parsed
        assert "already loaded" in parsed["status"]

    def test_logs_file(self):
        tracker = _make_tracker()
        _run(execute_tool("read_brand_guidelines", {}, tracker))
        assert "brand/guidelines.md (pre-loaded)" in tracker.files_loaded


# ---------------------------------------------------------------------------
# _handle_read_references
# ---------------------------------------------------------------------------

class TestHandleReadReferences:
    @patch("agent.tools.asyncio.to_thread", new_callable=AsyncMock, return_value="3 PDFs, 2 images")
    def test_returns_summary(self, mock_to_thread):
        tracker = _make_tracker()
        result = _run(execute_tool("read_references", {}, tracker))
        assert result == "3 PDFs, 2 images"
        assert "reference_inventory" in tracker.files_loaded


# ---------------------------------------------------------------------------
# _handle_list_skills
# ---------------------------------------------------------------------------

class TestHandleListSkills:
    @patch("agent.tools.json", wraps=json)
    def test_no_skills(self, _mock_json):
        with patch("agent.skills.load_registry", return_value=[]):
            result = _run(execute_tool("list_skills", {}, _make_tracker()))
            parsed = json.loads(result)
            assert parsed["skills"] == []
            assert "No skills" in parsed["message"]

    def test_with_skills(self):
        skills = [
            {"name": "meme-gen", "description": "Generate memes"},
            {"name": "trending", "description": "Find trending topics"},
        ]
        with patch("agent.skills.load_registry", return_value=skills):
            result = _run(execute_tool("list_skills", {}, _make_tracker()))
            parsed = json.loads(result)
            assert parsed["count"] == 2
            assert parsed["skills"][0]["name"] == "meme-gen"


# ---------------------------------------------------------------------------
# _handle_use_skill
# ---------------------------------------------------------------------------

class TestHandleUseSkill:
    def test_no_name(self):
        with patch("agent.skills.load_skill"):
            result = _run(execute_tool("use_skill", {"name": ""}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    def test_skill_not_found(self):
        with patch("agent.skills.load_skill", return_value=None):
            result = _run(execute_tool("use_skill", {"name": "nonexistent"}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed
            assert "not found" in parsed["error"]

    def test_skill_found(self):
        skill = {
            "name": "meme-gen",
            "content": "# Meme Generator\nSteps...",
            "scripts": {"make_meme.py": "print('meme')"},
            "references": [],
        }
        with patch("agent.skills.load_skill", return_value=skill):
            tracker = _make_tracker()
            result = _run(execute_tool("use_skill", {"name": "meme-gen"}, tracker))
            parsed = json.loads(result)
            assert parsed["name"] == "meme-gen"
            assert "instructions" in parsed
            assert "scripts" in parsed
            assert "meme-gen" in tracker.skills_used

    def test_skill_without_scripts_and_refs(self):
        skill = {
            "name": "simple",
            "content": "# Simple skill",
            "scripts": {},
            "references": [],
        }
        with patch("agent.skills.load_skill", return_value=skill):
            result = _run(execute_tool("use_skill", {"name": "simple"}, _make_tracker()))
            parsed = json.loads(result)
            assert "scripts" not in parsed
            assert "references" not in parsed


# ---------------------------------------------------------------------------
# _handle_create_skill
# ---------------------------------------------------------------------------

class TestHandleCreateSkill:
    def test_missing_required_fields(self):
        with patch("agent.skills.create_skill"):
            result = _run(execute_tool("create_skill", {"name": "test"}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    def test_successful_creation(self):
        with patch("agent.skills.create_skill", return_value={"status": "created", "name": "test-skill"}):
            tracker = _make_tracker()
            result = _run(execute_tool("create_skill", {
                "name": "test-skill",
                "description": "A test skill",
                "skill_md": "# Test\nDo stuff",
            }, tracker))
            parsed = json.loads(result)
            assert parsed["status"] == "created"
            assert "create_skill:test-skill" in tracker.apis_called

    def test_invalid_scripts_json(self):
        with patch("agent.skills.create_skill"):
            result = _run(execute_tool("create_skill", {
                "name": "test-skill",
                "description": "A test skill",
                "skill_md": "# Test",
                "scripts": "{invalid json",
            }, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed
            assert "Invalid scripts JSON" in parsed["error"]

    def test_scripts_as_dict(self):
        with patch("agent.skills.create_skill", return_value={"status": "created"}) as mock_create:
            _run(execute_tool("create_skill", {
                "name": "test-skill",
                "description": "A test skill",
                "skill_md": "# Test",
                "scripts": '{"run.py": "print(1)"}',
            }, _make_tracker()))
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["scripts"] == {"run.py": "print(1)"}


# ---------------------------------------------------------------------------
# _handle_execute_openclaw_script
# ---------------------------------------------------------------------------

class TestHandleExecuteOpenclawScript:
    def test_script_not_in_allowlist(self):
        result = _run(execute_tool("execute_openclaw_script", {"script_name": "evil.js"}, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not in allowlist" in parsed["error"]

    def test_unsafe_args_semicolon(self):
        with patch("agent.tools.Path") as MockPath:
            instance = MockPath.return_value
            instance.exists.return_value = True
            with patch("agent.tools.settings") as mock_settings:
                mock_settings.OPENCLAW_SCRIPTS_DIR = "/tmp/scripts"
                result = _run(execute_tool("execute_openclaw_script", {
                    "script_name": "read_vault.js",
                    "args": "foo; rm -rf /",
                }, _make_tracker()))
                parsed = json.loads(result)
                assert "error" in parsed
                assert "unsafe characters" in parsed["error"]

    def test_unsafe_args_pipe(self):
        with patch("agent.tools.Path") as MockPath:
            instance = MockPath.return_value
            instance.exists.return_value = True
            with patch("agent.tools.settings") as mock_settings:
                mock_settings.OPENCLAW_SCRIPTS_DIR = "/tmp/scripts"
                result = _run(execute_tool("execute_openclaw_script", {
                    "script_name": "browse_tasks.js",
                    "args": "foo | cat /etc/passwd",
                }, _make_tracker()))
                parsed = json.loads(result)
                assert "error" in parsed
                assert "unsafe characters" in parsed["error"]

    def test_unsafe_args_backtick(self):
        with patch("agent.tools.Path") as MockPath:
            instance = MockPath.return_value
            instance.exists.return_value = True
            with patch("agent.tools.settings") as mock_settings:
                mock_settings.OPENCLAW_SCRIPTS_DIR = "/tmp/scripts"
                result = _run(execute_tool("execute_openclaw_script", {
                    "script_name": "browse_tasks.js",
                    "args": "`whoami`",
                }, _make_tracker()))
                parsed = json.loads(result)
                assert "error" in parsed
                assert "unsafe characters" in parsed["error"]

    @patch("agent.tools.subprocess.run")
    @patch("agent.tools.settings")
    def test_script_not_found_on_disk(self, mock_settings, mock_subproc):
        mock_settings.OPENCLAW_SCRIPTS_DIR = "/tmp/scripts"
        result = _run(execute_tool("execute_openclaw_script", {
            "script_name": "read_vault.js",
        }, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"]
        mock_subproc.assert_not_called()

    @patch("agent.tools.asyncio.to_thread", new_callable=AsyncMock)
    @patch("agent.tools.settings")
    def test_successful_execution(self, mock_settings, mock_to_thread):
        mock_settings.OPENCLAW_SCRIPTS_DIR = "/tmp/scripts"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "vault balance: 1000\n"
        mock_result.stderr = ""
        mock_to_thread.return_value = mock_result

        # Create the script file path mock
        with patch("agent.tools.Path") as MockPath:
            script_path = MagicMock()
            script_path.exists.return_value = True
            script_path.__str__ = lambda self: "/tmp/scripts/read_vault.js"
            script_path.parent = "/tmp/scripts"
            MockPath.return_value.__truediv__ = MagicMock(return_value=script_path)

            tracker = _make_tracker()
            result = _run(execute_tool("execute_openclaw_script", {
                "script_name": "read_vault.js",
            }, tracker))
            assert "vault balance: 1000" in result
            assert "read_vault.js" in tracker.scripts_executed

    @patch("agent.tools.asyncio.to_thread", new_callable=AsyncMock)
    @patch("agent.tools.settings")
    def test_script_nonzero_exit(self, mock_settings, mock_to_thread):
        mock_settings.OPENCLAW_SCRIPTS_DIR = "/tmp/scripts"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: connection refused"
        mock_to_thread.return_value = mock_result

        with patch("agent.tools.Path") as MockPath:
            script_path = MagicMock()
            script_path.exists.return_value = True
            script_path.__str__ = lambda self: "/tmp/scripts/read_vault.js"
            script_path.parent = "/tmp/scripts"
            MockPath.return_value.__truediv__ = MagicMock(return_value=script_path)

            result = _run(execute_tool("execute_openclaw_script", {
                "script_name": "read_vault.js",
            }, _make_tracker()))
            parsed = json.loads(result)
            assert parsed["exit_code"] == 1
            assert "connection refused" in parsed["stderr"]

    @patch("agent.tools.asyncio.to_thread", new_callable=AsyncMock, side_effect=subprocess.TimeoutExpired(cmd="node", timeout=60))
    @patch("agent.tools.settings")
    def test_script_timeout(self, mock_settings, mock_to_thread):
        mock_settings.OPENCLAW_SCRIPTS_DIR = "/tmp/scripts"

        with patch("agent.tools.Path") as MockPath:
            script_path = MagicMock()
            script_path.exists.return_value = True
            script_path.__str__ = lambda self: "/tmp/scripts/read_vault.js"
            script_path.parent = "/tmp/scripts"
            MockPath.return_value.__truediv__ = MagicMock(return_value=script_path)

            result = _run(execute_tool("execute_openclaw_script", {
                "script_name": "read_vault.js",
            }, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed
            assert "timed out" in parsed["error"]


# ---------------------------------------------------------------------------
# _OPENCLAW_ALLOWLIST
# ---------------------------------------------------------------------------

class TestOpenclawAllowlist:
    EXPECTED_SCRIPTS = {
        "read_vault.js",
        "create_campaign.js",
        "schedule_content.js",
        "log_activity.js",
        "browse_tasks.js",
        "claim_task.js",
        "submit_task.js",
        "check_balance.js",
        "list_campaigns.js",
        "list_activities.js",
        "verify_contract.js",
        "get_task_details.js",
    }

    def test_allowlist_contents(self):
        assert _OPENCLAW_ALLOWLIST == self.EXPECTED_SCRIPTS

    def test_allowlist_count(self):
        assert len(_OPENCLAW_ALLOWLIST) == 12

    def test_all_are_js_files(self):
        for script in _OPENCLAW_ALLOWLIST:
            assert script.endswith(".js")


# ---------------------------------------------------------------------------
# _handle_check_figma_design
# ---------------------------------------------------------------------------

class TestHandleCheckFigmaDesign:
    @patch("agent.tools.figma.get_file_styles", new_callable=AsyncMock, return_value={"styles": []})
    @patch("agent.tools.settings")
    def test_styles_action(self, mock_settings, mock_figma):
        mock_settings.FIGMA_NODE_ID = "0:1"
        tracker = _make_tracker()
        result = _run(execute_tool("check_figma_design", {"action": "styles"}, tracker))
        parsed = json.loads(result)
        assert "styles" in parsed
        mock_figma.assert_awaited_once()
        assert "figma" in tracker.apis_called

    @patch("agent.tools.figma.get_design_tokens", new_callable=AsyncMock, return_value={"colors": ["#000"]})
    @patch("agent.tools.settings")
    def test_tokens_action(self, mock_settings, mock_figma):
        mock_settings.FIGMA_NODE_ID = "0:1"
        result = _run(execute_tool("check_figma_design", {"action": "tokens", "node_id": "0:5"}, _make_tracker()))
        mock_figma.assert_awaited_once_with("0:5")

    @patch("agent.tools.figma.get_node_metadata", new_callable=AsyncMock, return_value={"name": "Frame"})
    @patch("agent.tools.settings")
    def test_metadata_action(self, mock_settings, mock_figma):
        mock_settings.FIGMA_NODE_ID = "0:1"
        result = _run(execute_tool("check_figma_design", {"action": "metadata"}, _make_tracker()))
        parsed = json.loads(result)
        assert parsed["name"] == "Frame"

    @patch("agent.tools.figma.get_node_screenshot", new_callable=AsyncMock, return_value={"url": "https://..."})
    @patch("agent.tools.settings")
    def test_screenshot_action(self, mock_settings, mock_figma):
        mock_settings.FIGMA_NODE_ID = "0:1"
        result = _run(execute_tool("check_figma_design", {"action": "screenshot"}, _make_tracker()))
        parsed = json.loads(result)
        assert "url" in parsed

    @patch("agent.tools.settings")
    def test_unknown_action(self, mock_settings):
        mock_settings.FIGMA_NODE_ID = "0:1"
        result = _run(execute_tool("check_figma_design", {"action": "invalid"}, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Unknown action" in parsed["error"]


# ---------------------------------------------------------------------------
# _handle_generate_image
# ---------------------------------------------------------------------------

class TestHandleGenerateImage:
    def test_no_prompt_returns_error(self):
        with patch("agent.tools.asset_library"), \
             patch("agent.tools.image_gen"), \
             patch("agent.compositor_config._load_config_json", return_value=None):
            result = _run(execute_tool("generate_image", {"prompt": ""}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    @patch("agent.tools.image_gen.select_model", return_value=("black-forest-labs/flux-1.1-pro", "default"))
    @patch("agent.tools.image_gen.generate_image", new_callable=AsyncMock, return_value="https://image.url/img.png")
    @patch("agent.tools.asset_library.add")
    def test_text_to_image_success(self, mock_lib_add, mock_gen, mock_select):
        with patch("agent.compositor_config._load_config_json", return_value=None), \
             patch("agent.tools._state.get_active_profile", return_value=None), \
             patch("agent.tools._REFS_DIR") as mock_refs, \
             patch("agent.template_memory.get_image_region_aspect_ratio", return_value=None):
            mock_refs.glob.return_value = []
            tracker = _make_tracker()
            result = _run(execute_tool("generate_image", {
                "prompt": "A futuristic logo",
                "content_type": "announcement",
            }, tracker))
            parsed = json.loads(result)
            assert parsed["image_url"] == "https://image.url/img.png"
            assert "flux" in parsed.get("model", "")


# ---------------------------------------------------------------------------
# _handle_img2img
# ---------------------------------------------------------------------------

class TestHandleImg2Img:
    def test_no_prompt_returns_error(self):
        result = _run(execute_tool("img2img", {"prompt": ""}, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "No prompt" in parsed["error"]

    @patch("agent.tools.image_gen.generate_img2img", new_callable=AsyncMock, return_value="https://img.url/out.png")
    def test_with_reference_path(self, mock_img2img):
        tracker = _make_tracker()
        result = _run(execute_tool("img2img", {
            "prompt": "make it blue",
            "reference_image_path": "/tmp/ref.png",
        }, tracker))
        parsed = json.loads(result)
        assert parsed["image_url"] == "https://img.url/out.png"
        assert parsed["model"] == "flux-kontext-pro"
        assert "replicate:flux-kontext-pro" in tracker.apis_called

    @patch("agent.tools.image_gen.generate_image", new_callable=AsyncMock, return_value="https://fallback.url/img.png")
    def test_no_reference_no_mascot_fallback(self, mock_gen):
        result = _run(execute_tool("img2img", {
            "prompt": "a nice landscape",
        }, _make_tracker()))
        parsed = json.loads(result)
        assert "image_url" in parsed
        assert "text-to-image" in parsed.get("note", "")

    @patch("agent.tools.image_gen.generate_img2img", new_callable=AsyncMock, return_value=None)
    def test_img2img_failure(self, mock_img2img):
        result = _run(execute_tool("img2img", {
            "prompt": "make it blue",
            "reference_image_path": "/tmp/ref.png",
        }, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "failed" in parsed["error"]


# ---------------------------------------------------------------------------
# _handle_read_feedback_history
# ---------------------------------------------------------------------------

class TestHandleReadFeedbackHistory:
    def test_no_preferences(self):
        mock_session = MagicMock()
        mock_session.learned_preferences = []
        with patch("agent.session.load_session", return_value=mock_session):
            result = _run(execute_tool("read_feedback_history", {}, _make_tracker()))
            parsed = json.loads(result)
            assert parsed["preferences"] == []
            assert "No learned preferences" in parsed["message"]

    def test_with_preferences(self):
        mock_session = MagicMock()
        mock_session.learned_preferences = ["Use short sentences", "Avoid exclamation marks"]
        with patch("agent.session.load_session", return_value=mock_session):
            tracker = _make_tracker()
            result = _run(execute_tool("read_feedback_history", {}, tracker))
            parsed = json.loads(result)
            assert parsed["count"] == 2
            assert "distilled" in parsed["message"]
            assert "agent_session.json" in tracker.files_loaded


# ---------------------------------------------------------------------------
# _handle_delegate_task
# ---------------------------------------------------------------------------

class TestHandleDelegateTask:
    def test_no_task(self):
        with patch("agent.subagent.delegate_task", new_callable=AsyncMock):
            result = _run(execute_tool("delegate_task", {"task": ""}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    @patch("agent.subagent.delegate_task", new_callable=AsyncMock, return_value={"result": "analysis complete"})
    def test_successful_delegation(self, mock_delegate):
        tracker = _make_tracker()
        result = _run(execute_tool("delegate_task", {
            "task": "Research competitor landscape",
            "context": "crypto space",
        }, tracker))
        parsed = json.loads(result)
        assert parsed["result"] == "analysis complete"
        assert "subagent:delegate" in tracker.apis_called


# ---------------------------------------------------------------------------
# _handle_search_memory
# ---------------------------------------------------------------------------

class TestHandleSearchMemory:
    def test_no_query(self):
        with patch("agent.memory.search_past_generations"):
            result = _run(execute_tool("search_memory", {"query": ""}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    @patch("agent.memory.search_past_generations", return_value=[])
    def test_no_results(self, mock_search):
        result = _run(execute_tool("search_memory", {"query": "partnership"}, _make_tracker()))
        parsed = json.loads(result)
        assert parsed["results"] == []
        assert "No relevant" in parsed["message"]

    @patch("agent.memory.search_past_generations", return_value=[
        {"caption": "Partnership with X", "status": "approved"},
    ])
    def test_with_results(self, mock_search):
        tracker = _make_tracker()
        result = _run(execute_tool("search_memory", {
            "query": "partnership",
            "status_filter": "approved",
        }, tracker))
        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert "memory:search" in tracker.apis_called
        mock_search.assert_called_once_with("partnership", top_k=5, status_filter="approved")


# ---------------------------------------------------------------------------
# Unsafe chars regex (used internally by openclaw handler)
# ---------------------------------------------------------------------------

class TestUnsafeCharsRegex:
    """Test the _UNSAFE_CHARS pattern used to sanitize openclaw args."""

    UNSAFE_CHARS = re.compile(r"[;&|`$(){}!<>\\\n\r\t]")

    @pytest.mark.parametrize("char", [";", "&", "|", "`", "$", "(", ")", "{", "}", "!", "<", ">", "\\", "\n", "\r", "\t"])
    def test_rejects_unsafe_char(self, char):
        assert self.UNSAFE_CHARS.search(f"arg{char}rest")

    @pytest.mark.parametrize("safe_input", [
        "simple-arg",
        "arg_with_underscore",
        "arg.with.dots",
        "arg with spaces",
        "12345",
        "campaign-id-abc-123",
    ])
    def test_accepts_safe_input(self, safe_input):
        assert self.UNSAFE_CHARS.search(safe_input) is None


# ---------------------------------------------------------------------------
# Error handling — tools should return JSON errors, not raise
# ---------------------------------------------------------------------------

class TestToolErrorHandling:
    """Handlers should return JSON error strings rather than raising exceptions."""

    def test_generate_image_no_prompt(self):
        with patch("agent.compositor_config._load_config_json", return_value=None):
            result = _run(execute_tool("generate_image", {}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    def test_img2img_no_prompt(self):
        result = _run(execute_tool("img2img", {}, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed

    def test_openclaw_bad_script(self):
        result = _run(execute_tool("execute_openclaw_script", {"script_name": "bad.js"}, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed

    def test_use_skill_empty_name(self):
        with patch("agent.skills.load_skill"):
            result = _run(execute_tool("use_skill", {}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    def test_create_skill_missing_fields(self):
        with patch("agent.skills.create_skill"):
            result = _run(execute_tool("create_skill", {}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    def test_delegate_task_empty(self):
        with patch("agent.subagent.delegate_task", new_callable=AsyncMock):
            result = _run(execute_tool("delegate_task", {}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    def test_search_memory_empty_query(self):
        with patch("agent.memory.search_past_generations"):
            result = _run(execute_tool("search_memory", {}, _make_tracker()))
            parsed = json.loads(result)
            assert "error" in parsed

    def test_create_campaign_no_name(self):
        result = _run(execute_tool("create_campaign", {"posts": [{"day": 1, "time": "9am", "caption": "hi"}]}, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed

    def test_create_campaign_no_posts(self):
        result = _run(execute_tool("create_campaign", {"name": "test"}, _make_tracker()))
        parsed = json.loads(result)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# _handle_create_campaign
# ---------------------------------------------------------------------------

class TestHandleCreateCampaign:
    @patch("agent.campaigns.create_campaign", return_value={"success": True, "campaign": {"name": "test-campaign"}, "message": "Created"})
    @patch("agent.scheduling.schedule_queue.add_scheduled")
    @patch("agent.scheduling.schedule_queue.parse_time", return_value=(1711300000.0, "2026-03-25 10:00 AM PDT"))
    def test_basic_campaign_creation(self, mock_parse, mock_add, mock_create):
        mock_add.return_value = {"id": "abc12345", "prompt": "test", "scheduled_utc": 1711300000.0, "status": "pending"}
        tracker = _make_tracker()
        result = _run(execute_tool("create_campaign", {
            "name": "test-campaign",
            "brief": "A test campaign",
            "posts": [
                {"day": 1, "time": "10:00am", "caption": "First post"},
                {"day": 2, "time": "10:00am", "caption": "Second post"},
            ],
        }, tracker))
        parsed = json.loads(result)
        assert parsed["status"] == "campaign_created"
        assert parsed["campaign_name"] == "test-campaign"
        assert parsed["posts_scheduled"] == 2
        assert "create_campaign:test-campaign" in tracker.apis_called

    @patch("agent.campaigns.create_campaign", return_value={"success": True, "campaign": {"name": "dup-test"}, "message": "Created"})
    @patch("agent.scheduling.schedule_queue.add_scheduled", return_value=None)
    @patch("agent.scheduling.schedule_queue.parse_time", return_value=(1711300000.0, "2026-03-25 10:00 AM PDT"))
    def test_duplicate_posts_skipped(self, mock_parse, mock_add, mock_create):
        tracker = _make_tracker()
        result = _run(execute_tool("create_campaign", {
            "name": "dup-test",
            "posts": [
                {"day": 1, "time": "10:00am", "caption": "Duplicate post"},
            ],
        }, tracker))
        parsed = json.loads(result)
        assert parsed["posts_scheduled"] == 0
        assert len(parsed["errors"]) > 0
