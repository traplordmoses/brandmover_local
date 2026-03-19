"""
Comprehensive tests for agent/engine.py — the core agent engine module.

Tests cover:
- AgentResult dataclass defaults and initialization
- _sanitize_draft: hashtag removal, em-dash replacement, AI word stripping,
  title/subtitle truncation, hex color preservation, edge cases
- _try_parse_draft: JSON in code fences, raw JSON, missing caption, malformed JSON
- _trim_conversation: large messages, base64 stripping, tool result truncation, size cap
- _extract_image_url / _extract_image_urls: URL extraction from tool call logs
- _tool_description: known tool mappings and unknown tool fallback
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from agent.engine import (
    AgentResult,
    _sanitize_draft,
    _try_parse_draft,
    _trim_conversation,
    _extract_image_url,
    _extract_image_urls,
    _tool_description,
    _block_to_dict,
    MAX_HISTORY_SIZE_CHARS,
)
from agent.resource_log import ResourceTracker


# ---------------------------------------------------------------------------
# TestAgentResult
# ---------------------------------------------------------------------------

class TestAgentResult:
    """Tests for the AgentResult dataclass."""

    def test_default_values(self):
        r = AgentResult()
        assert r.final_text == ""
        assert r.draft == {}
        assert r.image_url is None
        assert r.image_urls == []
        assert isinstance(r.resources, ResourceTracker)
        assert r.tool_calls_made == []
        assert r.turns_used == 0
        assert r.total_time == 0.0
        assert r.conversation_history == []
        assert r._finished is False

    def test_custom_values(self):
        r = AgentResult(
            final_text="hello",
            draft={"caption": "test"},
            image_url="https://example.com/img.png",
            image_urls=["https://example.com/a.png", "https://example.com/b.png"],
            turns_used=3,
            total_time=1.5,
            _finished=True,
        )
        assert r.final_text == "hello"
        assert r.draft == {"caption": "test"}
        assert r.image_url == "https://example.com/img.png"
        assert len(r.image_urls) == 2
        assert r.turns_used == 3
        assert r.total_time == 1.5
        assert r._finished is True

    def test_mutable_defaults_are_independent(self):
        """Each instance should get its own mutable defaults."""
        a = AgentResult()
        b = AgentResult()
        a.tool_calls_made.append("x")
        a.image_urls.append("url")
        assert b.tool_calls_made == []
        assert b.image_urls == []


# ---------------------------------------------------------------------------
# TestSanitizeDraft
# ---------------------------------------------------------------------------

class TestSanitizeDraft:
    """Tests for _sanitize_draft()."""

    def test_strips_hashtags_from_caption(self):
        draft = {"caption": "Check out FOID #crypto #blockchain today"}
        result = _sanitize_draft(draft)
        assert "#crypto" not in result["caption"]
        assert "#blockchain" not in result["caption"]
        assert "FOID" in result["caption"]

    def test_strips_hashtags_from_title(self):
        draft = {"title": "Big #News", "caption": "ok"}
        result = _sanitize_draft(draft)
        assert "#News" not in result["title"]
        assert "Big" in result["title"]

    def test_strips_hashtags_from_subtitle(self):
        draft = {"subtitle": "The #future is here #now", "caption": "ok"}
        result = _sanitize_draft(draft)
        assert "#future" not in result["subtitle"]
        assert "#now" not in result["subtitle"]

    def test_preserves_hex_colors_starting_with_digit(self):
        """Hex color codes starting with a digit like #000000 should NOT be stripped.

        The regex _HASHTAG_RE is `#[A-Za-z]\\w*`, so it only matches hashtags
        that start with a letter. Hex codes like #0A0B0C (digit after #) are safe.
        Hex codes like #FF0000 (letter after #) ARE matched as hashtags — this is
        an accepted trade-off since those codes live in image_prompt, not caption.
        """
        draft = {"caption": "Use color #000000 and #1A2B3C in designs"}
        result = _sanitize_draft(draft)
        assert "#000000" in result["caption"]
        assert "#1A2B3C" in result["caption"]

    def test_hex_colors_starting_with_letter_are_stripped(self):
        """Hex codes starting with a letter (e.g. #FF0000) match the hashtag regex."""
        draft = {"caption": "Color #FF0000 here"}
        result = _sanitize_draft(draft)
        # This is an accepted behavior — text fields shouldn't have hex codes
        assert "#FF0000" not in result["caption"]

    def test_replaces_em_dashes_with_commas(self):
        draft = {"caption": "FOID — the future — starts now"}
        result = _sanitize_draft(draft)
        assert "—" not in result["caption"]
        assert "," in result["caption"]

    def test_strips_ai_words_revolutionizing(self):
        draft = {"caption": "We are revolutionizing the industry"}
        result = _sanitize_draft(draft)
        assert "revolutionizing" not in result["caption"].lower()

    def test_strips_ai_words_leveraging(self):
        draft = {"caption": "Leveraging blockchain for good"}
        result = _sanitize_draft(draft)
        assert "leveraging" not in result["caption"].lower()

    def test_strips_ai_words_cutting_edge(self):
        draft = {"caption": "Our cutting-edge solution"}
        result = _sanitize_draft(draft)
        assert "cutting-edge" not in result["caption"].lower()

    def test_strips_ai_words_game_changing(self):
        draft = {"caption": "A game-changing protocol"}
        result = _sanitize_draft(draft)
        assert "game-changing" not in result["caption"].lower()

    def test_strips_ai_words_groundbreaking(self):
        draft = {"caption": "A groundbreaking approach to DeFi"}
        result = _sanitize_draft(draft)
        assert "groundbreaking" not in result["caption"].lower()

    def test_strips_ai_words_pioneering(self):
        draft = {"caption": "Pioneering the next wave"}
        result = _sanitize_draft(draft)
        assert "pioneering" not in result["caption"].lower()

    def test_strips_ai_words_transformative(self):
        draft = {"caption": "This is transformative technology"}
        result = _sanitize_draft(draft)
        assert "transformative" not in result["caption"].lower()

    def test_strips_ai_words_innovative(self):
        draft = {"caption": "An innovative approach to yield"}
        result = _sanitize_draft(draft)
        assert "innovative" not in result["caption"].lower()

    def test_strips_ai_words_delve(self):
        draft = {"caption": "Let's delve into the details"}
        result = _sanitize_draft(draft)
        assert "delve" not in result["caption"].lower()

    def test_strips_ai_words_paradigm(self):
        draft = {"caption": "A new paradigm for finance"}
        result = _sanitize_draft(draft)
        assert "paradigm" not in result["caption"].lower()

    def test_strips_ai_words_synergy(self):
        draft = {"caption": "The synergy between DeFi and AI"}
        result = _sanitize_draft(draft)
        assert "synergy" not in result["caption"].lower()

    def test_strips_ai_words_empower(self):
        draft = {"caption": "Empower your portfolio today"}
        result = _sanitize_draft(draft)
        assert "empower" not in result["caption"].lower()

    def test_strips_ai_words_harness(self):
        draft = {"caption": "Harness the power of decentralization"}
        result = _sanitize_draft(draft)
        assert "harness" not in result["caption"].lower()

    def test_strips_ai_words_holistic(self):
        draft = {"caption": "A holistic view of the market"}
        result = _sanitize_draft(draft)
        assert "holistic" not in result["caption"].lower()

    def test_strips_ai_words_landscape(self):
        draft = {"caption": "Navigate the landscape of crypto"}
        result = _sanitize_draft(draft)
        assert "landscape" not in result["caption"].lower()

    def test_strips_ai_words_streamline(self):
        draft = {"caption": "We streamline your DeFi experience"}
        result = _sanitize_draft(draft)
        assert "streamline" not in result["caption"].lower()

    def test_strips_ai_words_elevate(self):
        draft = {"caption": "Elevate your trading strategy"}
        result = _sanitize_draft(draft)
        assert "elevate" not in result["caption"].lower()

    def test_strips_multiple_ai_words(self):
        draft = {"caption": "Revolutionizing and leveraging groundbreaking synergy"}
        result = _sanitize_draft(draft)
        text = result["caption"].lower()
        assert "revolutionizing" not in text
        assert "leveraging" not in text
        assert "groundbreaking" not in text
        assert "synergy" not in text

    def test_collapses_double_spaces(self):
        draft = {"caption": "FOID  is  great #launch"}
        result = _sanitize_draft(draft)
        assert "  " not in result["caption"]

    def test_title_truncated_to_4_words(self):
        draft = {"title": "This Is A Very Long Title", "caption": "ok"}
        result = _sanitize_draft(draft)
        assert result["title"] == "This Is A Very"

    def test_title_at_4_words_not_truncated(self):
        draft = {"title": "Four Words Are Fine", "caption": "ok"}
        result = _sanitize_draft(draft)
        assert result["title"] == "Four Words Are Fine"

    def test_title_under_4_words_not_truncated(self):
        draft = {"title": "Short Title", "caption": "ok"}
        result = _sanitize_draft(draft)
        assert result["title"] == "Short Title"

    def test_subtitle_truncated_to_10_words(self):
        draft = {
            "subtitle": "one two three four five six seven eight nine ten eleven twelve",
            "caption": "ok",
        }
        result = _sanitize_draft(draft)
        assert result["subtitle"] == "one two three four five six seven eight nine ten"

    def test_subtitle_at_10_words_not_truncated(self):
        draft = {
            "subtitle": "one two three four five six seven eight nine ten",
            "caption": "ok",
        }
        result = _sanitize_draft(draft)
        assert result["subtitle"] == "one two three four five six seven eight nine ten"

    def test_empty_draft(self):
        result = _sanitize_draft({})
        assert result == {}

    def test_missing_text_fields(self):
        draft = {"caption": "Hello world", "image_prompt": "some prompt #tag"}
        result = _sanitize_draft(draft)
        # image_prompt is not in _DRAFT_TEXT_FIELDS, should not be modified
        assert result["image_prompt"] == "some prompt #tag"

    def test_non_string_field_ignored(self):
        draft = {"caption": 42, "title": None}
        result = _sanitize_draft(draft)
        assert result["caption"] == 42
        assert result["title"] is None

    def test_clean_draft_unchanged(self):
        draft = {"caption": "FOID is live on mainnet", "title": "FOID Live"}
        result = _sanitize_draft(draft)
        assert result["caption"] == "FOID is live on mainnet"
        assert result["title"] == "FOID Live"

    def test_em_dash_leading_comma_cleaned(self):
        """An em-dash at the start shouldn't leave a leading comma."""
        draft = {"caption": "— starting with dash"}
        result = _sanitize_draft(draft)
        assert not result["caption"].startswith(",")
        assert "—" not in result["caption"]


# ---------------------------------------------------------------------------
# TestTryParseDraft
# ---------------------------------------------------------------------------

class TestTryParseDraft:
    """Tests for _try_parse_draft()."""

    def test_json_in_code_fence(self):
        text = 'Here is the draft:\n```json\n{"caption": "Hello world", "hashtags": ["#test"]}\n```'
        result = _try_parse_draft(text)
        assert result is not None
        assert result["caption"] == "Hello world"

    def test_json_in_code_fence_no_json_tag(self):
        text = 'Here:\n```\n{"caption": "No tag fence"}\n```'
        result = _try_parse_draft(text)
        assert result is not None
        assert result["caption"] == "No tag fence"

    def test_raw_json_in_text(self):
        text = 'The result is {"caption": "raw json", "alt_text": "desc"} end.'
        result = _try_parse_draft(text)
        assert result is not None
        assert result["caption"] == "raw json"

    def test_missing_caption_key_returns_none(self):
        text = '```json\n{"title": "No caption here"}\n```'
        result = _try_parse_draft(text)
        assert result is None

    def test_malformed_json_returns_none(self):
        text = '```json\n{caption: broken}\n```'
        result = _try_parse_draft(text)
        assert result is None

    def test_empty_string_returns_none(self):
        result = _try_parse_draft("")
        assert result is None

    def test_no_json_at_all(self):
        result = _try_parse_draft("Just some plain text, no JSON here.")
        assert result is None

    def test_multiple_json_objects_picks_one_with_caption(self):
        text = '{"title": "no caption"} and also {"caption": "found it", "extra": 1}'
        result = _try_parse_draft(text)
        assert result is not None
        assert result["caption"] == "found it"

    def test_json_with_nested_object(self):
        text = '{"caption": "Hello", "meta": {"key": "val"}}'
        result = _try_parse_draft(text)
        assert result is not None
        assert result["caption"] == "Hello"


# ---------------------------------------------------------------------------
# TestTrimConversation
# ---------------------------------------------------------------------------

class TestTrimConversation:
    """Tests for _trim_conversation()."""

    def test_empty_input(self):
        assert _trim_conversation([]) == []

    def test_simple_messages_pass_through(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _trim_conversation(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there"

    def test_truncates_long_tool_results(self):
        long_text = "x" * 5000
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": long_text, "tool_use_id": "123"}
            ]},
        ]
        result = _trim_conversation(msgs)
        block = result[0]["content"][0]
        assert len(block["content"]) < len(long_text)
        assert "[...truncated]" in block["content"]

    def test_tool_result_under_limit_not_truncated(self):
        short_text = "x" * 100
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": short_text, "tool_use_id": "123"}
            ]},
        ]
        result = _trim_conversation(msgs)
        block = result[0]["content"][0]
        assert block["content"] == short_text

    def test_strips_base64_image_data(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "abc123..."}}
            ]},
        ]
        result = _trim_conversation(msgs)
        block = result[0]["content"][0]
        assert block["type"] == "text"
        assert "stripped" in block["text"]

    def test_truncates_long_string_content(self):
        long_string = "a" * 10000
        msgs = [{"role": "user", "content": long_string}]
        result = _trim_conversation(msgs)
        assert len(result[0]["content"]) < len(long_string)
        assert "[...truncated]" in result[0]["content"]

    def test_string_content_under_limit_not_truncated(self):
        short_string = "Hello world"
        msgs = [{"role": "user", "content": short_string}]
        result = _trim_conversation(msgs)
        assert result[0]["content"] == short_string

    def test_sdk_block_with_model_dump(self):
        """Blocks with model_dump() should be converted to dicts."""
        mock_block = MagicMock()
        mock_block.model_dump.return_value = {"type": "text", "text": "hello"}

        msgs = [{"role": "assistant", "content": [mock_block]}]
        result = _trim_conversation(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "hello"}

    def test_caps_total_size(self):
        """History exceeding MAX_HISTORY_SIZE_CHARS is trimmed by removing message pairs."""
        # Create a conversation that is definitely too large
        big_content = "x" * 4000
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first reply"},
        ]
        # Add many pairs to exceed 50k
        for i in range(20):
            msgs.append({"role": "user", "content": big_content})
            msgs.append({"role": "assistant", "content": big_content})

        result = _trim_conversation(msgs)
        serialized = json.dumps(result, default=str)
        # Either under the limit or at minimum size (4 messages)
        assert len(serialized) <= MAX_HISTORY_SIZE_CHARS or len(result) <= 4

    def test_does_not_mutate_original(self):
        msgs = [{"role": "user", "content": "hello"}]
        original_content = msgs[0]["content"]
        _trim_conversation(msgs)
        assert msgs[0]["content"] == original_content


# ---------------------------------------------------------------------------
# TestBlockToDict
# ---------------------------------------------------------------------------

class TestBlockToDict:
    """Tests for _block_to_dict()."""

    def test_dict_passthrough(self):
        d = {"type": "text", "text": "hi"}
        assert _block_to_dict(d) == d

    def test_model_dump_object(self):
        mock = MagicMock()
        mock.model_dump.return_value = {"type": "tool_use", "name": "foo"}
        assert _block_to_dict(mock) == {"type": "tool_use", "name": "foo"}

    def test_fallback_to_str(self):
        result = _block_to_dict(42)
        assert result == {"type": "text", "text": "42"}


# ---------------------------------------------------------------------------
# TestExtractImageUrl
# ---------------------------------------------------------------------------

class TestExtractImageUrl:
    """Tests for _extract_image_url()."""

    def test_extracts_from_pre_extracted_url(self):
        calls = [
            {"name": "generate_image", "image_url": "https://example.com/img.png"}
        ]
        assert _extract_image_url(calls) == "https://example.com/img.png"

    def test_extracts_from_result_json(self):
        result_json = json.dumps({"image_url": "https://example.com/gen.png"})
        calls = [{"name": "generate_image", "result": result_json}]
        assert _extract_image_url(calls) == "https://example.com/gen.png"

    def test_extracts_from_img2img(self):
        calls = [
            {"name": "img2img", "image_url": "https://example.com/i2i.png"}
        ]
        assert _extract_image_url(calls) == "https://example.com/i2i.png"

    def test_extracts_via_regex_replicate_url(self):
        """The regex requires chars between https:// and the replicate domain,
        matching URLs like https://cdn.example.com/replicate.delivery/path."""
        result_str = 'Generated: https://cdn.example.com/replicate.delivery/abc123/image.png done'
        calls = [{"name": "generate_image", "result": result_str}]
        url = _extract_image_url(calls)
        assert url is not None
        assert "replicate.delivery" in url

    def test_extracts_via_regex_pbxt_url(self):
        result_str = 'Output at https://cdn.example.com/pbxt.replicate.com/xyz/out.png ok'
        calls = [{"name": "generate_image", "result": result_str}]
        url = _extract_image_url(calls)
        assert url is not None
        assert "pbxt.replicate.com" in url

    def test_regex_fallback_no_match_for_direct_domain(self):
        """Direct replicate.delivery URLs don't match the regex pattern
        (it requires chars between https:// and the domain)."""
        result_str = 'Output: https://replicate.delivery/abc/img.png'
        calls = [{"name": "generate_image", "result": result_str}]
        # No pre-extracted URL, no valid JSON, and direct domain doesn't match regex
        assert _extract_image_url(calls) is None

    def test_no_image_tools_returns_none(self):
        calls = [
            {"name": "read_brand_guidelines", "result": "guidelines loaded"}
        ]
        assert _extract_image_url(calls) is None

    def test_empty_list_returns_none(self):
        assert _extract_image_url([]) is None

    def test_malformed_result_returns_none(self):
        calls = [{"name": "generate_image", "result": "not json at all, no url"}]
        assert _extract_image_url(calls) is None

    def test_prefers_pre_extracted_over_result(self):
        """Pre-extracted image_url should be returned before parsing result."""
        result_json = json.dumps({"image_url": "https://example.com/result.png"})
        calls = [
            {
                "name": "generate_image",
                "image_url": "https://example.com/pre.png",
                "result": result_json,
            }
        ]
        assert _extract_image_url(calls) == "https://example.com/pre.png"

    def test_returns_first_image_tool(self):
        calls = [
            {"name": "generate_image", "image_url": "https://example.com/first.png"},
            {"name": "generate_image", "image_url": "https://example.com/second.png"},
        ]
        assert _extract_image_url(calls) == "https://example.com/first.png"


# ---------------------------------------------------------------------------
# TestExtractImageUrls
# ---------------------------------------------------------------------------

class TestExtractImageUrls:
    """Tests for _extract_image_urls()."""

    def test_extracts_single_url(self):
        result_json = json.dumps({"image_url": "https://example.com/a.png"})
        calls = [{"name": "generate_image", "result": result_json}]
        urls = _extract_image_urls(calls)
        assert urls == ["https://example.com/a.png"]

    def test_extracts_multiple_urls_from_image_urls_array(self):
        result_json = json.dumps({
            "image_urls": [
                "https://example.com/opt1.png",
                "https://example.com/opt2.png",
            ]
        })
        calls = [{"name": "generate_image", "result": result_json}]
        urls = _extract_image_urls(calls)
        assert len(urls) == 2
        assert "https://example.com/opt1.png" in urls
        assert "https://example.com/opt2.png" in urls

    def test_prefers_image_urls_array_over_single(self):
        """When both image_urls and image_url exist, image_urls is used."""
        result_json = json.dumps({
            "image_urls": ["https://example.com/a.png"],
            "image_url": "https://example.com/b.png",
        })
        calls = [{"name": "generate_image", "result": result_json}]
        urls = _extract_image_urls(calls)
        assert urls == ["https://example.com/a.png"]

    def test_multiple_tool_calls(self):
        result1 = json.dumps({"image_url": "https://example.com/1.png"})
        result2 = json.dumps({"image_url": "https://example.com/2.png"})
        calls = [
            {"name": "generate_image", "result": result1},
            {"name": "img2img", "result": result2},
        ]
        urls = _extract_image_urls(calls)
        assert len(urls) == 2

    def test_falls_back_to_pre_extracted_url(self):
        calls = [{"name": "generate_image", "result": "bad json", "image_url": "https://example.com/fb.png"}]
        urls = _extract_image_urls(calls)
        assert urls == ["https://example.com/fb.png"]

    def test_empty_list_returns_empty(self):
        assert _extract_image_urls([]) == []

    def test_non_image_tools_ignored(self):
        calls = [
            {"name": "read_brand_guidelines", "result": "ok"},
            {"name": "think", "result": "thinking..."},
        ]
        assert _extract_image_urls(calls) == []


# ---------------------------------------------------------------------------
# TestToolDescription
# ---------------------------------------------------------------------------

class TestToolDescription:
    """Tests for _tool_description()."""

    def test_known_tool_read_brand_guidelines(self):
        desc = _tool_description("read_brand_guidelines", {})
        assert "brand guidelines" in desc.lower()

    def test_known_tool_generate_image(self):
        desc = _tool_description("generate_image", {})
        assert "image" in desc.lower()

    def test_known_tool_check_figma_design(self):
        desc = _tool_description("check_figma_design", {"action": "colors"})
        assert "Figma" in desc
        assert "colors" in desc

    def test_known_tool_img2img(self):
        desc = _tool_description("img2img", {"reference_image_path": "logo.png"})
        assert "logo.png" in desc

    def test_known_tool_finish(self):
        desc = _tool_description("finish", {})
        assert "final" in desc.lower() or "draft" in desc.lower() or "submit" in desc.lower()

    def test_known_tool_think(self):
        desc = _tool_description("think", {})
        assert "reason" in desc.lower()

    def test_known_tool_execute_openclaw_script(self):
        desc = _tool_description("execute_openclaw_script", {"script_name": "deploy.py"})
        assert "deploy.py" in desc

    def test_known_tool_read_feedback_history(self):
        desc = _tool_description("read_feedback_history", {})
        assert "feedback" in desc.lower()

    def test_known_tool_log_resource_usage(self):
        desc = _tool_description("log_resource_usage", {})
        assert "resource" in desc.lower() or "log" in desc.lower()

    def test_unknown_tool_fallback(self):
        desc = _tool_description("some_unknown_tool", {})
        assert "some_unknown_tool" in desc

    def test_unknown_tool_format(self):
        desc = _tool_description("mystery_tool", {"arg": "val"})
        assert "mystery_tool" in desc
        assert "Executing" in desc or "executing" in desc.lower()
