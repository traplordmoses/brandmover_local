"""Tests for bulk photo upload auto-ingest and smart PDF handling."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.handlers import _merge_extracted, _process_bulk_upload, _delayed_bulk_process, handle_photo, handle_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_update(user_id=123, has_photo=True, caption=""):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.message.chat_id = 456
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    update.message.caption = caption or None
    if has_photo:
        mock_file = AsyncMock()
        mock_file.download_to_drive = AsyncMock()
        photo_obj = MagicMock()
        photo_obj.get_file = AsyncMock(return_value=mock_file)
        update.message.photo = [photo_obj]
        update.message.document = None
    else:
        update.message.photo = []
        update.message.document = None
    return update


def _mock_context(user_data=None):
    ctx = MagicMock()
    ctx.user_data = user_data or {}
    ctx.bot = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# _merge_extracted
# ---------------------------------------------------------------------------

class TestMergeExtracted:
    def test_deduplicates_colors_by_hex(self):
        results = [
            {"colors": [{"hex": "#ff0000", "name": "Red", "role": "primary"}], "fonts": [], "style_keywords": [], "logo_description": ""},
            {"colors": [{"hex": "#ff0000", "name": "Red", "role": "primary"}, {"hex": "#00ff00", "name": "Green", "role": "accent"}], "fonts": [], "style_keywords": [], "logo_description": ""},
        ]
        merged = _merge_extracted(results)
        assert len(merged["colors"]) == 2

    def test_deduplicates_fonts_by_family(self):
        results = [
            {"colors": [], "fonts": [{"family": "Inter", "weight": "Bold", "use": "display"}], "style_keywords": [], "logo_description": ""},
            {"colors": [], "fonts": [{"family": "Inter", "weight": "Regular", "use": "body"}], "style_keywords": [], "logo_description": ""},
        ]
        merged = _merge_extracted(results)
        assert len(merged["fonts"]) == 1

    def test_merges_style_keywords(self):
        results = [
            {"colors": [], "fonts": [], "style_keywords": ["minimal", "clean"], "logo_description": ""},
            {"colors": [], "fonts": [], "style_keywords": ["clean", "modern"], "logo_description": ""},
        ]
        merged = _merge_extracted(results)
        assert set(merged["style_keywords"]) == {"minimal", "clean", "modern"}

    def test_joins_logo_descriptions(self):
        results = [
            {"colors": [], "fonts": [], "style_keywords": [], "logo_description": "Blue circle logo"},
            {"colors": [], "fonts": [], "style_keywords": [], "logo_description": "Text wordmark"},
        ]
        merged = _merge_extracted(results)
        assert "Blue circle logo" in merged["logo_description"]
        assert "Text wordmark" in merged["logo_description"]

    def test_empty_results(self):
        merged = _merge_extracted([])
        assert merged["colors"] == []
        assert merged["fonts"] == []
        assert merged["style_keywords"] == []
        assert merged["logo_description"] == ""

    def test_skips_empty_hex_and_family(self):
        results = [
            {"colors": [{"hex": "", "name": "?", "role": "?"}], "fonts": [{"family": "", "weight": "", "use": ""}], "style_keywords": [], "logo_description": ""},
        ]
        merged = _merge_extracted(results)
        assert len(merged["colors"]) == 0
        assert len(merged["fonts"]) == 0


# ---------------------------------------------------------------------------
# Bulk upload batching in handle_photo
# ---------------------------------------------------------------------------

class TestBulkUploadBatching:
    def test_no_caption_batches_instead_of_prompting(self):
        """Photos without captions should batch, not immediately prompt."""
        async def _run():
            with patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.onboarding") as mock_onboard, \
                 patch("bot.handlers.state"), \
                 patch("bot.handlers._PILImage") as mock_pil, \
                 patch("bot.handlers._bulk_upload_tasks", {}):
                mock_onboard.get_session.return_value = None
                mock_img = MagicMock()
                mock_pil.open.return_value.convert.return_value = mock_img

                update = _mock_update(caption="")
                ctx = _mock_context()
                await handle_photo(update, ctx)

                # Should NOT have sent the old "what should I do" prompt
                for call in update.message.reply_text.call_args_list:
                    assert "reference / mascot" not in str(call)

                # Should have added to batch
                assert len(ctx.user_data["_bulk_uploads"]) == 1

        asyncio.run(_run())

    def test_multiple_photos_accumulate_in_batch(self):
        """Multiple rapid uploads should accumulate in the batch list."""
        async def _run():
            with patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.onboarding") as mock_onboard, \
                 patch("bot.handlers.state"), \
                 patch("bot.handlers._PILImage") as mock_pil, \
                 patch("bot.handlers._bulk_upload_tasks", {}):
                mock_onboard.get_session.return_value = None
                mock_img = MagicMock()
                mock_pil.open.return_value.convert.return_value = mock_img

                ctx = _mock_context()
                for _ in range(5):
                    update = _mock_update(caption="")
                    await handle_photo(update, ctx)

                assert len(ctx.user_data["_bulk_uploads"]) == 5

        asyncio.run(_run())

    def test_caption_bypasses_batching(self):
        """Photos with captions should NOT go through batching."""
        async def _run():
            with patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.onboarding") as mock_onboard, \
                 patch("bot.handlers.state") as mock_state, \
                 patch("bot.handlers._PILImage") as mock_pil, \
                 patch("bot.handlers._rate_limited", return_value=False), \
                 patch("bot.handlers._handle_pipeline_mode") as mock_pipeline, \
                 patch("bot.handlers.settings") as mock_settings:
                mock_onboard.get_session.return_value = None
                mock_img = MagicMock()
                mock_pil.open.return_value.convert.return_value = mock_img
                mock_state.has_pending.return_value = False
                mock_state.get_style_profiles.return_value = {}
                mock_settings.AGENT_MODE = "pipeline"
                mock_pipeline.return_value = None

                update = _mock_update(caption="write a post about this")
                ctx = _mock_context()
                await handle_photo(update, ctx)

                # Should NOT have batched
                assert "_bulk_uploads" not in ctx.user_data or len(ctx.user_data.get("_bulk_uploads", [])) == 0
                # Should have gone through pipeline
                mock_pipeline.assert_called_once()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# _process_bulk_upload
# ---------------------------------------------------------------------------

class TestProcessBulkUpload:
    def test_single_image_shows_prompt(self):
        """Single image batch should show the original prompt."""
        async def _run():
            ctx = MagicMock()
            ctx.user_data = {"_bulk_uploads": ["/tmp/img1.jpg"]}
            ctx.bot.send_message = AsyncMock()

            await _process_bulk_upload(ctx, user_id=123, chat_id=456)

            ctx.bot.send_message.assert_called_once()
            msg = ctx.bot.send_message.call_args[0][1]
            assert "reference / mascot" in msg

        asyncio.run(_run())

    def test_multiple_images_auto_ingest(self):
        """Multiple images should trigger auto-ingest."""
        async def _run():
            ctx = MagicMock()
            ctx.user_data = {
                "_bulk_uploads": ["/tmp/img1.jpg", "/tmp/img2.jpg", "/tmp/img3.jpg"],
            }
            ctx.bot.send_message = AsyncMock()
            ctx.bot.send_chat_action = AsyncMock()

            mock_extracted = {
                "colors": [{"hex": "#ff0000", "name": "Red", "role": "primary"}],
                "fonts": [],
                "style_keywords": ["bold"],
                "logo_description": "A logo",
            }

            with patch("agent.ingest.extract_brand_from_image", AsyncMock(return_value=mock_extracted)), \
                 patch("agent.ingest.diff_against_guidelines", AsyncMock(return_value="All good")), \
                 patch("shutil.copy2"), \
                 patch("bot.handlers.Path") as mock_path:
                mock_path.return_value.mkdir = MagicMock()
                mock_refs = MagicMock()
                mock_refs.mkdir = MagicMock()
                mock_path.return_value.__truediv__ = MagicMock(return_value=mock_refs)
                mock_refs.__truediv__ = MagicMock(return_value=Path("/tmp/fake"))

                await _process_bulk_upload(ctx, user_id=123, chat_id=456)

            # Should have sent "received 3 images" message
            first_msg = ctx.bot.send_message.call_args_list[0][0][1]
            assert "3 images" in first_msg

            # Should have stored merged extraction for /apply
            assert "last_ingest_extracted" in ctx.user_data

        asyncio.run(_run())

    def test_empty_batch_does_nothing(self):
        """Empty batch should return silently."""
        async def _run():
            ctx = MagicMock()
            ctx.user_data = {"_bulk_uploads": []}
            ctx.bot.send_message = AsyncMock()

            await _process_bulk_upload(ctx, user_id=123, chat_id=456)
            ctx.bot.send_message.assert_not_called()

        asyncio.run(_run())

    def test_batch_cleared_after_processing(self):
        """Batch should be removed from user_data after processing."""
        async def _run():
            ctx = MagicMock()
            ctx.user_data = {"_bulk_uploads": ["/tmp/img1.jpg"]}
            ctx.bot.send_message = AsyncMock()

            await _process_bulk_upload(ctx, user_id=123, chat_id=456)
            assert "_bulk_uploads" not in ctx.user_data

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Smart PDF handling
# ---------------------------------------------------------------------------

class TestSmartPdfHandling:
    def test_pdf_without_setup_saves_to_references(self):
        """PDF sent without /setup should be saved to brand/references/."""
        import tempfile, os

        async def _run():
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create brand folder structure
                brand_dir = os.path.join(tmpdir, "brand")
                os.makedirs(brand_dir)
                # Create existing guidelines.md
                guidelines_path = os.path.join(brand_dir, "guidelines.md")
                with open(guidelines_path, "w") as f:
                    f.write("# Brand\n")

                with patch("bot.handlers._authorized", return_value=True), \
                     patch("bot.handlers.onboarding") as mock_onboard, \
                     patch("bot.handlers.settings") as mock_settings:
                    mock_onboard.get_session.return_value = None
                    mock_settings.BRAND_FOLDER = brand_dir

                    update = MagicMock()
                    update.effective_user.id = 123
                    update.message.document = MagicMock()
                    update.message.document.file_name = "brand_guide.pdf"
                    update.message.document.mime_type = "application/pdf"
                    mock_tg_file = AsyncMock()
                    update.message.document.get_file = AsyncMock(return_value=mock_tg_file)
                    update.message.chat.send_action = AsyncMock()
                    update.message.reply_text = AsyncMock()

                    ctx = _mock_context()
                    await handle_document(update, ctx)

                    # Should have downloaded the file
                    mock_tg_file.download_to_drive.assert_called_once()
                    # Should have told user it's saved
                    reply_calls = update.message.reply_text.call_args_list
                    assert any("saved" in str(c).lower() for c in reply_calls)
                    # References dir should exist
                    assert os.path.isdir(os.path.join(brand_dir, "references"))

        asyncio.run(_run())

    def test_non_pdf_document_rejected(self):
        """Non-PDF documents should get a helpful message."""
        async def _run():
            with patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.onboarding") as mock_onboard:
                mock_onboard.get_session.return_value = None

                update = MagicMock()
                update.effective_user.id = 123
                update.message.document.file_name = "notes.txt"
                update.message.document.mime_type = "text/plain"
                update.message.reply_text = AsyncMock()

                ctx = _mock_context()
                await handle_document(update, ctx)

                msg = update.message.reply_text.call_args[0][0]
                assert "pdf" in msg.lower() or "PDF" in msg

        asyncio.run(_run())
