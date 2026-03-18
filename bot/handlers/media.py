"""
Media handlers — handle_photo, handle_voice, handle_document, bulk upload logic.
"""

__all__ = [
    "handle_photo",
    "handle_voice",
    "handle_document",
]

import asyncio as _aio
import io
import json
import logging
import tempfile
import time
from pathlib import Path

from PIL import Image as _PILImage
from telegram import Update
from telegram.ext import ContextTypes

from agent import asset_library, compositor, compositor_config, guidelines, onboarding, state, transcript
from agent import compositor_config as _cc
from config import settings

from bot.handlers.core import (
    _DIRECT_PHOTO_PATTERNS,
    _RATE_LIMIT_SECONDS,
    _authorized,
    _bulk_upload_tasks,
    _can_operate,
    _esc,
    _is_direct_photo_intent,
    _is_template_from_ref_intent,
    _rate_limited,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bulk upload helpers
# ---------------------------------------------------------------------------

def _merge_extracted(results: list[dict]) -> dict:
    """Merge multiple ingest extraction results into one deduplicated set."""
    colors: dict[str, dict] = {}
    fonts: dict[str, dict] = {}
    style_keywords: set[str] = set()
    logo_descriptions: list[str] = []

    for r in results:
        for c in r.get("colors", []):
            h = c.get("hex", "")
            if h and h not in colors:
                colors[h] = c
        for f in r.get("fonts", []):
            family = f.get("family", "")
            if family and family not in fonts:
                fonts[family] = f
        for kw in r.get("style_keywords", []):
            style_keywords.add(kw)
        desc = r.get("logo_description", "")
        if desc and desc not in logo_descriptions:
            logo_descriptions.append(desc)

    return {
        "colors": list(colors.values()),
        "fonts": list(fonts.values()),
        "style_keywords": sorted(style_keywords),
        "logo_description": " | ".join(logo_descriptions) if logo_descriptions else "",
    }


async def _delayed_bulk_process(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: int,
) -> None:
    """Wait 3 seconds for more photos, then process the batch."""
    import asyncio
    await asyncio.sleep(3)
    await _process_bulk_upload(context, user_id, chat_id)


async def _process_bulk_upload(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: int,
) -> None:
    """Process batched photo uploads — auto-ingest if multiple, prompt if single."""
    batch = context.user_data.pop("_bulk_uploads", [])
    if not batch:
        return

    if len(batch) == 1:
        # Single image — ask what to do
        await context.bot.send_message(
            chat_id,
            "got it. what should i do with this? reply with:\n"
            "reference / mascot / style <name> / background",
        )
        return

    # Multiple images — auto-ingest all
    count = len(batch)
    await context.bot.send_message(
        chat_id,
        f"received {count} images \u2014 analyzing with AI vision...",
    )

    from agent import ingest
    import shutil

    all_extracted = []
    for i, path in enumerate(batch, 1):
        try:
            await context.bot.send_chat_action(chat_id, "typing")
            extracted = await ingest.extract_brand_from_image(path)
            all_extracted.append(extracted)
        except Exception as e:
            logger.warning("Ingest failed for image %d: %s", i, e)

    # Save images to brand/references/
    refs_dir = Path(settings.BRAND_FOLDER) / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(batch, 1):
        try:
            dest = refs_dir / f"ref_{time.time_ns()}_{i}.jpg"
            await _aio.to_thread(shutil.copy2, path, str(dest))
        except Exception as e:
            logger.warning("Failed to save reference image %d: %s", i, e)

    if not all_extracted:
        await context.bot.send_message(
            chat_id,
            f"saved {count} images to brand/references/ but couldn't analyze them.\n"
            "try /ingest to analyze one at a time.",
        )
        return

    merged = _merge_extracted(all_extracted)
    extracted_text = json.dumps(merged, indent=2)
    if len(extracted_text) > 3000:
        extracted_text = extracted_text[:3000] + "\n..."

    await context.bot.send_message(
        chat_id,
        f"<b>analyzed {count} images \u2014 extracted brand elements:</b>\n"
        f"<pre>{_esc(extracted_text)}</pre>",
        parse_mode="HTML",
    )

    # Diff against guidelines
    try:
        report = await ingest.diff_against_guidelines(merged)
        await context.bot.send_message(
            chat_id,
            f"<b>Compliance report:</b>\n{_esc(report)}",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.warning("Diff against guidelines failed: %s", e)

    # Store for /apply
    context.user_data["last_ingest_extracted"] = merged

    await context.bot.send_message(
        chat_id,
        f"saved {count} images to brand/references/\n"
        "reply /apply to update your guidelines with the extracted info.",
    )


# ---------------------------------------------------------------------------
# handle_photo
# ---------------------------------------------------------------------------

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo uploads and image documents as reference images."""
    user_id = update.effective_user.id
    if not _can_operate(user_id):
        return
    if not update.message:
        return

    # Determine source: photo[] or image document
    tg_file = None
    if update.message.photo:
        tg_file = await update.message.photo[-1].get_file()
    elif update.message.document:
        mime = update.message.document.mime_type or ""
        if not mime.startswith("image/"):
            return
        tg_file = await update.message.document.get_file()
    else:
        return

    tmp_fd = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="brandmover_upload_", delete=False)
    tmp_path = tmp_fd.name
    tmp_fd.close()

    try:
        await tg_file.download_to_drive(tmp_path)
    except Exception as e:
        logger.error("Failed to download uploaded image: %s", e)
        await update.message.reply_text(
            "couldn't download that image, try sending it as a photo instead of a file"
        )
        return

    # File size guard — skip PIL processing for excessively large files (>10MB)
    try:
        _file_size = Path(tmp_path).stat().st_size
        if _file_size > 10 * 1024 * 1024:
            logger.warning("Uploaded image too large (%d bytes), skipping PIL processing", _file_size)
            await update.message.reply_text("That image is too large (>10 MB). Please send a smaller one.")
            return
    except OSError:
        pass

    # Convert to JPEG (off event loop)
    try:
        def _convert_jpeg():
            img = _PILImage.open(tmp_path).convert("RGB")
            img.save(tmp_path, "JPEG", quality=95)
        await _aio.to_thread(_convert_jpeg)
    except Exception as e:
        logger.warning("Image conversion failed (using as-is): %s", e)

    # --- Onboarding upload intercept (admin only) ---
    ob_session = onboarding.get_session(user_id) if _authorized(user_id) else None
    if ob_session and ob_session.state == onboarding.OnboardingState.UPLOADS.value:
        # Check if this looks like a template
        from agent import template_memory as _tm
        try:
            is_tpl = await _tm.detect_if_template(tmp_path)
        except Exception:
            is_tpl = False
        if is_tpl:
            try:
                template = await _tm.register_template(tmp_path, name=f"Onboarding Template")
                regions_str = ", ".join(f"{r.type}" for r in template.regions)
                await update.message.reply_text(
                    f"That looks like a template! Registered as <code>{_esc(template.id)}</code> "
                    f"({template.aspect_ratio}, regions: {_esc(regions_str) or 'none'}).\n"
                    f"Send more assets, or /onboard_skip when done.",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("Onboarding template registration failed: %s", e)
                # Fall through to normal asset handling
                ob_session.uploaded_assets.append({"path": tmp_path, "type": "image"})
                await onboarding.async_save_session(ob_session)
                count = len(ob_session.uploaded_assets)
                await update.message.reply_text(
                    f"Asset {count} received. Send more, or /onboard_skip when done.",
                )
            return

        ob_session.uploaded_assets.append({"path": tmp_path, "type": "image"})
        await onboarding.async_save_session(ob_session)
        count = len(ob_session.uploaded_assets)
        await update.message.reply_text(
            f"Asset {count} received. Send more, or /onboard_skip when done.",
        )
        return

    # --- Template upload intercept (admin only) ---
    user_data = context.user_data if context else {}
    if _authorized(user_id) and user_data.get("awaiting_template"):
        from bot.handlers.admin import _handle_template_upload
        await _handle_template_upload(update, context, tmp_path)
        return

    # --- Template from reference intercept (admin only) ---
    if _authorized(user_id) and user_data.get("awaiting_template_from_ref"):
        from bot.handlers.admin import _handle_template_from_reference
        state.clear_reference_image()
        await _handle_template_from_reference(update, context, tmp_path)
        return

    # --- /upload asset library intercept (admin only) ---
    if _authorized(user_id) and user_data.get("awaiting_asset_upload"):
        try:
            ct = asset_library._guess_content_type(Path(tmp_path))
            entry = asset_library.add(tmp_path, "uploaded", ct, tags=[ct, "uploaded"])
            upload_count = user_data.get("_asset_upload_count", 0) + 1
            user_data["_asset_upload_count"] = upload_count
            await update.message.reply_text(
                f"Added to library: <code>{entry.id}</code> ({ct})\n"
                f"{upload_count} asset(s) uploaded this session. Send more or /done when finished.",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Asset upload failed: %s", e)
            await update.message.reply_text(f"Failed to add asset: {_esc(str(e))}", parse_mode="HTML")
        return

    # --- Priority flag checks (admin only: logo > ingest > brand_check) ---
    if _authorized(user_id) and user_data.get("awaiting_logo_upload"):
        user_data["awaiting_logo_upload"] = False
        logo_dir = Path(settings.BRAND_FOLDER) / "assets"
        logo_dir.mkdir(parents=True, exist_ok=True)
        logo_dest = logo_dir / "logo.png"
        try:
            await _aio.to_thread(lambda: _PILImage.open(tmp_path).convert("RGBA").save(str(logo_dest), "PNG"))
            await update.message.reply_text(
                f"Brand logo saved to <code>{_esc(str(logo_dest))}</code>",
                parse_mode="HTML",
            )
            logger.info("Brand logo updated: %s", logo_dest)
        except Exception as e:
            logger.error("Failed to save logo: %s", e)
            await update.message.reply_text(f"Failed to save logo: {_esc(str(e))}", parse_mode="HTML")
        return

    if _authorized(user_id) and user_data.get("awaiting_ingest_image"):
        user_data["awaiting_ingest_image"] = False
        await update.message.chat.send_action("typing")
        await update.message.reply_text("Analyzing image for brand elements...")
        try:
            from agent import ingest
            extracted = await ingest.extract_brand_from_image(tmp_path)
            report = await ingest.diff_against_guidelines(extracted)
            # Send extracted info
            extracted_text = json.dumps(extracted, indent=2)
            if len(extracted_text) > 3000:
                extracted_text = extracted_text[:3000] + "\n..."
            await update.message.reply_text(
                f"<b>Extracted brand elements:</b>\n<pre>{_esc(extracted_text)}</pre>",
                parse_mode="HTML",
            )
            await update.message.reply_text(
                f"<b>Compliance report:</b>\n{_esc(report)}",
                parse_mode="HTML",
            )
            # Store extracted data for /apply
            context.user_data["last_ingest_extracted"] = extracted
            await update.message.reply_text(
                "Reply /apply to update guidelines with the extracted info.",
            )
        except Exception as e:
            logger.error("Brand ingestion failed: %s", e)
            await update.message.reply_text(f"Ingestion failed: {_esc(str(e))}", parse_mode="HTML")
        return

    if _authorized(user_id) and user_data.get("awaiting_brand_check"):
        user_data["awaiting_brand_check"] = False
        await update.message.chat.send_action("typing")
        await update.message.reply_text("Checking image against brand guidelines...")
        try:
            from agent import brand_check
            report = await brand_check.check_brand_compliance(tmp_path)
            formatted = brand_check.format_compliance_report(report)
            await update.message.reply_text(formatted, parse_mode="HTML")
        except Exception as e:
            logger.error("Brand check failed: %s", e)
            await update.message.reply_text(f"Brand check failed: {_esc(str(e))}", parse_mode="HTML")
        return

    # --- Normal flow: set as reference image ---
    # Clean up previous ref file to prevent temp file accumulation
    old_ref = state.get_reference_image()
    if old_ref and old_ref != tmp_path:
        try:
            Path(old_ref).unlink(missing_ok=True)
        except Exception:
            pass
    state.set_reference_image(tmp_path)
    logger.info("Reference image saved to state: %s", tmp_path)

    caption = (update.message.caption or "").strip()

    # Caption-based /template_upload — admin only
    if _authorized(user_id) and caption.lower().startswith("/template_upload"):
        template_name = caption[len("/template_upload"):].strip()
        if context:
            context.user_data["template_name"] = template_name
        state.clear_reference_image()  # Undo the reference save above
        from bot.handlers.admin import _handle_template_upload
        await _handle_template_upload(update, context, tmp_path)
        return

    # Caption-based /template_from_reference — admin only
    if _authorized(user_id) and caption.lower().startswith("/template_from_reference"):
        template_name = caption[len("/template_from_reference"):].strip()
        if context:
            context.user_data["template_from_ref_name"] = template_name
        state.clear_reference_image()
        from bot.handlers.admin import _handle_template_from_reference
        await _handle_template_from_reference(update, context, tmp_path)
        return

    # Caption-based /brand_check — admin only
    if _authorized(user_id) and caption.lower().startswith("/brand_check"):
        await update.message.chat.send_action("typing")
        await update.message.reply_text("Checking image against brand guidelines...")
        try:
            from agent import brand_check
            report = await brand_check.check_brand_compliance(tmp_path)
            formatted = brand_check.format_compliance_report(report)
            await update.message.reply_text(formatted, parse_mode="HTML")
        except Exception as e:
            logger.error("Brand check failed: %s", e)
            await update.message.reply_text(f"Brand check failed: {_esc(str(e))}", parse_mode="HTML")
        return

    # Natural language template-from-reference intent detection — admin only
    if _authorized(user_id) and caption and _is_template_from_ref_intent(caption):
        state.clear_reference_image()
        if context:
            context.user_data["template_from_ref_name"] = ""
        from bot.handlers.admin import _handle_template_from_reference
        await _handle_template_from_reference(update, context, tmp_path)
        return

    # --- Direct photo mode: user wants to use their photo as-is in a template ---
    if caption and _is_direct_photo_intent(caption):
        from agent import template_memory as _tm
        cfg = _cc.get_config()
        memory = _tm.TemplateMemory()
        template = memory.get_template_for_content_type("announcement")
        if template and cfg.compositor_enabled:
            state.clear_reference_image()
            if context:
                context.user_data["direct_photo_path"] = tmp_path
            await update.message.reply_text("got it, composing with your photo directly...")

            if _rate_limited(update.effective_user.id):
                await update.message.reply_text(
                    f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
                )
                return

            # Strip the direct-photo keywords from caption to get a content hint
            content_hint = caption
            for p in _DIRECT_PHOTO_PATTERNS:
                content_hint = content_hint.lower().replace(p, "").strip()
            if not content_hint:
                content_hint = "create an announcement"
            request = f"{content_hint}\n\n[DIRECT PHOTO: {tmp_path}]\n[generate text only, do NOT call generate_image]"

            from bot.handlers.generation import _handle_unified, _handle_agent_mode, _handle_pipeline_mode
            if settings.UNIFIED_BRAIN_ENABLED:
                await _handle_unified(update, context, request, user_id=user_id)
                return

            # Legacy path: pending draft blocker
            if state.has_pending(user_id=user_id):
                await update.message.reply_text(
                    "You have a pending draft. /approve, /reject, or /cancel it first.",
                    parse_mode="HTML",
                )
                return

            if settings.AGENT_MODE == "agent":
                await _handle_agent_mode(update, request, user_id=user_id)
            else:
                await _handle_pipeline_mode(update, request, user_id=user_id)
            return

    # Check if caption matches a style profile name (admin only)
    if _authorized(user_id) and caption:
        style_name = caption
        if caption.lower().startswith("style "):
            style_name = caption[6:].strip()

        profiles = state.get_style_profiles()
        if style_name in profiles:
            count = state.add_profile_image(style_name, tmp_path)
            await update.message.reply_text(
                f"added to <b>{_esc(style_name)}</b> profile ({count} images total)",
                parse_mode="HTML",
            )
            return

    if caption:
        await update.message.reply_text("got it, generating with your image as reference...")

        if _rate_limited(user_id):
            await update.message.reply_text(
                f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
            )
            return

        from bot.handlers.generation import _handle_unified, _handle_agent_mode, _handle_pipeline_mode
        if settings.UNIFIED_BRAIN_ENABLED:
            await _handle_unified(update, context, caption, user_id=user_id)
            return

        # Legacy path: pending draft blocker
        if state.has_pending(user_id=user_id):
            await update.message.reply_text(
                "You have a pending draft. /approve, /reject, or /cancel it first.",
                parse_mode="HTML",
            )
            return

        if settings.AGENT_MODE == "agent":
            await _handle_agent_mode(update, caption, user_id=user_id)
        else:
            await _handle_pipeline_mode(update, caption, user_id=user_id)
    else:
        from bot.handlers.generation import _handle_unified
        if settings.UNIFIED_BRAIN_ENABLED:
            await _handle_unified(update, context, "[User sent a photo]", user_id=user_id)
            return

        # Legacy: batch uploads — collect images for 3 seconds, then auto-ingest if bulk
        import asyncio as _aio_mod

        batch = context.user_data.setdefault("_bulk_uploads", [])
        batch.append(tmp_path)
        chat_id = update.message.chat_id
        user_id = update.effective_user.id

        # Cancel existing batch timer and reschedule
        existing = _bulk_upload_tasks.get(user_id)
        if existing and not existing.done():
            existing.cancel()

        _bulk_upload_tasks[user_id] = _aio_mod.create_task(
            _delayed_bulk_process(context, user_id, chat_id)
        )


# ---------------------------------------------------------------------------
# handle_voice
# ---------------------------------------------------------------------------

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice/audio messages — transcribe and process as text."""
    user_id = update.effective_user.id
    if not _can_operate(user_id):
        return

    if not update.message:
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    await update.message.chat.send_action("typing")

    try:
        tg_file = await voice.get_file()
        tmp_fd = tempfile.NamedTemporaryFile(suffix=".ogg", prefix="brandmover_voice_", delete=False)
        tmp_path = tmp_fd.name
        tmp_fd.close()
        await tg_file.download_to_drive(tmp_path)

        # Transcribe
        from agent import voice_transcribe
        text = await voice_transcribe.transcribe(tmp_path)

        # Clean up
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

        if not text:
            await update.message.reply_text("Couldn't transcribe the audio. Please try again or type your request.")
            return

        await update.message.reply_text(
            f"<i>Transcribed:</i> {_esc(text)}",
            parse_mode="HTML",
        )

        # Process as a text message
        from bot.handlers.generation import handle_message as _handle_text
        # Simulate a text message by setting the transcribed text
        # We need to route through handle_message, so we call the generation flow directly
        transcript.log_user_message(user_id, f"[voice] {text}")

        from bot.handlers.generation import _handle_unified, _handle_agent_mode, _handle_pipeline_mode

        if settings.UNIFIED_BRAIN_ENABLED:
            from bot.handlers.generation import _fast_path
            handled = await _fast_path(update, context, text)
            if not handled:
                await _handle_unified(update, context, text, user_id=user_id)
            return

        if settings.INTENT_ROUTER_ENABLED:
            from bot.handlers.generation import _route_intent
            try:
                handled = await _route_intent(update, context, text)
                if handled:
                    return
            except Exception as e:
                logger.warning("Intent router error on voice: %s", e)

        if _rate_limited(user_id):
            await update.message.reply_text(f"Please wait {_RATE_LIMIT_SECONDS}s between requests.")
            return

        from agent import state as _state
        if _state.has_pending(user_id=user_id):
            await update.message.reply_text(
                "You have a pending draft. /approve, /reject, or /cancel it first.",
                parse_mode="HTML",
            )
            return

        if settings.AGENT_MODE == "agent":
            await _handle_agent_mode(update, text, user_id=user_id)
        else:
            await _handle_pipeline_mode(update, text, user_id=user_id)

    except Exception as e:
        logger.error("Voice handler error: %s", e)
        await update.message.reply_text(
            f"Voice processing failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# handle_document
# ---------------------------------------------------------------------------

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads — PDF brand bootstrap when in setup mode."""
    if not _authorized(update.effective_user.id):
        return

    # Onboarding intercept — accept documents as brand assets during upload phase
    ob_session = onboarding.get_session(update.effective_user.id)
    if ob_session and ob_session.state == onboarding.OnboardingState.UPLOADS.value:
        document = update.message.document
        if document:
            tg_file = await document.get_file()
            tmp_fd = tempfile.NamedTemporaryFile(suffix="_" + (document.file_name or "doc"), prefix="onboard_doc_", delete=False)
            tmp_path = tmp_fd.name
            tmp_fd.close()
            await tg_file.download_to_drive(tmp_path)
            ob_session.uploaded_assets.append({"path": tmp_path, "type": "document"})
            await onboarding.async_save_session(ob_session)
            count = len(ob_session.uploaded_assets)
            await update.message.reply_text(
                f"Document received ({count} assets total). Send more, or /onboard_skip when done.",
            )
        return

    document = update.message.document
    if not document:
        return

    file_name = document.file_name or ""

    # Font upload handling
    is_font = file_name.lower().endswith((".ttf", ".otf"))
    if is_font or context.user_data.get("awaiting_font_upload"):
        context.user_data.pop("awaiting_font_upload", None)
        if not file_name.lower().endswith((".ttf", ".otf")):
            await update.message.reply_text("Please send a .ttf or .otf font file.")
            return
        try:
            import os as _os
            safe_name = _os.path.basename(file_name)
            if not safe_name.lower().endswith((".ttf", ".otf")):
                await update.message.reply_text("Invalid font filename.")
                return
            _fsize = getattr(document, "file_size", None)
            if isinstance(_fsize, int) and _fsize > 10 * 1024 * 1024:
                await update.message.reply_text("Font file too large (max 10 MB).")
                return
            fonts_dir = Path(settings.BRAND_FOLDER) / "assets" / "fonts"
            fonts_dir.mkdir(parents=True, exist_ok=True)
            tg_file = await document.get_file()
            save_path = fonts_dir / safe_name
            await tg_file.download_to_drive(str(save_path))
            try:
                from agent.font_manager import clear_cache as _fm_clear
                _fm_clear()
            except ImportError:
                pass
            try:
                from agent.compositor import clear_font_cache
                clear_font_cache()
            except ImportError:
                pass
            await update.message.reply_text(
                f"Font <b>{_esc(file_name)}</b> saved to brand fonts.\n"
                "It's now available for templates and compositions.",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("Font upload failed: %s", e)
            await update.message.reply_text("Font upload failed. Check logs for details.")
        return

    # Video file handling — save as reference and pass to agent
    is_video = file_name.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv"))
    if is_video:
        try:
            _fsize = getattr(document, "file_size", None)
            if isinstance(_fsize, int) and _fsize > 100 * 1024 * 1024:
                await update.message.reply_text("Video too large (max 100 MB).")
                return
            refs_dir = Path(settings.BRAND_FOLDER) / "references"
            refs_dir.mkdir(parents=True, exist_ok=True)
            import os as _os
            safe_name = _os.path.basename(file_name)
            save_path = refs_dir / safe_name
            tg_file = await document.get_file()
            await tg_file.download_to_drive(str(save_path))
            # Also save to state/outputs for agent tool access
            outputs_dir = Path(settings.STATE_FOLDER) / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            agent_path = outputs_dir / safe_name
            await _aio.to_thread(shutil.copy2, str(save_path), str(agent_path))

            caption = update.message.caption or ""
            user_id = update.effective_user.id

            await update.message.reply_text(
                f"Video <b>{_esc(safe_name)}</b> received and saved.\n"
                f"Path: <code>{_esc(str(agent_path))}</code>\n\n"
                "I'll use this as a reference. Send me instructions on what to do with it.",
                parse_mode="HTML",
            )

            # If there's a caption, treat it as a message with the video context
            if caption:
                context.user_data["last_video_path"] = str(agent_path)
                synthetic_text = f"[Video reference uploaded: {agent_path}]\n\n{caption}"
                transcript.log_user_message(user_id, synthetic_text)
                from bot.handlers.generation import _handle_unified, _handle_agent_mode
                if settings.UNIFIED_BRAIN_ENABLED:
                    await _handle_unified(update, context, synthetic_text, user_id=user_id)
                else:
                    await _handle_agent_mode(update, synthetic_text, user_id=user_id)
        except Exception as e:
            logger.error("Video upload failed: %s", e)
            await update.message.reply_text("Video upload failed. Check logs.")
        return

    is_pdf = file_name.lower().endswith(".pdf")

    if not is_pdf:
        await update.message.reply_text(
            "I can accept PDFs, images, fonts (.ttf/.otf), and videos (.mp4/.mov). "
            "Send a file to add to your brand references."
        )
        return

    # Smart PDF handling — save to references, auto-extract if no guidelines exist
    if not context.user_data.get("awaiting_setup_pdf"):
        await update.message.chat.send_action("typing")
        try:
            import os as _os
            safe_name = _os.path.basename(file_name)
            if not safe_name:
                await update.message.reply_text("Invalid filename.")
                return
            _fsize = getattr(document, "file_size", None)
            if isinstance(_fsize, int) and _fsize > 50 * 1024 * 1024:
                await update.message.reply_text("File too large (max 50 MB).")
                return
            tg_file = await document.get_file()
            refs_dir = Path(settings.BRAND_FOLDER) / "references"
            refs_dir.mkdir(parents=True, exist_ok=True)
            ref_path = refs_dir / safe_name
            await tg_file.download_to_drive(str(ref_path))

            guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
            if not guidelines_path.exists():
                await update.message.reply_text(
                    f"saved <code>{_esc(file_name)}</code> \u2014 extracting brand guidelines...",
                    parse_mode="HTML",
                )
                guidelines_md = await guidelines.extract_brand_from_pdf(str(ref_path))
                if guidelines_md:
                    await _aio.to_thread(guidelines_path.write_text, guidelines_md, encoding="utf-8")
                    compositor_config.invalidate_cache()
                    compositor.clear_font_cache()
                    guidelines.invalidate_brand_context()
                    preview = guidelines_md[:1500]
                    if len(guidelines_md) > 1500:
                        preview += "\n\n[... truncated ...]"
                    await update.message.reply_text(
                        f"<b>Guidelines Generated</b> ({len(guidelines_md)} chars)\n\n"
                        f"<pre>{_esc(preview)}</pre>\n\n"
                        "you're all set! send me a content request to try it out.",
                        parse_mode="HTML",
                    )
                else:
                    await update.message.reply_text(
                        "saved PDF to references but couldn't extract guidelines. try /setup for manual setup."
                    )
            else:
                await update.message.reply_text(
                    f"saved <code>{_esc(file_name)}</code> to brand references.\n\n"
                    "you already have brand guidelines. use /setup to rebuild from this PDF.",
                    parse_mode="HTML",
                )
        except Exception as e:
            logger.error("PDF handling failed: %s", e)
            await update.message.reply_text(
                f"failed to process PDF: {_esc(str(e))}",
                parse_mode="HTML",
            )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Extracting text from PDF and generating guidelines...")

    try:
        # Download the PDF
        tg_file = await document.get_file()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            await tg_file.download_to_drive(tmp_path)

        # Extract and generate guidelines
        guidelines_md = await guidelines.extract_brand_from_pdf(tmp_path)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        if not guidelines_md:
            await update.message.reply_text("Failed to extract guidelines from the PDF. Try a different file.")
            return

        # Save to brand folder
        brand_path = Path(settings.BRAND_FOLDER)
        guidelines_path = brand_path / "guidelines.md"

        # Backup existing guidelines
        if guidelines_path.exists():
            backup_path = brand_path / "guidelines.md.bak"
            import shutil
            await _aio.to_thread(shutil.copy2, guidelines_path, backup_path)
            await update.message.reply_text(
                f"Backed up existing guidelines to <code>guidelines.md.bak</code>",
                parse_mode="HTML",
            )

        await _aio.to_thread(guidelines_path.write_text, guidelines_md, encoding="utf-8")
        compositor_config.invalidate_cache()
        compositor.clear_font_cache()
        guidelines.invalidate_brand_context()

        # Also save PDF to references
        refs_dir = brand_path / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        ref_path = refs_dir / file_name
        await tg_file.download_to_drive(str(ref_path))

        context.user_data["awaiting_setup_pdf"] = False

        # Show preview
        preview = guidelines_md[:1500]
        if len(guidelines_md) > 1500:
            preview += "\n\n[... truncated ...]"

        await update.message.reply_text(
            f"<b>Guidelines Generated</b> ({len(guidelines_md)} chars)\n\n"
            f"<pre>{_esc(preview)}</pre>\n\n"
            f"Saved to: <code>{guidelines_path}</code>\n"
            f"PDF saved to: <code>{ref_path}</code>\n\n"
            f"You're all set! Send me a content request to try it out.",
            parse_mode="HTML",
        )
        logger.info("Brand setup complete: %d chars from %s", len(guidelines_md), file_name)

    except Exception as e:
        logger.error("Setup PDF extraction failed: %s", e)
        context.user_data["awaiting_setup_pdf"] = False
        await update.message.reply_text(
            f"Setup failed: {_esc(str(e))}\n\nTry again with /setup.",
            parse_mode="HTML",
        )
