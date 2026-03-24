"""
Media handlers — handle_photo, handle_voice, handle_document, bulk upload logic.
"""

__all__ = [
    "handle_photo",
    "handle_voice",
    "handle_document",
    "save_asset_command",
    "remake_command",
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

    # --- Brand asset analysis: photos with "add to brand" or "analyze" captions ---
    caption_lower = (update.message.caption or "").lower().strip()
    is_asset_intent = any(phrase in caption_lower for phrase in [
        "add to brand", "add to assets", "brand asset", "add this",
        "catalog this", "save to library", "analyze this",
    ])

    if _authorized(user_id) and (is_asset_intent or context.user_data.get("awaiting_asset_analysis")):
        context.user_data.pop("awaiting_asset_analysis", None)
        await update.message.reply_text("analyzing image for brand library...")

        try:
            from agent.asset_ingest import analyze_for_library

            analysis = await analyze_for_library(tmp_path)

            # Format the analysis as a nice Telegram message
            colors_str = ", ".join(
                f'{c["name"]} ({c["hex"]})' for c in analysis.get("dominant_colors", [])[:4]
            )
            captions_str = "\n".join(
                f'  \u2022 {c}' for c in analysis.get("suggested_captions", [])
            )
            tags_str = ", ".join(analysis.get("recommended_tags", [])[:8])

            msg = (
                f"<b>Asset Analysis</b>\n\n"
                f"<b>Category:</b> {_esc(analysis.get('category', 'unknown'))}\n"
                f"<b>Colors:</b> {_esc(colors_str)}\n"
                f"<b>Style:</b> {_esc(', '.join(analysis.get('style_keywords', [])[:5]))}\n"
                f"<b>Brand fit:</b> {_esc(analysis.get('brand_alignment', 'unknown'))}\n"
                f"<b>Notes:</b> {_esc(analysis.get('brand_alignment_notes', ''))}\n\n"
                f"<b>Suggested captions:</b>\n{_esc(captions_str)}\n\n"
                f"<b>Tags:</b> {_esc(tags_str)}\n\n"
                f"Reply /save_asset to add to the brand library\n"
                f"Reply /save_asset announcement (or meme, community, etc.) to set content type"
            )

            # Store analysis for /save_asset command
            context.user_data["_pending_asset_analysis"] = analysis
            context.user_data["_pending_asset_path"] = tmp_path

            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            logger.error("Asset analysis failed: %s", e)
            await update.message.reply_text(
                f"Asset analysis failed: {_esc(str(e))}", parse_mode="HTML"
            )
        return

    # --- Competitor analysis: photos with competitor-related captions ---
    _COMPETITOR_KEYWORDS = ("analyze", "competitor", "compare", "swipe", "breakdown")
    is_competitor_intent = any(kw in caption_lower for kw in _COMPETITOR_KEYWORDS)

    if _authorized(user_id) and is_competitor_intent:
        await update.message.reply_text("analyzing competitor content...")
        await update.message.chat.send_action("typing")

        try:
            from agent._client import get_anthropic as _get_anthropic
            from agent import guidelines as _guidelines
            import base64 as _b64

            # Read the image as base64
            img_bytes = await _aio.to_thread(Path(tmp_path).read_bytes)
            img_b64 = _b64.b64encode(img_bytes).decode("utf-8")

            # Load brand guidelines for comparison
            brand_ctx = await _aio.to_thread(_guidelines.get_brand_context)
            brand_summary = brand_ctx[:2000] if brand_ctx else "(no brand guidelines loaded)"

            # Call Claude Vision to analyze the competitor content
            _vision_client = _get_anthropic()
            _vision_response = await _vision_client.messages.create(
                model=settings.HAIKU_MODEL,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze this competitor's content. Extract:\n"
                                "1. Colors (hex codes and names)\n"
                                "2. Typography style (font weight, serif/sans-serif, size hierarchy)\n"
                                "3. Composition (layout, spacing, alignment)\n"
                                "4. Tone (formal/casual/playful/aggressive)\n"
                                "5. Messaging style (short/long copy, CTAs, hooks)\n\n"
                                "Then compare against this brand's guidelines:\n"
                                f"{brand_summary}\n\n"
                                "Return a structured comparison in this format:\n"
                                "THEIR BRAND: [summary]\n"
                                "YOUR BRAND: [summary]\n"
                                "KEY DIFFERENCES: [bullet points]\n"
                                "WAYS TO STAND OUT: [3 actionable suggestions]"
                            ),
                        },
                    ],
                }],
            )

            analysis_text = ""
            for block in _vision_response.content:
                if hasattr(block, "text"):
                    analysis_text += block.text

            # Save to state/competitor_analyses.json (cap at 50)
            analyses_path = Path(settings.STATE_FOLDER) / "competitor_analyses.json"
            analyses_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "timestamp": time.time(),
                "caption": caption_lower,
                "analysis": analysis_text,
            }

            existing = []
            if analyses_path.exists():
                try:
                    existing = json.loads(await _aio.to_thread(analyses_path.read_text, "utf-8"))
                except (json.JSONDecodeError, OSError):
                    existing = []

            existing.append(entry)
            if len(existing) > 50:
                existing = existing[-50:]
            await _aio.to_thread(
                analyses_path.write_text,
                json.dumps(existing, indent=2),
                "utf-8",
            )

            # Format and send
            if len(analysis_text) > 3800:
                analysis_text = analysis_text[:3800] + "\n..."
            await update.message.reply_text(
                f"<b>Competitor Analysis</b>\n\n{_esc(analysis_text)}",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("Competitor analysis failed: %s", e)
            await update.message.reply_text(
                f"Competitor analysis failed: {_esc(str(e))}",
                parse_mode="HTML",
            )
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

            from bot.handlers.generation import _handle_agent_mode
            await _handle_agent_mode(update, request, user_id=user_id)
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

        from bot.handlers.generation import _handle_agent_mode
        await _handle_agent_mode(update, caption, user_id=user_id)
    else:
        from bot.handlers.generation import _handle_agent_mode
        await _handle_agent_mode(update, "[User sent a photo]", user_id=user_id)


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

        from bot.handlers.generation import _handle_agent_mode
        await _handle_agent_mode(update, text, user_id=user_id)

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
        # Check for video style analysis intent
        caption = (update.message.caption or "").strip().lower()
        if caption and _is_video_analysis_intent(caption):
            await _handle_video_style_analysis(update, context, document)
            return

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
                from bot.handlers.generation import _handle_agent_mode
                await _handle_agent_mode(update, synthetic_text, user_id=user_id)
        except Exception as e:
            logger.error("Video upload failed: %s", e)
            await update.message.reply_text("Video upload failed. Check logs.")
        return

    is_pdf = file_name.lower().endswith(".pdf")

    # Accept any file type — save non-PDF files to brand references
    if not is_pdf:
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
            refs_dir = Path(settings.BRAND_FOLDER) / "references"
            refs_dir.mkdir(parents=True, exist_ok=True)
            save_path = refs_dir / safe_name
            tg_file = await document.get_file()
            await tg_file.download_to_drive(str(save_path))

            caption = update.message.caption or ""
            user_id = update.effective_user.id

            await update.message.reply_text(
                f"File <b>{_esc(safe_name)}</b> saved to brand references.\n"
                f"Path: <code>{_esc(str(save_path))}</code>",
                parse_mode="HTML",
            )

            # If there's a caption, route it through the agent with file context
            if caption:
                synthetic_text = f"[File uploaded: {save_path}]\n\n{caption}"
                transcript.log_user_message(user_id, synthetic_text)
                from bot.handlers.generation import _handle_agent_mode
                await _handle_agent_mode(update, synthetic_text, user_id=user_id)
        except Exception as e:
            logger.error("File upload failed: %s", e)
            await update.message.reply_text("File upload failed. Check logs.")
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


# ---------------------------------------------------------------------------
# /save_asset command
# ---------------------------------------------------------------------------

async def save_asset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Save a previously analyzed image to the brand asset library."""
    user_id = update.effective_user.id
    if not _authorized(user_id):
        return

    analysis = context.user_data.get("_pending_asset_analysis")
    asset_path = context.user_data.get("_pending_asset_path")

    if not analysis or not asset_path:
        await update.message.reply_text(
            "no pending asset to save. upload an image with 'add to brand' caption first."
        )
        return

    # Parse optional content type from command args
    args = (update.message.text or "").split(maxsplit=1)
    content_type = args[1].strip() if len(args) > 1 else None

    try:
        from agent.asset_ingest import add_to_library

        entry = await add_to_library(asset_path, analysis, content_type=content_type)

        # Clean up
        context.user_data.pop("_pending_asset_analysis", None)
        context.user_data.pop("_pending_asset_path", None)

        await update.message.reply_text(
            f"saved to brand library.\n\n"
            f"<b>ID:</b> <code>{entry.get('id', '?')}</code>\n"
            f"<b>Type:</b> {_esc(entry.get('content_type', 'general'))}\n"
            f"<b>Tags:</b> {_esc(', '.join(entry.get('tags', [])))}",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("save_asset failed: %s", e)
        await update.message.reply_text(
            f"failed to save asset: {_esc(str(e))}", parse_mode="HTML"
        )


# ---------------------------------------------------------------------------
# Video style reverse-engineering
# ---------------------------------------------------------------------------

_VIDEO_ANALYSIS_TRIGGERS = [
    "break this down",
    "break it down",
    "analyze style",
    "analyse style",
    "reverse engineer",
    "style breakdown",
    "video breakdown",
    "analyze this",
    "analyse this",
    "what style is this",
    "deconstruct",
    "study this",
    "break down the style",
]


def _is_video_analysis_intent(caption: str) -> bool:
    """Check if the caption indicates a video style analysis request."""
    caption_lower = caption.lower().strip()
    return any(trigger in caption_lower for trigger in _VIDEO_ANALYSIS_TRIGGERS)


async def _handle_video_style_analysis(
    update: Update, context: ContextTypes.DEFAULT_TYPE, document,
) -> None:
    """Handle video upload with style analysis intent."""
    import os as _os

    file_name = document.file_name or "video.mp4"

    # Size guard
    _fsize = getattr(document, "file_size", None)
    if isinstance(_fsize, int) and _fsize > 100 * 1024 * 1024:
        await update.message.reply_text("Video too large (max 100 MB).")
        return

    await update.message.reply_text("downloading video for style analysis...")
    await update.message.chat.send_action("typing")

    # Download to temp
    tg_file = await document.get_file()
    safe_name = _os.path.basename(file_name)
    tmp_fd = tempfile.NamedTemporaryFile(
        suffix="_" + safe_name, prefix="brandmover_vr_", delete=False,
    )
    tmp_path = tmp_fd.name
    tmp_fd.close()

    try:
        await tg_file.download_to_drive(tmp_path)
    except Exception as e:
        logger.error("Video download failed: %s", e)
        await update.message.reply_text("couldn't download the video, try again.")
        return

    try:
        from agent import video_reverse

        # Step 1: Extract keyframes
        await update.message.reply_text("extracting keyframes...")
        await update.message.chat.send_action("typing")
        frames = await video_reverse.extract_keyframes(tmp_path)

        if not frames:
            await update.message.reply_text(
                "couldn't extract frames from this video. "
                "make sure ffmpeg is installed and the video isn't corrupted."
            )
            return

        await update.message.reply_text(
            f"extracted {len(frames)} frames, analyzing style with Claude Vision..."
        )
        await update.message.chat.send_action("typing")

        # Step 2: Analyze with Claude Vision
        analysis = await video_reverse.analyze_video_style(tmp_path, frames)

        # Step 3: Format and send breakdown
        breakdown = await video_reverse.format_breakdown(analysis)

        # Split long messages for Telegram's 4096 char limit
        if len(breakdown) > 4000:
            parts = _split_html_message(breakdown, 3900)
            for part in parts:
                await update.message.reply_text(part, parse_mode="HTML")
        else:
            await update.message.reply_text(breakdown, parse_mode="HTML")

        # Step 4: Store analysis for /remake
        context.user_data["_pending_video_analysis"] = analysis
        context.user_data["_pending_video_path"] = tmp_path

        await update.message.reply_text(
            "reply /remake to recreate this in your brand style."
        )

        # Clean up frame temp files
        for frame in frames:
            try:
                Path(frame).unlink(missing_ok=True)
            except OSError:
                pass

    except RuntimeError as e:
        # ffmpeg not installed
        await update.message.reply_text(str(e))
    except Exception as e:
        logger.error("Video style analysis failed: %s", e)
        await update.message.reply_text(
            f"video analysis failed: {_esc(str(e))}", parse_mode="HTML"
        )


def _split_html_message(text: str, max_len: int) -> list[str]:
    """Split a long HTML message into chunks at line boundaries."""
    lines = text.split("\n")
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > max_len and current:
            parts.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len

    if current:
        parts.append("\n".join(current))

    return parts


async def remake_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /remake — recreate an analyzed video in brand style."""
    user_id = update.effective_user.id
    if not _authorized(user_id):
        return

    analysis = context.user_data.get("_pending_video_analysis")
    if not analysis:
        await update.message.reply_text(
            "no video analysis to remake. upload a video with the caption "
            '"break this down" first.'
        )
        return

    await update.message.reply_text("remapping to your brand style...")
    await update.message.chat.send_action("typing")

    try:
        from agent import video_reverse

        # Remap to brand
        scene_json = await video_reverse.remap_to_brand(analysis)

        # Format preview
        config = scene_json.get("config", {})
        brand = config.get("brand", {})
        scenes = scene_json.get("scenes", [])
        duration = config.get("durationInSeconds", 0)

        preview_lines = [
            "<b>Brand-Remapped Video</b>",
            "",
            f"<b>Brand:</b> {_esc(brand.get('name', '?'))}",
            f"<b>Duration:</b> {duration:.1f}s",
            f"<b>Format:</b> {config.get('width', '?')}x{config.get('height', '?')}",
            f"<b>Colors:</b> {brand.get('primaryColor', '')} / {brand.get('accentColor', '')} / {brand.get('backgroundColor', '')}",
            f"<b>Font:</b> {_esc(brand.get('fontFamily', '?'))}",
            "",
            f"<b>Scene Plan ({len(scenes)} scenes):</b>",
        ]

        for i, scene in enumerate(scenes):
            stype = scene.get("type", "?")
            frames = scene.get("durationFrames", 0)
            sec = frames / 30.0 if frames else 0

            # Get a short description based on scene type
            desc = ""
            if stype == "title":
                desc = scene.get("headline", "")
            elif stype == "tagline":
                line_texts = [ln.get("text", "") for ln in scene.get("lines", [])]
                desc = " ".join(line_texts)
            elif stype == "text_only":
                desc = scene.get("text", "")
            elif stype == "stat":
                desc = f'{scene.get("value", "")}{scene.get("suffix", "")} {scene.get("label", "")}'
            elif stype == "feature_list":
                items = scene.get("items", [])
                desc = f'{len(items)} items'
            elif stype == "cta":
                desc = scene.get("buttonText", "")
            elif stype == "chat_demo":
                desc = f'{len(scene.get("messages", []))} messages'
            elif stype == "steps":
                desc = f'{len(scene.get("items", []))} steps'
            else:
                desc = scene.get("narration", "")[:40] if scene.get("narration") else ""

            if desc:
                desc = f' - {_esc(desc[:50])}'
            preview_lines.append(f"  {i + 1}. <b>{_esc(stype)}</b> ({sec:.1f}s){desc}")

        preview = "\n".join(preview_lines)

        await update.message.reply_text(preview, parse_mode="HTML")

        # Send the raw JSON as a document for use with /render
        json_str = json.dumps(scene_json, indent=2)
        json_bytes = json_str.encode("utf-8")

        import io as _io
        json_doc = _io.BytesIO(json_bytes)
        json_doc.name = "brand_video_scenes.json"

        await update.message.reply_document(
            document=json_doc,
            caption="scene JSON for /render",
        )

        # Store for potential /render usage
        context.user_data["_remake_scene_json"] = scene_json

        await update.message.reply_text(
            "reply /render to generate this video, or edit the JSON and send it back."
        )

    except Exception as e:
        logger.error("Video remake failed: %s", e)
        await update.message.reply_text(
            f"remake failed: {_esc(str(e))}", parse_mode="HTML"
        )
