"""
Content generation handlers — handle_message (NL router), _route_intent,
_handle_unified, _handle_agent_mode, _handle_pipeline_mode, _fast_path, generate_command,
generate_callback.
"""

__all__ = [
    "handle_message",
    "generate_command",
    "generate_callback",
    "score_command",
    "approval_rate_command",
]

import asyncio as _aio
import io
import logging
import re
import tempfile
from pathlib import Path

from PIL import Image as _PILImage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from agent import asset_gen, auto_state, brain, chat, compositor_config, conversation_context, engine, feedback, generation_history, guidelines, hooks, image_gen, intent_router, onboarding, schedule_queue, state, transcript, unified_brain
from agent import compositor_config as _cc
from config import settings

from bot.handlers.core import (
    _ADMIN_ONLY_TOOLS,
    _RATE_LIMIT_SECONDS,
    _STEP_ICONS,
    _TOOL_ICONS,
    _authorized,
    _can_operate,
    _esc,
    _extract_commentary,
    _get_approve_lock,
    _is_template_region_update,
    _maybe_compose,
    _prepare_photo,
    _rate_limited,
    _truncate_reasoning,
)

from bot.handlers.draft import (
    _do_approve,
    _do_post,
    _do_reject,
    _send_draft,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified brain — fast path + unified handler
# ---------------------------------------------------------------------------

# Deterministic short-message lookup for draft actions + utility commands.
# Only used in unified brain mode. Maps normalized lowercase -> action.
_UNIFIED_FAST_PATH: dict[str, str] = {
    # Approve (save, do NOT post)
    "yes": "approve", "yep": "approve", "yeah": "approve", "y": "approve",
    "ok": "approve", "okay": "approve", "sure": "approve", "looks good": "approve",
    "lgtm": "approve", "approve": "approve", "approved": "approve",
    "i approve": "approve", "love it": "approve", "perfect": "approve",
    "thats good": "approve", "that's good": "approve",
    # Post (publish to X — requires approved draft)
    "post it": "post", "send it": "post", "ship it": "post",
    "publish": "post", "go": "post", "do it": "post",
    "post": "post", "send": "post",
    # Reroll
    "try again": "reroll", "again": "reroll", "another": "reroll",
    "another one": "reroll", "reroll": "reroll", "redo": "reroll",
    "regenerate": "reroll", "new one": "reroll",
}

# Suffix keywords that indicate approve intent even if the full message doesn't match.
# Catches "this is amazing i approve", "love it approve", etc.
_APPROVE_SUFFIXES = ("approve", "approved", "i approve", "looks good", "lgtm")
_POST_SUFFIXES = ("post it", "send it", "ship it", "publish it")

# Fast path actions that require a pending draft
_FAST_PATH_DRAFT_ACTIONS = {"approve", "reroll"}
# Fast path actions that target approved drafts
_FAST_PATH_APPROVED_ACTIONS = {"post"}


async def _fast_path(update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
    """Handle deterministic short-message actions for the unified brain path.

    Returns True if the message was handled, False to pass to unified brain.
    """
    user_id = update.effective_user.id
    normalized = message.lower().strip()

    action = _UNIFIED_FAST_PATH.get(normalized)

    # Fuzzy suffix match for longer messages like "this is amazing i approve"
    if not action:
        if any(normalized.endswith(s) for s in _APPROVE_SUFFIXES):
            action = "approve"
        elif any(normalized.endswith(s) for s in _POST_SUFFIXES):
            action = "post"

    if not action:
        return False

    has_draft = await state.async_has_pending(user_id=user_id)
    has_approved = await _aio.to_thread(state.has_approved, user_id=user_id)

    # Post action — guarded by per-user lock to prevent double-post race
    if action == "post":
        async with _get_approve_lock(user_id):
            if has_approved:
                await _do_post(update, context, source="unified_fast")
                return True
            elif has_draft:
                # One-shot shortcut: approve + post in one step
                await _do_approve(update, context, source="unified_fast")
                await _do_post(update, context, source="unified_fast_auto")
                return True
            else:
                # Nothing to post — pass to unified brain
                return False

    # Draft-dependent actions without a draft -> pass to unified brain
    if action in _FAST_PATH_DRAFT_ACTIONS and not has_draft:
        return False

    if action == "approve":
        async with _get_approve_lock(user_id):
            await _do_approve(update, context, source="unified_fast")
            return True

    if action == "reroll":
        if _rate_limited(user_id):
            await update.message.reply_text(f"Please wait {_RATE_LIMIT_SECONDS}s between requests.")
            return True
        pending = state.get_pending(user_id=user_id)
        if pending:
            original = pending.get("original_request", "")
            state.clear_pending(user_id=user_id)
            state.clear_draft_history(user_id=user_id)
            await update.message.reply_text("Regenerating...")
            if original:
                await _handle_unified(update, context, original, user_id=user_id)
            return True

    return False


async def _handle_unified(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    request: str,
    user_id: int | None = None,
) -> None:
    """Run the unified brain and handle the result."""
    await update.message.chat.send_action("typing")

    # Load conversation context and sync pending state
    ctx = conversation_context.get_context(user_id or update.effective_user.id)
    ctx.pending_draft_exists = state.has_pending(user_id=user_id)

    # Set user name if available
    if update.effective_user and update.effective_user.first_name and not ctx.user_name:
        ctx.user_name = update.effective_user.first_name

    # Inject reference image path if stored
    ref_path = state.get_reference_image()
    if ref_path and Path(ref_path).exists():
        request = f"{request}\n\n[REFERENCE IMAGE: {ref_path}]"

    # Status message for tool calls + live reasoning traces
    _status_msg = None
    _status_lines: list[str] = []
    _reasoning_line: str = ""  # Current reasoning line (replaced each turn)

    def _build_status_text() -> str:
        parts = list(_status_lines)
        if _reasoning_line:
            parts.append(f"<i>\U0001F4AD {_esc(_reasoning_line)}</i>")
        return "\n".join(parts)

    async def _update_status():
        nonlocal _status_msg
        text = _build_status_text()
        if not text:
            return
        if _status_msg is None:
            _status_msg = await update.message.reply_text(text, parse_mode="HTML")
        else:
            try:
                await _status_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass  # Telegram rejects edits if text unchanged

    async def on_tool_call(tool_name: str, description: str):
        nonlocal _reasoning_line
        _reasoning_line = ""
        icon = _TOOL_ICONS.get(tool_name, "\u26A1")
        _status_lines.append(f"{icon} {_esc(description)}")
        await _update_status()
        await update.message.chat.send_action("typing")

    async def on_reasoning(text: str):
        nonlocal _reasoning_line
        _reasoning_line = _truncate_reasoning(text)
        await _update_status()
        await update.message.chat.send_action("typing")

    try:
        result = await unified_brain.run(
            request=request,
            context=ctx,
            on_tool_call=on_tool_call,
            on_reasoning=on_reasoning,
        )

        # Delete the status message now that we're done
        if _status_msg:
            try:
                await _status_msg.delete()
            except Exception:
                pass

        # Check what the unified brain returned
        if result.draft:
            image_url = result.image_url
            image_urls = result.image_urls

            # Determine draft format
            draft_format = result.draft.get("format", "single")
            format_data = None
            if draft_format == "thread" and result.draft.get("thread_posts"):
                format_data = {"thread_posts": result.draft["thread_posts"]}

            # Save pending state
            state.save_pending(
                caption=result.draft["caption"],
                hashtags=result.draft.get("hashtags", []),
                image_url=image_url,
                alt_text=result.draft.get("alt_text", ""),
                image_prompt=result.draft.get("image_prompt", ""),
                original_request=request,
                image_urls=image_urls if image_urls and len(image_urls) > 1 else None,
                content_type=result.draft.get("content_type"),
                user_id=user_id,
                conversation_history=result.conversation_history,
                draft_format=draft_format,
                format_data=format_data,
            )

            await _send_draft(update, result.draft, image_url, resources=result.resources, image_urls=image_urls, user_id=user_id)

            # Fire hooks + transcript
            transcript.log_agent_response(
                user_id or 0, result.draft.get("caption", ""),
                turns=result.turns_used, tools=result.tool_calls_made,
            )
            await hooks.emit("draft:generated", {
                "draft": result.draft, "user_id": user_id,
                "turns": result.turns_used, "time": result.total_time,
            })
        else:
            text = result.response_text or "I processed your request but didn't generate a draft."
            max_len = 3900
            if len(text) > max_len:
                text = text[:max_len] + "..."
            await update.message.reply_text(
                _esc(text),
                parse_mode="HTML",
            )

    except Exception as e:
        logger.error("Unified brain error: %s", e)
        if _status_msg:
            try:
                await _status_msg.delete()
            except Exception:
                pass
        await update.message.reply_text(
            f"Something went wrong: {_esc(str(e))}\n\nPlease try again.",
            parse_mode="HTML",
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any plain text message — routes through intent router before generation."""
    user_id = update.effective_user.id
    if not _can_operate(user_id):
        return

    if not update.message or not update.message.text:
        return

    request = update.message.text.strip()
    if not request:
        return

    # Log to transcript
    transcript.log_user_message(user_id, request)

    # Template-from-reference adjustment intercept (admin only)
    if _authorized(user_id):
        tplref = (context.user_data or {}).get("tplref_pending") if context else None
        if isinstance(tplref, dict) and "design" in tplref:
            from bot.handlers.admin import _handle_tplref_adjustment
            await _handle_tplref_adjustment(update, context, request)
            return

        # Template region update intercept — user describing region positions after upload
        if _is_template_region_update(request, context):
            from bot.handlers.admin import _handle_template_region_update
            await _handle_template_region_update(update, context, request)
            return

    # Onboarding intercept — handle messages during onboarding flow (admin only)
    session = onboarding.get_session(user_id) if _authorized(user_id) else None
    if session and session.state not in (
        onboarding.OnboardingState.IDLE.value,
        onboarding.OnboardingState.COMPLETE.value,
    ):
        # DISCOVERY state uses Claude-driven async conversation
        if session.state == onboarding.OnboardingState.DISCOVERY.value:
            await update.message.chat.send_action("typing")
            session, response = await onboarding.advance_async(session, request)
        else:
            session, response = onboarding.advance(session, request)
        await onboarding.async_save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")

        # Handle async state transitions
        if session.state == onboarding.OnboardingState.AUDITING.value:
            from bot.handlers.admin import _run_onboarding_audit
            await _run_onboarding_audit(update, session)
        elif session.state == onboarding.OnboardingState.STRATEGY.value:
            from bot.handlers.admin import _run_onboarding_strategy
            await _run_onboarding_strategy(update, session)
        elif session.state == onboarding.OnboardingState.COMPLETE.value:
            summary = await onboarding.finalize_onboarding(session)
            await update.message.reply_text(summary, parse_mode="HTML")
        return

    # --- Unified brain path ---
    if settings.UNIFIED_BRAIN_ENABLED:
        handled = await _fast_path(update, context, request)
        if not handled:
            await _handle_unified(update, context, request, user_id=user_id)
        return

    # --- Legacy path: intent router + separate brains ---

    # Intent routing — classify message and dispatch if confident
    if settings.INTENT_ROUTER_ENABLED:
        try:
            handled = await _route_intent(update, context, request)
            if handled:
                return
        except Exception as e:
            logger.warning("Intent router error, falling through to generation: %s", e)

    # Fallback: generation path (rate limited, pending draft blocked)
    if _rate_limited(user_id):
        await update.message.reply_text(
            f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
        )
        return

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


async def _route_intent(update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
    """Classify intent and dispatch. Returns True if handled, False to fall through."""
    user_id = update.effective_user.id
    ctx = conversation_context.get_context(user_id)
    # Sync pending draft state from actual state file
    ctx.pending_draft_exists = state.has_pending(user_id=user_id)

    result = await intent_router.classify_intent(message, ctx)
    intent = result.intent
    confidence = result.confidence

    logger.info(
        "Intent: %s (%.2f) via %s for: %s",
        intent, confidence, result.routed_via, message[:60],
    )

    # Track classified intent
    try:
        recent = list(ctx.recent_intents) + [intent]
        conversation_context.update_context(user_id, recent_intents=recent)
    except Exception as e:
        logger.debug("Failed to track intent: %s", e)

    # High-confidence actions
    if intent == "approve" and confidence >= 0.8:
        await _do_approve(update, context, source="router")
        return True

    if intent == "reject" and confidence >= 0.8:
        if result.parameters.get("needs_feedback_prompt"):
            # Bare reject word (e.g. "no") — prompt for specific feedback
            fb = ""
        else:
            fb = result.parameters.get("feedback", message)
        await _do_reject(update, context, feedback_text=fb, source="router")
        return True

    if intent == "edit_request" and confidence >= 0.5:
        fb = result.parameters.get("feedback", message)
        if state.has_pending(user_id=user_id):
            await _do_reject(update, context, feedback_text=fb, source="router")
        else:
            return False  # Fall through to generation
        return True

    if intent == "reroll" and confidence >= 0.8:
        pending = state.get_pending(user_id=user_id)
        if pending:
            if _rate_limited(user_id):
                await update.message.reply_text(
                    f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
                )
                return True
            original = pending.get("original_request", "")
            state.clear_pending(user_id=user_id)
            state.clear_draft_history(user_id=user_id)
            await update.message.reply_text("Regenerating...")
            if original:
                if settings.AGENT_MODE == "agent":
                    await _handle_agent_mode(update, original, user_id=user_id)
                else:
                    await _handle_pipeline_mode(update, original, user_id=user_id)
            return True
        return False

    if intent == "modify_last" and confidence >= 0.5:
        fb = result.parameters.get("feedback", message)
        # Load existing pending to preserve conversation_history through edits
        existing_pending = state.get_pending(user_id=user_id)
        modified = await chat.handle_modify_last(fb, ctx, user_id=user_id)
        if modified:
            # Save modified draft and re-send (carry forward conversation_history)
            state.save_pending(
                caption=modified.get("caption", ""),
                hashtags=modified.get("hashtags", []),
                image_url=modified.get("image_url"),
                alt_text=modified.get("alt_text", ""),
                image_prompt=modified.get("image_prompt", ""),
                original_request=modified.get("original_request", ""),
                user_id=user_id,
                conversation_history=existing_pending.get("conversation_history") if existing_pending else None,
            )
            await _send_draft(update, modified, modified.get("image_url"), user_id=user_id)
            return True
        return False

    # Info commands — import lazily from debug module
    if intent == "show_status" and confidence >= 0.8:
        from bot.handlers.draft import status_command
        await status_command(update, context)
        return True

    if intent == "show_help" and confidence >= 0.8:
        from bot.handlers.admin import help_command
        await help_command(update, context)
        return True

    if intent == "show_analytics" and confidence >= 0.8:
        from bot.handlers.debug import analytics_command
        await analytics_command(update, context)
        return True

    if intent == "show_history" and confidence >= 0.8:
        from bot.handlers.debug import history_command
        await history_command(update, context)
        return True

    if intent == "brand_check" and confidence >= 0.8:
        from bot.handlers.admin import brand_check_command
        await brand_check_command(update, context)
        return True

    # Skill-matched intent — route to agent mode with skill hint
    if intent == "use_skill" and confidence >= 0.5:
        skill_name = result.parameters.get("skill", "")
        topic = result.parameters.get("topic", message)
        if skill_name:
            # Prefix the message with a skill hint so the agent picks it up
            augmented = f"[skill:{skill_name}] {topic}"
        else:
            augmented = message
        if _rate_limited(user_id):
            await update.message.reply_text(
                f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
            )
            return True
        if settings.AGENT_MODE == "agent":
            await _handle_agent_mode(update, augmented, user_id=user_id)
        else:
            await _handle_pipeline_mode(update, augmented, user_id=user_id)
        return True

    if intent == "upload_assets":
        await update.message.reply_text(
            "go ahead \u2014 send your images or PDFs and I'll analyze them automatically.\n\n"
            "bulk uploads are auto-ingested with AI vision. "
            "single images can be tagged as reference / mascot / style / background.",
        )
        return True

    if intent == "schedule_post" and confidence >= 0.5:
        time_expr = result.parameters.get("time", "")
        topic = result.parameters.get("topic", "")

        # If Haiku extracted time and topic, try to schedule directly
        if time_expr and topic:
            combined = f"{time_expr} {topic}"
            prompt, ts, recurrence, display = schedule_queue.parse_schedule_command(combined)
            if prompt and ts:
                item = schedule_queue.add_scheduled(prompt, ts, recurrence or "once")
                if item is None:
                    await update.message.reply_text(
                        "This post is already scheduled around that time. "
                        "Use /unschedule to cancel the existing one first.",
                        parse_mode="HTML",
                    )
                    return True
                await update.message.reply_text(
                    f"<b>Post scheduled</b>\n\n"
                    f"<b>Time:</b> {_esc(display)}\n"
                    f"<b>Prompt:</b> {_esc(prompt[:200])}\n"
                    f"<b>ID:</b> <code>{item['id']}</code>\n\n"
                    f"I'll generate a draft at the scheduled time and send it for your approval.\n"
                    f"Use /unschedule <code>{item['id']}</code> to cancel.",
                    parse_mode="HTML",
                )
                return True

        # If no time/topic extracted or parse failed, show bare keyword help
        # or list existing schedule
        items = schedule_queue.list_scheduled()
        if items:
            from bot.handlers.scheduling import scheduled_command
            await scheduled_command(update, context)
        else:
            await update.message.reply_text(
                "<b>Schedule a post</b>\n\n"
                "Tell me when and what to post:\n"
                "  <i>\"schedule a post about our launch at 3pm\"</i>\n"
                "  <i>\"post about community updates tomorrow 9am\"</i>\n\n"
                "Or use: /schedule <i>time</i> <i>prompt</i>",
                parse_mode="HTML",
            )
        return True

    # Conversational
    if intent == "greeting":
        user = update.effective_user
        name = user.first_name if user else ""
        if name and not ctx.user_name:
            ctx.user_name = name
        reply = await chat.handle_greeting(name, context=ctx)
        await update.message.reply_text(reply)
        conversation_context.update_context(
            user_id,
            last_bot_action="sent_content",
            user_name=ctx.user_name,
            conversation_history=ctx.conversation_history,
        )
        return True

    if intent == "casual_chat" and confidence >= 0.5:
        reply = await chat.handle_casual_chat(message, ctx)
        await update.message.reply_text(reply)
        conversation_context.update_context(
            user_id,
            last_bot_action="sent_content",
            conversation_history=ctx.conversation_history,
        )
        return True

    # generate_content or unknown / low confidence -> fall through to generation
    return False


async def _handle_agent_mode(update: Update, request: str, user_id: int | None = None) -> None:
    """Run the agent tool-use loop for a content request."""
    await update.message.chat.send_action("typing")

    # Extract direct photo path if embedded in the request
    direct_photo_path = None
    import re as _re_mod
    _dp_match = _re_mod.search(r"\[DIRECT PHOTO: (.+?)\]", request)
    if _dp_match:
        direct_photo_path = _dp_match.group(1)

    # If a reference image is stored, inject it into the request for the agent
    ref_path = state.get_reference_image()
    if ref_path and Path(ref_path).exists():
        request = f"{request}\n\n[REFERENCE IMAGE: {ref_path}]"

    _status_msg = None
    _status_lines: list[str] = []
    _reasoning_line: str = ""

    def _build_status_text() -> str:
        parts = list(_status_lines)
        if _reasoning_line:
            parts.append(f"<i>\U0001F4AD {_esc(_reasoning_line)}</i>")
        return "\n".join(parts)

    async def _update_status():
        nonlocal _status_msg
        text = _build_status_text()
        if not text:
            return
        if _status_msg is None:
            _status_msg = await update.message.reply_text(text, parse_mode="HTML")
        else:
            try:
                await _status_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass  # Telegram rejects edits if text unchanged

    async def on_tool_call(tool_name: str, description: str):
        nonlocal _reasoning_line
        _reasoning_line = ""
        icon = _TOOL_ICONS.get(tool_name, "\u26A1")
        _status_lines.append(f"{icon} {_esc(description)}")
        await _update_status()
        await update.message.chat.send_action("typing")

    async def on_reasoning(text: str):
        nonlocal _reasoning_line
        _reasoning_line = _truncate_reasoning(text)
        await _update_status()
        await update.message.chat.send_action("typing")

    # Restrict dangerous tools for non-admin operators
    _excluded = _ADMIN_ONLY_TOOLS if not _authorized(user_id) else None

    try:
        result = await engine.run_agent(
            request=request,
            on_tool_call=on_tool_call,
            on_reasoning=on_reasoning,
            excluded_tools=_excluded,
        )

        # Delete the status message now that we're done
        if _status_msg:
            try:
                await _status_msg.delete()
            except Exception:
                pass

        if not result.draft:
            text = result.response_text or "I processed your request but didn't generate a draft."
            max_len = 3900
            if len(text) > max_len:
                text = text[:max_len] + "..."
            await update.message.reply_text(
                _esc(text),
                parse_mode="HTML",
            )
            return

        image_url = result.image_url
        image_urls = result.image_urls

        # Direct photo mode: use the user's photo for template composition
        if direct_photo_path and Path(direct_photo_path).exists():
            image_url = direct_photo_path
            logger.info("Direct photo mode: using %s as image source", direct_photo_path)

        # Determine draft format and format-specific data
        draft_format = result.draft.get("format", "single")
        format_data = None
        if draft_format == "thread" and result.draft.get("thread_posts"):
            format_data = {"thread_posts": result.draft["thread_posts"]}
        elif draft_format == "report":
            format_data = {
                "report_type": result.draft.get("report_type", "custom"),
                "report_sections": result.draft.get("report_sections", []),
            }

        # Save pending state
        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft.get("hashtags", []),
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=request,
            image_urls=image_urls if len(image_urls) > 1 else None,
            content_type=result.draft.get("content_type"),
            user_id=user_id,
            conversation_history=result.conversation_history,
            draft_format=draft_format,
            format_data=format_data,
        )

        # Log to generation history (agent mode was previously missing this)
        try:
            await generation_history.async_log_generation(
                asset_type="social_post",
                content_type=result.draft.get("content_type", "unknown"),
                prompt=result.draft.get("image_prompt", ""),
                model_id=settings.AGENT_MODEL,
                image_urls=image_urls or ([image_url] if image_url else []),
                original_request=request,
            )
        except Exception as e:
            logger.debug("Agent generation history log failed: %s", e)

        await _send_draft(update, result.draft, image_url, resources=result.resources, image_urls=image_urls, user_id=user_id)

        # Fire hooks + transcript
        transcript.log_agent_response(
            user_id or 0, result.draft.get("caption", ""),
            turns=result.turns_used, tools=result.tool_calls_made,
        )
        await hooks.emit("draft:generated", {
            "draft": result.draft, "user_id": user_id,
            "turns": result.turns_used, "time": result.total_time,
        })

    except Exception as e:
        logger.error("Agent error: %s", e)
        import traceback as _tb
        tb_str = _tb.format_exc()

        # Auto-escalation to Claude Code if enabled
        escalation_msg = ""
        if getattr(settings, "CLAUDE_CODE_AUTO_ESCALATE", False):
            escalation_msg = "\n\nAttempting auto-fix via Claude Code..."

        await update.message.reply_text(
            f"Something went wrong: {_esc(str(e))}{escalation_msg}\n\nPlease try again.",
            parse_mode="HTML",
        )

        # Fire auto-escalation in background
        if getattr(settings, "CLAUDE_CODE_AUTO_ESCALATE", False):
            async def _notify(msg: str):
                try:
                    await update.message.reply_text(
                        f"<b>[Auto-Fix]</b> {_esc(msg)}",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass

            _aio.create_task(
                _escalate_agent_error(e, tb_str, _notify)
            )


async def _escalate_agent_error(
    error: Exception, tb_str: str, notify_callback
) -> None:
    """Background task: escalate an agent error to Claude Code."""
    try:
        from agent.claude_code import escalate_error
        await escalate_error(
            error,
            context="content generation (agent mode)",
            traceback_str=tb_str,
            notify_callback=notify_callback,
        )
    except Exception as esc_err:
        logger.error("Auto-escalation failed: %s", esc_err)


async def _handle_pipeline_mode(update: Update, request: str, user_id: int | None = None) -> None:
    """Run the existing multi-step pipeline for a content request.

    .. deprecated::
        Pipeline mode is deprecated. Set AGENT_MODE=agent.
    """
    logger.warning(
        "Pipeline mode is deprecated. Set AGENT_MODE=agent to use the active architecture."
    )
    await update.message.chat.send_action("typing")

    _pipe_status_msg = None
    _pipe_status_lines: list[str] = []

    async def on_step(step_num: int, total: int, step_name: str, summary: str):
        nonlocal _pipe_status_msg
        icon = _STEP_ICONS.get(step_name, "\u26A1")
        _pipe_status_lines.append(f"{icon} [{step_num}/{total}] {step_name}")
        text = "\n".join(_pipe_status_lines)
        if _pipe_status_msg is None:
            _pipe_status_msg = await update.message.reply_text(text, parse_mode="HTML")
        else:
            try:
                await _pipe_status_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass
        await update.message.chat.send_action("typing")

    try:
        brand_context = guidelines.get_brand_context()
        logger.info("Brand context loaded: %d chars", len(brand_context))

        pipeline_result = await brain.pipeline_generate(
            request=request,
            brand_context=brand_context,
            on_step=on_step,
        )

        draft = pipeline_result.draft

        # Delete the status message now that we're done
        if _pipe_status_msg:
            try:
                await _pipe_status_msg.delete()
            except Exception:
                pass

        if pipeline_result.fell_back:
            await update.message.reply_text(
                "<i>Note: Pipeline had an issue, used direct generation instead.</i>",
                parse_mode="HTML",
            )

        logger.info("Draft generated: %s", draft.get("caption", "")[:80])

        # Generate image with smart model routing (mode-aware)
        image_url = None
        cfg = _cc.get_config()
        should_gen = draft.get("image_prompt") and cfg.default_mode != "text_only"
        if should_gen:
            await update.message.chat.send_action("upload_photo")
            content_type = draft.get("content_type", "announcement")
            from agent import template_memory as _tm
            template_aspect = _tm.get_aspect_ratio_for_content_type(content_type)
            image_url = await image_gen.generate_image(draft["image_prompt"], content_type=content_type, aspect_ratio=template_aspect)
            if not image_url:
                await update.message.reply_text(
                    "Image generation failed \u2014 sending text draft only.",
                )

        # Save pending state
        state.save_pending(
            caption=draft["caption"],
            hashtags=draft.get("hashtags", []),
            image_url=image_url,
            alt_text=draft["alt_text"],
            image_prompt=draft["image_prompt"],
            original_request=request,
            user_id=user_id,
        )

        await _send_draft(update, draft, image_url, user_id=user_id)

    except Exception as e:
        logger.error("Pipeline error: %s", e)
        await update.message.reply_text(
            f"Something went wrong: {_esc(str(e))}\n\nPlease try again.",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /generate — standalone asset generation
# ---------------------------------------------------------------------------


async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /generate <type> <description> — generate a standalone branded asset."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=2)  # "/generate", "type", "description..."

    asset_types = ", ".join(asset_gen.SUPPORTED_ASSET_TYPES)
    if len(parts) < 3:
        await update.message.reply_text(
            f"Usage: /generate <i>type</i> <i>description</i>\n\n"
            f"Types: {_esc(asset_types)}\n\n"
            f"Example: <code>/generate logo a shield with lightning bolt</code>",
            parse_mode="HTML",
        )
        return

    asset_type = parts[1].lower()
    description = parts[2]

    if _rate_limited(update.effective_user.id):
        await update.message.reply_text(
            f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
        )
        return

    await update.message.chat.send_action("upload_photo")
    await update.message.reply_text(
        f"Generating <b>{_esc(asset_type)}</b> options...",
        parse_mode="HTML",
    )

    try:
        result = await asset_gen.generate_asset(asset_type, description)
    except Exception as e:
        logger.error("generate_asset failed: %s", e)
        await update.message.reply_text(f"Generation failed: {_esc(str(e))}", parse_mode="HTML")
        return

    if result.get("error"):
        await update.message.reply_text(
            f"Error: {_esc(result['error'])}", parse_mode="HTML"
        )
        return

    urls = result.get("urls", [])
    if not urls:
        await update.message.reply_text("No images were generated. Check logs for details.")
        return

    # Log to generation history
    try:
        await generation_history.async_log_generation(
            asset_type=asset_type,
            content_type=result.get("content_type", ""),
            prompt=result.get("prompt", description),
            model_id=result.get("model_id", "auto"),
            image_urls=urls,
            original_request=f"/generate {asset_type} {description}",
        )
    except Exception as e:
        logger.warning("Failed to log generation history: %s", e)

    # Build a 2x2 grid image and send with inline keyboard
    grid_image = await _build_asset_grid(urls)

    # Save as pending for /approve N (and callback buttons)
    state.save_pending(
        caption=f"[{asset_type}] {description}",
        hashtags=[],
        image_url=urls[0] if urls else "",
        alt_text=description,
        image_prompt=result.get("prompt", description),
        original_request=f"/generate {asset_type} {description}",
        image_urls=urls,
        content_type=result.get("content_type"),
    )

    if grid_image:
        # Build inline keyboard: Approve 1-N + Reject All
        buttons = [
            InlineKeyboardButton(f"Approve {i}", callback_data=f"gen_approve:{i}")
            for i in range(1, len(urls) + 1)
        ]
        buttons.append(
            InlineKeyboardButton("Reject All", callback_data="gen_reject")
        )
        # Arrange: approve buttons on first row, reject on second
        keyboard = InlineKeyboardMarkup([
            buttons[:-1],
            [buttons[-1]],
        ])

        await update.message.reply_photo(
            photo=grid_image,
            caption=f"<b>{_esc(asset_type)}</b> \u2014 {len(urls)} options generated",
            parse_mode="HTML",
            reply_markup=keyboard,
        )
    else:
        # Fallback: send individual photos if grid fails
        for i, url in enumerate(urls, 1):
            try:
                await update.message.reply_photo(
                    photo=url,
                    caption=f"Option {i}/{len(urls)} \u2014 {_esc(asset_type)}",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("Failed to send option %d: %s", i, e)

        await update.message.reply_text(
            f"{len(urls)} option(s) generated. Use /approve N to select one.",
            parse_mode="HTML",
        )


async def _build_asset_grid(urls: list[str]) -> io.BytesIO | None:
    """Download images and compose a labeled 2x2 grid. Returns BytesIO or None."""
    import httpx as _httpx
    from agent.net_guard import validate_url as _validate_url

    if not urls:
        return None

    images: list[_PILImage.Image] = []
    try:
        async with _httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            for url in urls[:4]:
                try:
                    _validate_url(url)
                    resp = await client.get(url)
                    resp.raise_for_status()
                    img = _PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning("Grid download failed for %s: %s", url[:60], e)
    except Exception as e:
        logger.warning("Grid build failed: %s", e)
        return None

    if not images:
        return None

    # Target cell size
    cell_w, cell_h = 640, 480
    pad = 8
    label_h = 32

    cols = 2 if len(images) > 1 else 1
    rows = (len(images) + cols - 1) // cols
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * (cell_h + label_h) + (rows + 1) * pad

    grid = _PILImage.new("RGB", (grid_w, grid_h), (20, 20, 40))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)

        try:
            font = ImageFont.load_default(size=20)
        except TypeError:
            font = ImageFont.load_default()

        for idx, img in enumerate(images):
            col = idx % cols
            row = idx // cols
            x = pad + col * (cell_w + pad)
            y = pad + row * (cell_h + label_h + pad)

            # Crop-fill to cell size
            sr = img.width / img.height
            tr = cell_w / cell_h
            if sr > tr:
                nw, nh = int(cell_h * sr), cell_h
            else:
                nw, nh = cell_w, int(cell_w / sr)
            resized = img.resize((nw, nh), _PILImage.LANCZOS)
            ox, oy = (nw - cell_w) // 2, (nh - cell_h) // 2
            cropped = resized.crop((ox, oy, ox + cell_w, oy + cell_h))

            # Draw label background
            draw.rectangle([x, y, x + cell_w, y + label_h], fill=(40, 40, 60))
            draw.text((x + 10, y + 6), f"Option {idx + 1}", fill=(255, 255, 255), font=font)

            grid.paste(cropped, (x, y + label_h))
    except Exception as e:
        logger.warning("Grid label drawing failed: %s", e)
        return None

    buf = io.BytesIO()
    await _aio.to_thread(lambda: grid.save(buf, format="PNG", optimize=True))
    buf.seek(0)
    return buf


async def generate_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks for /generate asset approval/rejection."""
    query = update.callback_query
    if not query or not query.data:
        return

    user_id = query.from_user.id if query.from_user else 0
    if not _authorized(user_id):
        await query.answer("Not authorized.")
        return

    data = query.data

    if data.startswith("gen_approve:"):
        # Parse option number
        try:
            option_num = int(data.split(":")[1])
        except (ValueError, IndexError):
            await query.answer("Invalid option.")
            return

        pending = state.get_pending(user_id=user_id)
        if not pending:
            await query.answer("Nothing pending to approve.")
            return

        image_urls = pending.get("image_urls", [])
        if not image_urls or option_num < 1 or option_num > len(image_urls):
            await query.answer(f"Invalid option {option_num}.")
            return

        # Select the chosen image
        pending["image_url"] = image_urls[option_num - 1]

        # Log feedback
        count = await feedback.async_log_feedback(
            request=pending.get("original_request", ""),
            draft=pending,
            accepted=True,
            resources_used=pending.get("resources_used", []),
        )

        # Update generation history
        try:
            ts = pending.get("timestamp", 0)
            if ts:
                await generation_history.async_update_generation_status(ts, "approved")
        except Exception as e:
            logger.debug("Generation history update failed: %s", e)

        # Add to LoRA training set
        if pending.get("image_url"):
            try:
                from agent import lora_pipeline
                lora_count, _ = await lora_pipeline.add_training_image_from_url(
                    pending["image_url"],
                    pending.get("image_prompt", ""),
                    pending.get("content_type", "brand_asset"),
                )
                logger.info("LoRA training image added (%d total)", lora_count)
            except Exception as e:
                logger.debug("LoRA training image add failed: %s", e)

        state.clear_pending()
        await query.answer(f"Option {option_num} approved!")
        await query.edit_message_caption(
            caption=f"Approved option {option_num}",
            parse_mode="HTML",
        )

    elif data == "gen_reject":
        pending = state.get_pending()
        if not pending:
            await query.answer("Nothing pending.")
            return

        # Log rejection
        await feedback.async_log_feedback(
            request=pending.get("original_request", ""),
            draft=pending,
            accepted=False,
            feedback_text="Rejected via button",
            resources_used=pending.get("resources_used", []),
        )

        try:
            ts = pending.get("timestamp", 0)
            if ts:
                await generation_history.async_update_generation_status(ts, "rejected")
        except Exception as e:
            logger.debug("Generation history update failed: %s", e)

        state.clear_pending()
        await query.answer("All options rejected.")
        await query.edit_message_caption(
            caption="Rejected. Use /generate again with feedback.",
            parse_mode="HTML",
        )

    else:
        await query.answer("Unknown action.")


# ---------------------------------------------------------------------------
# /score — show preference score for the current pending draft
# ---------------------------------------------------------------------------

async def score_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the preference engine score for the current pending draft."""
    if not _authorized(update.effective_user.id):
        return

    pending = state.get_pending()
    if not pending:
        await update.message.reply_text("No pending draft to score.")
        return

    await update.message.reply_text("Scoring draft against learned preferences...")

    try:
        from agent.preference_engine import score_draft
        caption = pending.get("caption", "")
        original_request = pending.get("original_request", caption)
        result = await score_draft(pending, original_request)

        flags_str = ", ".join(result.flags) if result.flags else "none"
        verdict = "REJECT" if result.should_reject else "PASS"
        text = (
            f"<b>Preference Score</b>\n\n"
            f"Score: <b>{result.score:.1f}</b> / 10\n"
            f"Verdict: <b>{verdict}</b>\n"
            f"Reasoning: {_esc(result.reasoning)}\n"
            f"Flags: {_esc(flags_str)}"
        )
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        logger.error("Score command failed: %s", e)
        await update.message.reply_text(f"Scoring failed: {e}")


# ---------------------------------------------------------------------------
# /approval_rate — show approval trend over the last 7 days
# ---------------------------------------------------------------------------

async def approval_rate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the approval rate trend from the preference engine."""
    if not _authorized(update.effective_user.id):
        return

    try:
        from agent.preference_engine import get_approval_trend
        trend = get_approval_trend(days=7)

        if not any(day["total"] > 0 for day in trend):
            await update.message.reply_text("No feedback data yet. Approve or reject some drafts first.")
            return

        lines = ["<b>Approval Rate (last 7 days)</b>\n"]
        for day in trend:
            if day["total"] == 0:
                bar = "  --"
            else:
                filled = round(day["rate"] / 10)
                bar = "=" * filled + "-" * (10 - filled)
                bar = f"[{bar}]"
            lines.append(
                f"<code>{day['date']}</code>  {bar}  "
                f"{day['rate']:.0f}% ({day['approved']}/{day['total']})"
            )

            # Per-content-type breakdown if there's data
            for ct, ct_stats in day.get("by_content_type", {}).items():
                ct_total = ct_stats["approved"] + ct_stats["rejected"]
                if ct_total > 0:
                    lines.append(
                        f"  {ct}: {ct_stats['rate']:.0f}% "
                        f"({ct_stats['approved']}/{ct_total})"
                    )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
    except Exception as e:
        logger.error("Approval rate command failed: %s", e)
        await update.message.reply_text(f"Failed to load approval rate: {e}")
