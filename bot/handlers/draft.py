"""
Draft management handlers — approve, reject, post, send_draft, draft_callback, revisions.
"""

__all__ = [
    "draft_callback",
    "approve_command",
    "reject_command",
    "refine_command",
    "edit_command",
    "cancel_command",
    "status_command",
]

import asyncio as _aio
import io
import logging
import re
import tempfile
import time
from pathlib import Path

from PIL import Image as _PILImage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from agent import auto_state, brain, compositor, compositor_config, conversation_context, engine, feedback, generation_history, guidelines, hooks, image_gen, state, transcript
from agent import compositor_config as _cc
from config import settings

from bot.handlers.core import (
    _ADMIN_ONLY_TOOLS,
    _RATE_LIMIT_SECONDS,
    _REVIEW_PROMPTS,
    _STEP_ICONS,
    _TOOL_ICONS,
    _authorized,
    _can_operate,
    _esc,
    _get_approve_lock,
    _maybe_compose,
    _prepare_photo,
    _rate_limited,
    _truncate_reasoning,
)

import random as _random

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared core logic — used by slash commands, NL router, and inline buttons
# ---------------------------------------------------------------------------

async def _do_approve(update: Update, context: ContextTypes.DEFAULT_TYPE, option_num: int = 1, source: str = "command") -> None:
    """Core approve logic shared by /approve, NL router, and inline buttons."""
    user_id = update.effective_user.id if update.effective_user else None
    pending = await state.async_get_pending(user_id=user_id)
    if not pending:
        await update.message.reply_text("Nothing to approve. Send me a content request first.")
        return

    # If multiple image options exist, select the chosen one
    image_urls = pending.get("image_urls", [])
    if image_urls and 1 <= option_num <= len(image_urls):
        pending["image_url"] = image_urls[option_num - 1]
        logger.info("Approve (%s): selected option %d of %d", source, option_num, len(image_urls))
    elif image_urls and option_num > len(image_urls):
        await update.message.reply_text(
            f"Only {len(image_urls)} options available. Use /approve 1-{len(image_urls)}."
        )
        return

    await update.message.chat.send_action("typing")

    # Log feedback
    count = await feedback.async_log_feedback(
        request=pending.get("original_request", ""),
        draft=pending,
        accepted=True,
        resources_used=pending.get("resources_used", []),
    )

    # Update generation history status
    try:
        ts = pending.get("timestamp", 0)
        if ts:
            await generation_history.async_update_generation_status(ts, "approved")
    except Exception as e:
        logger.debug("Generation history update failed: %s", e)

    # Save approved composed image to brand/references/ for style consistency
    composed_path, composed_ct = state.get_last_composed(user_id=user_id)
    if composed_path and Path(composed_path).exists():
        try:
            refs_dir = Path(settings.BRAND_FOLDER) / "references"
            refs_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            save_name = f"approved_{composed_ct}_{ts}.png"
            save_path = refs_dir / save_name
            import shutil
            await _aio.to_thread(shutil.copy2, composed_path, save_path)
            logger.info("Saved approved reference: %s", save_path)

            # Cap at 5 per content_type — delete oldest
            existing = sorted(refs_dir.glob(f"approved_{composed_ct}_*.png"))
            if len(existing) > 5:
                for old in existing[:-5]:
                    old.unlink(missing_ok=True)
                    logger.info("Pruned old reference: %s", old.name)
        except Exception as e:
            logger.warning("Failed to save approved reference: %s", e)

        # Save into active style profile if one is set for this content_type
        try:
            active_profile = state.get_active_profile(composed_ct)
            if active_profile:
                count_p = state.add_profile_image(active_profile, composed_path)
                logger.info(
                    "Saved approved image to profile %s (%d images)",
                    active_profile, count_p,
                )
        except Exception as e:
            logger.warning("Failed to save to style profile: %s", e)

        # Add to asset library
        try:
            from agent import asset_library
            asset_library.add(
                composed_path, "approved", composed_ct or "general",
                prompt=pending.get("image_prompt", ""),
                tags=["approved"],
            )
        except Exception as e:
            logger.debug("Asset library add failed: %s", e)

    # Save approved mascot outputs to grow character reference library
    _mascot_kw = re.compile(r"mascot|character", re.IGNORECASE)
    _is_mascot_draft = (
        _mascot_kw.search(pending.get("original_request", ""))
        or _mascot_kw.search(pending.get("image_prompt", ""))
    )
    if _is_mascot_draft and pending.get("image_url"):
        try:
            from agent._client import get_httpx as _get_httpx
            _r = await _get_httpx().get(pending["image_url"])
            _r.raise_for_status()
            ts = int(time.time())
            save_path = Path(settings.BRAND_FOLDER) / "assets" / f"mascot_approved_{ts}.png"
            await _aio.to_thread(lambda: _PILImage.open(io.BytesIO(_r.content)).convert("RGB").save(str(save_path), "PNG"))
            logger.info("Saved approved mascot output: %s", save_path)
        except Exception as e:
            logger.warning("Failed to save mascot output: %s", e)

    # Add to LoRA training set
    if pending.get("image_url"):
        try:
            from agent import lora_pipeline
            img_url = pending["image_url"]
            img_prompt = pending.get("image_prompt", "")
            ct = pending.get("content_type", composed_ct or "announcement")
            lora_count, threshold_hit = await lora_pipeline.add_training_image_from_url(
                img_url, img_prompt, ct,
            )
            logger.info("LoRA training image added (%d total)", lora_count)
            if threshold_hit:
                await update.message.reply_text(
                    f"Training set reached {lora_count} images! "
                    f"Use /train_lora to start LoRA training.",
                )
        except Exception as e:
            logger.warning("Failed to add LoRA training image: %s", e)

    # Move from pending -> approved (no posting yet)
    await _aio.to_thread(state.approve_pending, user_id=user_id)
    state.clear_draft_history(user_id=user_id)

    # Update session plan if active
    try:
        from agent import session_plan
        plan = session_plan.get_plan()
        if plan:
            current_id = plan.get("current_item")
            if current_id:
                session_plan.update_item(current_id, status="approved")
    except Exception as e:
        logger.debug("Session plan update failed in _do_approve: %s", e)

    await update.message.reply_text(
        "Approved! Want me to post this to X now, or schedule it for later?",
        parse_mode="HTML",
    )
    logger.info("Draft approved (%s), awaiting post/schedule (feedback #%d)", source, count)

    # Fire hooks + transcript
    transcript.log_draft_action(user_id or 0, "approved", caption=pending.get("caption", ""))
    await hooks.emit("draft:approved", {"draft": pending, "user_id": user_id, "source": source})

    # NOTE: Auto-summarize into learned_preferences.md disabled.
    # Preference extraction is now handled by pref_extractor.py -> session.learned_preferences.
    # The /learn command still works as a manual diagnostic tool.

    # Self-review: increment approval counter, trigger background review if threshold reached
    try:
        from agent import self_review_scheduler
        should_review = self_review_scheduler.record_approval()
        if should_review:
            await self_review_scheduler.maybe_trigger_review()
            logger.info("Self-review triggered after approval threshold")
    except Exception as e:
        logger.debug("Self-review scheduler failed in _do_approve: %s", e)


async def _do_post(update: Update, context: ContextTypes.DEFAULT_TYPE, source: str = "command") -> None:
    """Core post logic — publishes the approved draft to X/Discord."""
    from agent import publisher, campaigns

    user_id = update.effective_user.id if update.effective_user else None
    approved = await _aio.to_thread(state.get_approved, user_id=user_id)
    if not approved:
        await update.message.reply_text("Nothing to post. Approve a draft first.")
        return

    await update.message.chat.send_action("typing")

    draft_format = approved.get("format", "single")
    format_data = approved.get("format_data", {})

    composed_path, composed_ct = state.get_last_composed(user_id=user_id)

    # --- Thread publishing ---
    if draft_format == "thread" and format_data.get("thread_posts"):
        thread_posts = format_data["thread_posts"]
        tweet_url = None
        try:
            tweet_urls = await publisher.post_thread_to_x(thread_posts)
            tweet_url = tweet_urls[0] if tweet_urls else None
        except Exception as e:
            logger.error("Failed to post thread to X: %s", e)
            await update.message.reply_text(
                f"Thread posting failed. Check logs.\n"
                f"Your draft is still approved \u2014 try again with 'post it'.",
                parse_mode="HTML",
            )
            return

        # Record and clean up (shared with single post flow below)
        auto_slot = approved.get("auto_slot")
        if auto_slot:
            auto_state.record_post(
                slot_name=auto_slot,
                caption=approved.get("caption", ""),
                tweet_url=tweet_url,
                event_ids=approved.get("auto_event_ids"),
            )
        state.clear_approved(user_id=user_id)

        try:
            from agent.session import record_approved_post
            record_approved_post(
                caption=approved.get("caption", ""),
                slot=auto_slot or "",
                tweet_url=tweet_url,
            )
        except Exception as e:
            logger.debug("Session record failed: %s", e)

        # Track post performance for analytics feedback loop
        try:
            from agent.performance import record_post as _perf_record
            if tweet_url:
                _thread_tweet_id = tweet_url.rstrip("/").split("/")[-1]
                _perf_record(
                    tweet_id=_thread_tweet_id,
                    content_type=approved.get("content_type", ""),
                    caption=approved.get("caption", ""),
                )
        except Exception as e:
            logger.debug("Performance tracking failed: %s", e)

        url_list = "\n".join(f"  {i+1}/ {url}" for i, url in enumerate(tweet_urls))
        await update.message.reply_text(
            f"Thread posted to X! ({len(tweet_urls)} posts)\n{url_list}",
            parse_mode="HTML",
        )
        transcript.log_publish(user_id or 0, "twitter", url=tweet_url)
        await hooks.emit("post:published", {"platform": "twitter", "url": tweet_url, "user_id": user_id, "format": "thread"})
        return

    # Publish to all configured platforms concurrently
    publish_results: dict[str, str | None] = {}
    try:
        publish_results = await publisher.publish_to_all(
            draft=approved,
            image_url=approved.get("image_url"),
            composed_path=composed_path,
        )
    except Exception as e:
        logger.error("publish_to_all failed: %s", e)
        await update.message.reply_text(
            f"Publishing failed. Check logs for details.\n"
            f"Your draft is still approved \u2014 try again with 'post it'.",
            parse_mode="HTML",
        )
        return

    # Extract specific URLs for backward compatibility
    tweet_url = publish_results.get("x")
    discord_url = publish_results.get("discord")

    # If X was requested but failed, report it (but don't block other results)
    if "x" in (settings.PUBLISH_PLATFORMS or ["x"]) and tweet_url is None:
        logger.warning("X publishing returned no URL -- may have failed")

    # If this draft came from the auto-post scheduler, record it
    auto_slot = approved.get("auto_slot")
    if auto_slot:
        auto_state.record_post(
            slot_name=auto_slot,
            caption=approved.get("caption", ""),
            tweet_url=tweet_url,
            event_ids=approved.get("auto_event_ids"),
        )
        logger.info("Auto-post slot '%s' recorded via post (%s)", auto_slot, source)

        # Update campaign slot progress if this was a campaign post
        if auto_slot.startswith("scheduled:"):
            queue_id = auto_slot.removeprefix("scheduled:")
            campaigns.update_slot_by_queue_id(queue_id, "posted", post_url=tweet_url or "")

    state.clear_approved(user_id=user_id)

    # Track context — draft posted, nothing pending
    try:
        if user_id:
            conversation_context.update_context(
                user_id,
                last_bot_action="sent_content",
                pending_draft_exists=False,
                last_command="/post",
            )
    except Exception as e:
        logger.debug("Context tracking failed in _do_post: %s", e)

    # Clean up temp composed file (after publish so it's still available for X upload)
    if composed_path and Path(composed_path).exists():
        try:
            Path(composed_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("Composed cleanup failed for %s: %s", composed_path, e)
        state.clear_last_composed(user_id=user_id)

    # Record in session memory for agent context
    try:
        from agent.session import record_approved_post
        record_approved_post(
            caption=approved.get("caption", ""),
            slot=auto_slot or "",
            tweet_url=tweet_url,
        )
    except Exception as e:
        logger.debug("Session record_approved_post failed: %s", e)

    # Track post performance for analytics feedback loop
    try:
        from agent.performance import record_post as _perf_record
        if tweet_url:
            _tweet_id = tweet_url.rstrip("/").split("/")[-1]
            _perf_record(
                tweet_id=_tweet_id,
                content_type=approved.get("content_type", ""),
                caption=approved.get("caption", ""),
            )
    except Exception as e:
        logger.debug("Performance tracking failed: %s", e)

    # Build response message showing all platform results
    slot_note = f"  (auto-slot: {_esc(auto_slot)})" if auto_slot else ""
    platform_lines = []
    for plat, url in publish_results.items():
        if url:
            platform_lines.append(f"{plat.capitalize()}: {_esc(url)}")
        else:
            # Only show failures for platforms that were explicitly configured
            if plat in (settings.PUBLISH_PLATFORMS or ["x"]):
                platform_lines.append(f"{plat.capitalize()}: (failed)")
    platforms_str = "\n".join(platform_lines) if platform_lines else "(no platforms published)"
    await update.message.reply_text(
        f"Published!{slot_note}\n{platforms_str}",
        parse_mode="HTML",
    )
    logger.info("Draft posted (%s): %s", source, publish_results)

    # Fire hooks + transcript for each successful platform
    user_id = update.effective_user.id if update.effective_user else 0
    for plat, url in publish_results.items():
        if url:
            transcript.log_publish(user_id, plat, url=url)
    await hooks.emit("post:published", {"platforms": publish_results, "user_id": user_id})


async def _do_reject(update: Update, context: ContextTypes.DEFAULT_TYPE, feedback_text: str = "", source: str = "command") -> None:
    """Core reject logic shared by /reject, NL router, and inline buttons."""
    user_id = update.effective_user.id if update.effective_user else None
    pending = state.get_pending(user_id=user_id)
    if not pending:
        await update.message.reply_text("Nothing to reject. Send me a content request first.")
        return

    if not feedback_text:
        await update.message.reply_text(
            "Please include feedback: /reject <i>make it more urgent and add a CTA</i>",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")

    # Log the rejection
    count = await feedback.async_log_feedback(
        request=pending.get("original_request", ""),
        draft=pending,
        accepted=False,
        feedback_text=feedback_text,
        resources_used=pending.get("resources_used", []),
    )
    logger.info("Draft rejected (%s, feedback #%d): %s", source, count, feedback_text[:100])

    # Record in session memory for agent context
    try:
        from agent.session import record_rejected_draft
        record_rejected_draft(
            caption=pending.get("caption", ""),
            feedback=feedback_text,
            slot=pending.get("auto_slot", ""),
        )
    except Exception as e:
        logger.debug("Session record_rejected_draft failed: %s", e)

    # Update generation history status
    try:
        ts = pending.get("timestamp", 0)
        if ts:
            await generation_history.async_update_generation_status(ts, "rejected")
    except Exception as e:
        logger.debug("Generation history update failed: %s", e)

    # NOTE: Auto-summarize into learned_preferences.md disabled.
    # Preference extraction is now handled by pref_extractor.py -> session.learned_preferences.

    # Clear the old pending before running revision
    state.clear_pending(user_id=user_id)

    # Update session plan if active
    try:
        from agent import session_plan
        plan = session_plan.get_plan()
        if plan:
            current_id = plan.get("current_item")
            if current_id:
                session_plan.update_item(current_id, status="rejected", notes=feedback_text[:200])
                logger.info("Session plan: item #%d rejected", current_id)
    except Exception as e:
        logger.debug("Session plan update failed in _do_reject: %s", e)

    # Track context — draft rejected, revision incoming
    try:
        if user_id:
            conversation_context.update_context(
                user_id,
                last_bot_action="idle",
                pending_draft_exists=False,
                last_command="/reject",
            )
    except Exception as e:
        logger.debug("Context tracking failed in _do_reject: %s", e)

    # Fire hooks + transcript
    transcript.log_draft_action(user_id or 0, "rejected", caption=pending.get("caption", ""), feedback=feedback_text)
    await hooks.emit("draft:rejected", {"draft": pending, "feedback": feedback_text, "user_id": user_id})

    # Branch: agent mode re-runs with revision context, pipeline mode uses revise_draft
    if settings.AGENT_MODE == "agent":
        await _handle_agent_revision(update, pending, feedback_text, user_id=user_id)
    else:
        await _handle_pipeline_revision(update, pending, feedback_text, user_id=user_id)


async def _handle_pipeline_revision(update: Update, pending: dict, feedback_text: str, user_id: int | None = None) -> None:
    """Revise a draft using the pipeline mode."""
    try:
        brand_context = guidelines.get_brand_context()
        draft = await brain.revise_draft(
            original_draft=pending,
            feedback=feedback_text,
            brand_context=brand_context,
        )

        # Generate new image if prompt changed
        image_url = pending.get("image_url")
        if draft.get("image_prompt") != pending.get("image_prompt"):
            await update.message.chat.send_action("upload_photo")
            content_type = draft.get("content_type", "announcement")
            from agent import template_memory as _tm
            template_aspect = _tm.get_aspect_ratio_for_content_type(content_type)
            image_url = await image_gen.generate_image(draft["image_prompt"], content_type=content_type, aspect_ratio=template_aspect)

        # Save revised draft (carry forward auto-post metadata through revisions)
        state.save_pending(
            caption=draft["caption"],
            hashtags=draft.get("hashtags", []),
            image_url=image_url,
            alt_text=draft["alt_text"],
            image_prompt=draft["image_prompt"],
            original_request=pending["original_request"],
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
            user_id=user_id,
        )

        await _send_draft(update, draft, image_url, user_id=user_id)

    except Exception as e:
        logger.error("Revision failed: %s", e)
        # Restore the old pending so user can retry
        state.save_pending(
            caption=pending.get("caption", ""),
            hashtags=pending.get("hashtags", []),
            image_url=pending.get("image_url"),
            alt_text=pending.get("alt_text", ""),
            image_prompt=pending.get("image_prompt", ""),
            original_request=pending.get("original_request", ""),
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
            user_id=user_id,
        )
        await update.message.reply_text(
            f"Revision failed: {_esc(str(e))}\n\nOriginal draft still pending. Try again or /cancel.",
            parse_mode="HTML",
        )


async def _handle_agent_revision(update: Update, pending: dict, feedback_text: str, user_id: int | None = None) -> None:
    """Revise a draft using agent mode — continues the conversation thread if history exists."""
    history = pending.get("conversation_history")

    _rev_status_msg = None
    _rev_status_lines: list[str] = []
    _rev_reasoning_line: str = ""

    def _build_rev_status_text() -> str:
        parts = list(_rev_status_lines)
        if _rev_reasoning_line:
            parts.append(f"<i>\U0001F4AD {_esc(_rev_reasoning_line)}</i>")
        return "\n".join(parts)

    async def _update_rev_status():
        nonlocal _rev_status_msg
        text = _build_rev_status_text()
        if not text:
            return
        if _rev_status_msg is None:
            _rev_status_msg = await update.message.reply_text(text, parse_mode="HTML")
        else:
            try:
                await _rev_status_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass

    async def on_tool_call(tool_name: str, description: str):
        nonlocal _rev_reasoning_line
        _rev_reasoning_line = ""
        icon = _TOOL_ICONS.get(tool_name, "\u26A1")
        _rev_status_lines.append(f"{icon} {_esc(description)}")
        await _update_rev_status()
        await update.message.chat.send_action("typing")

    async def on_reasoning(text: str):
        nonlocal _rev_reasoning_line
        _rev_reasoning_line = _truncate_reasoning(text)
        await _update_rev_status()
        await update.message.chat.send_action("typing")

    try:
        if history:
            # Continue the conversation — agent sees its full prior reasoning
            history.append({
                "role": "user",
                "content": (
                    f"Your draft was rejected. Here is the feedback:\n\n"
                    f"\"{feedback_text}\"\n\n"
                    f"Please use think to analyze what went wrong, then revise and submit "
                    f"an improved draft via finish. Address the feedback directly."
                ),
            })
            _rev_excluded = _ADMIN_ONLY_TOOLS if not _authorized(user_id or 0) else None
            result = await engine.run_agent_with_history(
                history, on_tool_call=on_tool_call,
                on_reasoning=on_reasoning,
                excluded_tools=_rev_excluded,
            )
        else:
            # Fallback: no history available (legacy pending drafts)
            revision_context = (
                f"PREVIOUS DRAFT (REJECTED):\n"
                f"Caption: {pending.get('caption', '')}\n"
                f"Image prompt: {pending.get('image_prompt', '')}\n\n"
                f"USER FEEDBACK: {feedback_text}\n\n"
                f"Please revise the draft based on this feedback. Address the specific concerns raised."
            )
            result = await engine.run_agent(
                request=pending.get("original_request", ""),
                on_tool_call=on_tool_call,
                on_reasoning=on_reasoning,
                revision_context=revision_context,
            )

        # Delete the status message now that we're done
        if _rev_status_msg:
            try:
                await _rev_status_msg.delete()
            except Exception:
                pass

        if not result.draft:
            await update.message.reply_text(
                f"Agent couldn't produce a valid revision.\n\n<pre>{_esc(result.final_text[:500])}</pre>",
                parse_mode="HTML",
            )
            return

        image_url = result.image_url or pending.get("image_url")

        # Carry forward auto-post slot metadata through revisions
        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft.get("hashtags", []),
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=pending["original_request"],
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
            user_id=user_id,
            conversation_history=result.conversation_history,
        )

        await _send_draft(update, result.draft, image_url, resources=result.resources, user_id=user_id)
        return

    except Exception as e:
        logger.error("Agent revision failed: %s", e)
        # Restore the old pending so user can retry
        state.save_pending(
            caption=pending.get("caption", ""),
            hashtags=pending.get("hashtags", []),
            image_url=pending.get("image_url"),
            alt_text=pending.get("alt_text", ""),
            image_prompt=pending.get("image_prompt", ""),
            original_request=pending.get("original_request", ""),
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
        )
        await update.message.reply_text(
            f"Agent revision failed: {_esc(str(e))}\n\nOriginal draft still pending. Try again or /cancel.",
            parse_mode="HTML",
        )


async def _send_draft(
    update: Update,
    draft: dict,
    image_url: str | None,
    resources=None,
    image_urls: list[str] | None = None,
    user_id: int | None = None,
) -> None:
    """Send the generated draft to the user for review.

    When image_urls has >1 item, sends each as a numbered option with its own
    composed brand template so the CMO can compare side-by-side.
    """
    import asyncio as _asyncio

    caption = draft["caption"]
    content_type = draft.get("content_type", "default")
    draft_format = draft.get("format", "single")

    # --- Thread display ---
    if draft_format == "thread" and draft.get("thread_posts"):
        thread_posts = draft["thread_posts"]
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Approve Thread", callback_data="draft_approve"),
                InlineKeyboardButton("Reject", callback_data="draft_reject"),
            ],
        ])
        lines = [f"<b>Thread ({len(thread_posts)} posts)</b>\n"]
        for i, post in enumerate(thread_posts, 1):
            text = _esc(post.get("text", ""))
            char_count = len(post.get("text", ""))
            lines.append(f"<b>{i}/</b> {text} <i>({char_count})</i>\n")
        lines.append(f"\n/approve to post thread | /reject <i>feedback</i> to revise")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML", reply_markup=keyboard)
        return

    # --- Report display ---
    if draft_format == "report" and draft.get("_report_path"):
        report_path = draft["_report_path"]
        try:
            _report_data = await _aio.to_thread(Path(report_path).read_bytes)
            await update.message.reply_document(
                document=io.BytesIO(_report_data),
                filename=Path(report_path).name,
                caption=f"Report: {_esc(draft.get('title', 'Report'))}",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Failed to send report file: %s", e)
            await update.message.reply_text(
                f"Report generated at: <code>{_esc(report_path)}</code>",
                parse_mode="HTML",
            )
        return

    # --- Calendar display ---
    if draft_format == "calendar" and draft.get("_calendar_path"):
        cal_path = draft["_calendar_path"]
        try:
            _cal_data = await _aio.to_thread(Path(cal_path).read_bytes)
            await update.message.reply_document(
                document=io.BytesIO(_cal_data),
                filename=Path(cal_path).name,
                caption=f"Content Calendar: {_esc(draft.get('title', 'Calendar'))}",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Failed to send calendar file: %s", e)
            await update.message.reply_text(
                f"Calendar saved to: <code>{_esc(cal_path)}</code>",
                parse_mode="HTML",
            )
        return

    # Ensure the compositor has title/subtitle — synthesize from caption if missing
    if not draft.get("title") and not draft.get("subtitle") and caption:
        sentences = caption.split(". ", 1)
        draft["title"] = sentences[0].rstrip(".")
        draft["subtitle"] = sentences[1] if len(sentences) > 1 else ""

    # Inline keyboard for quick actions
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Approve", callback_data="draft_approve"),
            InlineKeyboardButton("Reject", callback_data="draft_reject"),
        ],
        [
            InlineKeyboardButton("Edit", callback_data="draft_edit"),
            InlineKeyboardButton("Reroll", callback_data="draft_reroll"),
        ],
    ])

    review = _random.choice(_REVIEW_PROMPTS)

    # --- Multi-option path (N>1 images) ---
    if image_urls and len(image_urls) > 1:
        # Compose all options in parallel for faster response
        compose_tasks = [_maybe_compose(draft, url, content_type) for url in image_urls]
        compose_results = await _asyncio.gather(*compose_tasks, return_exceptions=True)

        for idx, result in enumerate(compose_results, 1):
            if isinstance(result, Exception):
                logger.warning("Failed to compose option %d: %s", idx, result)
                continue
            photo, composed = result

            # Save composed image for approve-time archiving (save last option)
            if composed and isinstance(composed, io.BytesIO):
                try:
                    tmp_fd_c = tempfile.NamedTemporaryFile(suffix=".png", prefix=f"brandmover_composed_opt{idx}_", delete=False)
                    tmp_composed = tmp_fd_c.name
                    tmp_fd_c.close()
                    _data = composed.getvalue()
                    await _aio.to_thread(Path(tmp_composed).write_bytes, _data)
                    composed.seek(0)
                    if idx == 1:
                        state.set_last_composed(tmp_composed, content_type, user_id=user_id)
                except Exception as e:
                    logger.warning("Failed to save composed option %d: %s", idx, e)

            photo = _prepare_photo(photo)
            opt_caption = f"<b>Option {idx} of {len(image_urls)}</b>"
            if photo:
                try:
                    await update.message.reply_photo(
                        photo=photo,
                        caption=opt_caption,
                        parse_mode="HTML",
                    )
                except Exception as e:
                    logger.warning("Failed to send option %d image: %s", idx, e)
            else:
                logger.warning("Option %d image unavailable", idx)

        # Send summary text after all options
        approve_hints = " | ".join(f"/approve {i}" for i in range(1, len(image_urls) + 1))
        text_msg = f"{_esc(caption)}\n\n{approve_hints}\n/reject <i>feedback</i> to revise"
        await update.message.reply_text(text_msg, parse_mode="HTML")
        return

    # --- Single image path ---
    if image_url:
        photo, composed = await _maybe_compose(draft, image_url, content_type)

        # Save composed image to temp for approve-time archiving
        if composed and isinstance(composed, io.BytesIO):
            try:
                tmp_fd_c = tempfile.NamedTemporaryFile(suffix=".png", prefix="brandmover_last_composed_", delete=False)
                tmp_composed = tmp_fd_c.name
                tmp_fd_c.close()
                _data = composed.getvalue()
                await _aio.to_thread(Path(tmp_composed).write_bytes, _data)
                composed.seek(0)  # reset for Telegram send
                state.set_last_composed(tmp_composed, content_type, user_id=user_id)
            except Exception as e:
                logger.warning("Failed to save composed image for archiving: %s", e)

        photo = _prepare_photo(photo)
        if photo:
            # Slim caption: just the post text + review prompt
            photo_caption = f"{_esc(caption)}\n\n{review}"
            try:
                await update.message.reply_photo(
                    photo=photo,
                    caption=photo_caption[:1024],
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
            except Exception as e:
                logger.warning("Failed to send image via Telegram: %s \u2014 sending text only", e)
                text_msg = f"{_esc(caption)}\n\n<i>(image unavailable)</i>\n\n{review}"
                await update.message.reply_text(text_msg, parse_mode="HTML", reply_markup=keyboard)
        else:
            text_msg = f"{_esc(caption)}\n\n<i>(image unavailable)</i>\n\n{review}"
            await update.message.reply_text(text_msg, parse_mode="HTML", reply_markup=keyboard)
    else:
        # Text-only draft
        text_msg = f"{_esc(caption)}\n\n{review}"
        await update.message.reply_text(text_msg, parse_mode="HTML", reply_markup=keyboard)

    # Track context — draft was sent
    try:
        user_id = update.effective_user.id if update.effective_user else 0
        if user_id:
            conversation_context.update_context(
                user_id,
                last_bot_action="sent_draft",
                pending_draft_exists=True,
                last_content_type=content_type,
            )
    except Exception as e:
        logger.debug("Context tracking failed in _send_draft: %s", e)


# ---------------------------------------------------------------------------
# Inline draft button callbacks
# ---------------------------------------------------------------------------


class _CallbackProxy:
    """Lightweight proxy so callback query responses go through query.message."""
    def __init__(self, update, query):
        self._update = update
        self.message = query.message
        self.effective_user = query.from_user
    def __getattr__(self, name):
        return getattr(self._update, name)


async def draft_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses on draft messages."""
    query = update.callback_query
    await query.answer()

    cb_user_id = query.from_user.id
    if not _can_operate(cb_user_id):
        return

    action = query.data.split("_", 1)[1]  # approve|reject|edit|reroll

    # Proxy so _do_approve/_do_reject reply via query.message
    proxy = _CallbackProxy(update, query)

    if action == "approve":
        await _do_approve(proxy, context, source="button")
    elif action == "reject":
        await query.message.reply_text(
            "What should I change? Reply with your feedback, e.g.:\n"
            "<i>make it more urgent and add a CTA</i>",
            parse_mode="HTML",
        )
    elif action == "edit":
        await query.message.reply_text(
            "What should I edit? Reply with your feedback, e.g.:\n"
            "<i>change the background to blue</i>",
            parse_mode="HTML",
        )
    elif action == "reroll":
        if _rate_limited(cb_user_id):
            await query.message.reply_text(
                f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
            )
            return
        pending = state.get_pending(user_id=cb_user_id)
        if pending:
            original = pending.get("original_request", "")
            state.clear_pending(user_id=cb_user_id)
            state.clear_draft_history(user_id=cb_user_id)
            await query.message.reply_text("Regenerating...")
            if original:
                # Import generation handlers lazily to avoid circular imports
                from bot.handlers.generation import _handle_unified, _handle_agent_mode, _handle_pipeline_mode
                if settings.UNIFIED_BRAIN_ENABLED:
                    await _handle_unified(proxy, context, original, user_id=cb_user_id)
                elif settings.AGENT_MODE == "agent":
                    await _handle_agent_mode(proxy, original, user_id=cb_user_id)
                else:
                    await _handle_pipeline_mode(proxy, original, user_id=cb_user_id)


# ---------------------------------------------------------------------------
# Approve/reject/edit commands
# ---------------------------------------------------------------------------


async def approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /approve [N] — approve the pending draft and log feedback."""
    if not _can_operate(update.effective_user.id):
        return

    # Parse optional option number from "/approve N"
    text = (update.message.text or "").strip()
    parts = text.split()
    option_num = 1
    if len(parts) >= 2:
        try:
            option_num = int(parts[1])
        except ValueError:
            pass

    user_id = update.effective_user.id
    async with _get_approve_lock(user_id):
        await _do_approve(update, context, option_num=option_num, source="command")


async def reject_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reject [reason] — revise draft with feedback."""
    if not _can_operate(update.effective_user.id):
        return

    # Extract feedback after "/reject"
    text = update.message.text or ""
    feedback_text = text.partition("/reject")[2].strip()

    await _do_reject(update, context, feedback_text=feedback_text, source="command")


async def refine_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /refine <instruction> — focused edit on the current pending draft.

    Unlike /reject, this does NOT reject the draft or log negative feedback.
    It applies a targeted edit and preserves conversation history.
    """
    user_id = update.effective_user.id
    if not _can_operate(user_id):
        return

    text = (update.message.text or "").strip()
    instruction = text.partition("/refine")[2].strip()

    if not instruction:
        await update.message.reply_text(
            "Usage: /refine <i>instruction</i>\n\n"
            "Examples:\n"
            "  /refine make the tone more casual\n"
            "  /refine shorten to under 100 chars\n"
            "  /refine add a call to action",
            parse_mode="HTML",
        )
        return

    pending = state.get_pending(user_id=user_id)
    if not pending:
        await update.message.reply_text("Nothing to refine. Send me a content request first.")
        return

    await update.message.chat.send_action("typing")

    from agent.refinement import refine_artifact, extract_artifact_from_pending

    artifact = extract_artifact_from_pending(user_id=user_id)
    if not artifact:
        await update.message.reply_text("Could not load pending draft.")
        return

    history = artifact.pop("conversation_history", None)

    # Progress display — same pattern as agent revision
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
                pass

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
        result = await refine_artifact(
            artifact=artifact,
            instruction=instruction,
            history=history,
            on_tool_call=on_tool_call,
            on_reasoning=on_reasoning,
        )

        # Clean up status message
        if _status_msg:
            try:
                await _status_msg.delete()
            except Exception:
                pass

        if not result.draft:
            await update.message.reply_text(
                f"Refinement couldn't produce a valid draft.\n\n<pre>{_esc(result.final_text[:500])}</pre>",
                parse_mode="HTML",
            )
            return

        # Save refined draft — preserve all metadata from original pending
        image_url = result.image_url or pending.get("image_url")
        state.save_pending(
            caption=result.draft["caption"],
            hashtags=result.draft.get("hashtags", []),
            image_url=image_url,
            alt_text=result.draft.get("alt_text", ""),
            image_prompt=result.draft.get("image_prompt", ""),
            original_request=pending["original_request"],
            auto_slot=pending.get("auto_slot"),
            auto_event_ids=pending.get("auto_event_ids"),
            content_type=result.draft.get("content_type", pending.get("content_type")),
            user_id=user_id,
            conversation_history=result.conversation_history,
            draft_format=result.draft.get("format", pending.get("format", "single")),
            format_data=result.draft.get("format_data", pending.get("format_data")),
        )

        await _send_draft(update, result.draft, image_url, resources=result.resources, user_id=user_id)

    except Exception as e:
        logger.error("Refinement failed: %s", e)
        # Original pending is still intact (save_pending archives, but we
        # only call it on success above)
        await update.message.reply_text(
            f"Refinement failed: {_esc(str(e))}\n\nOriginal draft still pending. Try again or /cancel.",
            parse_mode="HTML",
        )


async def edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /edit <feedback> — surgical img2img edit on the last generated image."""
    user_id = update.effective_user.id
    if not _can_operate(user_id):
        return

    text = (update.message.text or "").strip()
    feedback_text = text.partition("/edit")[2].strip()
    if not feedback_text:
        await update.message.reply_text(
            "Usage: /edit <i>make the background darker</i>\n\n"
            "This applies a light img2img edit to the last generated image.",
            parse_mode="HTML",
        )
        return

    if _rate_limited(user_id):
        await update.message.reply_text(
            f"Please wait {_RATE_LIMIT_SECONDS}s between requests."
        )
        return

    last_url, content_type = state.get_last_generated(user_id=user_id)
    if not last_url:
        await update.message.reply_text("No image to edit \u2014 generate one first with a brand_3d request.")
        return

    await update.message.chat.send_action("upload_photo")
    await update.message.reply_text(f"\U0001F58C Editing: {_esc(feedback_text)}", parse_mode="HTML")

    try:
        # Download the last generated image to a temp file
        from agent._client import get_httpx
        tmp_fd = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="edit_ref_", delete=False)
        tmp_path = tmp_fd.name
        tmp_fd.close()
        client = get_httpx()
        resp = await client.get(last_url)
        resp.raise_for_status()
        await _aio.to_thread(lambda: _PILImage.open(io.BytesIO(resp.content)).convert("RGB").save(tmp_path, "JPEG", quality=95))

        # Build edit prompt with brand constraints
        edit_prompt = (
            f"Edit this image: {feedback_text}. "
            f"Keep everything else identical. Maintain the brand's visual style."
        )

        # Low strength for surgical edits
        url = await image_gen.generate_img2img(edit_prompt, tmp_path, strength=0.2)

        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("Temp cleanup failed for %s: %s", tmp_path, e)

        if not url:
            await update.message.reply_text("Edit failed \u2014 image generation returned no result. Try again.")
            return

        # Save as new last generated
        state.save_last_generated(url, content_type or "brand_3d", user_id=user_id)

        # Get existing pending draft for compositing context
        pending = state.get_pending(user_id=user_id)
        ct = content_type or "brand_3d"

        # Build a draft dict the compositor can use (needs title + subtitle)
        if pending:
            draft = dict(pending)
            ct = content_type or pending.get("content_type", ct)
        else:
            draft = {"caption": "Edited image", "alt_text": "Edited brand image"}

        # Ensure the compositor has title/subtitle — synthesize from caption if missing
        if not draft.get("title") and not draft.get("subtitle"):
            caption_text = draft.get("caption", "")
            if caption_text:
                # First sentence or first 60 chars -> title, rest -> subtitle
                sentences = caption_text.split(". ", 1)
                draft["title"] = sentences[0].rstrip(".")
                draft["subtitle"] = sentences[1] if len(sentences) > 1 else ""

        photo, composed = await _maybe_compose(draft, url, ct)

        # Save composed for archiving
        if composed and isinstance(composed, io.BytesIO):
            try:
                tmp_fd_c = tempfile.NamedTemporaryFile(suffix=".png", prefix="brandmover_edit_composed_", delete=False)
                tmp_composed = tmp_fd_c.name
                tmp_fd_c.close()
                _data = composed.getvalue()
                await _aio.to_thread(Path(tmp_composed).write_bytes, _data)
                composed.seek(0)
                photo = composed  # reset after reading for save
                state.set_last_composed(tmp_composed, ct, user_id=user_id)
            except Exception as e:
                logger.debug("Failed to save edit composed image: %s", e)

        # Update pending with the new image URL
        if pending:
            state.save_pending(
                caption=pending.get("caption", ""),
                hashtags=pending.get("hashtags", []),
                image_url=url,
                alt_text=pending.get("alt_text", ""),
                image_prompt=pending.get("image_prompt", ""),
                original_request=pending.get("original_request", ""),
                user_id=user_id,
            )

        photo = _prepare_photo(photo)
        edit_keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Approve", callback_data="draft_approve"),
                InlineKeyboardButton("Reject", callback_data="draft_reject"),
            ],
            [
                InlineKeyboardButton("Edit", callback_data="draft_edit"),
                InlineKeyboardButton("Reroll", callback_data="draft_reroll"),
            ],
        ])
        review = _random.choice(_REVIEW_PROMPTS)
        if photo:
            edit_caption = f"<b>Edited</b>: {_esc(feedback_text)}\n\n{review}"
            try:
                await update.message.reply_photo(
                    photo=photo,
                    caption=edit_caption[:1024],
                    parse_mode="HTML",
                    reply_markup=edit_keyboard,
                )
            except Exception as e:
                logger.warning("Failed to send edited image: %s", e)
                await update.message.reply_text(
                    f"<b>Edited</b>: {_esc(feedback_text)}\n\n<i>(image unavailable)</i>\n\n{review}",
                    parse_mode="HTML",
                    reply_markup=edit_keyboard,
                )
        else:
            await update.message.reply_text(
                f"<b>Edited</b>: {_esc(feedback_text)}\n\n<i>(image unavailable)</i>\n\n{review}",
                parse_mode="HTML",
                reply_markup=edit_keyboard,
            )

    except Exception as e:
        logger.error("Edit command failed: %s", e)
        await update.message.reply_text(
            f"Edit failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel — clear pending draft."""
    user_id = update.effective_user.id
    if not _can_operate(user_id):
        return

    has_pending = state.has_pending(user_id=user_id)
    has_approved = state.has_approved(user_id=user_id)

    if not has_pending and not has_approved:
        await update.message.reply_text("Nothing to cancel \u2014 no pending or approved draft.")
        return

    state.clear_pending(user_id=user_id)
    state.clear_approved(user_id=user_id)
    state.clear_draft_history(user_id=user_id)
    await update.message.reply_text("Draft cancelled. Send a new request whenever you're ready.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show pending draft info."""
    user_id = update.effective_user.id
    if not _can_operate(user_id):
        return

    pending = state.get_pending(user_id=user_id)
    if not pending:
        await update.message.reply_text("No pending draft. Send me a content request to get started.")
        return

    age = int(time.time() - pending.get("timestamp", 0))
    minutes = age // 60
    revision = state.get_draft_revision_count(user_id=user_id)
    rev_tag = f" (revision {revision})" if revision > 1 else ""
    msg = (
        f"<b>Pending Draft{_esc(rev_tag)}</b>\n\n"
        f"<b>Request:</b> {_esc(pending['original_request'])}\n\n"
        f"<b>Caption:</b>\n{_esc(pending['caption'])}\n\n"
        f"<b>Image:</b> {'Yes' if pending.get('image_url') else 'No'}\n"
        f"<b>Waiting:</b> {minutes} min\n\n"
        f"Reply /approve to post or /reject <i>feedback</i> to revise."
    )
    await update.message.reply_text(msg, parse_mode="HTML")
