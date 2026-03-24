"""
Debug/diagnostic handlers — /logs, /export, /status, /analytics, /history,
/feedback, /learn, /review, /style, /pref, /unpref, /preferences, /topics,
/heartbeat, /library, /skills, /strategy, /preview.
"""

__all__ = [
    "feedback_command",
    "learn_command",
    "pref_command",
    "unpref_command",
    "preferences_command",
    "topics_command",
    "heartbeat_command",
    "review_command",
    "style_command",
    "history_command",
    "analytics_command",
    "library_command",
    "skills_command",
    "strategy_command",
    "preview_command",
]

import asyncio as _aio
import logging
import time
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from agent import asset_library, feedback, generation_history, guidelines, state
from agent import compositor_config as _cc
from config import settings

from bot.handlers.core import (
    _authorized,
    _can_operate,
    _esc,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# /feedback, /learn
# ---------------------------------------------------------------------------


async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /feedback — show feedback stats."""
    if not _authorized(update.effective_user.id):
        return

    stats = feedback.get_feedback_stats()
    await update.message.reply_text(
        f"<b>Feedback Stats</b>\n\n<pre>{_esc(stats)}</pre>",
        parse_mode="HTML",
    )


async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /learn — trigger preference learning from feedback history."""
    if not _authorized(update.effective_user.id):
        return

    await update.message.chat.send_action("typing")

    try:
        summary = await feedback.summarize_preferences()
        await update.message.reply_text(
            f"<b>Preferences Updated</b>\n\n{_esc(summary[:2000])}",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Learn command failed: %s", e)
        await update.message.reply_text(
            f"Failed to summarize preferences: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /pref, /unpref, /preferences
# ---------------------------------------------------------------------------


async def pref_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /pref — add a learned preference to session memory."""
    if not _authorized(update.effective_user.id):
        return

    text = update.message.text or ""
    pref = text.split(maxsplit=1)[1].strip() if " " in text else ""
    if not pref:
        await update.message.reply_text(
            "Usage: /pref <i>prefer shorter captions with more emoji</i>",
            parse_mode="HTML",
        )
        return

    from agent.session import add_learned_preference
    added = add_learned_preference(pref)
    if added:
        await update.message.reply_text(f"Preference added: <i>{_esc(pref)}</i>", parse_mode="HTML")
    else:
        await update.message.reply_text("That preference already exists.")


async def unpref_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /unpref — remove a learned preference by index."""
    if not _authorized(update.effective_user.id):
        return

    text = update.message.text or ""
    parts = text.split()
    if len(parts) < 2 or not parts[1].isdigit():
        await update.message.reply_text(
            "Usage: /unpref <i>2</i>  (use /preferences to see indices)",
            parse_mode="HTML",
        )
        return

    idx = int(parts[1]) - 1  # 1-indexed for user, 0-indexed internally
    from agent.session import remove_learned_preference
    removed = remove_learned_preference(idx)
    if removed:
        await update.message.reply_text(f"Removed preference: <i>{_esc(removed)}</i>", parse_mode="HTML")
    else:
        await update.message.reply_text("Invalid index. Use /preferences to see the list.")


async def preferences_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /preferences — list learned preferences."""
    if not _authorized(update.effective_user.id):
        return

    from agent.session import load_session
    session = load_session()
    prefs = session.learned_preferences
    if not prefs:
        await update.message.reply_text(
            "No learned preferences yet. Add one with /pref <i>your preference</i>",
            parse_mode="HTML",
        )
        return

    lines = [f"  {i + 1}. {_esc(p)}" for i, p in enumerate(prefs)]
    await update.message.reply_text(
        f"<b>Learned Preferences</b>\n\n" + "\n".join(lines) +
        "\n\nRemove with /unpref <i>number</i>",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /topics
# ---------------------------------------------------------------------------


async def topics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /topics — manage the topic bank. Subcommands: refresh, add, retire."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    sub = parts[1].strip() if len(parts) > 1 else ""

    # /topics refresh
    if sub.lower() == "refresh":
        await update.message.chat.send_action("typing")
        try:
            from agent.topic_refresh import refresh_topic_bank
            result = await refresh_topic_bank()
            await update.message.reply_text(
                f"Topic bank refreshed: +{result['added']} added, -{result['retired']} retired.",
            )
        except Exception as e:
            logger.error("Topic refresh failed: %s", e)
            await update.message.reply_text(f"Refresh failed: {_esc(str(e))}", parse_mode="HTML")
        return

    # /topics add <category> | <angle>
    if sub.lower().startswith("add "):
        raw = sub[4:].strip()
        if "|" not in raw:
            await update.message.reply_text(
                "Usage: /topics add <i>category</i> | <i>angle description</i>",
                parse_mode="HTML",
            )
            return
        cat, angle_text = raw.split("|", 1)
        cat = cat.strip()
        angle_text = angle_text.strip()
        if not cat or not angle_text:
            await update.message.reply_text("Both category and angle are required.")
            return
        from agent.topic_bank import add_angle
        angle_id = add_angle(cat, angle_text)
        await update.message.reply_text(
            f"Added angle <code>{_esc(angle_id)}</code> to category <b>{_esc(cat)}</b>",
            parse_mode="HTML",
        )
        return

    # /topics retire <angle_id>
    if sub.lower().startswith("retire "):
        angle_id = sub[7:].strip()
        if not angle_id:
            await update.message.reply_text("Usage: /topics retire <i>angle_id</i>", parse_mode="HTML")
            return
        from agent.topic_bank import retire_angle
        if retire_angle(angle_id):
            await update.message.reply_text(f"Retired angle: <code>{_esc(angle_id)}</code>", parse_mode="HTML")
        else:
            await update.message.reply_text(f"Angle not found: {_esc(angle_id)}")
        return

    # /topics (no subcommand) — show summary
    from agent.topic_bank import load_bank, get_fresh_angles, seed_bank_if_empty
    from agent.session import _relative_time

    seed_bank_if_empty()
    bank = load_bank()

    # Count per category
    cat_counts: dict[str, int] = {}
    active_count = 0
    for a in bank.angles:
        if not a.get("retired", False):
            cat = a.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            active_count += 1

    cats_line = ", ".join(f"{c}: {n}" for c, n in sorted(cat_counts.items()))
    refreshed = _relative_time(bank.last_refreshed) if bank.last_refreshed else "never"

    # Next 3 suggested
    fresh = get_fresh_angles(3)
    suggestions = ""
    if fresh:
        suggestions = "\n\n<b>Next suggested angles:</b>\n"
        for a in fresh:
            suggestions += f"\u2022 [{a.get('category')}] {_esc(a.get('angle', '')[:80])}\n"

    await update.message.reply_text(
        f"<b>Topic Bank</b>\n\n"
        f"Active angles: {active_count}\n"
        f"Categories: {_esc(cats_line)}\n"
        f"Last refreshed: {refreshed}"
        f"{suggestions}\n"
        f"\n/topics refresh \u2014 regenerate angles"
        f"\n/topics add <i>cat</i> | <i>angle</i>"
        f"\n/topics retire <i>id</i>",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /heartbeat
# ---------------------------------------------------------------------------


async def heartbeat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /heartbeat — show recent heartbeat log entries."""
    if not _authorized(update.effective_user.id):
        return

    from agent.heartbeat import get_recent_heartbeat_entries
    from agent.session import _relative_time

    entries = get_recent_heartbeat_entries(5)
    if not entries:
        await update.message.reply_text("No heartbeat log entries yet.")
        return

    lines = []
    for e in reversed(entries):
        ts = e.get("timestamp", 0)
        action = e.get("decision", "?")
        reason = e.get("reason", "")[:60]
        sigs = ", ".join(e.get("signals", []))
        claude = "claude" if e.get("used_claude_reasoning") else "fast"
        taken = "yes" if e.get("action_taken") else "no"
        lines.append(
            f"\u2022 {_relative_time(ts)} \u2014 <b>{_esc(action)}</b> ({claude})\n"
            f"  Signals: {_esc(sigs)} | Acted: {taken}\n"
            f"  {_esc(reason)}"
        )

    await update.message.reply_text(
        f"<b>Recent Heartbeat</b>\n\n" + "\n\n".join(lines),
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /review
# ---------------------------------------------------------------------------


async def review_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /review — trigger a self-review of agent performance."""
    if not _authorized(update.effective_user.id):
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Running self-review... analyzing feedback and generation history.")

    try:
        from agent.self_review import run_self_review
        from agent.self_review_scheduler import mark_review_complete

        result = await run_self_review()

        if result.get("error"):
            await update.message.reply_text(
                f"<b>Self-Review</b> (partial)\n\n{_esc(result['error'])}",
                parse_mode="HTML",
            )
            return

        mark_review_complete()

        stats = result.get("stats", {})
        approval_rate = stats.get("approval_rate", 0)
        avg_rejections = stats.get("avg_rejections_before_approval", 0)
        best_type = stats.get("best_content_type", "unknown")
        reasons = stats.get("common_rejection_reasons", [])

        insights = result.get("insights", [])
        top_insights = "\n".join(f"  - {i}" for i in insights[:5]) if insights else "  (none)"
        top_reasons = "\n".join(f"  - {r}" for r in reasons[:3]) if reasons else "  (none)"

        msg = (
            f"<b>Self-Review Complete</b>\n\n"
            f"<b>Stats:</b>\n"
            f"  Approval rate: {approval_rate * 100:.0f}%\n"
            f"  Avg rejections before approval: {avg_rejections:.1f}\n"
            f"  Best content type: {_esc(str(best_type))}\n\n"
            f"<b>Top rejection reasons:</b>\n{top_reasons}\n\n"
            f"<b>Key insights:</b>\n{top_insights}\n\n"
            f"Learned preferences updated."
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        logger.error("Review command failed: %s", e)
        await update.message.reply_text(
            f"Self-review failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /style
# ---------------------------------------------------------------------------


async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /style — manage style profiles."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    args = text.partition("/style")[2].strip().split()

    # /style — list all profiles
    if not args:
        profiles = state.list_profiles()
        if not profiles:
            await update.message.reply_text(
                "<b>Style Profiles</b>\n\n"
                "No profiles yet.\n\n"
                "<code>/style create &lt;name&gt; &lt;description&gt;</code> \u2014 create one\n"
                "<code>/style &lt;name&gt; &lt;content_type&gt;</code> \u2014 set active\n"
                "Upload a photo with the profile name as caption to add references.",
                parse_mode="HTML",
            )
            return

        lines = ["<b>Style Profiles</b>\n"]
        for p in profiles:
            active = ", ".join(p["active_for"]) if p["active_for"] else "none"
            lines.append(
                f"<b>{_esc(p['name'])}</b> \u2014 {_esc(p['description'])}\n"
                f"  images: {p['image_count']} | strength: {p['strength']} | active for: {active}"
            )
        await update.message.reply_text("\n\n".join(lines), parse_mode="HTML")
        return

    # /style create <name> <description...>
    if args[0] == "create":
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: <code>/style create &lt;name&gt; &lt;description&gt;</code>",
                parse_mode="HTML",
            )
            return
        name = args[1]
        description = " ".join(args[2:]) if len(args) > 2 else ""
        try:
            state.add_style_profile(name, description=description)
            await update.message.reply_text(
                f"Created style profile <b>{_esc(name)}</b>\n"
                f"Upload reference photos with caption <code>{_esc(name)}</code> to add images.",
                parse_mode="HTML",
            )
        except ValueError as e:
            await update.message.reply_text(f"Error: {_esc(str(e))}", parse_mode="HTML")
        return

    # First arg is a profile name
    profile_name = args[0]
    profiles = state.get_style_profiles()
    if profile_name not in profiles:
        await update.message.reply_text(
            f"Profile <b>{_esc(profile_name)}</b> not found. "
            f"Use <code>/style create {_esc(profile_name)} description</code> to create it.",
            parse_mode="HTML",
        )
        return

    # /style <name> info
    if len(args) >= 2 and args[1] == "info":
        p_data = profiles[profile_name]
        refs = state.get_profile_refs(profile_name)
        data = state._read_styles()
        active_for = [ct for ct, p in data["active"].items() if p == profile_name]
        active_str = ", ".join(active_for) if active_for else "none"
        await update.message.reply_text(
            f"<b>{_esc(profile_name)}</b>\n\n"
            f"<b>Description:</b> {_esc(p_data.get('description', ''))}\n"
            f"<b>Strength:</b> {p_data.get('strength', 0.3)}\n"
            f"<b>Prompt prefix:</b> {_esc(p_data.get('prompt_prefix', '') or '(none)')}\n"
            f"<b>Images:</b> {len(refs)}\n"
            f"<b>Active for:</b> {active_str}",
            parse_mode="HTML",
        )
        return

    # /style <name> remove
    if len(args) >= 2 and args[1] == "remove":
        state.remove_active_profile(profile_name)
        await update.message.reply_text(
            f"Removed <b>{_esc(profile_name)}</b> from all active mappings (images kept).",
            parse_mode="HTML",
        )
        return

    # /style <name> <content_type> — set active
    if len(args) >= 2:
        content_type = args[1]
        try:
            state.set_active_profile(content_type, profile_name)
            await update.message.reply_text(
                f"Set <b>{_esc(profile_name)}</b> as active style for <b>{_esc(content_type)}</b>",
                parse_mode="HTML",
            )
        except ValueError as e:
            await update.message.reply_text(f"Error: {_esc(str(e))}", parse_mode="HTML")
        return

    # Shouldn't reach here, but show info as fallback
    await update.message.reply_text(
        f"Usage:\n"
        f"<code>/style</code> \u2014 list profiles\n"
        f"<code>/style create &lt;name&gt; &lt;desc&gt;</code> \u2014 create\n"
        f"<code>/style &lt;name&gt; &lt;content_type&gt;</code> \u2014 set active\n"
        f"<code>/style &lt;name&gt; info</code> \u2014 details\n"
        f"<code>/style &lt;name&gt; remove</code> \u2014 deactivate",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /history, /analytics
# ---------------------------------------------------------------------------


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history — show generation stats and recent entries."""
    if not _authorized(update.effective_user.id):
        return

    stats = generation_history.get_generation_stats()
    recent = generation_history.get_recent_generations(5)

    lines = [f"<b>Generation History</b> ({stats['total']} total)\n"]

    if stats["by_status"]:
        status_parts = [f"{k}: {v}" for k, v in sorted(stats["by_status"].items())]
        lines.append(f"<b>By status:</b> {', '.join(status_parts)}")

    if stats["by_type"]:
        type_parts = [f"{k}: {v}" for k, v in sorted(stats["by_type"].items())]
        lines.append(f"<b>By type:</b> {', '.join(type_parts)}")

    if stats["by_model"]:
        model_parts = [f"{k}: {v}" for k, v in sorted(stats["by_model"].items())]
        lines.append(f"<b>By model:</b> {', '.join(model_parts)}")

    total_cost = stats.get("estimated_total_cost_usd", 0)
    if total_cost > 0:
        lines.append(f"<b>Est. total cost:</b> ${total_cost:.2f}")

    if recent:
        lines.append("\n<b>Recent:</b>")
        for e in recent:
            import datetime
            ts = datetime.datetime.fromtimestamp(e.get("timestamp", 0)).strftime("%m/%d %H:%M")
            status = e.get("status", "?")
            at = e.get("asset_type", e.get("content_type", "?"))
            req = e.get("original_request", "")[:50]
            lines.append(f"  [{status}] {ts} {at} \u2014 {_esc(req)}")

    if not stats["total"]:
        lines.append("No generations recorded yet.")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analytics — show approval rates by content type and model."""
    if not _authorized(update.effective_user.id):
        return

    data = generation_history.get_approval_analytics()
    lines = ["<b>Approval Rate Analytics</b>\n"]

    ct_data = data.get("by_content_type", {})
    if ct_data:
        lines.append("<b>By content type:</b>")
        for ct, stats in ct_data.items():
            total = stats["approved"] + stats["rejected"]
            lines.append(f"  {ct}: {stats['rate']:.0f}% ({stats['approved']}/{total})")
    else:
        lines.append("No reviewed drafts yet.")

    model_data = data.get("by_model", {})
    if model_data:
        lines.append("\n<b>By model:</b>")
        for model, stats in model_data.items():
            total = stats["approved"] + stats["rejected"]
            lines.append(f"  {model}: {stats['rate']:.0f}% ({stats['approved']}/{total})")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /library, /skills, /strategy, /preview
# ---------------------------------------------------------------------------


async def library_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /library [query] — list or search the asset library."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    query = parts[1] if len(parts) > 1 else ""

    entries = asset_library.find(query=query, limit=10) if query else asset_library.list_all(limit=10)

    if not entries:
        await update.message.reply_text("Asset library is empty." if not query else f"No assets matching '{_esc(query)}'.")
        return

    lines = [f"<b>Asset Library</b> ({len(entries)} shown)\n"]
    for e in entries:
        used = f", used {e.used_count}x" if e.used_count else ""
        tags = f" [{', '.join(e.tags[:3])}]" if e.tags else ""
        prompt_short = (e.prompt[:40] + "...") if len(e.prompt) > 40 else e.prompt
        lines.append(f"<code>{e.id}</code> {e.source}/{e.content_type}{tags}{used}\n  {_esc(prompt_short)}")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def skills_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /skills — list available agent skills with stats."""
    if not _authorized(update.effective_user.id):
        return

    from agent.skills import load_registry, get_skill_stats

    registry = load_registry()
    active = [s for s in registry if s.get("status", "active") == "active"]

    if not active:
        await update.message.reply_text("No skills registered yet.")
        return

    lines = ["<b>Agent Skills</b>\n"]
    for s in active:
        stats = get_skill_stats(s["name"])
        uses = stats.get("uses", 0)
        rate = stats.get("approval_rate", 0)
        learnings = stats.get("learnings_count", 0)
        rate_str = f"{rate:.0%}" if uses > 0 else "n/a"
        lines.append(
            f"<b>{_esc(s['name'])}</b> \u2014 {_esc(s['description'])}\n"
            f"  uses: {uses} | approval: {rate_str} | learnings: {learnings}"
        )

    lines.append(f"\n<i>{len(active)} skills active</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def strategy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /strategy — show current brand strategy and config."""
    if not _authorized(update.effective_user.id):
        return

    config_path = Path(settings.BRAND_FOLDER) / "config.json"
    strategy_path = Path(settings.BRAND_FOLDER) / "strategy.md"

    if not config_path.exists() and not strategy_path.exists():
        await update.message.reply_text(
            "No strategy configured yet. Run /onboard to set up your brand."
        )
        return

    lines = ["<b>Brand Strategy</b>\n"]

    if config_path.exists():
        try:
            import json as _json
            cfg = _json.loads(await _aio.to_thread(config_path.read_text, encoding="utf-8"))
            pipeline = cfg.get("pipeline", {})
            lines.append(f"<b>Brand:</b> {_esc(cfg.get('brand_name', 'N/A'))}")
            lines.append(f"<b>Archetype:</b> {_esc(cfg.get('onboarding', {}).get('archetype', 'N/A'))}")
            lines.append(f"<b>Compositor:</b> {'ON' if pipeline.get('compositor_enabled') else 'OFF'}")
            badge = pipeline.get("badge_text")
            lines.append(f"<b>Badge:</b> {_esc(badge) if badge else '(none)'}")
            lines.append(f"<b>Mode:</b> {_esc(pipeline.get('default_mode', 'N/A'))}")
            platforms = cfg.get("platforms", [])
            if platforms:
                lines.append(f"<b>Platforms:</b> {', '.join(platforms)}")
            vs = cfg.get("visual_source", {})
            if vs:
                lines.append(f"<b>Visual source:</b> {_esc(vs.get('primary', 'N/A'))}")
            types = cfg.get("content_types_enabled", [])
            if types:
                lines.append(f"<b>Content types:</b> {', '.join(types[:8])}")
        except Exception as e:
            lines.append(f"<i>Error reading config.json: {_esc(str(e))}</i>")

    if strategy_path.exists():
        try:
            md = await _aio.to_thread(strategy_path.read_text, encoding="utf-8")
            preview = md[:500]
            if len(md) > 500:
                preview += "..."
            lines.append(f"\n<b>Strategy Notes:</b>\n<pre>{_esc(preview)}</pre>")
        except Exception as e:
            lines.append(f"<i>Error reading strategy.md: {_esc(str(e))}</i>")

    cal_path = Path(settings.BRAND_FOLDER) / "content_calendar.md"
    if cal_path.exists():
        lines.append("\nContent calendar available \u2014 see <code>brand/content_calendar.md</code>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def preview_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /preview [topic] — generate a sample post without rate limits or history."""
    if not _authorized(update.effective_user.id):
        return

    topic = " ".join(context.args) if context.args else ""
    if not topic:
        await update.message.reply_text(
            "Usage: /preview <topic>\n\n"
            "Example: /preview weekly product update"
        )
        return

    await update.message.chat.send_action("typing")

    try:
        from agent import engine
        result = await engine.run_agent(request=topic)
        draft = result.draft

        if not draft.get("caption"):
            await update.message.reply_text("Preview generation failed \u2014 no caption produced.")
            return

        lines = [
            "<b>Preview</b>\n",
            f"{_esc(draft['caption'])}",
        ]
        hashtags = draft.get("hashtags", [])
        if hashtags:
            lines.append(f"\n{' '.join('#' + h for h in hashtags)}")
        if draft.get("content_type"):
            lines.append(f"\n<i>Type: {_esc(draft['content_type'])}</i>")
        if draft.get("image_prompt"):
            lines.append(f"<i>Image prompt: {_esc(draft['image_prompt'][:150])}</i>")

        lines.append("\n<i>This is a preview \u2014 not saved or tracked.</i>")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error("Preview error: %s", e)
        await update.message.reply_text(
            f"Preview failed: {_esc(str(e))}",
            parse_mode="HTML",
        )
