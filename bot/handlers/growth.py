"""Growth engine commands -- target management, reply drafting, growth reports."""

__all__ = [
    "growth_command",
    "targets_command",
    "target_add_command",
    "target_remove_command",
    "replies_command",
    "growth_report_command",
    "growth_callback",
]

import json
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from agent import growth_engine
from config import settings

from bot.handlers.core import (
    _authorized,
    _can_operate,
    _esc,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# /growth -- dashboard summary
# ---------------------------------------------------------------------------


async def growth_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show growth dashboard summary."""
    if not _can_operate(update.effective_user.id):
        return

    dashboard = growth_engine.get_growth_dashboard()
    weekly = dashboard["weekly"]
    monthly = dashboard["monthly"]

    lines = ["<b>Growth Dashboard</b>\n"]

    # Follower stats
    if weekly.get("current") is not None:
        lines.append(f"Followers: <b>{weekly['current']:,}</b>")
        if weekly.get("change") is not None:
            sign = "+" if weekly["change"] >= 0 else ""
            lines.append(
                f"This week: {sign}{weekly['change']:,} "
                f"({sign}{weekly['pct_change']}%)"
            )
        if monthly.get("change") is not None:
            sign = "+" if monthly["change"] >= 0 else ""
            lines.append(
                f"This month: {sign}{monthly['change']:,} "
                f"({sign}{monthly['pct_change']}%)"
            )
    else:
        lines.append(
            "No follower data yet. Enable GROWTH_ENGINE_ENABLED=true "
            "to start tracking automatically."
        )

    lines.append(f"\nTargets monitored: {dashboard['target_count']}")
    lines.append(f"\n<i>{_esc(dashboard['suggestion'])}</i>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /targets -- list target accounts
# ---------------------------------------------------------------------------


async def targets_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List target accounts being monitored."""
    if not _can_operate(update.effective_user.id):
        return

    targets = growth_engine.list_targets()
    if not targets:
        await update.message.reply_text(
            "No target accounts.\n\n"
            "Add targets with:\n"
            "<code>/target_add @username reason for monitoring</code>",
            parse_mode="HTML",
        )
        return

    lines = ["<b>Target Accounts</b>\n"]
    for i, t in enumerate(targets, 1):
        reason = f" -- {_esc(t['reason'])}" if t.get("reason") else ""
        lines.append(f"{i}. @{_esc(t['username'])}{reason}")

    lines.append(f"\n{len(targets)} target(s) total.")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /target_add @username [reason]
# ---------------------------------------------------------------------------


async def target_add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add a target account."""
    if not _authorized(update.effective_user.id):
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/target_add @username reason for targeting</code>",
            parse_mode="HTML",
        )
        return

    username = args[0].lstrip("@")
    reason = " ".join(args[1:]) if len(args) > 1 else ""

    result = growth_engine.add_target(username, reason)
    if "error" in result:
        await update.message.reply_text(f"{_esc(result['error'])}")
    else:
        await update.message.reply_text(
            f"Added @{_esc(username)} to targets."
            + (f"\nReason: {_esc(reason)}" if reason else ""),
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /target_remove @username
# ---------------------------------------------------------------------------


async def target_remove_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove a target account."""
    if not _authorized(update.effective_user.id):
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/target_remove @username</code>",
            parse_mode="HTML",
        )
        return

    username = args[0].lstrip("@")
    if growth_engine.remove_target(username):
        await update.message.reply_text(f"Removed @{_esc(username)} from targets.")
    else:
        await update.message.reply_text(f"@{_esc(username)} is not in the target list.")


# ---------------------------------------------------------------------------
# /replies -- draft replies to target accounts
# ---------------------------------------------------------------------------


async def replies_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch recent tweets from targets and generate reply drafts."""
    if not _can_operate(update.effective_user.id):
        return

    targets = growth_engine.list_targets()
    if not targets:
        await update.message.reply_text(
            "No targets configured. Use /target_add first."
        )
        return

    status_msg = await update.message.reply_text("Fetching tweets from targets...")

    # Collect recent tweets from targets
    tweets_found: list[dict] = []
    try:
        from agent.publishing.publisher import _get_client_v2
        client = _get_client_v2()

        for t in targets[:5]:  # Limit to 5 targets per batch
            try:
                user_resp = client.get_user(username=t["username"])
                if not user_resp or not user_resp.data:
                    continue
                user_id = user_resp.data.id
                tweets_resp = client.get_users_tweets(
                    user_id,
                    max_results=5,
                    tweet_fields=["created_at", "text"],
                    exclude=["retweets", "replies"],
                )
                if tweets_resp and tweets_resp.data:
                    for tw in tweets_resp.data[:3]:
                        tweets_found.append({
                            "username": t["username"],
                            "tweet_id": str(tw.id),
                            "text": tw.text,
                        })
            except Exception as e:
                logger.debug("Failed to fetch tweets for @%s: %s", t["username"], e)
    except Exception as e:
        await status_msg.edit_text(
            f"Could not access Twitter API: {_esc(str(e)[:200])}\n\n"
            "Check your Twitter API credentials.",
            parse_mode="HTML",
        )
        return

    if not tweets_found:
        await status_msg.edit_text("No recent tweets found from target accounts.")
        return

    # Generate reply drafts using Haiku
    try:
        from agent._client import get_anthropic
        from agent import guidelines

        client = get_anthropic()
        brand_ctx = guidelines.get_brand_context()
        brand_snippet = brand_ctx[:500] if brand_ctx else "Professional brand voice."

        tweets_text = "\n\n".join(
            f"@{tw['username']}: {tw['text']}" for tw in tweets_found[:5]
        )

        prompt = (
            f"Generate short, engaging reply drafts for these tweets. "
            f"Stay in brand voice:\n{brand_snippet[:300]}\n\n"
            f"Tweets:\n{tweets_text}\n\n"
            f"For each tweet, write ONE concise reply (max 200 chars) that:\n"
            f"- Adds genuine value (insight, question, or agreement)\n"
            f"- Is conversational and authentic\n"
            f"- Positions the brand as a thought leader\n"
            f"- Does NOT hard-sell or shill\n\n"
            f"Return JSON array: [{{\"tweet_index\": int, \"reply\": str}}]"
        )

        response = await client.messages.create(
            model=settings.HAIKU_MODEL,
            max_tokens=1000,
            system="You are a social media engagement strategist. Write replies that build relationships.",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text if response.content else "[]"
        # Strip fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            replies = json.loads(cleaned)
        except json.JSONDecodeError:
            replies = []

    except Exception as e:
        await status_msg.edit_text(
            f"Reply generation failed: {_esc(str(e)[:200])}",
            parse_mode="HTML",
        )
        return

    if not replies:
        await status_msg.edit_text("Could not generate reply drafts.")
        return

    # Present each reply with inline buttons
    await status_msg.delete()
    for r in replies:
        idx = r.get("tweet_index", 0)
        reply_text = r.get("reply", "")
        if idx < len(tweets_found):
            tw = tweets_found[idx]
            msg_text = (
                f"<b>Reply to @{_esc(tw['username'])}</b>\n"
                f"<i>{_esc(tw['text'][:200])}</i>\n\n"
                f"Draft: {_esc(reply_text)}"
            )

            # Store reply data in context for callback
            reply_key = f"reply_{tw['tweet_id']}"
            context.bot_data[reply_key] = {
                "tweet_id": tw["tweet_id"],
                "reply_text": reply_text,
                "username": tw["username"],
            }

            buttons = [
                [
                    InlineKeyboardButton("Send", callback_data=f"growth_send_{tw['tweet_id']}"),
                    InlineKeyboardButton("Skip", callback_data=f"growth_skip_{tw['tweet_id']}"),
                    InlineKeyboardButton("Edit", callback_data=f"growth_edit_{tw['tweet_id']}"),
                ]
            ]
            await update.message.reply_text(
                msg_text,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(buttons),
            )


# ---------------------------------------------------------------------------
# /growth_report [days]
# ---------------------------------------------------------------------------


async def growth_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed growth analytics report."""
    if not _can_operate(update.effective_user.id):
        return

    args = context.args or []
    days = 30
    if args:
        try:
            days = int(args[0])
            days = max(1, min(365, days))
        except ValueError:
            pass

    history = growth_engine.get_follower_history(days=days)
    stats = growth_engine.get_growth_stats(days=days)
    targets = growth_engine.list_targets()

    lines = [f"<b>Growth Report ({days} days)</b>\n"]

    # Follower stats
    if stats.get("current") is not None:
        lines.append(f"Current followers: <b>{stats['current']:,}</b>")
        if stats.get("change") is not None:
            sign = "+" if stats["change"] >= 0 else ""
            lines.append(f"Change: {sign}{stats['change']:,} ({sign}{stats['pct_change']}%)")
        lines.append(f"Data points: {stats['data_points']}")
    else:
        lines.append("No follower data collected yet.")

    # Top performing content types (from generation history)
    try:
        from agent.generation_history import load_history
        hist = load_history()
        if hist:
            type_counts: dict[str, int] = {}
            approved_counts: dict[str, int] = {}
            for entry in hist:
                ct = entry.get("content_type", "unknown")
                type_counts[ct] = type_counts.get(ct, 0) + 1
                if entry.get("status") == "approved":
                    approved_counts[ct] = approved_counts.get(ct, 0) + 1

            if type_counts:
                lines.append("\n<b>Content Type Performance</b>")
                for ct in sorted(type_counts, key=type_counts.get, reverse=True)[:5]:
                    total = type_counts[ct]
                    approved = approved_counts.get(ct, 0)
                    rate = (approved / total * 100) if total > 0 else 0
                    lines.append(f"  {_esc(ct)}: {approved}/{total} approved ({rate:.0f}%)")
    except Exception:
        pass

    # Targets summary
    lines.append(f"\n<b>Targets</b>: {len(targets)} accounts monitored")
    for t in targets[:5]:
        lines.append(f"  @{_esc(t['username'])}")

    # Stalling check
    if growth_engine.is_growth_stalling():
        lines.append(
            "\n<b>Action needed:</b> Growth is stalling (<1% weekly). "
            "Consider a growth thread or more engagement."
        )

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# Inline button callback for reply actions
# ---------------------------------------------------------------------------


async def growth_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle growth-related inline button callbacks."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    parts = data.split("_", 2)  # growth_action_tweetid
    if len(parts) < 3:
        return

    action = parts[1]
    tweet_id = parts[2]
    reply_key = f"reply_{tweet_id}"
    reply_data = context.bot_data.get(reply_key)

    if action == "skip":
        await query.edit_message_text(
            f"{query.message.text}\n\n<i>Skipped.</i>",
            parse_mode="HTML",
        )
        context.bot_data.pop(reply_key, None)

    elif action == "send":
        if not reply_data:
            await query.edit_message_text("Reply data expired. Generate new replies with /replies.")
            return

        try:
            from agent.publishing.publisher import _get_client_v2
            client = _get_client_v2()
            client.create_tweet(
                text=reply_data["reply_text"],
                in_reply_to_tweet_id=int(tweet_id),
            )
            await query.edit_message_text(
                f"{query.message.text}\n\n<b>Sent!</b>",
                parse_mode="HTML",
            )
        except Exception as e:
            await query.edit_message_text(
                f"{query.message.text}\n\n<b>Failed:</b> {_esc(str(e)[:200])}",
                parse_mode="HTML",
            )
        context.bot_data.pop(reply_key, None)

    elif action == "edit":
        await query.edit_message_text(
            f"{query.message.text}\n\n"
            f"<i>Send your edited reply as a text message. "
            f"Start it with</i> <code>reply {tweet_id}</code>",
            parse_mode="HTML",
        )
