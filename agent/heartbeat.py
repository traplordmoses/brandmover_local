"""
Heartbeat reasoning layer.
Cheap Python gate + optional Claude reasoning call to decide what to do.

Replaces the dumb cron loop with an assess → reason → dispatch cycle.
"""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum

import anthropic

from pathlib import Path

from agent import auto_state, schedule_queue, scheduler, state
from agent.session import build_session_context, load_session
from config import settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT_LOG_PATH = _PROJECT_ROOT / "state" / "heartbeat_log.jsonl"

# Gate daily_maintenance() to run at most once per calendar day.
_last_maintenance_date: str | None = None


class HeartbeatAction(Enum):
    SLEEP = "sleep"
    GENERATE_SLOT = "generate_slot"
    GENERATE_SCHEDULED = "generate_scheduled"
    REVISE_DRAFT = "revise_draft"
    PROACTIVE = "proactive"


@dataclass
class HeartbeatDecision:
    action: HeartbeatAction
    reason: str
    slot_name: str | None = None
    slot_config: dict | None = None
    scheduled_item: dict | None = None
    prompt_override: str | None = None
    feedback: str | None = None
    topic_angle_id: str | None = None


# ---------------------------------------------------------------------------
# Signal detection — pure Python, no API calls
# ---------------------------------------------------------------------------

async def cheap_gate() -> list[dict]:
    """
    Pure Python signal check. No API calls. Returns a list of hot signals.
    Each signal: {"type": str, "priority": int, "data": dict}

    Priority: 1 = urgent, 2 = normal, 3 = low
    """
    signals: list[dict] = []

    # 1. Due scheduled items (priority 2)
    try:
        from scripts.auto_post import SCHEDULER_INTERVAL_SECONDS
        due_items = schedule_queue.get_due_items(
            window_seconds=SCHEDULER_INTERVAL_SECONDS
        )
        for item in due_items:
            signals.append({
                "type": "pending_scheduled",
                "priority": 2,
                "data": {
                    "item_id": item["id"],
                    "prompt": item.get("prompt", "")[:200],
                    "label": item.get("label", ""),
                    "item": item,
                },
            })
    except Exception as e:
        logger.debug("cheap_gate: schedule_queue check failed: %s", e)

    # 2. Due predefined slots (priority 2)
    try:
        schedule = scheduler.load_schedule()
        due_slots = scheduler.get_due_slots(schedule)
        slots_config = schedule.get("slots", {})
        for slot_name in due_slots:
            signals.append({
                "type": "due_slot",
                "priority": 2,
                "data": {
                    "slot_name": slot_name,
                    "slot_config": slots_config.get(slot_name, {}),
                },
            })
    except Exception as e:
        logger.debug("cheap_gate: scheduler check failed: %s", e)

    # 3. Proactive trigger (priority 3) — only if auto-posting enabled and not paused
    if settings.AUTO_POST_ENABLED and not auto_state.is_paused():
        try:
            session = load_session()
            last_ts = 0.0

            # Check recent_posts timestamps
            if session.recent_posts:
                last_ts = max(
                    p.get("timestamp", 0) for p in session.recent_posts
                )

            # Also check last_run
            if session.last_run:
                run_ts = session.last_run.get("timestamp", 0)
                last_ts = max(last_ts, run_ts)

            # Also check auto_state for last_post_timestamp
            try:
                auto_st = auto_state._read_state()
                auto_ts = auto_st.get("last_post_timestamp", 0)
                last_ts = max(last_ts, auto_ts)
            except Exception:
                pass

            if last_ts > 0:
                hours_since = (time.time() - last_ts) / 3600
                if hours_since >= settings.HEARTBEAT_PROACTIVE_HOURS:
                    signals.append({
                        "type": "proactive",
                        "priority": 3,
                        "data": {"hours_since_last": round(hours_since, 1)},
                    })
        except Exception as e:
            logger.debug("cheap_gate: proactive check failed: %s", e)

    # Sort by priority (lower = higher priority)
    signals.sort(key=lambda s: s["priority"])
    return signals


# ---------------------------------------------------------------------------
# Reasoning — decides what to do with the signals
# ---------------------------------------------------------------------------

async def heartbeat_reason(signals: list[dict]) -> HeartbeatDecision:
    """
    Decide what to do based on detected signals.

    Fast path: single unambiguous signal → dispatch directly, no Claude call.
    Slow path: multiple competing signals or proactive → Claude reasoning.
    """
    if not signals:
        return HeartbeatDecision(HeartbeatAction.SLEEP, reason="No signals")

    # --- Fast path: single signal, dispatch directly ---
    if len(signals) == 1:
        sig = signals[0]

        if sig["type"] == "due_slot":
            return HeartbeatDecision(
                HeartbeatAction.GENERATE_SLOT,
                reason=f"Single due slot: {sig['data']['slot_name']}",
                slot_name=sig["data"]["slot_name"],
                slot_config=sig["data"].get("slot_config"),
            )

        if sig["type"] == "pending_scheduled":
            return HeartbeatDecision(
                HeartbeatAction.GENERATE_SCHEDULED,
                reason=f"Single scheduled item: {sig['data']['item_id']}",
                scheduled_item=sig["data"]["item"],
            )

        if sig["type"] == "proactive":
            # Proactive requires Claude to decide what to post about
            return await _claude_reason(signals)

    # --- Slow path: multiple signals or ambiguous situation ---
    return await _claude_reason(signals)


async def _claude_reason(signals: list[dict]) -> HeartbeatDecision:
    """Use a lightweight Claude call to decide what to do."""
    from agent._client import get_anthropic
    from agent.topic_bank import get_fresh_angles, seed_bank_if_empty

    client = get_anthropic()

    session_context = build_session_context()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Ensure topic bank exists
    seed_bank_if_empty()

    # Get fresh topic angles for proactive posts
    fresh_angles = get_fresh_angles(n=5)
    angles_block = ""
    if fresh_angles and any(s["type"] == "proactive" for s in signals):
        angles_block = (
            f"\nAvailable content angles (least recently used):\n"
            f"{json.dumps(fresh_angles, indent=2, default=str)}\n"
        )

    # Build a clean version of signals for Claude (strip non-serializable data)
    clean_signals = []
    for sig in signals:
        clean = {"type": sig["type"], "priority": sig["priority"]}
        data = sig.get("data", {})
        # Only include serializable fields
        for k, v in data.items():
            if k != "item" and k != "slot_config":
                clean[k] = v
        clean_signals.append(clean)

    reasoning_prompt = (
        f"You are BrandMover's planning agent. Decide what to do right now.\n\n"
        f"{session_context}\n\n"
        f"Active signals:\n{json.dumps(clean_signals, indent=2, default=str)}\n"
        f"{angles_block}\n"
        f"Current time: {current_time}\n\n"
        f"Decide ONE action. Use the decide tool to submit your choice."
        f"{' For proactive posts, reference a topic angle ID and craft a specific prompt.' if angles_block else ''}"
    )

    decide_tool = {
        "name": "decide",
        "description": "Submit your decision on what action to take",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "generate_slot", "generate_scheduled",
                        "revise_draft", "proactive", "sleep",
                    ],
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why",
                },
                "slot_name": {
                    "type": "string",
                    "description": "For generate_slot — which slot to process",
                },
                "prompt_override": {
                    "type": "string",
                    "description": "For proactive — what should we post about?",
                },
                "topic_angle_id": {
                    "type": "string",
                    "description": "For proactive posts — which topic bank angle ID to use",
                },
            },
            "required": ["action", "reason"],
        },
    }

    try:
        response = await client.messages.create(
            model=settings.AGENT_MODEL,
            max_tokens=500,
            system="You are a concise planning agent. Pick the single best action.",
            tools=[decide_tool],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": reasoning_prompt}],
        )
    except anthropic.APIError as e:
        logger.error("Heartbeat Claude reasoning failed: %s", e)
        # Fallback: pick the highest-priority signal
        return _fallback_decision(signals)

    # Extract the decide tool call
    for block in response.content:
        if block.type == "tool_use" and block.name == "decide":
            inp = block.input
            action_str = inp.get("action", "sleep")
            reason = inp.get("reason", "Claude decided")

            try:
                action = HeartbeatAction(action_str)
            except ValueError:
                action = HeartbeatAction.SLEEP

            # Resolve slot_name and scheduled_item from signals
            slot_name = inp.get("slot_name")
            slot_config = None
            scheduled_item = None

            if action == HeartbeatAction.GENERATE_SLOT and slot_name:
                for sig in signals:
                    if (sig["type"] == "due_slot"
                            and sig["data"].get("slot_name") == slot_name):
                        slot_config = sig["data"].get("slot_config")
                        break

            if action == HeartbeatAction.GENERATE_SCHEDULED:
                for sig in signals:
                    if sig["type"] == "pending_scheduled":
                        scheduled_item = sig["data"].get("item")
                        break

            return HeartbeatDecision(
                action=action,
                reason=reason,
                slot_name=slot_name,
                slot_config=slot_config,
                scheduled_item=scheduled_item,
                prompt_override=inp.get("prompt_override"),
                topic_angle_id=inp.get("topic_angle_id"),
            )

    # No decide tool call found — fallback
    logger.warning("Heartbeat: Claude did not call decide tool, falling back")
    return _fallback_decision(signals)


def _fallback_decision(signals: list[dict]) -> HeartbeatDecision:
    """Fallback when Claude reasoning fails: pick highest-priority signal."""
    if not signals:
        return HeartbeatDecision(HeartbeatAction.SLEEP, reason="No signals")

    sig = signals[0]  # Already sorted by priority
    if sig["type"] == "due_slot":
        return HeartbeatDecision(
            HeartbeatAction.GENERATE_SLOT,
            reason=f"Fallback: due slot {sig['data']['slot_name']}",
            slot_name=sig["data"]["slot_name"],
            slot_config=sig["data"].get("slot_config"),
        )
    if sig["type"] == "pending_scheduled":
        return HeartbeatDecision(
            HeartbeatAction.GENERATE_SCHEDULED,
            reason=f"Fallback: scheduled item {sig['data']['item_id']}",
            scheduled_item=sig["data"].get("item"),
        )
    return HeartbeatDecision(
        HeartbeatAction.SLEEP,
        reason=f"Fallback: skipping {sig['type']} signal",
    )


# ---------------------------------------------------------------------------
# Observability log
# ---------------------------------------------------------------------------

def _log_heartbeat(entry: dict) -> None:
    """Append one JSON line to the heartbeat log."""
    entry["timestamp"] = time.time()
    HEARTBEAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(HEARTBEAT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        _prune_log()
    except Exception as e:
        logger.debug("Heartbeat log write failed: %s", e)


def _prune_log(max_lines: int = 500) -> None:
    """Keep only the last max_lines entries."""
    if not HEARTBEAT_LOG_PATH.exists():
        return
    try:
        lines = HEARTBEAT_LOG_PATH.read_text().strip().split("\n")
        if len(lines) > max_lines:
            HEARTBEAT_LOG_PATH.write_text("\n".join(lines[-max_lines:]) + "\n")
    except Exception:
        pass


def get_recent_heartbeat_entries(n: int = 5) -> list[dict]:
    """Read the last N heartbeat log entries."""
    if not HEARTBEAT_LOG_PATH.exists():
        return []
    try:
        lines = HEARTBEAT_LOG_PATH.read_text().strip().split("\n")
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Dispatch — executes the decision
# ---------------------------------------------------------------------------

async def heartbeat_tick(bot=None) -> bool:
    """
    Single heartbeat tick. Called by the scheduler loop.
    Returns True if an action was taken.
    """
    # Import here to avoid circular imports
    from scripts.auto_post import process_slot, process_scheduled_item

    # Always check for user-scheduled items, even if auto-posting is off
    signals = await cheap_gate()

    if not settings.AUTO_POST_ENABLED or auto_state.is_paused():
        # Only process user-scheduled items when auto-posting is off/paused
        signals = [s for s in signals if s["type"] == "pending_scheduled"]
        if not signals:
            return False

    if not signals:
        return False

    # Skip if there's already a pending draft awaiting review
    if await state.async_has_pending():
        logger.debug("Heartbeat: draft pending approval, skipping this cycle")
        return False

    # Get decision — track whether Claude reasoning was used
    used_claude = len(signals) > 1 or any(s["type"] == "proactive" for s in signals)
    decision = await heartbeat_reason(signals)

    if decision.action == HeartbeatAction.SLEEP:
        logger.info("Heartbeat: sleeping — %s", decision.reason)
        _log_heartbeat({
            "signals": [s["type"] for s in signals],
            "decision": decision.action.value,
            "reason": decision.reason,
            "slot": decision.slot_name,
            "used_claude_reasoning": used_claude,
            "action_taken": False,
        })
        return False

    logger.info(
        "Heartbeat: action=%s reason=%s",
        decision.action.value, decision.reason,
    )

    # Load global config for process_slot/process_scheduled_item
    schedule = scheduler.load_schedule()
    global_config = schedule.get("global", {})
    slots_config = schedule.get("slots", {})
    action_taken = False

    # --- Dispatch ---
    if decision.action == HeartbeatAction.GENERATE_SLOT:
        slot_name = decision.slot_name
        if not slot_name:
            logger.warning("Heartbeat: GENERATE_SLOT but no slot_name")
        else:
            slot_config = decision.slot_config or slots_config.get(slot_name, {})
            action_taken = await process_slot(
                slot_name, slot_config, global_config,
                dry_run=settings.AUTO_POST_DRY_RUN, bot=bot,
            )

    elif decision.action == HeartbeatAction.GENERATE_SCHEDULED:
        item = decision.scheduled_item
        if not item:
            logger.warning("Heartbeat: GENERATE_SCHEDULED but no item")
        else:
            action_taken = await process_scheduled_item(
                item, global_config,
                dry_run=settings.AUTO_POST_DRY_RUN, bot=bot,
            )

    elif decision.action == HeartbeatAction.PROACTIVE:
        prompt = decision.prompt_override
        if not prompt:
            logger.info("Heartbeat: PROACTIVE but no prompt — skipping")
        else:
            # Mark topic angle as used
            if decision.topic_angle_id:
                from agent.topic_bank import mark_angle_used
                mark_angle_used(decision.topic_angle_id)

            logger.info("Heartbeat: proactive generation — %s", prompt[:100])
            from agent import engine
            try:
                result = await engine.run_agent(request=prompt)
            except Exception as e:
                logger.error("Heartbeat proactive generation failed: %s", e)
                result = None

            if result and result.draft:
                caption = result.draft.get("caption", "")
                await state.async_save_pending(
                    caption=caption,
                    hashtags=result.draft.get("hashtags", []),
                    image_url=result.image_url,
                    alt_text=result.draft.get("alt_text", ""),
                    image_prompt=result.draft.get("image_prompt", ""),
                    original_request=prompt,
                    image_urls=result.image_urls if len(result.image_urls) > 1 else None,
                    auto_slot="proactive",
                    conversation_history=result.conversation_history,
                )

                if result.image_url:
                    state.save_last_generated(
                        result.image_url,
                        result.draft.get("content_type", "default"),
                    )

                if bot:
                    from bot.handlers import send_auto_draft
                    await send_auto_draft(bot, result.draft, result.image_url, "proactive")
                else:
                    from scripts.auto_post import _notify_telegram
                    await _notify_telegram(
                        f"<b>Proactive Draft Ready</b>\n\n"
                        f"{caption[:200]}\n\n"
                        f"/approve to post  |  /reject <i>feedback</i>  |  /cancel"
                    )

                logger.info("Heartbeat: proactive draft queued for approval")
                action_taken = True

    elif decision.action == HeartbeatAction.REVISE_DRAFT:
        logger.info("Heartbeat: revision requested but not yet implemented")

    # Log the heartbeat decision
    _log_heartbeat({
        "signals": [s["type"] for s in signals],
        "decision": decision.action.value,
        "reason": decision.reason,
        "slot": decision.slot_name,
        "topic_angle_id": decision.topic_angle_id,
        "used_claude_reasoning": used_claude,
        "action_taken": action_taken,
    })

    # Run periodic maintenance (once per day)
    await daily_maintenance(bot=bot)

    return action_taken


# ---------------------------------------------------------------------------
# Daily maintenance — periodic tasks that run once per calendar day
# ---------------------------------------------------------------------------

async def daily_maintenance(bot=None) -> None:
    """Run periodic housekeeping tasks, gated to once per calendar day.

    This fixes the gap where HEARTBEAT_ENABLED=true skips the periodic
    tasks that previously lived inside run_cron() (steps 3-6).
    """
    global _last_maintenance_date
    import datetime

    today = datetime.date.today().isoformat()
    if _last_maintenance_date == today:
        return
    _last_maintenance_date = today

    logger.info("Daily maintenance starting (%s)", today)

    # 1. Prune old scheduled queue items
    try:
        schedule_queue.prune_old()
    except Exception as e:
        logger.debug("Maintenance: queue prune failed: %s", e)

    # 2. Daily self-review check
    try:
        from agent.self_review_scheduler import maybe_trigger_daily_review
        await maybe_trigger_daily_review()
    except Exception as e:
        logger.debug("Maintenance: self-review check failed: %s", e)

    # 3. Topic bank refresh (if stale)
    try:
        from agent.topic_bank import load_bank, seed_bank_if_empty
        seed_bank_if_empty()
        bank = load_bank()
        hours_since_refresh = (time.time() - (bank.last_refreshed or 0)) / 3600
        if hours_since_refresh > settings.TOPIC_BANK_REFRESH_INTERVAL_HOURS:
            from agent.topic_refresh import refresh_topic_bank
            result = await refresh_topic_bank()
            logger.info("Maintenance: topic bank refreshed: %s", result)
    except Exception as e:
        logger.debug("Maintenance: topic bank refresh failed: %s", e)

    # 4. Auto preference extraction (if due)
    if settings.PREF_EXTRACTION_ENABLED:
        try:
            from agent.pref_extractor import extract_preferences
            new_prefs = await extract_preferences()
            if new_prefs:
                logger.info(
                    "Maintenance: auto-extracted %d preferences: %s",
                    len(new_prefs), new_prefs,
                )
                if bot:
                    msg = "<b>Auto-learned preferences</b>\n\n"
                    for p in new_prefs:
                        msg += f"\u2022 {p}\n"
                    msg += "\nUse /preferences to view all. /unpref <number> to remove any."
                    try:
                        await bot.send_message(
                            chat_id=settings.TELEGRAM_ALLOWED_USER_ID,
                            text=msg,
                            parse_mode="HTML",
                        )
                    except Exception as send_err:
                        logger.debug("Maintenance: pref notification failed: %s", send_err)
        except Exception as e:
            logger.debug("Maintenance: preference extraction failed: %s", e)

    logger.info("Daily maintenance complete")
