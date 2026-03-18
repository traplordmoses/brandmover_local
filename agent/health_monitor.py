"""
Health monitor -- checks system health and alerts on failures.

Runs periodic checks on:
- Last post recency (>24h without a post = degraded)
- Error rate from generation_history (>30% = degraded)
- State file integrity
- Pending drafts stale >2h

Alerts are rate-limited: max 1 per error_type per hour.
State lives in state/health_state.json via FileStore.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agent.paths import STATE_DIR
from agent.state_manager import FileStore

logger = logging.getLogger(__name__)

_STATE_FILE = STATE_DIR / "health_state.json"
_store = FileStore(_STATE_FILE, default_factory=lambda: {
    "last_alert_timestamps": {},
    "last_check_at": 0.0,
})

# How often to run health checks (seconds)
_CHECK_INTERVAL_SECONDS = 1800  # 30 minutes

# Alert rate limit: max 1 per error_type per hour
_ALERT_COOLDOWN_SECONDS = 3600


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    """Result of a health check run."""
    status: str  # "healthy" | "degraded" | "down"
    checks: dict[str, dict] = field(default_factory=dict)
    last_post_age_hours: float = 0.0
    error_rate_24h: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of health status."""
        lines = [f"Status: {self.status.upper()}"]
        if self.last_post_age_hours > 0:
            lines.append(f"Last post: {self.last_post_age_hours:.1f}h ago")
        lines.append(f"Error rate (24h): {self.error_rate_24h:.0f}%")
        for name, check in self.checks.items():
            icon = "OK" if check.get("ok") else "WARN"
            detail = check.get("detail", "")
            lines.append(f"  [{icon}] {name}: {detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

async def run_health_checks() -> HealthStatus:
    """Run all health checks and return a HealthStatus.

    Checks:
    1. last_post_recency: >24h without a published post = degraded
    2. error_rate: >30% generation failures in last 24h = degraded
    3. state_files: state directory and key files exist and are readable
    4. stale_drafts: pending draft older than 2h = warning
    """
    checks: dict[str, dict] = {}
    issues = 0

    # --- 1. Last post recency ---
    last_post_age_hours = 0.0
    try:
        from agent.auto_state import _read_state as read_auto_state
        auto = await asyncio.to_thread(read_auto_state)
        posts = auto.get("posts_today", [])
        # Also check historical posts
        last_ts = 0.0
        for p in posts:
            ts = p.get("timestamp", 0)
            if ts > last_ts:
                last_ts = ts
        if not last_ts:
            last_ts = auto.get("last_post_at", 0.0)

        if last_ts:
            last_post_age_hours = (time.time() - last_ts) / 3600
            if last_post_age_hours > 24:
                checks["last_post_recency"] = {
                    "ok": False,
                    "detail": f"No post in {last_post_age_hours:.0f}h (threshold: 24h)",
                }
                issues += 1
            else:
                checks["last_post_recency"] = {
                    "ok": True,
                    "detail": f"{last_post_age_hours:.1f}h ago",
                }
        else:
            checks["last_post_recency"] = {
                "ok": True,
                "detail": "No post history yet",
            }
    except Exception as e:
        checks["last_post_recency"] = {"ok": False, "detail": f"Check failed: {e}"}
        issues += 1

    # --- 2. Error rate (24h) ---
    error_rate_24h = 0.0
    try:
        from agent.generation_history import _read_history
        entries = await asyncio.to_thread(_read_history)
        cutoff = time.time() - 86400  # 24 hours
        recent = [e for e in entries if e.get("timestamp", 0) > cutoff]
        if recent:
            failures = sum(
                1 for e in recent
                if e.get("status") in ("failed", "error", "rejected")
            )
            error_rate_24h = (failures / len(recent)) * 100
            if error_rate_24h > 30:
                checks["error_rate"] = {
                    "ok": False,
                    "detail": f"{error_rate_24h:.0f}% failures ({failures}/{len(recent)} in 24h)",
                }
                issues += 1
            else:
                checks["error_rate"] = {
                    "ok": True,
                    "detail": f"{error_rate_24h:.0f}% ({failures}/{len(recent)} in 24h)",
                }
        else:
            checks["error_rate"] = {"ok": True, "detail": "No generations in 24h"}
    except Exception as e:
        checks["error_rate"] = {"ok": False, "detail": f"Check failed: {e}"}
        issues += 1

    # --- 3. State file integrity ---
    try:
        key_files = [
            STATE_DIR / "state.json",
            STATE_DIR / "feedback.json",
            STATE_DIR / "generation_history.json",
        ]
        missing = [f.name for f in key_files if not f.exists()]
        if missing:
            checks["state_files"] = {
                "ok": False,
                "detail": f"Missing: {', '.join(missing)}",
            }
            issues += 1
        else:
            checks["state_files"] = {"ok": True, "detail": "All present"}
    except Exception as e:
        checks["state_files"] = {"ok": False, "detail": f"Check failed: {e}"}
        issues += 1

    # --- 4. Stale pending drafts ---
    try:
        from agent.state import get_pending
        pending = await asyncio.to_thread(get_pending)
        if pending:
            pending_ts = pending.get("timestamp", 0)
            if pending_ts:
                age_hours = (time.time() - pending_ts) / 3600
                if age_hours > 2:
                    checks["stale_drafts"] = {
                        "ok": False,
                        "detail": f"Pending draft is {age_hours:.1f}h old (threshold: 2h)",
                    }
                    issues += 1
                else:
                    checks["stale_drafts"] = {
                        "ok": True,
                        "detail": f"Pending draft: {age_hours:.1f}h old",
                    }
            else:
                checks["stale_drafts"] = {"ok": True, "detail": "Pending draft (no timestamp)"}
        else:
            checks["stale_drafts"] = {"ok": True, "detail": "No pending drafts"}
    except Exception as e:
        checks["stale_drafts"] = {"ok": False, "detail": f"Check failed: {e}"}
        issues += 1

    # --- Determine overall status ---
    if issues >= 3:
        status = "down"
    elif issues >= 1:
        status = "degraded"
    else:
        status = "healthy"

    return HealthStatus(
        status=status,
        checks=checks,
        last_post_age_hours=last_post_age_hours,
        error_rate_24h=error_rate_24h,
    )


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

async def alert_on_failure(error_type: str, details: str, bot=None) -> None:
    """Send a Telegram alert, rate-limited to max 1 per error_type per hour.

    Args:
        error_type: Category of the error (e.g. "high_error_rate", "stale_draft").
        details: Human-readable description of the issue.
        bot: Telegram bot instance (optional). Falls back to HTTP notification.
    """
    from config import settings

    if not settings.HEALTH_ALERT_ENABLED:
        return

    state = _store.read()
    alert_timestamps = state.get("last_alert_timestamps", {})

    last_alert = alert_timestamps.get(error_type, 0.0)
    if (time.time() - last_alert) < _ALERT_COOLDOWN_SECONDS:
        logger.debug("Alert rate-limited for %s (last: %.0fs ago)", error_type, time.time() - last_alert)
        return

    # Update alert timestamp
    alert_timestamps[error_type] = time.time()
    state["last_alert_timestamps"] = alert_timestamps
    _store.write(state)

    message = (
        f"<b>Health Alert: {error_type}</b>\n\n"
        f"{details}"
    )

    if bot:
        try:
            await bot.send_message(
                chat_id=settings.TELEGRAM_ALLOWED_USER_ID,
                text=message,
                parse_mode="HTML",
            )
            logger.info("Health alert sent: %s", error_type)
        except Exception as e:
            logger.error("Failed to send health alert via bot: %s", e)
    else:
        try:
            from scripts.auto_post import _notify_telegram
            await _notify_telegram(message)
            logger.info("Health alert sent via HTTP: %s", error_type)
        except Exception as e:
            logger.error("Failed to send health alert via HTTP: %s", e)


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

async def maybe_run_health_check(bot=None) -> None:
    """Called from the scheduler loop. Runs checks every 30 min, alerts on issues.

    Reads last_check_at from state to avoid running too frequently.
    """
    from config import settings

    if not settings.HEALTH_CHECK_ENABLED:
        return

    state = _store.read()
    last_check = state.get("last_check_at", 0.0)
    if (time.time() - last_check) < _CHECK_INTERVAL_SECONDS:
        return

    # Update check timestamp
    state["last_check_at"] = time.time()
    _store.write(state)

    health = await run_health_checks()
    logger.info("Health check: %s (%d checks)", health.status, len(health.checks))

    if health.status in ("degraded", "down"):
        # Collect all failing checks into alert details
        failing = [
            f"- {name}: {check['detail']}"
            for name, check in health.checks.items()
            if not check.get("ok")
        ]
        details = "\n".join(failing)
        await alert_on_failure(
            error_type=f"system_{health.status}",
            details=f"System is {health.status}.\n\n{details}",
            bot=bot,
        )
