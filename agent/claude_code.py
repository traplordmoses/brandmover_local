"""
Claude Code CLI integration — run Claude Code from within the bot.

Wraps the `claude` CLI as an async subprocess with streaming NDJSON output
parsing, progress callbacks, session resume, and auto-escalation support.

Used by the /code Telegram command and the auto-escalation hook to let the
bot operator fix issues from their phone.
"""

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent.audit_log import audit
from agent.paths import STATE_DIR

logger = logging.getLogger(__name__)

# Persistent state for daily invocation tracking
_DAILY_STATE_FILE = STATE_DIR / "claude_code_daily.json"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Concurrency lock — only one Claude Code session at a time
_run_lock = asyncio.Lock()


@dataclass
class ClaudeCodeResult:
    """Result from a Claude Code CLI invocation."""
    success: bool
    result_text: str
    cost_usd: float
    session_id: str
    duration_ms: int
    num_turns: int
    error_message: str = ""
    tools_used: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)


# ── Daily limit tracking ─────────────────────────────────────────────────

def _load_daily_state() -> dict:
    try:
        if _DAILY_STATE_FILE.exists():
            data = json.loads(_DAILY_STATE_FILE.read_text())
            # Reset if it's a new day
            today = time.strftime("%Y-%m-%d")
            if data.get("date") != today:
                return {"date": today, "count": 0, "total_cost": 0.0}
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"date": time.strftime("%Y-%m-%d"), "count": 0, "total_cost": 0.0}


def _save_daily_state(state: dict) -> None:
    try:
        _DAILY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DAILY_STATE_FILE.write_text(json.dumps(state))
    except OSError as e:
        logger.warning("Failed to save Claude Code daily state: %s", e)


def check_daily_limit(limit: int) -> tuple[bool, int]:
    """Check if daily invocation limit is reached. Returns (allowed, count)."""
    state = _load_daily_state()
    return state["count"] < limit, state["count"]


def _increment_daily_count(cost: float = 0.0) -> None:
    state = _load_daily_state()
    state["count"] = state.get("count", 0) + 1
    state["total_cost"] = state.get("total_cost", 0.0) + cost
    _save_daily_state(state)


# ── Git state capture for revert ──────────────────────────────────────────

async def _capture_git_state() -> str | None:
    """Capture current git state for potential revert. Returns stash ref or None."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "stash", "create",
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        ref = stdout.decode().strip()
        return ref if ref else None
    except Exception as e:
        logger.debug("Git stash create failed: %s", e)
        return None


async def _get_changed_files() -> list[str]:
    """Get list of files changed since last commit."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only",
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return [f for f in stdout.decode().strip().split("\n") if f]
    except Exception:
        return []


async def revert_files(files: list[str]) -> bool:
    """Revert specific files to their last committed state."""
    if not files:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", "--", *files,
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except Exception as e:
        logger.error("Revert failed: %s", e)
        return False


async def get_diff(files: list[str] | None = None) -> str:
    """Get git diff for changed files."""
    cmd = ["git", "diff"]
    if files:
        cmd.extend(["--", *files])
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()[:8000]  # Cap at 8KB
    except Exception:
        return "(diff unavailable)"


async def validate_syntax(files: list[str]) -> list[str]:
    """Run py_compile on changed .py files. Returns list of errors."""
    errors = []
    for f in files:
        if not f.endswith(".py"):
            continue
        path = _PROJECT_ROOT / f
        if not path.exists():
            continue
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "py_compile", str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                errors.append(f"{f}: {stderr.decode().strip()}")
        except Exception as e:
            errors.append(f"{f}: {e}")
    return errors


# ── NDJSON stream parsing ─────────────────────────────────────────────────

def _parse_progress(line: str) -> tuple[str, dict]:
    """Parse a stream-json line. Returns (event_type, data)."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return "unknown", {}

    msg_type = data.get("type", "unknown")
    return msg_type, data


def _extract_tool_info(message: dict) -> list[tuple[str, str]]:
    """Extract tool name and summary from an assistant message."""
    tools = []
    content = message.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return tools

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            # Build a human-readable summary
            if name == "Read":
                summary = f"Reading {inp.get('file_path', '?')}"
            elif name == "Edit":
                path = inp.get("file_path", "?")
                summary = f"Editing {path}"
            elif name == "Write":
                path = inp.get("file_path", "?")
                summary = f"Creating {path}"
            elif name == "Bash":
                cmd = inp.get("command", "?")
                summary = f"Running: {cmd[:60]}"
            elif name in ("Grep", "Glob"):
                pattern = inp.get("pattern", "?")
                summary = f"Searching for {pattern[:40]}"
            else:
                summary = f"Using {name}"
            tools.append((name, summary))
    return tools


def _extract_text(message: dict) -> str:
    """Extract text content from an assistant message."""
    content = message.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(parts)[:200]


# ── Main CLI runner ───────────────────────────────────────────────────────

async def run_claude_code(
    instruction: str,
    *,
    model: str = "sonnet",
    max_budget_usd: float = 1.0,
    timeout_seconds: int = 300,
    session_id: str | None = None,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
    system_prompt: str | None = None,
    user_id: int | None = None,
) -> ClaudeCodeResult:
    """Run Claude Code CLI with the given instruction.

    Args:
        instruction: What to tell Claude Code to do.
        model: Model to use (sonnet, opus, haiku).
        max_budget_usd: Maximum cost cap per invocation.
        timeout_seconds: Kill subprocess after this many seconds.
        session_id: Resume a previous session.
        on_progress: Async callback for progress updates.
        system_prompt: Optional system prompt override.
        user_id: Telegram user ID for audit logging.

    Returns:
        ClaudeCodeResult with outcome, cost, session_id, etc.
    """
    # Check that claude CLI exists
    claude_path = shutil.which("claude")
    if not claude_path:
        return ClaudeCodeResult(
            success=False,
            result_text="",
            cost_usd=0,
            session_id="",
            duration_ms=0,
            num_turns=0,
            error_message="Claude Code CLI not found. Install it first.",
        )

    # Build the command
    cmd = [
        claude_path,
        "-p", instruction,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--model", model,
        "--max-budget-usd", str(max_budget_usd),
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    start_time = time.time()
    tools_used: list[str] = []
    files_before = set(await _get_changed_files())

    # Capture pre-run git state
    git_ref = await _capture_git_state()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=None,  # inherit environment
        )

        result_data: dict = {}
        last_progress_time = 0.0

        async def _read_stream():
            nonlocal result_data, last_progress_time
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                msg_type, data = _parse_progress(line)

                if msg_type == "result":
                    result_data = data
                elif msg_type == "assistant":
                    # Extract tool calls for progress
                    tool_info = _extract_tool_info(data)
                    for name, summary in tool_info:
                        tools_used.append(name)
                        if on_progress:
                            now = time.time()
                            # Throttle updates to every 2 seconds
                            if now - last_progress_time >= 2.0:
                                last_progress_time = now
                                await on_progress(summary)
                    # Extract reasoning text
                    if not tool_info and on_progress:
                        text = _extract_text(data)
                        if text:
                            now = time.time()
                            if now - last_progress_time >= 2.0:
                                last_progress_time = now
                                await on_progress(text[:80])

        # Run with timeout
        await asyncio.wait_for(_read_stream(), timeout=timeout_seconds)
        await proc.wait()

    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ClaudeCodeResult(
            success=False,
            result_text="",
            cost_usd=0,
            session_id=session_id or "",
            duration_ms=elapsed_ms,
            num_turns=0,
            error_message=f"Timed out after {timeout_seconds}s",
            tools_used=tools_used,
        )
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ClaudeCodeResult(
            success=False,
            result_text="",
            cost_usd=0,
            session_id="",
            duration_ms=elapsed_ms,
            num_turns=0,
            error_message=str(e),
            tools_used=tools_used,
        )

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Parse result
    is_error = result_data.get("is_error", proc.returncode != 0)
    result_text = result_data.get("result", "")
    cost = result_data.get("total_cost_usd", 0) or 0
    sid = result_data.get("session_id", "")
    num_turns = result_data.get("num_turns", 0)

    # If no result data from stream, read stderr
    if not result_text and is_error:
        stderr = await proc.stderr.read() if proc.stderr else b""
        result_text = stderr.decode("utf-8", errors="replace")[:2000]

    # Detect changed files
    files_after = set(await _get_changed_files())
    files_changed = sorted(files_after - files_before)

    # Track daily usage
    _increment_daily_count(cost)

    # Audit log
    audit(
        "claude_code_run",
        user_id=user_id,
        instruction=instruction[:500],
        model=model,
        cost_usd=cost,
        duration_ms=elapsed_ms,
        files_changed=files_changed,
        session_id=sid,
        success=not is_error,
    )

    result = ClaudeCodeResult(
        success=not is_error,
        result_text=result_text,
        cost_usd=cost,
        session_id=sid,
        duration_ms=elapsed_ms,
        num_turns=num_turns,
        error_message="" if not is_error else result_text,
        tools_used=tools_used,
        files_changed=files_changed,
    )

    logger.info(
        "Claude Code %s in %dms (cost=$%.3f, files=%d, turns=%d)",
        "succeeded" if result.success else "failed",
        elapsed_ms, cost, len(files_changed), num_turns,
    )

    return result


# ── Auto-escalation ──────────────────────────────────────────────────────

async def escalate_error(
    error: Exception,
    context: str,
    traceback_str: str,
    *,
    notify_callback: Callable[[str], Awaitable[None]] | None = None,
) -> ClaudeCodeResult | None:
    """Auto-escalate a bot error to Claude Code for self-repair.

    Args:
        error: The exception that was raised.
        context: What was happening when the error occurred.
        traceback_str: Full traceback string.
        notify_callback: Async function to send notification to admin.

    Returns:
        ClaudeCodeResult if escalation was attempted, None if skipped.
    """
    from config import settings

    if not getattr(settings, "CLAUDE_CODE_ENABLED", False):
        return None
    if not getattr(settings, "CLAUDE_CODE_AUTO_ESCALATE", False):
        return None

    # Check daily limit
    limit = getattr(settings, "CLAUDE_CODE_DAILY_LIMIT", 20)
    allowed, count = check_daily_limit(limit)
    if not allowed:
        logger.warning("Claude Code auto-escalation skipped: daily limit reached (%d)", count)
        return None

    # Don't auto-escalate Claude Code's own errors
    if "claude_code" in context.lower():
        return None

    budget = getattr(settings, "CLAUDE_CODE_ESCALATION_BUDGET_USD", 0.50)
    timeout = getattr(settings, "CLAUDE_CODE_TIMEOUT_SECONDS", 300)
    model = getattr(settings, "CLAUDE_CODE_MODEL", "sonnet")

    instruction = (
        f"The BrandMover bot hit an error during {context}.\n\n"
        f"Error: {error}\n\n"
        f"Traceback:\n{traceback_str[-2000:]}\n\n"
        "Diagnose and fix this issue. Focus on the specific error — "
        "do not refactor unrelated code. Keep changes minimal."
    )

    if notify_callback:
        await notify_callback(
            f"Auto-fix triggered for: {context}\n"
            f"Error: {str(error)[:200]}"
        )

    result = await run_claude_code(
        instruction,
        model=model,
        max_budget_usd=budget,
        timeout_seconds=timeout,
    )

    if notify_callback:
        if result.success:
            files_str = "\n".join(f"  - {f}" for f in result.files_changed) or "  (none)"
            await notify_callback(
                f"Auto-fix completed ({result.duration_ms // 1000}s)\n\n"
                f"Files changed:\n{files_str}\n\n"
                f"Result: {result.result_text[:500]}"
            )
        else:
            await notify_callback(
                f"Auto-fix failed: {result.error_message[:300]}"
            )

    return result
