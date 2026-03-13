"""
Event hook system — async pub/sub for decoupled side effects.

Inspired by OpenClaw's hook architecture. Events are fired from core code
(handlers, engine, scheduler) and subscribers run asynchronously without
blocking the main flow.

Events:
- draft:generated — after agent produces a draft
- draft:approved — after operator approves
- draft:rejected — after operator rejects
- post:published — after posting to X/Twitter
- generation:started — when agent run begins
- generation:complete — when agent run finishes
- skill:created — when a new skill is saved
- heartbeat:tick — on each heartbeat cycle
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Type for hook handlers — async functions that receive event data
HookHandler = Callable[[dict[str, Any]], Awaitable[None]]

# Registry: event_name → list of handlers
_hooks: dict[str, list[HookHandler]] = defaultdict(list)


def on(event: str, handler: HookHandler) -> None:
    """Register a handler for an event type."""
    _hooks[event].append(handler)
    logger.debug("Hook registered: %s → %s", event, handler.__name__)


def off(event: str, handler: HookHandler) -> None:
    """Unregister a handler."""
    try:
        _hooks[event].remove(handler)
    except ValueError:
        pass


async def emit(event: str, data: dict[str, Any] | None = None) -> None:
    """Fire an event. All handlers run concurrently, fire-and-forget.

    Errors in handlers are logged but never propagate — hooks must not
    break core functionality.
    """
    handlers = _hooks.get(event, [])
    if not handlers:
        return

    payload = dict(data) if data else {}
    payload.setdefault("event", event)

    async def _safe_call(h):
        try:
            await h(payload)
        except Exception as e:
            logger.warning("Hook %s failed for event %s: %s", h.__name__, event, e)

    # Run all handlers concurrently
    await asyncio.gather(*[_safe_call(h) for h in handlers])


def clear() -> None:
    """Remove all registered hooks. Used in tests."""
    _hooks.clear()


def list_hooks() -> dict[str, list[str]]:
    """Return registered hooks for debugging."""
    return {event: [h.__name__ for h in handlers] for event, handlers in _hooks.items() if handlers}
