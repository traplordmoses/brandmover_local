"""
Natural language intent router using Claude Haiku for classification.

Short messages hit a lookup table (zero latency, zero cost).
Longer messages are classified by Haiku with conversation context.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import anthropic

from agent.conversation_context import ConversationContext
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_INTENTS = (
    "approve",
    "reject",
    "edit_request",
    "reroll",
    "generate_content",
    "schedule_post",
    "change_style",
    "modify_last",
    "upload_assets",
    "show_status",
    "show_help",
    "show_analytics",
    "show_history",
    "brand_check",
    "casual_chat",
    "greeting",
    "unknown",
)

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_HAIKU_TIMEOUT_SECONDS = 5
_RATE_LIMIT_PER_HOUR = 30
_CACHE_MAX_SIZE = 50
_CACHE_TTL_SECONDS = 60

# ---------------------------------------------------------------------------
# Short message lookup table — bypasses Claude entirely
# ---------------------------------------------------------------------------

_SHORT_MESSAGE_TABLE: dict[str, str] = {
    # Approve
    "yes": "approve",
    "yep": "approve",
    "yeah": "approve",
    "y": "approve",
    "ok": "approve",
    "okay": "approve",
    "sure": "approve",
    "looks good": "approve",
    "lgtm": "approve",
    "post it": "approve",
    "send it": "approve",
    "ship it": "approve",
    "approve": "approve",
    "approved": "approve",
    "go": "approve",
    "do it": "approve",
    "publish": "approve",
    # Reject
    "no": "reject",
    "nah": "reject",
    "nope": "reject",
    "n": "reject",
    "skip": "reject",
    "reject": "reject",
    "pass": "reject",
    "cancel": "reject",
    "delete": "reject",
    "discard": "reject",
    # Reroll
    "try again": "reroll",
    "again": "reroll",
    "another": "reroll",
    "another one": "reroll",
    "reroll": "reroll",
    "redo": "reroll",
    "regenerate": "reroll",
    "new one": "reroll",
    # Greeting
    "hi": "greeting",
    "hello": "greeting",
    "hey": "greeting",
    "yo": "greeting",
    "sup": "greeting",
    "gm": "greeting",
    "good morning": "greeting",
    # Upload
    "upload": "upload_assets",
    "ingest": "upload_assets",
    "send images": "upload_assets",
    "send pictures": "upload_assets",
    "add images": "upload_assets",
    # Schedule
    "schedule": "schedule_post",
    "scheduled": "schedule_post",
    "schedule posts": "schedule_post",
    "show schedule": "schedule_post",
    # Utility
    "help": "show_help",
    "status": "show_status",
    "analytics": "show_analytics",
    "history": "show_history",
}

# Intents from the table that require a pending draft
_DRAFT_DEPENDENT_INTENTS = {"approve", "reject", "reroll"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    intent: str
    confidence: float
    parameters: dict = field(default_factory=dict)
    raw_message: str = ""
    routed_via: str = "fallback"  # "table" | "haiku" | "fallback"


# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-user)
# ---------------------------------------------------------------------------

_rate_counts: dict[int, list[float]] = {}


def _check_rate_limit(user_id: int) -> bool:
    """Return True if under rate limit, False if exceeded."""
    now = time.time()
    cutoff = now - 3600
    if user_id not in _rate_counts:
        _rate_counts[user_id] = []
    _rate_counts[user_id] = [t for t in _rate_counts[user_id] if t > cutoff]
    return len(_rate_counts[user_id]) < _RATE_LIMIT_PER_HOUR


def _record_rate(user_id: int) -> None:
    """Record a Haiku call for rate limiting."""
    if user_id not in _rate_counts:
        _rate_counts[user_id] = []
    _rate_counts[user_id].append(time.time())


def reset_rate_limits() -> None:
    """Clear all rate limit state (for testing)."""
    _rate_counts.clear()


# ---------------------------------------------------------------------------
# Response cache (LRU-ish, TTL-based)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[RoutingResult, float]] = {}


def _cache_key(message: str, context: ConversationContext) -> str:
    """Build a cache key from normalized message + context state."""
    ctx_hash = f"{context.last_bot_action}:{context.pending_draft_exists}:{context.last_content_type}"
    return f"{message}|{ctx_hash}"


def _cache_get(key: str) -> RoutingResult | None:
    """Return cached result if fresh, else None."""
    if key in _cache:
        result, ts = _cache[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return result
        del _cache[key]
    return None


def _cache_put(key: str, result: RoutingResult) -> None:
    """Store result in cache, evicting oldest if full."""
    if len(_cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_cache, key=lambda k: _cache[k][1])
        del _cache[oldest_key]
    _cache[key] = (result, time.time())


def clear_cache() -> None:
    """Clear the response cache (for testing)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Haiku classification prompt
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """\
You are a message intent classifier for a social media content bot.
Classify the user's message into exactly one intent.

Available intents:
- approve: User wants to approve/post a pending draft
- reject: User wants to reject/discard a pending draft
- edit_request: User wants to modify a pending draft (has specific feedback)
- reroll: User wants to regenerate content from scratch
- generate_content: User wants to create new content (has a topic/idea)
- schedule_post: User wants to schedule a post for a future time (says "schedule", "post at", "post tomorrow", "queue up", etc.)
- change_style: User wants to change the visual or writing style
- modify_last: User wants to tweak the last generated content
- show_status: User wants to see bot/system status
- show_help: User wants help or command list
- show_analytics: User wants analytics/stats
- show_history: User wants generation history
- upload_assets: User wants to upload, share, send, or update brand assets, logos, images, pictures, renders, files, or creative work. Questions like "can I give you..." or "I want to send you..." about files/images are upload_assets, NOT casual_chat.
- brand_check: User wants to check brand compliance
- casual_chat: User is making small talk, not requesting any action
- greeting: User is saying hello
- unknown: Cannot determine intent

Respond with ONLY valid JSON:
{"intent": "<intent>", "confidence": <0.0-1.0>, "parameters": {}}

For edit_request, include: {"feedback": "the user's feedback"}
For generate_content, include: {"topic": "extracted topic"}
For schedule_post, include: {"time": "the time expression", "topic": "what to post about"}
For change_style, include: {"style": "requested style"}
"""


def _build_classify_user_message(message: str, context: ConversationContext) -> str:
    """Build the user message for Haiku classification."""
    parts = [f"Message: {message}"]
    parts.append(f"\nContext:")
    parts.append(f"- Last bot action: {context.last_bot_action}")
    parts.append(f"- Pending draft exists: {context.pending_draft_exists}")
    if context.last_content_type:
        parts.append(f"- Last content type: {context.last_content_type}")
    if context.recent_intents:
        parts.append(f"- Recent intents: {', '.join(context.recent_intents[-3:])}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------

async def classify_intent(
    message: str,
    context: ConversationContext,
) -> RoutingResult:
    """Classify user message into an intent with confidence score.

    Flow:
    1. Check short message table (instant, free)
    2. Check cache (instant, free)
    3. Call Haiku (fast, cheap)
    4. Fallback to unknown on error
    """
    normalized = message.strip().lower()

    # 1. Short message table lookup
    if normalized in _SHORT_MESSAGE_TABLE:
        intent = _SHORT_MESSAGE_TABLE[normalized]
        # Context-aware remapping: draft-dependent intents without a draft → casual_chat
        if intent in _DRAFT_DEPENDENT_INTENTS and not context.pending_draft_exists:
            return RoutingResult(
                intent="casual_chat",
                confidence=0.9,
                raw_message=message,
                routed_via="table",
            )
        params = {}
        if intent == "reject" and len(normalized.split()) <= 2:
            params["needs_feedback_prompt"] = True
        return RoutingResult(
            intent=intent,
            confidence=0.95,
            parameters=params,
            raw_message=message,
            routed_via="table",
        )

    # 2. Cache check
    key = _cache_key(normalized, context)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # 3. Rate limit check
    if not _check_rate_limit(context.user_id):
        logger.warning("Router rate limit exceeded for user %d", context.user_id)
        return RoutingResult(
            intent="generate_content",
            confidence=0.5,
            parameters={"topic": message},
            raw_message=message,
            routed_via="fallback",
        )

    # 4. Call Haiku
    try:
        result = await asyncio.wait_for(
            _call_haiku(message, context),
            timeout=_HAIKU_TIMEOUT_SECONDS,
        )
        _record_rate(context.user_id)
        _cache_put(key, result)
        return result
    except asyncio.TimeoutError:
        logger.warning("Haiku classification timed out for: %s", message[:50])
        return RoutingResult(
            intent="generate_content",
            confidence=0.5,
            parameters={"topic": message},
            raw_message=message,
            routed_via="fallback",
        )
    except Exception as e:
        logger.warning("Haiku classification failed: %s", e)
        return RoutingResult(
            intent="generate_content",
            confidence=0.5,
            parameters={"topic": message},
            raw_message=message,
            routed_via="fallback",
        )


async def _call_haiku(message: str, context: ConversationContext) -> RoutingResult:
    """Call Claude Haiku for intent classification."""
    from agent._client import get_anthropic
    client = get_anthropic()
    user_msg = _build_classify_user_message(message, context)

    response = await client.messages.create(
        model=_HAIKU_MODEL,
        max_tokens=256,
        system=_CLASSIFY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences (Haiku often wraps JSON in ```json ... ```)
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("Haiku returned non-JSON: %s", raw_text[:100])
        return RoutingResult(
            intent="unknown",
            confidence=0.0,
            raw_message=message,
            routed_via="haiku",
        )

    intent = data.get("intent", "unknown")
    confidence = float(data.get("confidence", 0.0))
    parameters = data.get("parameters", {})

    # Validate intent
    if intent not in KNOWN_INTENTS:
        intent = "unknown"
        confidence = 0.0

    # Low confidence floor
    if confidence < 0.3:
        intent = "unknown"

    # Context-aware: draft-dependent intents without a draft
    if intent in _DRAFT_DEPENDENT_INTENTS and not context.pending_draft_exists:
        intent = "casual_chat"
        confidence = max(confidence, 0.5)

    return RoutingResult(
        intent=intent,
        confidence=confidence,
        parameters=parameters,
        raw_message=message,
        routed_via="haiku",
    )
