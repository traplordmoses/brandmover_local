"""
X/Twitter growth engine — monitors target accounts, generates reply drafts,
and tracks follower growth analytics.

Provides:
    - Target account management (add/remove/list accounts to monitor)
    - Timeline monitoring via X API v2 (fetch recent tweets from targets)
    - Reply draft generation using Claude Haiku (witty, on-brand replies)
    - Follower growth tracking with daily snapshots
    - Engagement opportunity scoring (prioritize high-impact replies)
    - Content type share-ratio analysis

State files (in STATE_DIR):
    - growth_targets.json: accounts to monitor/engage with
    - growth_state.json: reply history, engagement metrics, share ratios
    - growth_history.json: daily follower count snapshots

Public API:
    # Target management
    await add_target_account("elonmusk", reason="Tech thought leader")
    await remove_target_account("elonmusk")
    targets = get_target_accounts()

    # Timeline monitoring
    tweets = await fetch_target_timelines(limit_per_account=5)

    # Reply generation
    drafts = await generate_reply_drafts(tweets, brand_context="...")

    # Analytics
    metrics = await fetch_account_metrics()
    track_follower_growth(current_count=1234)
    report = get_growth_report(days=7)
    ratios = get_share_ratio_by_type()

    # Scoring
    score = score_engagement_opportunity(tweet)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from agent._client import get_anthropic
from agent.paths import STATE_DIR
from agent.state_manager import FileStore
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_X_API_BASE = "https://api.twitter.com/2"

# Maximum tweets to fetch per target account per request
_DEFAULT_LIMIT_PER_ACCOUNT = 5

# Reply scoring weights
_SCORE_WEIGHT_FOLLOWERS = 0.25
_SCORE_WEIGHT_RECENCY = 0.30
_SCORE_WEIGHT_RELEVANCE = 0.20
_SCORE_WEIGHT_CONVERSATION_SIZE = 0.15
_SCORE_WEIGHT_REPLY_RATE = 0.10

# Recency: tweets older than this many seconds get a zero recency score
_RECENCY_WINDOW_SECONDS = 3600  # 60 minutes

# Growth history: keep at most this many daily snapshots
_MAX_HISTORY_SNAPSHOTS = 365

# Thread safety for state writes
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# State files — each backed by FileStore for atomic reads/writes
# ---------------------------------------------------------------------------

_TARGETS_FILE = STATE_DIR / "growth_targets.json"
_STATE_FILE = STATE_DIR / "growth_state.json"
_HISTORY_FILE = STATE_DIR / "growth_history.json"

_targets_store = FileStore(_TARGETS_FILE, default_factory=list)
_state_store = FileStore(
    _STATE_FILE,
    default_factory=lambda: {
        "reply_history": [],
        "engagement_by_target": {},
        "share_ratios": {},
        "last_timeline_fetch": 0,
    },
)
_history_store = FileStore(_HISTORY_FILE, default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bearer_headers() -> dict[str, str]:
    """Return Authorization header using the X Bearer Token."""
    return {"Authorization": f"Bearer {settings.X_BEARER_TOKEN}"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# 1. Target Account Management
# ---------------------------------------------------------------------------

async def add_target_account(username: str, reason: str = "") -> dict:
    """Add an account to the growth monitoring list.

    Args:
        username: X/Twitter handle (without @).
        reason: Why this account is being targeted (e.g. "industry leader").

    Returns:
        The newly created target entry dict.
    """
    username = username.lstrip("@").lower()
    with _lock:
        targets: list[dict] = _targets_store.read()

        # Deduplicate
        for t in targets:
            if t.get("username") == username:
                logger.info("Target account @%s already exists, updating reason", username)
                t["reason"] = reason or t.get("reason", "")
                _targets_store.write(targets)
                return t

        entry = {
            "username": username,
            "added_at": _now_iso(),
            "reason": reason,
            "follower_count": 0,
            "avg_engagement": 0.0,
            "last_checked": None,
            "engagement_count": 0,
            "follows_gained": 0,
        }
        targets.append(entry)
        _targets_store.write(targets)

    logger.info("Added growth target: @%s (%s)", username, reason or "no reason")
    return entry


async def remove_target_account(username: str) -> bool:
    """Remove an account from the growth monitoring list.

    Returns:
        True if the account was found and removed, False otherwise.
    """
    username = username.lstrip("@").lower()
    with _lock:
        targets: list[dict] = _targets_store.read()
        original_len = len(targets)
        targets = [t for t in targets if t.get("username") != username]
        if len(targets) == original_len:
            return False
        _targets_store.write(targets)

    logger.info("Removed growth target: @%s", username)
    return True


def get_target_accounts() -> list[dict]:
    """Return the current list of target accounts."""
    return _targets_store.read()


# ---------------------------------------------------------------------------
# 2. Timeline Monitoring
# ---------------------------------------------------------------------------

async def _resolve_user_id(username: str) -> str | None:
    """Resolve a username to a user ID via X API v2.

    Args:
        username: X handle (without @).

    Returns:
        User ID string, or None if the lookup fails.
    """
    if not settings.X_BEARER_TOKEN:
        logger.warning("X_BEARER_TOKEN not set — cannot resolve user ID")
        return None

    url = f"{_X_API_BASE}/users/by/username/{username}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_bearer_headers())
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return data.get("id")
    except httpx.HTTPStatusError as e:
        logger.error(
            "X API %d resolving @%s: %s",
            e.response.status_code, username, e.response.text[:200],
        )
    except httpx.HTTPError as e:
        logger.error("HTTP error resolving @%s: %s", username, e)
    return None


async def fetch_target_timelines(limit_per_account: int = _DEFAULT_LIMIT_PER_ACCOUNT) -> list[dict]:
    """Fetch recent tweets from all target accounts.

    Uses X API v2 ``GET /2/users/:id/tweets`` with Bearer Token auth.

    Args:
        limit_per_account: Max tweets to fetch per target account.

    Returns:
        List of tweet dicts: ``{tweet_id, author, text, metrics, created_at,
        conversation_id}``.
    """
    if not settings.X_BEARER_TOKEN:
        logger.warning("X_BEARER_TOKEN not set — cannot fetch timelines")
        return []

    targets = get_target_accounts()
    if not targets:
        logger.info("No growth targets configured — nothing to fetch")
        return []

    all_tweets: list[dict] = []

    for target in targets:
        username = target["username"]
        user_id = await _resolve_user_id(username)
        if not user_id:
            logger.warning("Could not resolve user ID for @%s, skipping", username)
            continue

        url = f"{_X_API_BASE}/users/{user_id}/tweets"
        params = {
            "max_results": min(limit_per_account, 100),
            "tweet.fields": "created_at,public_metrics,conversation_id,author_id",
            "exclude": "retweets,replies",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=_bearer_headers(), params=params)
                resp.raise_for_status()
                body = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "X API %d fetching tweets for @%s: %s",
                e.response.status_code, username, e.response.text[:200],
            )
            continue
        except httpx.HTTPError as e:
            logger.error("HTTP error fetching tweets for @%s: %s", username, e)
            continue

        for tweet in body.get("data", []):
            pm = tweet.get("public_metrics", {})
            all_tweets.append({
                "tweet_id": tweet["id"],
                "author": username,
                "text": tweet.get("text", ""),
                "metrics": {
                    "likes": pm.get("like_count", 0),
                    "retweets": pm.get("retweet_count", 0),
                    "replies": pm.get("reply_count", 0),
                    "impressions": pm.get("impression_count", 0),
                },
                "created_at": tweet.get("created_at", ""),
                "conversation_id": tweet.get("conversation_id", ""),
            })

        # Update target metadata
        with _lock:
            targets_fresh = _targets_store.read()
            for t in targets_fresh:
                if t["username"] == username:
                    t["last_checked"] = _now_iso()
                    break
            _targets_store.write(targets_fresh)

    # Record fetch timestamp in growth state
    with _lock:
        state = _state_store.read()
        state["last_timeline_fetch"] = _now_ts()
        _state_store.write(state)

    logger.info("Fetched %d tweets from %d target accounts", len(all_tweets), len(targets))
    return all_tweets


# ---------------------------------------------------------------------------
# 3. Reply Draft Generation
# ---------------------------------------------------------------------------

_REPLY_SYSTEM_PROMPT = """\
You are a social media engagement specialist. Your goal is to craft reply tweets
that are witty, insightful, and showcase genuine expertise — never promotional
or corporate. Each reply should:

1. Be directly relevant to the original tweet's topic
2. Add real value: a unique perspective, data point, or insight
3. Sound conversational and human, not like a brand account
4. Be concise (under 280 characters)
5. Invite further discussion naturally

{brand_context}

Respond with a JSON array of objects. Each object must have:
- "reply_draft": the actual reply text (under 280 chars)
- "strategy": brief description of the engagement approach used
- "confidence": 0.0-1.0 rating of how well this reply fits
"""

_REPLY_USER_PROMPT = """\
Generate reply drafts for these tweets. Return ONLY a JSON array.

Tweets to reply to:
{tweets_block}
"""


async def generate_reply_drafts(
    tweets: list[dict],
    brand_context: str = "",
) -> list[dict]:
    """Generate witty, on-brand reply drafts for target tweets using Claude Haiku.

    Args:
        tweets: List of tweet dicts from ``fetch_target_timelines()``.
        brand_context: Optional brand guidelines text to inform tone/expertise.

    Returns:
        List of dicts: ``{target_tweet_id, target_author, target_text,
        reply_draft, strategy, confidence}``.
    """
    if not tweets:
        return []

    # Format tweets for the prompt
    tweets_block_parts: list[str] = []
    for i, tw in enumerate(tweets, 1):
        tweets_block_parts.append(
            f"[{i}] @{tw['author']}: {tw['text']}\n"
            f"    (tweet_id: {tw['tweet_id']}, "
            f"likes: {tw['metrics'].get('likes', 0)}, "
            f"replies: {tw['metrics'].get('replies', 0)})"
        )
    tweets_block = "\n\n".join(tweets_block_parts)

    brand_section = ""
    if brand_context:
        brand_section = f"\nBrand context (use for tone, not promotion):\n{brand_context[:2000]}"

    system_prompt = _REPLY_SYSTEM_PROMPT.format(brand_context=brand_section)
    user_prompt = _REPLY_USER_PROMPT.format(tweets_block=tweets_block)

    try:
        client = get_anthropic()
        response = await client.messages.create(
            model=settings.HAIKU_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from response
        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        # Parse JSON from response — handle markdown code blocks
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            # Strip markdown code fences
            lines = raw_text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw_text = "\n".join(lines)

        drafts_raw: list[dict] = json.loads(raw_text)

    except json.JSONDecodeError as e:
        logger.error("Failed to parse reply drafts JSON: %s", e)
        return []
    except Exception as e:
        logger.error("Error generating reply drafts: %s", e)
        return []

    # Map drafts back to original tweets
    results: list[dict] = []
    for i, draft_obj in enumerate(drafts_raw):
        if i >= len(tweets):
            break
        tw = tweets[i]
        results.append({
            "target_tweet_id": tw["tweet_id"],
            "target_author": tw["author"],
            "target_text": tw["text"][:200],
            "reply_draft": draft_obj.get("reply_draft", ""),
            "strategy": draft_obj.get("strategy", ""),
            "confidence": float(draft_obj.get("confidence", 0.5)),
        })

    logger.info("Generated %d reply drafts", len(results))
    return results


# ---------------------------------------------------------------------------
# 4. Growth Analytics
# ---------------------------------------------------------------------------

async def fetch_account_metrics() -> dict:
    """Fetch the authenticated account's metrics from X API v2.

    Returns:
        Dict with ``{follower_count, following_count, tweet_count, username,
        fetched_at}``, or empty dict on failure.
    """
    if not settings.X_BEARER_TOKEN:
        logger.warning("X_BEARER_TOKEN not set — cannot fetch account metrics")
        return {}

    url = f"{_X_API_BASE}/users/me"
    params = {"user.fields": "public_metrics,username"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # /users/me requires OAuth 2.0 User Context or OAuth 1.0a.
            # With Bearer Token, we use the authenticated user's token.
            resp = await client.get(url, headers=_bearer_headers(), params=params)
            resp.raise_for_status()
            data = resp.json().get("data", {})
    except httpx.HTTPStatusError as e:
        # /users/me may not work with app-only Bearer Token.
        # Fall back gracefully.
        logger.warning(
            "X API %d fetching /users/me (may need user-context token): %s",
            e.response.status_code, e.response.text[:200],
        )
        return {}
    except httpx.HTTPError as e:
        logger.error("HTTP error fetching account metrics: %s", e)
        return {}

    pm = data.get("public_metrics", {})
    metrics = {
        "username": data.get("username", ""),
        "follower_count": pm.get("followers_count", 0),
        "following_count": pm.get("following_count", 0),
        "tweet_count": pm.get("tweet_count", 0),
        "fetched_at": _now_iso(),
    }

    # Auto-track growth if we got a follower count
    if metrics["follower_count"] > 0:
        track_follower_growth(metrics["follower_count"])

    return metrics


def track_follower_growth(current_count: int) -> None:
    """Append a daily follower-count snapshot to growth_history.json.

    Only writes one snapshot per calendar day (UTC). If a snapshot for today
    already exists, it updates the count in place.

    Args:
        current_count: Current follower count.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with _lock:
        history: list[dict] = _history_store.read()

        # Check if we already have a snapshot for today
        for entry in history:
            if entry.get("date") == today:
                entry["follower_count"] = current_count
                entry["updated_at"] = _now_iso()
                _history_store.write(history)
                return

        # New daily snapshot
        history.append({
            "date": today,
            "follower_count": current_count,
            "recorded_at": _now_iso(),
            "updated_at": _now_iso(),
        })

        # Cap history length
        if len(history) > _MAX_HISTORY_SNAPSHOTS:
            history = history[-_MAX_HISTORY_SNAPSHOTS:]

        _history_store.write(history)

    logger.debug("Tracked follower count: %d for %s", current_count, today)


def get_growth_report(days: int = 7) -> dict:
    """Generate a follower growth report for the specified period.

    Args:
        days: Number of days to analyze (default 7).

    Returns:
        Dict with ``{period, start_followers, end_followers, net_growth,
        growth_rate, daily_snapshots, top_growth_drivers, best_content_types,
        share_ratios}``.
    """
    history: list[dict] = _history_store.read()

    if not history:
        return {
            "period": days,
            "start_followers": 0,
            "end_followers": 0,
            "net_growth": 0,
            "growth_rate": 0.0,
            "daily_snapshots": [],
            "top_growth_drivers": [],
            "best_content_types": [],
            "share_ratios": {},
        }

    # Filter to requested period
    recent = history[-days:] if len(history) >= days else history

    start_followers = recent[0].get("follower_count", 0)
    end_followers = recent[-1].get("follower_count", 0)
    net_growth = end_followers - start_followers
    growth_rate = 0.0
    if start_followers > 0:
        growth_rate = round(net_growth / start_followers * 100, 2)

    # Identify biggest daily jumps
    daily_deltas: list[dict] = []
    for i in range(1, len(recent)):
        prev = recent[i - 1].get("follower_count", 0)
        curr = recent[i].get("follower_count", 0)
        delta = curr - prev
        daily_deltas.append({
            "date": recent[i].get("date", ""),
            "delta": delta,
            "count": curr,
        })

    top_growth_drivers = sorted(daily_deltas, key=lambda d: d["delta"], reverse=True)[:3]

    # Analyze best content types from reply history
    state = _state_store.read()
    type_engagement: dict[str, list[float]] = {}
    for reply in state.get("reply_history", []):
        ctype = reply.get("content_type", "reply")
        conf = reply.get("confidence", 0.5)
        type_engagement.setdefault(ctype, []).append(conf)

    best_content_types = sorted(
        [
            {"type": ctype, "avg_confidence": round(sum(scores) / len(scores), 2), "count": len(scores)}
            for ctype, scores in type_engagement.items()
        ],
        key=lambda x: x["avg_confidence"],
        reverse=True,
    )[:5]

    return {
        "period": days,
        "start_followers": start_followers,
        "end_followers": end_followers,
        "net_growth": net_growth,
        "growth_rate": growth_rate,
        "daily_snapshots": recent,
        "top_growth_drivers": top_growth_drivers,
        "best_content_types": best_content_types,
        "share_ratios": get_share_ratio_by_type(),
    }


def get_share_ratio_by_type() -> dict:
    """Analyze which content types get the most retweets relative to impressions.

    Reads from growth state's share ratio tracking data. Returns a dict
    mapping content type to ``{retweets, impressions, share_ratio}``.
    """
    state = _state_store.read()
    ratios = state.get("share_ratios", {})

    # Compute ratios from raw data
    result: dict[str, dict] = {}
    for ctype, data in ratios.items():
        retweets = data.get("retweets", 0)
        impressions = data.get("impressions", 0)
        ratio = 0.0
        if impressions > 0:
            ratio = round(retweets / impressions * 100, 4)
        result[ctype] = {
            "retweets": retweets,
            "impressions": impressions,
            "share_ratio": ratio,
        }

    return result


def update_share_ratios(content_type: str, retweets: int, impressions: int) -> None:
    """Update cumulative share ratio data for a content type.

    Args:
        content_type: The content type label (e.g. "educational", "meme").
        retweets: Number of retweets for the post.
        impressions: Number of impressions for the post.
    """
    with _lock:
        state = _state_store.read()
        ratios = state.setdefault("share_ratios", {})
        entry = ratios.setdefault(content_type, {"retweets": 0, "impressions": 0})
        entry["retweets"] = entry.get("retweets", 0) + retweets
        entry["impressions"] = entry.get("impressions", 0) + impressions
        _state_store.write(state)


# ---------------------------------------------------------------------------
# 5. Engagement Scoring
# ---------------------------------------------------------------------------

def score_engagement_opportunity(tweet: dict) -> dict:
    """Score a tweet for reply worthiness on a 0-100 scale.

    Factors considered:
    - Author follower count (higher = more exposure potential)
    - Tweet recency (replies within 30 min get max visibility)
    - Topic relevance to brand (based on keyword overlap)
    - Conversation size (fewer existing replies = more visibility)
    - Author reply rate (do they engage back?)

    Args:
        tweet: A tweet dict from ``fetch_target_timelines()``.

    Returns:
        Dict with ``{score, factors, recommendation}``.
    """
    metrics = tweet.get("metrics", {})
    factors: dict[str, dict[str, Any]] = {}

    # --- Factor 1: Author follower count ---
    # Look up target follower count from stored targets
    author = tweet.get("author", "")
    targets = get_target_accounts()
    author_followers = 0
    for t in targets:
        if t.get("username") == author:
            author_followers = t.get("follower_count", 0)
            break

    # Logarithmic scale: 1K=30, 10K=50, 100K=70, 1M=90
    import math
    if author_followers > 0:
        follower_score = min(100, max(0, 15 * math.log10(max(1, author_followers))))
    else:
        follower_score = 20  # Unknown — moderate default

    factors["follower_count"] = {
        "raw": author_followers,
        "score": round(follower_score, 1),
        "weight": _SCORE_WEIGHT_FOLLOWERS,
    }

    # --- Factor 2: Recency ---
    created_at = tweet.get("created_at", "")
    recency_score = 0
    if created_at:
        try:
            tweet_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_seconds = (datetime.now(timezone.utc) - tweet_time).total_seconds()
            if age_seconds <= _RECENCY_WINDOW_SECONDS:
                # Linear decay: 100 at 0s, 0 at 3600s
                recency_score = max(0, 100 * (1 - age_seconds / _RECENCY_WINDOW_SECONDS))
            # Else: 0 (too old for high-visibility replies)
        except (ValueError, TypeError):
            recency_score = 30  # Parse error — moderate default

    factors["recency"] = {
        "age_seconds": round((datetime.now(timezone.utc) - datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        )).total_seconds()) if created_at else None,
        "score": round(recency_score, 1),
        "weight": _SCORE_WEIGHT_RECENCY,
    }

    # --- Factor 3: Topic relevance (heuristic) ---
    # Without full NLP, use reply count as a proxy for "interesting topic"
    # (more replies = trending topic = more visibility for our reply too)
    reply_count = metrics.get("replies", 0)
    like_count = metrics.get("likes", 0)
    total_engagement = reply_count + like_count
    if total_engagement > 100:
        relevance_score = 90
    elif total_engagement > 50:
        relevance_score = 70
    elif total_engagement > 10:
        relevance_score = 50
    elif total_engagement > 0:
        relevance_score = 30
    else:
        relevance_score = 15

    factors["topic_relevance"] = {
        "engagement_signals": total_engagement,
        "score": relevance_score,
        "weight": _SCORE_WEIGHT_RELEVANCE,
    }

    # --- Factor 4: Conversation size ---
    # Fewer existing replies = better chance of being seen
    if reply_count == 0:
        convo_score = 100  # First reply — maximum visibility
    elif reply_count < 5:
        convo_score = 80
    elif reply_count < 20:
        convo_score = 50
    elif reply_count < 100:
        convo_score = 25
    else:
        convo_score = 10  # Crowded — low visibility

    factors["conversation_size"] = {
        "existing_replies": reply_count,
        "score": convo_score,
        "weight": _SCORE_WEIGHT_CONVERSATION_SIZE,
    }

    # --- Factor 5: Author reply rate ---
    # Check engagement history with this author
    state = _state_store.read()
    engagement_data = state.get("engagement_by_target", {}).get(author, {})
    replies_sent = engagement_data.get("replies_sent", 0)
    replies_received = engagement_data.get("replies_received", 0)
    if replies_sent > 0:
        reply_rate = replies_received / replies_sent
        reply_rate_score = min(100, reply_rate * 100)
    else:
        reply_rate_score = 50  # No history — neutral default

    factors["author_reply_rate"] = {
        "replies_sent": replies_sent,
        "replies_received": replies_received,
        "score": round(reply_rate_score, 1),
        "weight": _SCORE_WEIGHT_REPLY_RATE,
    }

    # --- Weighted composite score ---
    weighted_score = (
        factors["follower_count"]["score"] * _SCORE_WEIGHT_FOLLOWERS
        + factors["recency"]["score"] * _SCORE_WEIGHT_RECENCY
        + factors["topic_relevance"]["score"] * _SCORE_WEIGHT_RELEVANCE
        + factors["conversation_size"]["score"] * _SCORE_WEIGHT_CONVERSATION_SIZE
        + factors["author_reply_rate"]["score"] * _SCORE_WEIGHT_REPLY_RATE
    )
    final_score = round(min(100, max(0, weighted_score)), 1)

    # Generate recommendation
    if final_score >= 75:
        recommendation = "High priority — reply immediately for maximum impact"
    elif final_score >= 50:
        recommendation = "Good opportunity — worth crafting a thoughtful reply"
    elif final_score >= 25:
        recommendation = "Moderate — reply if you have a strong angle"
    else:
        recommendation = "Low priority — skip unless highly relevant to brand"

    return {
        "score": final_score,
        "factors": factors,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 6. Growth State Management (reply history, engagement tracking)
# ---------------------------------------------------------------------------

def record_reply(
    target_tweet_id: str,
    target_author: str,
    reply_text: str,
    confidence: float = 0.5,
    content_type: str = "reply",
) -> None:
    """Record a sent reply in the growth state for tracking and analysis.

    Args:
        target_tweet_id: The tweet ID we replied to.
        target_author: The author of the target tweet.
        reply_text: The reply text we sent.
        confidence: The confidence score from draft generation.
        content_type: The content type label.
    """
    with _lock:
        state = _state_store.read()
        history = state.setdefault("reply_history", [])
        history.append({
            "target_tweet_id": target_tweet_id,
            "target_author": target_author,
            "reply_text": reply_text[:280],
            "confidence": confidence,
            "content_type": content_type,
            "sent_at": _now_iso(),
            "outcome": None,  # Updated later if the author engages back
        })

        # Cap reply history at 1000 entries
        if len(history) > 1000:
            state["reply_history"] = history[-1000:]

        # Update per-target engagement counts
        eng = state.setdefault("engagement_by_target", {})
        target_eng = eng.setdefault(target_author, {
            "replies_sent": 0,
            "replies_received": 0,
            "first_engaged": _now_iso(),
        })
        target_eng["replies_sent"] = target_eng.get("replies_sent", 0) + 1
        target_eng["last_engaged"] = _now_iso()

        _state_store.write(state)

    logger.info("Recorded reply to @%s (tweet %s)", target_author, target_tweet_id)


def get_daily_reply_count() -> int:
    """Return the number of replies sent today (UTC).

    Used to enforce the daily reply limit from settings.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    state = _state_store.read()
    count = 0
    for reply in state.get("reply_history", []):
        sent_at = reply.get("sent_at", "")
        if sent_at.startswith(today):
            count += 1
    return count


def can_reply() -> tuple[bool, str]:
    """Check if we are allowed to send another reply today.

    Returns:
        (allowed, reason) tuple.
    """
    if not getattr(settings, "GROWTH_ENGINE_ENABLED", False):
        return False, "Growth engine is disabled"

    daily_limit = getattr(settings, "GROWTH_REPLY_DAILY_LIMIT", 20)
    sent_today = get_daily_reply_count()
    if sent_today >= daily_limit:
        return False, f"Daily reply limit reached ({sent_today}/{daily_limit})"

    return True, "OK"
