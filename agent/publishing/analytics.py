"""
X/Twitter engagement analytics — pulls post performance and feeds back into learning.

Uses X API v2 with Bearer Token to batch-fetch public metrics for recent posts,
then writes results to state/performance_data.json. The learning system reads
this data to understand what content resonates with the audience.

Public API:
    metrics = await fetch_post_metrics(["123", "456"])
    await update_performance_data()
    top = get_top_performing(limit=5)
    summary = get_performance_summary()
"""

import json
import logging
import time
from pathlib import Path

import httpx

from agent.paths import STATE_DIR
from config import settings

logger = logging.getLogger(__name__)

PERFORMANCE_DATA_FILE = STATE_DIR / "performance_data.json"

# X API v2 tweets endpoint — fetches public metrics in batch (up to 100 IDs)
_X_API_TWEETS_URL = "https://api.twitter.com/2/tweets"

# Maximum tweet IDs per request (X API limit)
_MAX_IDS_PER_REQUEST = 100

# Only fetch metrics for posts from the last 7 days
_LOOKBACK_SECONDS = 7 * 24 * 3600


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _load_performance_data() -> list[dict]:
    """Load performance data from disk."""
    if not PERFORMANCE_DATA_FILE.exists():
        return []
    try:
        return json.loads(PERFORMANCE_DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []


def _save_performance_data(data: list[dict]) -> None:
    """Save performance data to disk."""
    PERFORMANCE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_DATA_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Core API: fetch metrics from X
# ---------------------------------------------------------------------------

async def fetch_post_metrics(tweet_ids: list[str]) -> list[dict]:
    """Fetch engagement metrics from X API v2 for a batch of tweet IDs.

    Uses the Bearer Token from settings.X_BEARER_TOKEN. Returns a list of
    dicts with keys: tweet_id, likes, retweets, replies, impressions,
    engagement_rate.

    Args:
        tweet_ids: List of tweet ID strings to fetch metrics for.

    Returns:
        List of metric dicts. Only includes tweets that were successfully
        fetched (missing/deleted tweets are silently skipped).
    """
    if not tweet_ids:
        return []

    bearer_token = settings.X_BEARER_TOKEN
    if not bearer_token:
        logger.warning("X_BEARER_TOKEN not set — cannot fetch post metrics")
        return []

    # Deduplicate and limit
    unique_ids = list(dict.fromkeys(tweet_ids))  # preserve order, remove dupes
    results: list[dict] = []

    # Process in batches of 100 (X API limit)
    for batch_start in range(0, len(unique_ids), _MAX_IDS_PER_REQUEST):
        batch = unique_ids[batch_start : batch_start + _MAX_IDS_PER_REQUEST]
        ids_param = ",".join(batch)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    _X_API_TWEETS_URL,
                    params={
                        "ids": ids_param,
                        "tweet.fields": "public_metrics",
                    },
                    headers={
                        "Authorization": f"Bearer {bearer_token}",
                    },
                )
                resp.raise_for_status()
                body = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "X API returned %d fetching metrics: %s",
                e.response.status_code,
                e.response.text[:200],
            )
            continue
        except httpx.HTTPError as e:
            logger.error("HTTP error fetching X metrics: %s", e)
            continue

        tweets = body.get("data", [])
        for tweet in tweets:
            pm = tweet.get("public_metrics", {})
            likes = pm.get("like_count", 0)
            retweets = pm.get("retweet_count", 0)
            replies = pm.get("reply_count", 0)
            impressions = pm.get("impression_count", 0)

            engagement_rate = 0.0
            if impressions > 0:
                engagement_rate = round(
                    (likes + retweets + replies) / impressions * 100, 4
                )

            results.append({
                "tweet_id": tweet["id"],
                "likes": likes,
                "retweets": retweets,
                "replies": replies,
                "impressions": impressions,
                "engagement_rate": engagement_rate,
            })

        # Log errors for individual tweets (e.g., deleted)
        errors = body.get("errors", [])
        for err in errors:
            logger.debug(
                "X API error for tweet %s: %s",
                err.get("resource_id", "?"),
                err.get("detail", "unknown"),
            )

    logger.info("Fetched metrics for %d/%d tweets", len(results), len(unique_ids))
    return results


# ---------------------------------------------------------------------------
# Update performance data from generation history + auto_post_state
# ---------------------------------------------------------------------------

def _extract_tweet_ids_from_history() -> dict[str, dict]:
    """Scan auto_post_state and performance.json for recent tweet IDs.

    Returns {tweet_id: {content_type, caption_preview, posted_at}}.
    """
    tweet_map: dict[str, dict] = {}
    now = time.time()
    cutoff = now - _LOOKBACK_SECONDS

    # Source 1: performance.json (recorded by performance.record_post)
    perf_file = STATE_DIR / "performance.json"
    if perf_file.exists():
        try:
            entries = json.loads(perf_file.read_text(encoding="utf-8"))
            for entry in entries:
                tid = entry.get("tweet_id", "")
                posted_at = entry.get("posted_at", 0)
                if tid and posted_at > cutoff:
                    tweet_map[tid] = {
                        "content_type": entry.get("content_type", ""),
                        "caption_preview": entry.get("caption_preview", ""),
                        "posted_at": posted_at,
                    }
        except (json.JSONDecodeError, IOError):
            pass

    # Source 2: auto_post_state.json (posts_today entries with tweet_url)
    auto_state_file = STATE_DIR / "auto_post_state.json"
    if auto_state_file.exists():
        try:
            state = json.loads(auto_state_file.read_text(encoding="utf-8"))
            for post in state.get("posts_today", []):
                tweet_url = post.get("tweet_url", "")
                ts = post.get("timestamp", 0)
                if tweet_url and ts > cutoff:
                    tid = tweet_url.rstrip("/").split("/")[-1]
                    if tid and tid not in tweet_map:
                        tweet_map[tid] = {
                            "content_type": "",
                            "caption_preview": post.get("caption", "")[:100],
                            "posted_at": ts,
                        }
        except (json.JSONDecodeError, IOError):
            pass

    return tweet_map


async def update_performance_data() -> int:
    """Fetch fresh metrics for all recent posts and write to performance_data.json.

    Reads tweet IDs from generation history / auto_post_state, fetches metrics
    from X API, merges with existing data, and saves. Returns number of tweets
    updated.
    """
    tweet_map = _extract_tweet_ids_from_history()
    if not tweet_map:
        logger.info("No recent tweets to fetch metrics for")
        return 0

    tweet_ids = list(tweet_map.keys())
    metrics_list = await fetch_post_metrics(tweet_ids)

    # Build a lookup from fetched metrics
    metrics_by_id = {m["tweet_id"]: m for m in metrics_list}

    # Merge with existing data
    existing = _load_performance_data()
    existing_by_id = {e["tweet_id"]: e for e in existing}

    now = time.time()
    for tid, meta in tweet_map.items():
        fetched = metrics_by_id.get(tid, {})
        entry = existing_by_id.get(tid, {})

        entry.update({
            "tweet_id": tid,
            "content_type": meta.get("content_type", entry.get("content_type", "")),
            "caption_preview": meta.get("caption_preview", entry.get("caption_preview", "")),
            "posted_at": meta.get("posted_at", entry.get("posted_at", 0)),
            "likes": fetched.get("likes", entry.get("likes", 0)),
            "retweets": fetched.get("retweets", entry.get("retweets", 0)),
            "replies": fetched.get("replies", entry.get("replies", 0)),
            "impressions": fetched.get("impressions", entry.get("impressions", 0)),
            "engagement_rate": fetched.get("engagement_rate", entry.get("engagement_rate", 0.0)),
            "last_checked": now if fetched else entry.get("last_checked", 0),
        })
        existing_by_id[tid] = entry

    # Write merged data (keep last 200 entries, sorted by posted_at)
    all_entries = sorted(existing_by_id.values(), key=lambda e: e.get("posted_at", 0))
    if len(all_entries) > 200:
        all_entries = all_entries[-200:]

    _save_performance_data(all_entries)

    updated_count = len(metrics_list)
    logger.info(
        "Updated performance data: %d tweets fetched, %d total tracked",
        updated_count,
        len(all_entries),
    )
    return updated_count


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_top_performing(limit: int = 5) -> list[dict]:
    """Return the top posts by engagement rate.

    Args:
        limit: Maximum number of posts to return.

    Returns:
        List of post dicts sorted by engagement_rate descending.
    """
    data = _load_performance_data()
    if not data:
        return []

    # Sort by engagement rate, then by likes as tiebreaker
    sorted_data = sorted(
        data,
        key=lambda p: (p.get("engagement_rate", 0), p.get("likes", 0)),
        reverse=True,
    )
    return sorted_data[:limit]


def get_performance_summary() -> str:
    """Generate a human-readable performance summary for agent context injection.

    Returns a compact multi-line string summarizing recent engagement trends,
    top-performing content types, and best posts.
    """
    data = _load_performance_data()
    if not data:
        return "No engagement data available yet."

    # Only consider posts with actual metrics (last_checked > 0)
    measured = [p for p in data if p.get("last_checked", 0) > 0]
    if not measured:
        return f"{len(data)} posts tracked but no metrics fetched yet."

    total = len(measured)
    avg_likes = sum(p.get("likes", 0) for p in measured) / total
    avg_rts = sum(p.get("retweets", 0) for p in measured) / total
    avg_engagement = sum(p.get("engagement_rate", 0) for p in measured) / total

    lines = [
        f"X engagement ({total} posts measured): "
        f"avg {avg_likes:.1f} likes, {avg_rts:.1f} RTs, "
        f"{avg_engagement:.2f}% engagement rate.",
    ]

    # Best content type by engagement
    type_stats: dict[str, dict] = {}
    for p in measured:
        ct = p.get("content_type") or "unknown"
        stats = type_stats.setdefault(ct, {"count": 0, "total_engagement": 0.0})
        stats["count"] += 1
        stats["total_engagement"] += p.get("engagement_rate", 0)

    if type_stats:
        best_type = max(
            type_stats.items(),
            key=lambda x: x[1]["total_engagement"] / x[1]["count"] if x[1]["count"] else 0,
        )
        avg_eng = best_type[1]["total_engagement"] / best_type[1]["count"]
        lines.append(
            f"Best type: {best_type[0]} ({avg_eng:.2f}% avg engagement, "
            f"{best_type[1]['count']} posts)."
        )

    # Top 3 posts
    top = get_top_performing(limit=3)
    if top:
        top_lines = []
        for p in top:
            preview = p.get("caption_preview", "")[:50]
            top_lines.append(
                f"  - \"{preview}...\" ({p.get('likes', 0)} likes, "
                f"{p.get('engagement_rate', 0):.2f}% eng)"
            )
        lines.append("Top posts:\n" + "\n".join(top_lines))

    return "\n".join(lines)
