"""
Content Performance Tracker — monitors X/Twitter post engagement.

Tracks likes, retweets, impressions for posted content and feeds
performance data back into the generation pipeline to learn what works.

Public API:
    await track_post_performance(tweet_id)
    stats = get_performance_summary()
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

PERFORMANCE_FILE = PROJECT_ROOT / "state" / "performance.json"


@dataclass
class PostMetrics:
    """Metrics for a single posted piece of content."""
    tweet_id: str = ""
    posted_at: float = 0.0
    content_type: str = ""
    caption_preview: str = ""
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0
    last_checked: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tweet_id": self.tweet_id,
            "posted_at": self.posted_at,
            "content_type": self.content_type,
            "caption_preview": self.caption_preview,
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "impressions": self.impressions,
            "engagement_rate": self.engagement_rate,
            "last_checked": self.last_checked,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PostMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _load_performance() -> list[dict]:
    """Load performance data from disk."""
    if not PERFORMANCE_FILE.exists():
        return []
    try:
        return json.loads(PERFORMANCE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []


def _save_performance(data: list[dict]) -> None:
    """Save performance data to disk."""
    PERFORMANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def record_post(tweet_id: str, content_type: str = "", caption: str = "") -> None:
    """Record a new post for tracking."""
    data = _load_performance()
    data.append({
        "tweet_id": tweet_id,
        "posted_at": time.time(),
        "content_type": content_type,
        "caption_preview": caption[:100] if caption else "",
        "likes": 0,
        "retweets": 0,
        "replies": 0,
        "impressions": 0,
        "engagement_rate": 0.0,
        "last_checked": 0.0,
    })
    # Keep last 200 posts
    if len(data) > 200:
        data = data[-200:]
    _save_performance(data)
    logger.info("Recorded post for tracking: %s", tweet_id)


def update_metrics(tweet_id: str, likes: int, retweets: int,
                   replies: int = 0, impressions: int = 0) -> None:
    """Update metrics for a tracked post."""
    data = _load_performance()
    for entry in data:
        if entry["tweet_id"] == tweet_id:
            entry["likes"] = likes
            entry["retweets"] = retweets
            entry["replies"] = replies
            entry["impressions"] = impressions
            if impressions > 0:
                entry["engagement_rate"] = round(
                    (likes + retweets + replies) / impressions * 100, 2
                )
            entry["last_checked"] = time.time()
            break
    _save_performance(data)


async def fetch_post_metrics(tweet_id: str) -> dict | None:
    """Fetch current metrics for a tweet from X API.

    Returns dict with likes, retweets, replies, impressions or None if failed.
    """
    try:
        from agent.publisher import get_tweepy_client
        client = get_tweepy_client()
        if not client:
            return None

        tweet = await asyncio.to_thread(
            client.get_tweet,
            tweet_id,
            tweet_fields=["public_metrics"],
        )

        if tweet and tweet.data:
            metrics = tweet.data.get("public_metrics", {})
            return {
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "impressions": metrics.get("impression_count", 0),
            }
    except Exception as e:
        logger.warning("Failed to fetch metrics for tweet %s: %s", tweet_id, e)
    return None


async def track_post_performance(tweet_id: str) -> PostMetrics | None:
    """Fetch and update metrics for a tracked post."""
    metrics = await fetch_post_metrics(tweet_id)
    if metrics:
        update_metrics(tweet_id, **metrics)
        data = _load_performance()
        for entry in data:
            if entry["tweet_id"] == tweet_id:
                return PostMetrics.from_dict(entry)
    return None


def get_performance_summary(last_n: int = 20) -> dict:
    """Get a summary of recent post performance.

    Returns dict with top posts, averages, and content type breakdown.
    """
    data = _load_performance()
    if not data:
        return {"total_posts": 0, "message": "No posts tracked yet."}

    recent = data[-last_n:]

    # Averages
    total = len(recent)
    avg_likes = sum(p.get("likes", 0) for p in recent) / total
    avg_retweets = sum(p.get("retweets", 0) for p in recent) / total
    avg_engagement = sum(p.get("engagement_rate", 0) for p in recent) / total

    # Top posts by engagement
    sorted_posts = sorted(recent, key=lambda p: p.get("likes", 0) + p.get("retweets", 0), reverse=True)
    top_posts = sorted_posts[:5]

    # Content type breakdown
    type_stats = {}
    for p in recent:
        ct = p.get("content_type", "unknown")
        if ct not in type_stats:
            type_stats[ct] = {"count": 0, "total_likes": 0, "total_retweets": 0}
        type_stats[ct]["count"] += 1
        type_stats[ct]["total_likes"] += p.get("likes", 0)
        type_stats[ct]["total_retweets"] += p.get("retweets", 0)

    for ct in type_stats:
        n = type_stats[ct]["count"]
        type_stats[ct]["avg_likes"] = round(type_stats[ct]["total_likes"] / n, 1)
        type_stats[ct]["avg_retweets"] = round(type_stats[ct]["total_retweets"] / n, 1)

    return {
        "total_posts": len(data),
        "recent_count": total,
        "avg_likes": round(avg_likes, 1),
        "avg_retweets": round(avg_retweets, 1),
        "avg_engagement_rate": round(avg_engagement, 2),
        "top_posts": [
            {
                "tweet_id": p["tweet_id"],
                "caption": p.get("caption_preview", ""),
                "likes": p.get("likes", 0),
                "retweets": p.get("retweets", 0),
                "engagement_rate": p.get("engagement_rate", 0),
            }
            for p in top_posts
        ],
        "by_content_type": type_stats,
    }


async def refresh_recent_metrics(max_posts: int = 10) -> int:
    """Fetch fresh metrics for the most recent tracked posts.

    Called periodically from the auto-post scheduler to keep
    performance data current. Returns number of posts updated.
    """
    data = _load_performance()
    if not data:
        return 0

    now = time.time()
    refresh_interval = 6 * 3600  # 6 hours
    try:
        from config import settings
        refresh_interval = settings.PERFORMANCE_REFRESH_HOURS * 3600
    except Exception:
        pass

    # Find posts that haven't been checked recently, most recent first
    candidates = sorted(data, key=lambda p: p.get("posted_at", 0), reverse=True)
    stale = [
        p for p in candidates
        if now - p.get("last_checked", 0) > refresh_interval
    ][:max_posts]

    updated = 0
    for entry in stale:
        tweet_id = entry.get("tweet_id", "")
        if not tweet_id:
            continue
        metrics = await fetch_post_metrics(tweet_id)
        if metrics:
            update_metrics(tweet_id, **metrics)
            updated += 1

    if updated:
        logger.info("Refreshed metrics for %d/%d posts", updated, len(stale))
    return updated


def get_performance_context() -> str:
    """Generate a brief performance context string for the system prompt.

    This helps the LLM understand what content performs well.
    """
    summary = get_performance_summary()
    if summary.get("total_posts", 0) == 0:
        return ""

    lines = [
        f"Content performance (last {summary['recent_count']} posts): "
        f"avg {summary['avg_likes']} likes, {summary['avg_retweets']} RTs, "
        f"{summary['avg_engagement_rate']}% engagement rate."
    ]

    if summary.get("by_content_type"):
        best_type = max(
            summary["by_content_type"].items(),
            key=lambda x: x[1].get("avg_likes", 0),
        )
        lines.append(f"Best performing type: {best_type[0]} ({best_type[1]['avg_likes']} avg likes).")

    return " ".join(lines)
