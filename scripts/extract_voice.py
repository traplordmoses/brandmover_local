#!/usr/bin/env python3
"""
Voice Extractor — scrape a personal X account and generate a writing voice profile.

Analyzes tweet patterns (sentence structure, capitalization, punctuation, humor,
slang, vocabulary) and outputs a voice_profile.md that the BrandMover agent
loads automatically from brand/references/.

Usage:
    python scripts/extract_voice.py --handle @yourhandle
    python scripts/extract_voice.py --handle @yourhandle --max-tweets 100
    python scripts/extract_voice.py --handle @yourhandle --output brand/references/voice_profile.md

Requires:
    - X_BEARER_TOKEN in .env
    - ANTHROPIC_API_KEY in .env
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import anthropic
import tweepy

from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# X Scraping — deep tweet pull
# ---------------------------------------------------------------------------

def scrape_tweets(handle: str, max_tweets: int = 100) -> dict:
    """Fetch tweets, profile, and engagement data from an X account."""
    handle = handle.lstrip("@")
    bearer = settings.X_BEARER_TOKEN
    if not bearer:
        logger.error("X_BEARER_TOKEN not set in .env — cannot scrape X")
        sys.exit(1)

    client = tweepy.Client(bearer_token=bearer)

    # Profile
    user = client.get_user(
        username=handle,
        user_fields=["description", "name", "public_metrics", "url", "created_at"],
    )
    if not user.data:
        logger.error("User @%s not found", handle)
        sys.exit(1)

    user_data = user.data
    user_id = user_data.id

    logger.info("Found @%s (%s) — %s followers",
                handle, user_data.name,
                f"{user_data.public_metrics.get('followers_count', 0):,}" if user_data.public_metrics else "?")

    # Fetch tweets in batches (API max 100 per request)
    all_tweets = []
    pagination_token = None
    remaining = max_tweets

    while remaining > 0:
        batch_size = min(remaining, 100)
        try:
            tweets = client.get_users_tweets(
                user_id,
                max_results=batch_size,
                tweet_fields=["created_at", "public_metrics", "text", "in_reply_to_user_id"],
                exclude=["retweets"],
                pagination_token=pagination_token,
            )
        except Exception as e:
            logger.warning("Tweet fetch error: %s", e)
            break

        if not tweets.data:
            break

        for t in tweets.data:
            is_reply = t.in_reply_to_user_id is not None
            metrics = t.public_metrics or {}
            all_tweets.append({
                "text": t.text,
                "is_reply": is_reply,
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "created_at": str(t.created_at) if t.created_at else "",
            })
            remaining -= 1

        # Check for next page
        if tweets.meta and tweets.meta.get("next_token"):
            pagination_token = tweets.meta["next_token"]
        else:
            break

    # Separate original posts from replies
    originals = [t for t in all_tweets if not t["is_reply"]]
    replies = [t for t in all_tweets if t["is_reply"]]

    # Sort originals by engagement
    originals.sort(key=lambda x: x["likes"] + x["retweets"], reverse=True)

    logger.info("Scraped %d tweets (%d original, %d replies)",
                len(all_tweets), len(originals), len(replies))

    return {
        "handle": f"@{handle}",
        "name": user_data.name,
        "bio": user_data.description or "",
        "followers": user_data.public_metrics.get("followers_count", 0) if user_data.public_metrics else 0,
        "originals": originals,
        "replies": replies,
        "total_scraped": len(all_tweets),
    }


# ---------------------------------------------------------------------------
# Voice Analysis via Claude
# ---------------------------------------------------------------------------

VOICE_ANALYSIS_PROMPT = """You are a linguistic analyst specializing in writing voice and personal brand. Analyze the following tweets from a personal X/Twitter account and create a detailed **Voice Profile** that can be used to make an AI agent write in this person's exact style.

## Tweet Data

**Account:** {handle} ({name})
**Bio:** {bio}

### Top Original Tweets (by engagement):
{top_originals}

### Recent Original Tweets:
{recent_originals}

### Sample Replies:
{sample_replies}

## Analysis Instructions

Create a comprehensive voice profile covering ALL of these dimensions. Be extremely specific — use actual examples from the tweets to illustrate each pattern. The goal is that someone reading this profile could write tweets that are indistinguishable from this person's actual posts.

Output a markdown document with this structure:

# Voice Profile: {handle}

## Core Voice Identity
[2-3 sentence summary of their overall writing personality]

## Sentence Structure
- Average sentence length pattern
- Fragment usage
- Run-on tendencies
- How they open tweets (hooks, questions, statements, etc.)
- How they close tweets (CTAs, trailing thoughts, punchlines, etc.)

## Capitalization & Formatting
- Case patterns (lowercase, Title Case, ALL CAPS, mixed)
- When they break pattern (what triggers caps, what stays lowercase)
- Line break usage
- Use of ellipsis, em dashes, colons

## Punctuation Style
- Period vs no-period at end of tweets
- Question mark usage patterns
- Exclamation mark frequency and context
- Comma patterns
- Quotation mark usage

## Vocabulary & Word Choice
- Signature words/phrases they repeat
- Technical vs casual language balance
- Slang and informal language
- Words they never seem to use
- How they refer to themselves (I, we, impersonal)

## Humor & Wit Style
- Type of humor (dry, absurdist, self-deprecating, observational, none)
- How they deliver punchlines
- Sarcasm patterns
- Meme literacy level

## Emoji & Punctuation Decoration
- Emoji frequency (per-tweet average)
- Preferred emojis
- Placement patterns (start, end, inline)
- Use of special characters (arrows, bullets, etc.)

## Topic Tendencies
- What they talk about most
- How they frame technical topics for general audience
- Recurring themes or obsessions
- What they avoid talking about

## Engagement Style
- How they ask questions
- How they make CTAs
- Reply style vs original tweet style
- Thread structure (if applicable)

## Rhetorical Devices
- Repetition patterns
- Parallel structure
- Contrast/juxtaposition
- Storytelling tendencies

## Voice Rules (Do / Don't)
### DO:
[Bullet list of specific patterns to follow]

### DON'T:
[Bullet list of patterns to avoid]

## Example Voice Samples
[5-8 of their most characteristic tweets that best exemplify the voice, with brief annotation of WHY each is characteristic]

Output ONLY the markdown document. Be specific and actionable — every pattern should be backed by evidence from the actual tweets."""


def analyze_voice(data: dict) -> str:
    """Send tweet data to Claude for deep voice analysis."""
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — cannot analyze voice")
        sys.exit(1)

    # Prepare tweet samples for the prompt
    originals = data["originals"]
    replies = data["replies"]

    top_originals = "\n\n".join(
        f"[{t['likes']} likes, {t['retweets']} RTs]\n{t['text']}"
        for t in originals[:25]
    )

    recent_originals = "\n\n".join(
        f"{t['text']}"
        for t in originals[25:60]
    ) if len(originals) > 25 else "(same as above)"

    sample_replies = "\n\n".join(
        f"{t['text']}"
        for t in replies[:20]
    ) if replies else "(no replies in sample)"

    prompt = VOICE_ANALYSIS_PROMPT.format(
        handle=data["handle"],
        name=data["name"],
        bio=data["bio"],
        top_originals=top_originals,
        recent_originals=recent_originals,
        sample_replies=sample_replies,
    )

    logger.info("Analyzing voice patterns with Claude (%d originals, %d replies)...",
                len(originals), len(replies))

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract a writing voice profile from a personal X account.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The voice profile is saved to brand/references/voice_profile.md and automatically
loaded by the BrandMover agent as part of brand context. This means the agent will
adopt your personal writing style when generating content for your brand.

Examples:
  python scripts/extract_voice.py --handle @yourhandle
  python scripts/extract_voice.py --handle @yourhandle --max-tweets 100
  python scripts/extract_voice.py --handle @yourhandle --dry-run
        """,
    )
    parser.add_argument("--handle", required=True,
                        help="X/Twitter handle to scrape (e.g. @yourhandle)")
    parser.add_argument("--max-tweets", type=int, default=100,
                        help="Max tweets to fetch (default: 100)")
    parser.add_argument("--output",
                        default=str(_project_root / "brand" / "references" / "voice_profile.md"),
                        help="Output path (default: brand/references/voice_profile.md)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scrape and print tweet data without LLM analysis")
    parser.add_argument("--save-cache", metavar="PATH",
                        help="Save scraped tweet data to JSON file for reuse")
    parser.add_argument("--from-cache", metavar="PATH",
                        help="Load tweet data from a cached JSON file instead of scraping")

    args = parser.parse_args()

    # Load from cache or scrape
    if args.from_cache:
        cache_path = Path(args.from_cache)
        if not cache_path.exists():
            logger.error("Cache file not found: %s", cache_path)
            sys.exit(1)
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("Loaded %d tweets from cache: %s", data.get("total_scraped", 0), cache_path)
    else:
        data = scrape_tweets(args.handle, max_tweets=args.max_tweets)

    # Save cache if requested (or always save to a default location)
    cache_out = args.save_cache or str(_project_root / ".cache" / f"tweets_{args.handle.lstrip('@')}.json")
    Path(cache_out).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_out).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Tweet data cached to: %s", cache_out)

    if args.dry_run:
        print(f"\n=== {data.get('handle', '?')} — {data.get('name', '?')} ===")
        print(f"Bio: {data.get('bio', '')}")
        print(f"Followers: {data.get('followers', 0):,}")
        print(f"Tweets scraped: {data.get('total_scraped', 0)} ({len(data.get('originals', []))} original, {len(data.get('replies', []))} replies)")
        print(f"\n--- Top 10 tweets by engagement ---")
        for t in data.get("originals", [])[:10]:
            print(f"\n[{t['likes']} likes] {t['text'][:200]}")
        print(f"\nCached to: {cache_out}")
        return

    # Analyze
    profile = analyze_voice(data)
    if not profile:
        logger.error("Voice analysis failed")
        sys.exit(1)

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(profile, encoding="utf-8")
    logger.info("Voice profile written to: %s", output_path)

    print(f"\nVoice profile saved to: {output_path}")
    print(f"The agent will automatically load this as part of brand context.")
    print(f"\nTo regenerate with more data:")
    print(f"  python scripts/extract_voice.py --handle {args.handle} --max-tweets 200")


if __name__ == "__main__":
    main()
