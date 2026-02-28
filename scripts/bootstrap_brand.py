#!/usr/bin/env python3
"""
Brand Bootstrap — auto-generate brand guidelines by scraping X accounts and websites.

Usage:
    python scripts/bootstrap_brand.py --x-handle @FOIDFUN --website https://foid.fun
    python scripts/bootstrap_brand.py --x-handle @FOIDFUN --website https://foid.fun --reference pitch_deck.pdf
    python scripts/bootstrap_brand.py --x-handle @YourBrand  # X only, no website

Requires:
    - ANTHROPIC_API_KEY in .env (for Claude analysis)
    - X_BEARER_TOKEN in .env (for X/Twitter scraping)

Output:
    Writes a filled-in brand/guidelines.md to the brand folder.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import anthropic
import httpx
import tweepy
from bs4 import BeautifulSoup

from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# X / Twitter Scraping
# ---------------------------------------------------------------------------

def scrape_x_account(handle: str, max_tweets: int = 50) -> dict:
    """Fetch recent tweets and profile info from an X account using Bearer Token."""
    handle = handle.lstrip("@")
    bearer = settings.X_BEARER_TOKEN
    if not bearer:
        logger.warning("X_BEARER_TOKEN not set — skipping X scraping")
        return {}

    try:
        client = tweepy.Client(bearer_token=bearer)

        # Get user info
        user = client.get_user(
            username=handle,
            user_fields=["description", "profile_image_url", "public_metrics", "url", "name"],
        )
        if not user.data:
            logger.error("User @%s not found", handle)
            return {}

        user_data = user.data
        user_id = user_data.id

        # Get recent tweets (exclude replies and retweets for cleaner voice sample)
        tweets = client.get_users_tweets(
            user_id,
            max_results=min(max_tweets, 100),
            tweet_fields=["created_at", "public_metrics", "text"],
            exclude=["replies", "retweets"],
        )

        tweet_texts = []
        if tweets.data:
            for t in tweets.data:
                tweet_texts.append({
                    "text": t.text,
                    "likes": t.public_metrics.get("like_count", 0) if t.public_metrics else 0,
                    "retweets": t.public_metrics.get("retweet_count", 0) if t.public_metrics else 0,
                })
            # Sort by engagement
            tweet_texts.sort(key=lambda x: x["likes"] + x["retweets"], reverse=True)

        return {
            "handle": f"@{handle}",
            "name": user_data.name,
            "bio": user_data.description or "",
            "followers": user_data.public_metrics.get("followers_count", 0) if user_data.public_metrics else 0,
            "url": user_data.url or "",
            "tweets": tweet_texts[:max_tweets],
            "tweet_count": len(tweet_texts),
        }

    except Exception as e:
        logger.error("X scraping failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Website Scraping
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_RGB_RE = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_FONT_RE = re.compile(r"font-family\s*:\s*([^;}{]+)", re.IGNORECASE)
_GFONT_RE = re.compile(r"fonts\.googleapis\.com/css2?\?family=([^&\"']+)")


def _normalize_hex(c: str) -> str:
    """Normalize 3-char hex to 6-char."""
    c = c.lower()
    if len(c) == 4:  # #abc
        return f"#{c[1]*2}{c[2]*2}{c[3]*2}"
    return c


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def scrape_website(url: str) -> dict:
    """Scrape a website for brand signals: colors, fonts, meta tags, copy."""
    if not url.startswith("http"):
        url = f"https://{url}"

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (compatible; BrandMover/1.0)"
        })
        resp.raise_for_status()
    except Exception as e:
        logger.error("Website fetch failed for %s: %s", url, e)
        return {}

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # --- Meta tags ---
    meta = {}
    for tag in soup.find_all("meta"):
        name = tag.get("name", "") or tag.get("property", "")
        content = tag.get("content", "")
        if name and content:
            meta[name.lower()] = content

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # --- Colors from HTML + inline styles ---
    all_text = html
    # Also try to fetch linked CSS (first 3 stylesheets only)
    css_texts = []
    for link in soup.find_all("link", rel="stylesheet")[:3]:
        href = link.get("href", "")
        if href:
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            try:
                css_resp = httpx.get(href, follow_redirects=True, timeout=10, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; BrandMover/1.0)"
                })
                if css_resp.status_code == 200:
                    css_texts.append(css_resp.text)
            except Exception:
                pass

    all_text += "\n".join(css_texts)

    # Extract hex colors
    hex_colors = [_normalize_hex(c) for c in _HEX_RE.findall(all_text)]
    # Extract rgb colors
    for m in _RGB_RE.finditer(all_text):
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hex_colors.append(_rgb_to_hex(r, g, b))

    # Count frequency, filter out pure black/white/grays
    color_freq = {}
    for c in hex_colors:
        if c in ("#000000", "#ffffff", "#fff", "#000"):
            continue
        # Skip near-grays
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        if max(r, g, b) - min(r, g, b) < 20:
            continue
        color_freq[c] = color_freq.get(c, 0) + 1

    # Top colors by frequency
    top_colors = sorted(color_freq.items(), key=lambda x: -x[1])[:12]

    # --- Fonts ---
    fonts_found = set()
    for m in _FONT_RE.finditer(all_text):
        raw = m.group(1).strip().strip("'\"")
        # Take first font in stack
        first = raw.split(",")[0].strip().strip("'\"")
        if first and first.lower() not in ("inherit", "initial", "sans-serif", "serif", "monospace", "system-ui", "ui-sans-serif", "ui-monospace"):
            fonts_found.add(first)

    # Google Fonts
    for m in _GFONT_RE.finditer(all_text):
        for fam in m.group(1).split("|"):
            name = fam.split(":")[0].replace("+", " ")
            if name:
                fonts_found.add(name)

    # --- Key copy: headings, hero text ---
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"])[:10]:
        text = tag.get_text(strip=True)
        if text and len(text) < 200:
            headings.append(text)

    # Paragraphs (first few meaningful ones)
    paragraphs = []
    for p in soup.find_all("p")[:15]:
        text = p.get_text(strip=True)
        if text and len(text) > 30 and len(text) < 500:
            paragraphs.append(text)

    # Theme color meta
    theme_color = meta.get("theme-color", "")

    return {
        "url": url,
        "title": title,
        "description": meta.get("description", meta.get("og:description", "")),
        "og_image": meta.get("og:image", ""),
        "theme_color": theme_color,
        "top_colors": [{"hex": c, "count": n} for c, n in top_colors],
        "fonts": sorted(fonts_found),
        "headings": headings,
        "sample_copy": paragraphs[:8],
    }


# ---------------------------------------------------------------------------
# Reference file reading (PDF, text, markdown)
# ---------------------------------------------------------------------------

def read_reference_file(path: str) -> str:
    """Read a reference file and return its text content."""
    p = Path(path)
    if not p.exists():
        logger.warning("Reference file not found: %s", path)
        return ""

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        try:
            import pymupdf
            doc = pymupdf.open(str(p))
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text[:15000]  # Cap at 15k chars
        except ImportError:
            logger.warning("pymupdf not installed — cannot read PDF. pip install pymupdf")
            return ""

    # Text, markdown, etc.
    try:
        return p.read_text(encoding="utf-8")[:15000]
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return ""


# ---------------------------------------------------------------------------
# LLM Analysis — synthesize scraped data into brand guidelines
# ---------------------------------------------------------------------------

_GUIDELINES_TEMPLATE = Path(__file__).resolve().parent.parent / "brand" / "guidelines.md"

ANALYSIS_PROMPT = """You are a brand strategist. Analyze the following scraped data from a brand's X/Twitter account and website, and generate a complete brand guidelines document.

## Scraped Data

### X/Twitter Profile & Tweets
{x_data}

### Website Analysis
{web_data}

### Reference Material
{ref_data}

## Instructions

Generate a complete brand guidelines markdown document following this EXACT structure. Fill in every section with specific, actionable details based on the scraped data. If data is insufficient for a section, make reasonable inferences from the available content and note your confidence level.

For voice & tone: analyze the tweet texts carefully for patterns — sentence length, capitalization habits, emoji usage, punctuation style, vocabulary, slang, humor style.

For colors: use the extracted colors from the website. Identify which are likely primary, secondary, accent, background, and text colors based on frequency and context.

For visual style: infer from the website's design aesthetic, colors, and any og:image content.

The output must be a complete markdown document starting with "# [Brand Name] Brand Guidelines" that can be saved directly as brand/guidelines.md. Use the same section structure as a typical BrandMover guidelines file:

1. BRAND IDENTITY (name, tagline, website, category, X handle, product, positioning, themes, cadence)
2. VOICE & TONE (personality traits, writing style, tone by content type, emoji usage, never-use list)
3. COLORS (table with hex values and usage notes, color rules)
4. TYPOGRAPHY (font table with weights and styles)
5. LOGO USAGE (description + don'ts)
6. ILLUSTRATION STYLE (visual aesthetic, image prompt guidance, avoid list)
7. MASCOT (if evidence suggests one — otherwise note "No mascot detected")
8. HASHTAGS (brand hashtags if used, max per post)
9. CONTENT CATEGORIES (at least 6 categories with tone and length notes)
10. BRAND PHRASES & SLANG (established phrases, slang, CTAs)
11. TARGET AUDIENCE (primary, secondary, psychographics, tone match)

Be specific and actionable. Every field should have real values, not placeholders.
Output ONLY the markdown document, nothing else."""


def analyze_with_llm(x_data: dict, web_data: dict, ref_text: str) -> str:
    """Use Claude to synthesize scraped data into brand guidelines."""
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — cannot run LLM analysis")
        return ""

    client = anthropic.Anthropic(api_key=api_key)

    x_str = json.dumps(x_data, indent=2) if x_data else "No X/Twitter data available."
    web_str = json.dumps(web_data, indent=2, default=str) if web_data else "No website data available."
    ref_str = ref_text if ref_text else "No reference material provided."

    prompt = ANALYSIS_PROMPT.format(
        x_data=x_str,
        web_data=web_str,
        ref_data=ref_str,
    )

    logger.info("Sending scraped data to Claude for analysis...")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ---------------------------------------------------------------------------
# Compositor config generation
# ---------------------------------------------------------------------------

def generate_compositor_patch(colors: list[dict]) -> str:
    """Generate compositor.py color constants from extracted brand colors."""
    if not colors or len(colors) < 2:
        return ""

    # Heuristic: darkest color = BG, most saturated bright = PRIMARY, second = ACCENT
    def _brightness(hex_c: str) -> float:
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        return (r + g + b) / 3

    def _saturation(hex_c: str) -> float:
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        mx, mn = max(r, g, b), min(r, g, b)
        return (mx - mn) / max(mx, 1)

    sorted_dark = sorted(colors, key=lambda c: _brightness(c["hex"]))
    sorted_sat = sorted(colors, key=lambda c: _saturation(c["hex"]), reverse=True)

    bg = sorted_dark[0]["hex"]
    primary = sorted_sat[0]["hex"]
    accent = sorted_sat[1]["hex"] if len(sorted_sat) > 1 else sorted_sat[0]["hex"]

    def _hex_to_rgb(h: str) -> str:
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return f"({r}, {g}, {b})"

    return f"""# Auto-generated brand colors from bootstrap_brand.py
# Update these in agent/compositor.py:
BRAND_PRIMARY = {_hex_to_rgb(primary)}  # {primary}
BRAND_BG      = {_hex_to_rgb(bg)}  # {bg}
BRAND_TEXT     = (255, 255, 255)  # White
BRAND_ACCENT   = {_hex_to_rgb(accent)}  # {accent}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate brand guidelines by scraping X accounts and websites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bootstrap_brand.py --x-handle @FOIDFUN --website https://foid.fun
  python scripts/bootstrap_brand.py --x-handle @Nike --website https://nike.com
  python scripts/bootstrap_brand.py --x-handle @YourBrand --reference pitch_deck.pdf
  python scripts/bootstrap_brand.py --website https://yourbrand.com  # website only
        """,
    )
    parser.add_argument("--x-handle", help="X/Twitter handle to scrape (e.g. @FOIDFUN)")
    parser.add_argument("--website", help="Website URL to scrape (e.g. https://foid.fun)")
    parser.add_argument("--reference", action="append", default=[],
                        help="Reference file(s) — PDF, text, or markdown. Can be used multiple times.")
    parser.add_argument("--max-tweets", type=int, default=50,
                        help="Max tweets to fetch (default: 50)")
    parser.add_argument("--output", default=str(_project_root / "brand" / "guidelines.md"),
                        help="Output path for guidelines (default: brand/guidelines.md)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scraped data without running LLM analysis")
    parser.add_argument("--compositor", action="store_true",
                        help="Also print suggested compositor.py color constants")

    args = parser.parse_args()

    if not args.x_handle and not args.website:
        parser.error("Provide at least --x-handle or --website (or both)")

    # --- Scrape ---
    x_data = {}
    web_data = {}
    ref_text = ""

    if args.x_handle:
        logger.info("Scraping X account: %s", args.x_handle)
        x_data = scrape_x_account(args.x_handle, max_tweets=args.max_tweets)
        if x_data:
            logger.info("  Found %d tweets from %s (%d followers)",
                        x_data.get("tweet_count", 0),
                        x_data.get("name", "?"),
                        x_data.get("followers", 0))
        else:
            logger.warning("  No X data retrieved")

    if args.website:
        logger.info("Scraping website: %s", args.website)
        web_data = scrape_website(args.website)
        if web_data:
            logger.info("  Found %d colors, %d fonts, %d headings",
                        len(web_data.get("top_colors", [])),
                        len(web_data.get("fonts", [])),
                        len(web_data.get("headings", [])))
        else:
            logger.warning("  No website data retrieved")

    if args.reference:
        ref_parts = []
        for ref_path in args.reference:
            logger.info("Reading reference: %s", ref_path)
            text = read_reference_file(ref_path)
            if text:
                ref_parts.append(f"--- {ref_path} ---\n{text}")
        ref_text = "\n\n".join(ref_parts)

    if args.dry_run:
        print("\n=== X DATA ===")
        print(json.dumps(x_data, indent=2))
        print("\n=== WEBSITE DATA ===")
        print(json.dumps(web_data, indent=2, default=str))
        if ref_text:
            print("\n=== REFERENCE TEXT (first 2000 chars) ===")
            print(ref_text[:2000])
        if args.compositor and web_data.get("top_colors"):
            print("\n=== COMPOSITOR COLORS ===")
            print(generate_compositor_patch(web_data["top_colors"]))
        return

    # --- Analyze with LLM ---
    guidelines = analyze_with_llm(x_data, web_data, ref_text)
    if not guidelines:
        logger.error("LLM analysis failed — no guidelines generated")
        sys.exit(1)

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(guidelines, encoding="utf-8")
    logger.info("Brand guidelines written to: %s", output_path)

    # --- Compositor suggestion ---
    if args.compositor and web_data.get("top_colors"):
        patch = generate_compositor_patch(web_data["top_colors"])
        if patch:
            print("\n" + patch)

    print(f"\nDone. Review your guidelines at: {output_path}")
    print("Next steps:")
    print("  1. Review and edit brand/guidelines.md")
    print("  2. Place your logo as brand/assets/logo.png")
    print("  3. Update compositor colors in agent/compositor.py if needed")
    print("  4. Run the bot: python main.py")


if __name__ == "__main__":
    main()
