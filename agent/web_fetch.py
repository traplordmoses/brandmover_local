"""
Web fetch tool — fetches and extracts readable content from URLs.

Returns structured metadata (title, OG tags) + extracted text content.
Handles tweet URLs, articles, and general web pages.
"""

import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENT = "BrandMover/1.0 (AI Marketing Agent)"
_TIMEOUT_SECONDS = 10

# Tags to strip before extracting text content
_STRIP_TAGS = {"nav", "header", "footer", "script", "style", "aside", "noscript", "svg", "iframe"}

# Pattern to detect tweet/X URLs
_TWEET_URL_RE = re.compile(r"https?://(?:www\.)?(?:twitter\.com|x\.com)/\w+/status/\d+", re.IGNORECASE)


def _extract_meta(soup: BeautifulSoup, name: str) -> str:
    """Extract content from <meta name="..."> or <meta property="...">."""
    tag = soup.find("meta", attrs={"name": name})
    if tag and tag.get("content"):
        return tag["content"].strip()
    tag = soup.find("meta", attrs={"property": name})
    if tag and tag.get("content"):
        return tag["content"].strip()
    return ""


def _extract_content(soup: BeautifulSoup, max_chars: int) -> str:
    """Extract main text content, stripping non-content tags."""
    # Work on a copy so we don't mutate the original
    body = soup.find("body")
    if not body:
        return ""

    # Remove non-content tags
    for tag_name in _STRIP_TAGS:
        for tag in body.find_all(tag_name):
            tag.decompose()

    # Get text with newline separators
    text = body.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"

    return text


async def fetch_url(url: str, max_chars: int = 15000) -> str:
    """Fetch a URL and return structured metadata + text content.

    Args:
        url: Full URL to fetch.
        max_chars: Max characters of page text to return.

    Returns:
        Formatted string with metadata and content.
    """
    # --- SSRF protection ---
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"URL: {url}\nError: Only http and https URLs are allowed."
    hostname = parsed.hostname
    if not hostname:
        return f"URL: {url}\nError: Could not parse hostname from URL."
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return f"URL: {url}\nError: Could not resolve hostname '{hostname}'."
    _BLOCKED_NETWORKS = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
    ]
    for info in addrinfos:
        addr = ipaddress.ip_address(info[4][0])
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                return f"URL: {url}\nError: Access to private/internal network addresses is blocked."

    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    headers = {"User-Agent": _USER_AGENT}

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True, ssl=False) as resp:
                if resp.status >= 400:
                    return f"URL: {url}\nError: HTTP {resp.status} — {resp.reason}"

                # Only process HTML content
                content_type = resp.headers.get("Content-Type", "")
                if "html" not in content_type and "text" not in content_type:
                    return (
                        f"URL: {url}\n"
                        f"Content-Type: {content_type}\n"
                        f"Note: Not an HTML page. Cannot extract text content."
                    )

                html = await resp.text(errors="replace")

    except aiohttp.ClientError as e:
        return f"URL: {url}\nError: Connection failed — {type(e).__name__}: {e}"
    except TimeoutError:
        return f"URL: {url}\nError: Request timed out after {_TIMEOUT_SECONDS}s"
    except Exception as e:
        return f"URL: {url}\nError: {type(e).__name__}: {e}"

    soup = BeautifulSoup(html, "html.parser")

    # Extract metadata
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    description = _extract_meta(soup, "description")
    og_title = _extract_meta(soup, "og:title")
    og_description = _extract_meta(soup, "og:description")
    og_image = _extract_meta(soup, "og:image")

    # For tweet URLs, try additional meta tags
    is_tweet = bool(_TWEET_URL_RE.match(url))
    tweet_text = ""
    if is_tweet:
        # Twitter puts tweet text in og:description or twitter:description
        tweet_text = _extract_meta(soup, "twitter:description") or og_description
        tweet_author = _extract_meta(soup, "twitter:title") or og_title

    # Extract page text content
    content = _extract_content(soup, max_chars)

    # Build output
    parts = [f"URL: {url}"]

    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    if og_title and og_title != title:
        parts.append(f"OG Title: {og_title}")
    if og_description and og_description != description:
        parts.append(f"OG Description: {og_description}")
    if og_image:
        parts.append(f"OG Image: {og_image}")

    if is_tweet and tweet_text:
        parts.append(f"\n--- Tweet Content ---\n{tweet_text}")
        if tweet_author:
            parts.append(f"Author: {tweet_author}")

    if content:
        content_label = "Page Content (limited — site uses client-side rendering)" if len(content) < 200 and not is_tweet else "Page Content"
        parts.append(f"\n--- {content_label} ---\n{content}")
    elif not is_tweet:
        parts.append("\nNote: Could not extract page content (site may use client-side rendering). Metadata above should still be useful.")

    result = "\n".join(parts)
    logger.info("web_fetch: %s — %d chars extracted", url[:80], len(result))
    return result
