"""
Singleton API clients for connection reuse.

Reusing a single AsyncAnthropic client avoids TCP+TLS handshake overhead
(~100-300ms) on every LLM call. Same for httpx.AsyncClient.

Tests can mock ``anthropic.AsyncAnthropic`` at the call site and reset
singletons via :func:`reset` (done automatically by the conftest fixture).
"""

import anthropic
import httpx

from config import settings

_anthropic_client: anthropic.AsyncAnthropic | None = None
_httpx_client: httpx.AsyncClient | None = None


def get_anthropic() -> anthropic.AsyncAnthropic:
    """Return a shared AsyncAnthropic client (lazy-initialized).

    Uses this module's ``anthropic`` reference so test patches on
    ``agent._client.anthropic.AsyncAnthropic`` propagate correctly.
    The autouse conftest fixture calls :func:`reset` before each test
    so the singleton is rebuilt through the (potentially mocked) constructor.
    """
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic_client


def get_httpx() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient (lazy-initialized)."""
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=120, follow_redirects=True)
    return _httpx_client


def reset() -> None:
    """Reset singleton clients so the next call rebuilds them.

    Called by the autouse conftest fixture before each test.
    """
    global _anthropic_client, _httpx_client
    _anthropic_client = None
    _httpx_client = None


async def close() -> None:
    """Close shared clients. Call on shutdown if needed."""
    global _anthropic_client, _httpx_client
    if _httpx_client is not None:
        await _httpx_client.aclose()
        _httpx_client = None
    if _anthropic_client is not None:
        _anthropic_client = None
