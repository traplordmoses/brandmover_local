"""Tests for agent/web_fetch.py — URL fetching and content extraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SSRF protection (blocks private IPs via net_guard)
# ---------------------------------------------------------------------------

class TestSSRFProtection:
    @pytest.mark.asyncio
    async def test_blocks_private_ip(self):
        """Fetching a URL that resolves to a private IP returns an error."""
        with patch("agent.web_fetch.validate_url", side_effect=ValueError("blocked")):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://192.168.1.1/admin")

        assert "Error" in result
        assert "blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_localhost(self):
        """Fetching localhost returns an error."""
        with patch("agent.web_fetch.validate_url", side_effect=ValueError("private address")):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://127.0.0.1:8080/secret")

        assert "Error" in result
        assert "private" in result.lower()


# ---------------------------------------------------------------------------
# HTTPS downgrade prevention
# ---------------------------------------------------------------------------

class TestHTTPSDowngrade:
    @pytest.mark.asyncio
    async def test_blocks_https_to_http_redirect(self):
        """HTTPS -> HTTP redirect is blocked."""
        mock_resp = AsyncMock()
        mock_resp.status = 301
        mock_resp.headers = {"Location": "http://example.com/page"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("https://example.com/start")

        assert "HTTPS to HTTP downgrade" in result


# ---------------------------------------------------------------------------
# Redirect following with validation
# ---------------------------------------------------------------------------

class TestRedirects:
    @pytest.mark.asyncio
    async def test_redirect_target_validated(self):
        """Redirect targets are checked against SSRF protection."""
        # First response: redirect to internal address
        mock_redirect_resp = AsyncMock()
        mock_redirect_resp.status = 302
        mock_redirect_resp.headers = {"Location": "http://10.0.0.1/internal"}
        mock_redirect_resp.__aenter__ = AsyncMock(return_value=mock_redirect_resp)
        mock_redirect_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_redirect_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        call_count = 0
        def validate_side_effect(url):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ValueError("Redirect to private IP blocked")

        with patch("agent.web_fetch.validate_url", side_effect=validate_side_effect), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/redirect")

        assert "Redirect blocked" in result

    @pytest.mark.asyncio
    async def test_too_many_redirects(self):
        """More than 5 redirects returns an error."""
        mock_resp = AsyncMock()
        mock_resp.status = 302
        mock_resp.headers = {"Location": "http://example.com/loop"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/start")

        assert "Too many redirects" in result


# ---------------------------------------------------------------------------
# Content size limits
# ---------------------------------------------------------------------------

class TestContentSizeLimits:
    @pytest.mark.asyncio
    async def test_rejects_oversized_response(self):
        """Responses over 5MB are rejected based on Content-Length."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {
            "Content-Length": "10000000",
            "Content-Type": "text/html",
        }
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/huge")

        assert "too large" in result.lower() or "Response too large" in result


# ---------------------------------------------------------------------------
# Successful fetch
# ---------------------------------------------------------------------------

class TestSuccessfulFetch:
    @pytest.mark.asyncio
    async def test_extracts_title_and_content(self):
        """Successful HTML fetch returns title and page text."""
        html = b"""
        <html>
        <head><title>Test Page</title>
        <meta name="description" content="A test page">
        </head>
        <body>
        <nav>Skip this nav</nav>
        <main><p>Important content here.</p></main>
        </body>
        </html>
        """
        mock_content = AsyncMock()
        mock_content.read = AsyncMock(return_value=html)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {
            "Content-Type": "text/html; charset=utf-8",
        }
        # No Content-Length header so size check is skipped
        mock_resp.content = mock_content
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/page")

        assert "Test Page" in result
        assert "A test page" in result
        assert "Important content here" in result

    @pytest.mark.asyncio
    async def test_non_html_content_type(self):
        """Non-HTML content types return a note instead of extracting text."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {
            "Content-Type": "application/json",
        }
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/api")

        assert "Not an HTML page" in result


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_timeout_returns_error_string(self):
        """Timeout produces a descriptive error, not an exception."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/slow")

        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_connection_error_returns_error_string(self):
        """Connection errors produce a descriptive error string."""
        import aiohttp

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            side_effect=aiohttp.ClientError("connection refused"),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/down")

        assert "Connection failed" in result

    @pytest.mark.asyncio
    async def test_http_error_status(self):
        """HTTP 4xx/5xx returns error with status code."""
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.reason = "Not Found"
        mock_resp.headers = {}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("agent.web_fetch.validate_url"), \
             patch("agent.web_fetch.aiohttp.ClientSession", return_value=mock_session):
            from agent.web_fetch import fetch_url
            result = await fetch_url("http://example.com/missing")

        assert "404" in result
        assert "Not Found" in result
