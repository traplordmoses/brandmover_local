"""Tests for agent.net_guard — SSRF protection module."""

import socket
from unittest.mock import patch

import pytest

from agent.net_guard import is_private_ip, validate_url


def _fake_addrinfo(ip: str):
    """Build a minimal getaddrinfo result list for a single IP."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


def _fake_addrinfo_v6(ip: str):
    """Build a minimal getaddrinfo result for an IPv6 address."""
    return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]


# ---------------------------------------------------------------------------
# validate_url — scheme checks
# ---------------------------------------------------------------------------


class TestValidateUrlScheme:
    def test_rejects_ftp(self):
        with pytest.raises(ValueError, match="Only http/https"):
            validate_url("ftp://example.com/file.txt")

    def test_rejects_file(self):
        with pytest.raises(ValueError, match="Only http/https"):
            validate_url("file:///etc/passwd")

    def test_rejects_javascript(self):
        with pytest.raises(ValueError, match="Only http/https"):
            validate_url("javascript:alert(1)")

    def test_rejects_empty_scheme(self):
        with pytest.raises(ValueError):
            validate_url("://example.com")

    def test_rejects_no_scheme(self):
        with pytest.raises(ValueError):
            validate_url("example.com")


# ---------------------------------------------------------------------------
# validate_url — hostname checks
# ---------------------------------------------------------------------------


class TestValidateUrlHostname:
    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError, match="Could not parse hostname"):
            validate_url("http://")

    def test_rejects_empty_hostname(self):
        with pytest.raises(ValueError, match="Could not parse hostname"):
            validate_url("http:///path")


# ---------------------------------------------------------------------------
# validate_url — blocked IP addresses
# ---------------------------------------------------------------------------


class TestValidateUrlBlockedIPs:
    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("127.0.0.1"))
    def test_rejects_localhost(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://localhost/admin")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("127.0.0.2"))
    def test_rejects_loopback_range(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://something.local/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("10.0.0.1"))
    def test_rejects_10_network(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://internal.corp/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("10.255.255.255"))
    def test_rejects_10_network_upper(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://internal.corp/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("172.16.0.1"))
    def test_rejects_172_16_network(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://internal.corp/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("172.31.255.255"))
    def test_rejects_172_31_network(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://internal.corp/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("192.168.1.1"))
    def test_rejects_192_168_network(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://router.local/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("169.254.169.254"))
    def test_rejects_link_local_metadata(self, _mock):
        """Cloud metadata endpoint — critical SSRF target."""
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://169.254.169.254/latest/meta-data/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("169.254.0.1"))
    def test_rejects_link_local(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://link-local.example/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("0.0.0.0"))
    def test_rejects_zero_network(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://zero.example/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo_v6("::1"))
    def test_rejects_ipv6_loopback(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://[::1]/")

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo_v6("fc00::1"))
    def test_rejects_ipv6_unique_local(self, _mock):
        with pytest.raises(ValueError, match="private/internal"):
            validate_url("http://[fc00::1]/")


# ---------------------------------------------------------------------------
# validate_url — DNS resolution failure
# ---------------------------------------------------------------------------


class TestValidateUrlDNSFailure:
    @patch(
        "agent.net_guard.socket.getaddrinfo",
        side_effect=socket.gaierror("Name or service not known"),
    )
    def test_rejects_unresolvable_host(self, _mock):
        with pytest.raises(ValueError, match="Could not resolve hostname"):
            validate_url("http://does-not-exist.invalid/")


# ---------------------------------------------------------------------------
# validate_url — valid public URLs
# ---------------------------------------------------------------------------


class TestValidateUrlAcceptsPublic:
    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("8.8.8.8"))
    def test_accepts_http_public(self, _mock):
        validate_url("http://example.com/page")  # should not raise

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("8.8.8.8"))
    def test_accepts_https_public(self, _mock):
        validate_url("https://example.com/page")  # should not raise

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34"))
    def test_accepts_public_ip(self, _mock):
        validate_url("https://example.com/")  # should not raise

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("1.1.1.1"))
    def test_accepts_cloudflare_dns(self, _mock):
        validate_url("https://cloudflare.com/")  # should not raise


# ---------------------------------------------------------------------------
# is_private_ip — private addresses
# ---------------------------------------------------------------------------


class TestIsPrivateIp:
    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("127.0.0.1"))
    def test_localhost_is_private(self, _mock):
        assert is_private_ip("localhost") is True

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("10.0.0.5"))
    def test_10_network_is_private(self, _mock):
        assert is_private_ip("internal.corp") is True

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("172.20.0.1"))
    def test_172_network_is_private(self, _mock):
        assert is_private_ip("internal.corp") is True

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("192.168.0.1"))
    def test_192_168_is_private(self, _mock):
        assert is_private_ip("router.local") is True

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("169.254.169.254"))
    def test_link_local_is_private(self, _mock):
        assert is_private_ip("metadata.internal") is True


# ---------------------------------------------------------------------------
# is_private_ip — DNS failure (fail-closed)
# ---------------------------------------------------------------------------


class TestIsPrivateIpDNSFailure:
    @patch(
        "agent.net_guard.socket.getaddrinfo",
        side_effect=socket.gaierror("Name or service not known"),
    )
    def test_dns_failure_returns_true(self, _mock):
        """Fail-closed: unresolvable hosts are treated as private."""
        assert is_private_ip("nonexistent.invalid") is True


# ---------------------------------------------------------------------------
# is_private_ip — public addresses
# ---------------------------------------------------------------------------


class TestIsPrivateIpPublic:
    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("8.8.8.8"))
    def test_public_ip_returns_false(self, _mock):
        assert is_private_ip("dns.google") is False

    @patch("agent.net_guard.socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34"))
    def test_another_public_ip_returns_false(self, _mock):
        assert is_private_ip("example.com") is False
