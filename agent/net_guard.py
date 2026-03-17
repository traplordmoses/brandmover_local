"""
Shared SSRF protection utilities.

Provides URL validation and private-IP detection used by web_fetch,
take_screenshot, demo_recorder, and any other module that makes
outbound network requests to user-supplied URLs.
"""

import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("::ffff:0:0/96"),
]


def validate_url(url: str) -> None:
    """Validate URL scheme and block private/internal IPs.

    Raises:
        ValueError: If scheme is not http/https, hostname cannot be resolved,
                    or resolved IP is in a blocked private network.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs allowed, got: {parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Could not parse hostname from URL: {url}")
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname {hostname!r}: {e}")
    for info in addrinfos:
        addr = ipaddress.ip_address(info[4][0])
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(f"Access to private/internal address {addr} is blocked")
        # Check IPv6-mapped IPv4 addresses (e.g. ::ffff:127.0.0.1)
        mapped = getattr(addr, "ipv4_mapped", None)
        if mapped:
            for net in _BLOCKED_NETWORKS:
                if isinstance(net, ipaddress.IPv4Network) and mapped in net:
                    raise ValueError(f"URL resolves to blocked IP (mapped): {mapped}")


def is_private_ip(hostname: str) -> bool:
    """Check if a hostname resolves to a private/internal IP.

    Returns True if any resolved address is in a blocked network,
    False otherwise. Returns True on DNS resolution failure (fail-closed).
    """
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True  # fail closed
    for info in addrinfos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                return True
    return False
