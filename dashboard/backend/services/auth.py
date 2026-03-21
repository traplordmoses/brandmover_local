"""
Telegram WebApp initData HMAC-SHA-256 validation.

Validates the `initData` string sent by Telegram Mini Apps according to
https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app

Public API:
    validate_telegram_init_data(init_data, bot_token) -> dict | None
"""

import hashlib
import hmac
import json
import time
from urllib.parse import parse_qs, unquote

# Maximum age of auth_date before we reject the payload (seconds).
_MAX_AUTH_AGE = 3600  # 1 hour


def validate_telegram_init_data(init_data: str, bot_token: str) -> dict | None:
    """Validate Telegram WebApp initData.

    Returns the parsed ``user`` dict if the signature is valid and the
    auth_date is within the allowed window, or *None* otherwise.
    """
    # Parse key=value pairs from the URL-encoded string.
    parsed = parse_qs(init_data, keep_blank_values=True)

    # parse_qs returns lists; flatten to single values.
    params: dict[str, str] = {}
    for key, values in parsed.items():
        params[key] = values[0] if values else ""

    received_hash = params.pop("hash", None)
    if not received_hash:
        return None

    # Build the data-check string: all remaining params sorted by key,
    # joined with newlines as "key=value".
    sorted_pairs = sorted(params.items(), key=lambda kv: kv[0])
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted_pairs)

    # secret_key = HMAC-SHA-256("WebAppData", bot_token)
    secret_key = hmac.new(
        b"WebAppData",
        bot_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    # computed_hash = HMAC-SHA-256(secret_key, data_check_string)
    computed_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        return None

    # Verify auth_date freshness.
    try:
        auth_date = int(params.get("auth_date", "0"))
    except (ValueError, TypeError):
        return None

    if time.time() - auth_date > _MAX_AUTH_AGE:
        return None

    # Extract the user JSON payload.
    user_raw = params.get("user")
    if not user_raw:
        return None

    try:
        user = json.loads(unquote(user_raw))
    except (json.JSONDecodeError, TypeError):
        return None

    return user
