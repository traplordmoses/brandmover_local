"""Tests for operator authorization — _can_operate vs _authorized."""

from unittest.mock import patch

from config import settings


def test_can_operate_admin():
    """Admin always has operator access."""
    from bot.handlers import _can_operate
    with patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 111):
        assert _can_operate(111) is True


def test_can_operate_operator():
    """Users in TELEGRAM_OPERATOR_IDS have operator access."""
    from bot.handlers import _can_operate
    with patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 111), \
         patch.object(settings, "TELEGRAM_OPERATOR_IDS", [222, 333]):
        assert _can_operate(222) is True
        assert _can_operate(333) is True


def test_can_operate_unknown():
    """Unknown users are denied."""
    from bot.handlers import _can_operate
    with patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 111), \
         patch.object(settings, "TELEGRAM_OPERATOR_IDS", [222]):
        assert _can_operate(999) is False


def test_authorized_excludes_operators():
    """_authorized only allows admin, not operators."""
    from bot.handlers import _authorized
    with patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 111), \
         patch.object(settings, "TELEGRAM_OPERATOR_IDS", [222]):
        assert _authorized(111) is True
        assert _authorized(222) is False
        assert _authorized(999) is False


def test_empty_operator_ids():
    """With no operator IDs configured, only admin works."""
    from bot.handlers import _can_operate
    with patch.object(settings, "TELEGRAM_ALLOWED_USER_ID", 111), \
         patch.object(settings, "TELEGRAM_OPERATOR_IDS", []):
        assert _can_operate(111) is True
        assert _can_operate(222) is False
