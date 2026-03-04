"""Tests for agent.discord_bot — channel storage and readiness checks."""

import json
from unittest.mock import patch

from agent import discord_bot


def test_channel_id_storage_roundtrip(tmp_path):
    channels_file = tmp_path / "discord_channels.json"

    with patch.object(discord_bot, "_CHANNELS_FILE", channels_file), \
         patch.object(discord_bot, "_STATE_DIR", tmp_path):
        # Initially empty
        assert discord_bot.get_channel_id("announcements") is None

        # Write some channels
        discord_bot._write_channels({
            "announcements": 123456789,
            "memes": 987654321,
        })

        assert discord_bot.get_channel_id("announcements") == 123456789
        assert discord_bot.get_channel_id("memes") == 987654321
        assert discord_bot.get_channel_id("nonexistent") is None


def test_missing_file_returns_none(tmp_path):
    channels_file = tmp_path / "does_not_exist.json"

    with patch.object(discord_bot, "_CHANNELS_FILE", channels_file):
        assert discord_bot.get_channel_id("anything") is None


def test_corrupt_file_returns_empty(tmp_path):
    channels_file = tmp_path / "discord_channels.json"
    channels_file.write_text("not valid json", encoding="utf-8")

    with patch.object(discord_bot, "_CHANNELS_FILE", channels_file):
        assert discord_bot.get_channel_id("announcements") is None


def test_is_ready_without_client():
    with patch.object(discord_bot, "_client", None):
        assert discord_bot.is_ready() is False


def test_read_channels_returns_ints(tmp_path):
    """Values stored as strings should be coerced to int."""
    channels_file = tmp_path / "discord_channels.json"
    channels_file.write_text(
        json.dumps({"general": "555"}), encoding="utf-8"
    )

    with patch.object(discord_bot, "_CHANNELS_FILE", channels_file):
        assert discord_bot.get_channel_id("general") == 555
