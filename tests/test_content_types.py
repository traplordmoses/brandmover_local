"""Tests for agent.content_types — canonical type definitions."""

from agent.content_types import (
    ALL_CONTENT_TYPES,
    AGENT_SELECTABLE_TYPES,
    COMPOSITOR_PROFILE_MAP,
    COMPOSITOR_PROFILE_TYPES,
    LORA_ELIGIBLE_TYPES,
)


def test_lora_eligible_subset_of_all():
    """Every LoRA-eligible type must be in ALL_CONTENT_TYPES."""
    assert LORA_ELIGIBLE_TYPES <= set(ALL_CONTENT_TYPES)


def test_compositor_profile_types_subset_of_all():
    assert COMPOSITOR_PROFILE_TYPES <= set(ALL_CONTENT_TYPES)


def test_agent_selectable_subset_of_all():
    assert set(AGENT_SELECTABLE_TYPES) <= set(ALL_CONTENT_TYPES)


def test_compositor_map_covers_all():
    """Every ALL_CONTENT_TYPES value should have a map entry."""
    for ct in ALL_CONTENT_TYPES:
        assert ct in COMPOSITOR_PROFILE_MAP, f"{ct} missing from COMPOSITOR_PROFILE_MAP"


def test_compositor_map_values_valid():
    """Map values should be known compositor profile keys or 'default'."""
    valid = COMPOSITOR_PROFILE_TYPES | {"default"}
    for ct, profile in COMPOSITOR_PROFILE_MAP.items():
        assert profile in valid, f"Map value '{profile}' for '{ct}' not a valid profile"


def test_no_duplicate_all_types():
    assert len(ALL_CONTENT_TYPES) == len(set(ALL_CONTENT_TYPES))
