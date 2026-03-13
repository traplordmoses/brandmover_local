"""
Periodic topic bank refresh — uses Claude to generate new angles
based on brand config and recent post history.
"""

import json
import logging
import time

import anthropic

from agent.session import build_session_context
from agent.topic_bank import load_bank, save_bank, add_angle, retire_angle
from config import settings

logger = logging.getLogger(__name__)


async def refresh_topic_bank() -> dict:
    """
    Ask Claude to refresh the topic bank with new angles and retire stale ones.
    Returns: {"added": int, "retired": int}
    """
    from agent._client import get_anthropic
    from agent import guidelines

    client = get_anthropic()
    bank = load_bank()
    session_context = build_session_context()

    # Load brand context
    brand_context = guidelines.get_brand_context()
    # Truncate to avoid blowing context
    if len(brand_context) > 5000:
        brand_context = brand_context[:5000] + "\n[...truncated...]"

    # Build angles summary for Claude
    angles_summary = json.dumps(
        [
            {
                "id": a.get("id"),
                "category": a.get("category"),
                "angle": a.get("angle"),
                "times_used": a.get("times_used", 0),
                "retired": a.get("retired", False),
            }
            for a in bank.angles
        ],
        indent=2,
    )

    prompt = (
        f"You are a content strategist reviewing a brand's topic bank for social media.\n\n"
        f"Brand context:\n{brand_context}\n\n"
        f"Current topic bank ({len(bank.angles)} angles):\n{angles_summary}\n\n"
        f"Recent post history:\n{session_context or '(no recent posts)'}\n\n"
        f"Tasks:\n"
        f"1. Suggest 3-5 NEW angles that aren't covered by existing ones. "
        f"Focus on fresh approaches that complement recent posts rather than repeating them.\n"
        f"2. Identify any angles that should be RETIRED (overused with times_used > 5, "
        f"or no longer relevant).\n\n"
        f"Use the update_bank tool to submit your changes."
    )

    update_bank_tool = {
        "name": "update_bank",
        "description": "Submit topic bank updates",
        "input_schema": {
            "type": "object",
            "properties": {
                "new_angles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "angle": {"type": "string"},
                            "example_hooks": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["category", "angle"],
                    },
                },
                "retire_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of angles to retire",
                },
            },
            "required": ["new_angles", "retire_ids"],
        },
    }

    try:
        response = await client.messages.create(
            model=settings.AGENT_MODEL,
            max_tokens=1000,
            system="You are a concise content strategist. Suggest fresh angles and retire stale ones.",
            tools=[update_bank_tool],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        logger.error("Topic bank refresh failed: %s", e)
        return {"added": 0, "retired": 0}

    added = 0
    retired = 0

    for block in response.content:
        if block.type == "tool_use" and block.name == "update_bank":
            inp = block.input

            # Add new angles
            for new in inp.get("new_angles", []):
                cat = new.get("category", "engagement")
                angle_text = new.get("angle", "")
                hooks = new.get("example_hooks", [])
                if angle_text:
                    add_angle(cat, angle_text, hooks)
                    added += 1

            # Retire stale angles
            for rid in inp.get("retire_ids", []):
                if retire_angle(rid):
                    retired += 1

            break

    # Update refresh timestamp
    bank = load_bank()
    bank.last_refreshed = time.time()
    save_bank(bank)

    logger.info("Topic bank refreshed: +%d added, -%d retired", added, retired)
    return {"added": added, "retired": retired}
