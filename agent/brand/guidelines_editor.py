"""
Conversational brand guidelines editor.
Parse natural language edit instructions and apply them to guidelines.md.
"""

import json
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


async def apply_edit(instruction: str) -> dict:
    """
    Take a natural language instruction and generate a proposed edit to guidelines.md.

    Returns a dict with keys:
        success: bool
        section_modified: str  (which ## section was changed)
        change_summary: str    (one-line description)
        diff_preview: str      (key lines that changed, old -> new)
        new_content: str       (full updated guidelines.md)
        error: str             (only present on failure)
    """
    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"

    if not guidelines_path.exists():
        return {
            "success": False,
            "error": "guidelines.md not found. run /setup or /onboard first.",
        }

    try:
        guidelines_content = guidelines_path.read_text(encoding="utf-8")
    except OSError as e:
        return {"success": False, "error": f"failed to read guidelines.md: {e}"}

    if not guidelines_content.strip():
        return {"success": False, "error": "guidelines.md is empty."}

    system_prompt = (
        "You are editing a brand guidelines markdown document. The user wants to make a change.\n\n"
        f"Current document:\n{guidelines_content}\n\n"
        f"User instruction: {instruction}\n\n"
        "Respond with a JSON object (no markdown fences, just raw JSON):\n"
        "{\n"
        '  "section": "which ## section was modified",\n'
        '  "summary": "one-line description of what changed",\n'
        '  "diff": "show the key lines that changed (old -> new)",\n'
        '  "full_document": "the COMPLETE updated guidelines.md with the change applied"\n'
        "}\n\n"
        "Rules:\n"
        "- Only change what the user asked for. Do not rewrite other sections.\n"
        "- Preserve all existing formatting, tables, and structure.\n"
        "- If adding to a list, add at the end of the relevant list.\n"
        "- If the instruction is ambiguous, make the most reasonable interpretation.\n"
        "- The full_document must be the ENTIRE file, not just the changed section.\n"
        "- Do not wrap the JSON in markdown code fences."
    )

    try:
        from agent._client import get_anthropic

        client = get_anthropic()
        response = await client.messages.create(
            model=settings.SONNET_MODEL,
            max_tokens=8192,
            system="You are a precise brand guidelines editor. Respond only with valid JSON.",
            messages=[{"role": "user", "content": system_prompt}],
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if the model added them despite instructions
        if raw.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = raw.index("\n")
            raw = raw[first_newline + 1 :]
            # Remove closing fence
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3].rstrip()

        parsed = json.loads(raw)

        section = parsed.get("section", "unknown")
        summary = parsed.get("summary", "")
        diff = parsed.get("diff", "")
        full_document = parsed.get("full_document", "")

        if not full_document or not full_document.strip():
            return {
                "success": False,
                "error": "Claude returned an empty document. try again.",
            }

        return {
            "success": True,
            "section_modified": section,
            "change_summary": summary,
            "diff_preview": diff,
            "new_content": full_document,
        }

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s", e)
        return {
            "success": False,
            "error": "failed to parse edit response. try rephrasing.",
        }
    except Exception as e:
        logger.error("Guidelines edit failed: %s", e)
        return {"success": False, "error": str(e)}


async def confirm_edit(new_content: str) -> bool:
    """
    Write the new content to guidelines.md and invalidate all caches.

    Returns True on success, False on failure.
    """
    from agent import compositor_config, guidelines

    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"

    try:
        guidelines_path.write_text(new_content, encoding="utf-8")
        guidelines.invalidate_brand_context()
        compositor_config.invalidate_cache()
        logger.info(
            "Guidelines updated via editor (%d chars)", len(new_content)
        )
        return True
    except OSError as e:
        logger.error("Failed to write guidelines.md: %s", e)
        return False
