"""
Content calendar generator — produces markdown calendars from agent output.

When the agent calls finish with format="calendar", the engine routes here
to save a structured content calendar to brand/content_calendar.md.
"""

import logging
from datetime import datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_CALENDAR_PATH = Path(settings.BRAND_FOLDER) / "content_calendar.md"


def generate_calendar(draft: dict) -> str | None:
    """Generate a content calendar markdown file from agent draft data.

    The draft should contain:
    - calendar_entries: list of {date, time, theme, type, topic, status}
    - title: calendar title (optional)
    - subtitle: date range (optional)

    Returns the file path on success, None on failure.
    """
    entries = draft.get("calendar_entries") or draft.get("thread_posts") or []
    title = draft.get("title", "Content Calendar")
    subtitle = draft.get("subtitle", "")

    if not entries:
        # If no structured entries, use caption as freeform calendar
        caption = draft.get("caption", "")
        if caption:
            content = f"# {title}\n\n"
            if subtitle:
                content += f"*{subtitle}*\n\n"
            content += caption
            _CALENDAR_PATH.write_text(content, encoding="utf-8")
            logger.info("Freeform calendar saved: %s", _CALENDAR_PATH)
            return str(_CALENDAR_PATH)
        return None

    # Build markdown table
    lines = [f"# {title}"]
    if subtitle:
        lines.append(f"\n*{subtitle}*")
    lines.append(f"\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("| Date | Time | Theme | Type | Topic | Status |")
    lines.append("|------|------|-------|------|-------|--------|")

    for entry in entries:
        if isinstance(entry, str):
            # Plain text entry — put in topic column
            lines.append(f"| | | | | {entry} | planned |")
            continue
        date = entry.get("date", "")
        time_ = entry.get("time", "")
        theme = entry.get("theme", "")
        type_ = entry.get("type", entry.get("content_type", ""))
        topic = entry.get("topic", entry.get("text", ""))
        status = entry.get("status", "planned")
        lines.append(f"| {date} | {time_} | {theme} | {type_} | {topic} | {status} |")

    content = "\n".join(lines) + "\n"
    _CALENDAR_PATH.write_text(content, encoding="utf-8")
    logger.info("Content calendar saved: %s (%d entries)", _CALENDAR_PATH, len(entries))
    return str(_CALENDAR_PATH)
