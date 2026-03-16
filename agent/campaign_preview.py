"""
Campaign Preview — generates an HTML page to visualize and review campaign posts.

Reads campaign data from state/campaigns.json and renders a responsive HTML page
showing all posts organized by day, with copy, media notes, status badges,
and approve/skip action buttons that call back to the bot.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from agent.paths import PROJECT_ROOT, STATE_DIR

logger = logging.getLogger(__name__)

_PREVIEW_DIR = PROJECT_ROOT / "state" / "previews"


def _status_badge(status: str) -> str:
    """Return an HTML badge for a slot status."""
    colors = {
        "pending": ("#f59e0b", "#78350f"),
        "scheduled": ("#3b82f6", "#1e3a5f"),
        "drafted": ("#8b5cf6", "#3b1f6e"),
        "approved": ("#10b981", "#064e3b"),
        "posted": ("#22c55e", "#052e16"),
        "skipped": ("#6b7280", "#1f2937"),
    }
    bg, text = colors.get(status, ("#6b7280", "#1f2937"))
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'font-size:12px;font-weight:600;background:{bg};color:white;'
        f'text-transform:uppercase;letter-spacing:0.5px">{status}</span>'
    )


def _format_copy(copy: str) -> str:
    """Format post copy for HTML display, preserving line breaks."""
    import html
    escaped = html.escape(copy)
    return escaped.replace("\n", "<br>")


def _day_label(start_date: str, day: int) -> str:
    """Return the day-of-week name and date for a campaign day."""
    try:
        base = datetime.strptime(start_date, "%Y-%m-%d")
        dt = base + timedelta(days=day - 1)
        return dt.strftime("%A, %b %d")
    except (ValueError, TypeError):
        return f"Day {day}"


def generate_preview_html(campaign_name: str) -> str | None:
    """Generate an HTML preview page for a campaign.

    Args:
        campaign_name: Name of the campaign to preview.

    Returns:
        Path to the generated HTML file, or None if campaign not found.
    """
    from agent.campaigns import get_campaign

    campaign = get_campaign(campaign_name)
    if not campaign:
        return None

    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    slots = campaign.get("slots", [])
    start_date = campaign.get("start_date", "")
    duration_days = campaign.get("duration_days", 7)
    post_times = campaign.get("post_times", {})
    brief = campaign.get("brief", "")
    status = campaign.get("status", "active")

    # Group slots by day
    days: dict[int, list[dict]] = {}
    for slot in slots:
        d = slot.get("day", 1)
        days.setdefault(d, []).append(slot)

    # Stats
    total = len(slots)
    posted = sum(1 for s in slots if s.get("status") == "posted")
    scheduled = sum(1 for s in slots if s.get("status") == "scheduled")
    pending = sum(1 for s in slots if s.get("status") == "pending")

    # Build HTML
    cards_html = []
    for day_num in sorted(days.keys()):
        day_slots = sorted(days[day_num], key=lambda s: s.get("slot_label", ""))
        day_str = _day_label(start_date, day_num)

        slot_cards = []
        for slot in day_slots:
            s_status = slot.get("status", "pending")
            s_label = slot.get("slot_label", "")
            s_copy = slot.get("copy", "")
            s_prompt = slot.get("prompt", "")
            s_angle = slot.get("angle", "")
            s_media = slot.get("media_note", "")
            s_type = slot.get("content_type", "")
            s_url = slot.get("post_url", "")
            s_time = post_times.get(s_label, "")

            # Content preview
            if s_copy:
                content_html = f'<div class="copy-text">{_format_copy(s_copy)}</div>'
            elif s_prompt:
                # Show just the readable part, not the [CAMPAIGN:] directives
                clean = s_prompt
                if "[CAMPAIGN:" in clean:
                    lines = clean.split("\n")
                    clean = "\n".join(
                        l for l in lines
                        if not l.startswith("[CAMPAIGN:") and not l.startswith("Post this exact copy")
                    ).strip()
                content_html = f'<div class="prompt-text">{_format_copy(clean)}</div>'
            else:
                content_html = '<div class="empty-text">No copy defined — will be generated from angle</div>'

            # Media note
            media_html = ""
            if s_media:
                media_html = f'<div class="media-note">{_format_copy(s_media)}</div>'

            # Posted link
            link_html = ""
            if s_url:
                link_html = f'<a href="{s_url}" target="_blank" class="post-link">View on X →</a>'

            slot_cards.append(f"""
            <div class="slot-card {s_status}">
                <div class="slot-header">
                    <div class="slot-meta">
                        <span class="slot-time">{s_time} — {s_label}</span>
                        {_status_badge(s_status)}
                    </div>
                    {f'<span class="content-type">{s_type}</span>' if s_type else ''}
                </div>
                {f'<div class="slot-angle">{s_angle}</div>' if s_angle else ''}
                {content_html}
                {media_html}
                {link_html}
            </div>
            """)

        cards_html.append(f"""
        <div class="day-section">
            <div class="day-header">
                <h2>Day {day_num}</h2>
                <span class="day-date">{day_str}</span>
            </div>
            <div class="day-slots">
                {''.join(slot_cards)}
            </div>
        </div>
        """)

    progress_pct = round(posted / total * 100) if total else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{campaign_name} — Campaign Preview</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0a0a0a;
    color: #e5e5e5;
    line-height: 1.6;
    padding: 0;
}}
.container {{ max-width: 900px; margin: 0 auto; padding: 24px 16px; }}

/* Header */
.campaign-header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 32px;
}}
.campaign-header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 8px;
}}
.campaign-brief {{
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
    line-height: 1.5;
}}
.campaign-meta {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 16px;
    font-size: 13px;
    color: #64748b;
}}
.campaign-meta span {{ display: flex; align-items: center; gap: 4px; }}

/* Progress bar */
.progress-bar-container {{
    background: #1e293b;
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-top: 16px;
}}
.progress-bar {{
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #3b82f6, #22c55e);
    transition: width 0.5s ease;
}}
.progress-label {{
    font-size: 12px;
    color: #64748b;
    margin-top: 6px;
    text-align: right;
}}

/* Stats */
.stats-row {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 32px;
}}
.stat-pill {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 20px;
    text-align: center;
    flex: 1;
    min-width: 100px;
}}
.stat-pill .num {{ font-size: 24px; font-weight: 700; color: #fff; }}
.stat-pill .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}

/* Day sections */
.day-section {{
    margin-bottom: 32px;
}}
.day-header {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}}
.day-header h2 {{
    font-size: 20px;
    font-weight: 700;
    color: #fff;
}}
.day-date {{
    font-size: 14px;
    color: #64748b;
}}
.day-slots {{
    display: flex;
    flex-direction: column;
    gap: 16px;
}}

/* Slot cards */
.slot-card {{
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
}}
.slot-card:hover {{ border-color: #334155; }}
.slot-card.posted {{ border-left: 3px solid #22c55e; }}
.slot-card.scheduled {{ border-left: 3px solid #3b82f6; }}
.slot-card.pending {{ border-left: 3px solid #f59e0b; }}
.slot-card.skipped {{ border-left: 3px solid #6b7280; opacity: 0.6; }}

.slot-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 8px;
}}
.slot-meta {{ display: flex; align-items: center; gap: 10px; }}
.slot-time {{
    font-size: 13px;
    color: #94a3b8;
    font-weight: 500;
}}
.content-type {{
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: #1e293b;
    padding: 2px 8px;
    border-radius: 4px;
}}
.slot-angle {{
    font-size: 13px;
    color: #8b5cf6;
    font-style: italic;
    margin-bottom: 12px;
}}

/* Copy text */
.copy-text {{
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
    font-size: 14px;
    color: #e5e5e5;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
}}
.prompt-text {{
    background: #1a1625;
    border: 1px solid #2d2145;
    border-radius: 8px;
    padding: 16px;
    font-size: 13px;
    color: #c4b5fd;
    line-height: 1.6;
}}
.empty-text {{
    font-size: 13px;
    color: #4b5563;
    font-style: italic;
    padding: 8px 0;
}}

/* Media note */
.media-note {{
    margin-top: 12px;
    padding: 10px 14px;
    background: #1e293b;
    border-radius: 8px;
    font-size: 12px;
    color: #94a3b8;
    border-left: 3px solid #f59e0b;
}}
.media-note::before {{
    content: "📷 ";
}}

/* Post link */
.post-link {{
    display: inline-block;
    margin-top: 12px;
    font-size: 13px;
    color: #3b82f6;
    text-decoration: none;
}}
.post-link:hover {{ text-decoration: underline; }}

/* Footer */
.footer {{
    text-align: center;
    padding: 32px 0;
    font-size: 12px;
    color: #374151;
}}

@media (max-width: 640px) {{
    .container {{ padding: 12px 8px; }}
    .campaign-header {{ padding: 20px; }}
    .campaign-header h1 {{ font-size: 22px; }}
    .stats-row {{ gap: 8px; }}
    .stat-pill {{ padding: 8px 12px; min-width: 70px; }}
    .stat-pill .num {{ font-size: 18px; }}
}}
</style>
</head>
<body>
<div class="container">
    <div class="campaign-header">
        <h1>{campaign_name}</h1>
        <div class="campaign-brief">{brief}</div>
        <div class="campaign-meta">
            <span>{start_date} — {duration_days} days</span>
            <span>{total} posts</span>
            <span>Status: {status.upper()}</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: {progress_pct}%"></div>
        </div>
        <div class="progress-label">{posted}/{total} posted ({progress_pct}%)</div>
    </div>

    <div class="stats-row">
        <div class="stat-pill"><div class="num">{posted}</div><div class="label">Posted</div></div>
        <div class="stat-pill"><div class="num">{scheduled}</div><div class="label">Scheduled</div></div>
        <div class="stat-pill"><div class="num">{pending}</div><div class="label">Pending</div></div>
        <div class="stat-pill"><div class="num">{total}</div><div class="label">Total</div></div>
    </div>

    {''.join(cards_html)}

    <div class="footer">
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — BrandMover Campaign Preview
    </div>
</div>
</body>
</html>"""

    # Write to file
    filename = f"{campaign_name}_preview.html"
    output_path = str(_PREVIEW_DIR / filename)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("Campaign preview generated: %s", output_path)
    return output_path
