"""
Report Generator — produces branded HTML reports from bot data.

Supports multiple report types:
- performance: generation stats, approval rates, cost tracking
- campaign: campaign progress and post breakdown
- feedback: approval/rejection patterns and learned preferences
- custom: freeform sections passed by the agent

All reports use the FOID brand template — iridescent blue background,
frosted glass panels, Orbitron headings, Frutiger Aero aesthetic.
Output is an HTML file in state/reports/, ready to send via send_file.
"""

import html
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from agent.paths import PROJECT_ROOT, STATE_DIR

logger = logging.getLogger(__name__)

_REPORT_DIR = STATE_DIR / "reports"


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------

def _collect_performance_data() -> dict:
    """Gather generation stats, approval analytics, and recent posts."""
    from agent.generation_history import (
        get_generation_stats,
        get_approval_analytics,
        get_recent_generations,
    )
    return {
        "stats": get_generation_stats(),
        "analytics": get_approval_analytics(),
        "recent": get_recent_generations(20),
    }


def _collect_campaign_data(campaign_name: str) -> dict | None:
    """Gather campaign progress and slot details."""
    from agent.campaigns import get_campaign, get_campaign_progress
    campaign = get_campaign(campaign_name)
    if not campaign:
        return None
    return {
        "campaign": campaign,
        "progress": get_campaign_progress(campaign_name),
    }


def _collect_feedback_data() -> dict:
    """Gather feedback stats and recent entries."""
    from agent.feedback import get_feedback_stats, _read_feedback
    entries = _read_feedback()
    total = len(entries)
    approved = sum(1 for e in entries if e.get("accepted"))
    rejected = total - approved
    rate = round(approved / total * 100, 1) if total else 0

    recent = entries[-20:]
    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "rate": rate,
        "recent": recent,
        "stats_text": get_feedback_stats(),
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@400;500;600&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    background: linear-gradient(135deg, #5B8FC4 0%, #8BBAE8 30%, #4AA8B8 60%, #5B8FC4 100%);
    background-attachment: fixed;
    color: #1A2E44;
    line-height: 1.6;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}
/* Aurora gradient overlay */
body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(74, 168, 184, 0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 60%, rgba(139, 186, 232, 0.4) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 90%, rgba(62, 238, 196, 0.1) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}
.container { max-width: 900px; margin: 0 auto; padding: 32px 16px; position: relative; z-index: 1; }

/* Frosted glass panel mixin */
.glass {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 16px;
}

/* .EXE window chrome */
.exe-window {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 24px;
}
.exe-titlebar {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: rgba(13, 27, 42, 0.6);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    font-weight: 700;
    color: #D4A843;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.exe-dot { width: 10px; height: 10px; border-radius: 50%; }
.exe-dot-red { background: #FF5F57; }
.exe-dot-yellow { background: #FEBC2E; }
.exe-dot-green { background: #28C840; }
.exe-titlebar .dots { display: flex; gap: 5px; margin-right: 10px; }
.exe-body { padding: 28px; }

/* Header */
.report-header {
    text-align: center;
    padding: 40px 32px;
}
.report-header h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 26px;
    font-weight: 900;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 3px;
    text-shadow: 0 2px 20px rgba(91, 143, 196, 0.5);
    margin-bottom: 8px;
}
.report-subtitle {
    color: rgba(255, 255, 255, 0.7);
    font-size: 14px;
    font-weight: 400;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 28px;
}
.stat-pill {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    flex: 1;
    min-width: 100px;
}
.stat-pill .num {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF;
    text-shadow: 0 1px 8px rgba(62, 238, 196, 0.3);
}
.stat-pill .label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.6);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
    font-weight: 600;
}

/* Section */
.section {
    margin-bottom: 28px;
}
.section h2 {
    font-family: 'Orbitron', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #D4A843;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(212, 168, 67, 0.3);
}

/* Table */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.data-table th {
    text-align: left;
    padding: 10px 14px;
    background: rgba(13, 27, 42, 0.4);
    color: rgba(255, 255, 255, 0.7);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 10px;
    font-family: 'Orbitron', monospace;
}
.data-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    color: #FFFFFF;
}
.data-table tr:hover td { background: rgba(255, 255, 255, 0.05); }

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.card-meta {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
    margin-bottom: 8px;
}
.card-content {
    font-size: 14px;
    color: #FFFFFF;
    line-height: 1.6;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-family: 'Orbitron', monospace;
}
.badge-green { background: rgba(76, 175, 80, 0.8); color: white; }
.badge-red { background: rgba(255, 105, 180, 0.8); color: white; }
.badge-blue { background: rgba(107, 159, 212, 0.8); color: white; }
.badge-yellow { background: rgba(212, 168, 67, 0.8); color: white; }
.badge-gray { background: rgba(255, 255, 255, 0.2); color: rgba(255, 255, 255, 0.7); }
.badge-cyan { background: rgba(62, 238, 196, 0.8); color: #0D1B2A; }

/* Bar chart */
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.bar-label { width: 120px; font-size: 12px; color: rgba(255, 255, 255, 0.7); text-align: right; flex-shrink: 0; }
.bar-track {
    flex: 1; height: 24px;
    background: rgba(13, 27, 42, 0.3);
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
    display: flex;
    align-items: center;
    padding-left: 8px;
    font-size: 11px;
    font-weight: 600;
    color: white;
    min-width: 24px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

/* Progress bar */
.progress-container {
    background: rgba(13, 27, 42, 0.3);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 24px;
}
.progress-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6B9FD4, #3EEEC4);
}

/* Footer */
.footer {
    text-align: center;
    padding: 32px 0;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    font-family: 'Orbitron', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Custom content */
.custom-section {
    background: rgba(13, 27, 42, 0.5);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(62, 238, 196, 0.2);
    border-radius: 8px;
    padding: 20px;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
    color: #E8F0FF;
    font-family: 'Inter', sans-serif;
}

/* Terminal panel (for data-heavy sections) */
.terminal-panel {
    background: rgba(13, 27, 42, 0.7);
    border: 1px solid rgba(62, 238, 196, 0.15);
    border-radius: 8px;
    padding: 20px;
    color: #3EEEC4;
    font-family: monospace;
    font-size: 13px;
}

@media (max-width: 640px) {
    .container { padding: 12px 8px; }
    .report-header { padding: 24px 16px; }
    .report-header h1 { font-size: 18px; letter-spacing: 2px; }
    .stats-row { gap: 8px; }
    .stat-pill { padding: 10px 12px; min-width: 70px; }
    .stat-pill .num { font-size: 18px; }
    .bar-label { width: 80px; font-size: 11px; }
    .exe-body { padding: 16px; }
}
"""


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def _badge(text: str, color: str = "gray") -> str:
    return f'<span class="badge badge-{color}">{_esc(text)}</span>'


def _stat_pill(num: str | int | float, label: str) -> str:
    return f'<div class="stat-pill"><div class="num">{_esc(str(num))}</div><div class="label">{_esc(label)}</div></div>'


def _bar_chart(items: list[tuple[str, int | float]], color: str = "#6B9FD4", max_val: float | None = None) -> str:
    """Render a horizontal bar chart. items = [(label, value), ...]"""
    if not items:
        return ""
    if max_val is None:
        max_val = max(v for _, v in items) if items else 1
    if max_val == 0:
        max_val = 1
    rows = []
    for label, value in items:
        pct = round(value / max_val * 100)
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{_esc(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}">{value}</div></div>'
            f'</div>'
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _build_performance_report(title: str, subtitle: str) -> str:
    """Build a performance/analytics report."""
    data = _collect_performance_data()
    stats = data["stats"]
    analytics = data["analytics"]
    recent = data["recent"]

    # Stats pills
    pills = [
        _stat_pill(stats["total"], "Total Generated"),
        _stat_pill(stats.get("by_status", {}).get("approved", 0), "Approved"),
        _stat_pill(stats.get("by_status", {}).get("rejected", 0), "Rejected"),
        _stat_pill(f"${stats.get('estimated_total_cost_usd', 0):.2f}", "Total Cost"),
    ]

    # Content type breakdown
    type_items = sorted(stats.get("by_type", {}).items(), key=lambda x: -x[1])
    type_chart = _bar_chart(type_items, "#FF69B4")

    # Model breakdown
    model_items = sorted(stats.get("by_model", {}).items(), key=lambda x: -x[1])
    model_chart = _bar_chart(model_items, "#4AA8B8")

    # Approval rates by content type
    approval_rows = []
    for ct, info in sorted(analytics.get("by_content_type", {}).items()):
        rate = info.get("rate", 0)
        color = "green" if rate >= 70 else "yellow" if rate >= 40 else "red"
        approval_rows.append(
            f'<tr>'
            f'<td>{_esc(ct)}</td>'
            f'<td>{info.get("approved", 0)}</td>'
            f'<td>{info.get("rejected", 0)}</td>'
            f'<td>{_badge(f"{rate}%", color)}</td>'
            f'</tr>'
        )

    # Recent generations table
    recent_rows = []
    for entry in reversed(recent[-10:]):
        ts = entry.get("timestamp", 0)
        dt = datetime.fromtimestamp(ts).strftime("%b %d %H:%M") if ts else "—"
        caption = _esc(entry.get("caption", "")[:60])
        status = entry.get("status", "draft")
        s_color = {"approved": "green", "rejected": "red", "posted": "blue"}.get(status, "gray")
        ct = _esc(entry.get("content_type", "—"))
        cost = entry.get("estimated_cost_usd", 0)
        recent_rows.append(
            f'<tr>'
            f'<td>{dt}</td>'
            f'<td>{caption}</td>'
            f'<td>{ct}</td>'
            f'<td>{_badge(status, s_color)}</td>'
            f'<td>${cost:.3f}</td>'
            f'</tr>'
        )

    sections = f"""
    <div class="stats-row">{''.join(pills)}</div>

    <div class="section">
        <h2>Content Types</h2>
        {type_chart}
    </div>

    <div class="section">
        <h2>Models Used</h2>
        {model_chart}
    </div>

    <div class="section">
        <h2>Approval Rates by Content Type</h2>
        <table class="data-table">
            <thead><tr><th>Type</th><th>Approved</th><th>Rejected</th><th>Rate</th></tr></thead>
            <tbody>{''.join(approval_rows)}</tbody>
        </table>
    </div>

    <div class="section">
        <h2>Recent Generations</h2>
        <table class="data-table">
            <thead><tr><th>Date</th><th>Caption</th><th>Type</th><th>Status</th><th>Cost</th></tr></thead>
            <tbody>{''.join(recent_rows)}</tbody>
        </table>
    </div>
    """
    return _wrap_html(title, subtitle, sections)


def _build_campaign_report(title: str, subtitle: str, campaign_name: str) -> str | None:
    """Build a campaign progress report."""
    data = _collect_campaign_data(campaign_name)
    if not data:
        return None

    campaign = data["campaign"]
    progress = data["progress"]
    slots = campaign.get("slots", [])
    by_status = progress.get("by_status", {})

    pills = [
        _stat_pill(progress.get("total_slots", 0), "Total Posts"),
        _stat_pill(by_status.get("posted", 0), "Posted"),
        _stat_pill(by_status.get("scheduled", 0), "Scheduled"),
        _stat_pill(by_status.get("pending", 0), "Pending"),
    ]

    # Progress bar
    pct = progress.get("progress_pct", 0)
    progress_bar = (
        f'<div class="progress-container">'
        f'<div class="progress-fill" style="width:{pct}%"></div>'
        f'</div>'
        f'<div style="font-size:11px;color:rgba(255,255,255,0.5);text-align:right;margin-top:-18px;margin-bottom:24px;'
        f'font-family:Orbitron,monospace;letter-spacing:1px;text-transform:uppercase">{pct}% complete</div>'
    )

    # Status breakdown chart
    status_items = sorted(by_status.items(), key=lambda x: -x[1])
    status_chart = _bar_chart(status_items, "#D4A843")

    # Slot table
    slot_rows = []
    for slot in slots[:50]:
        s_status = slot.get("status", "pending")
        s_color = {"posted": "green", "scheduled": "blue", "pending": "yellow", "skipped": "gray"}.get(s_status, "gray")
        day = slot.get("day", "—")
        label = _esc(slot.get("slot_label", ""))
        copy_preview = _esc(slot.get("copy", slot.get("angle", ""))[:80])
        ct = _esc(slot.get("content_type", "—"))
        slot_rows.append(
            f'<tr>'
            f'<td>Day {day}</td>'
            f'<td>{label}</td>'
            f'<td>{copy_preview}</td>'
            f'<td>{ct}</td>'
            f'<td>{_badge(s_status, s_color)}</td>'
            f'</tr>'
        )

    brief = _esc(campaign.get("brief", ""))

    sections = f"""
    <div class="card" style="margin-bottom:24px">
        <div class="card-meta">Campaign Brief</div>
        <div class="card-content">{brief}</div>
    </div>

    {progress_bar}
    <div class="stats-row">{''.join(pills)}</div>

    <div class="section">
        <h2>Status Breakdown</h2>
        {status_chart}
    </div>

    <div class="section">
        <h2>All Posts</h2>
        <table class="data-table">
            <thead><tr><th>Day</th><th>Slot</th><th>Content</th><th>Type</th><th>Status</th></tr></thead>
            <tbody>{''.join(slot_rows)}</tbody>
        </table>
    </div>
    """
    return _wrap_html(title, subtitle, sections)


def _build_feedback_report(title: str, subtitle: str) -> str:
    """Build a feedback/preferences report."""
    data = _collect_feedback_data()

    pills = [
        _stat_pill(data["total"], "Total Reviews"),
        _stat_pill(data["approved"], "Approved"),
        _stat_pill(data["rejected"], "Rejected"),
        _stat_pill(f"{data['rate']}%", "Approval Rate"),
    ]

    # Recent feedback entries
    cards = []
    for entry in reversed(data["recent"][-15:]):
        accepted = entry.get("accepted", False)
        status = "approved" if accepted else "rejected"
        color = "green" if accepted else "red"
        request = _esc(entry.get("request", "")[:100])
        caption = _esc(entry.get("draft", {}).get("caption", "")[:120])
        feedback_text = _esc(entry.get("feedback_text", ""))
        ts = entry.get("timestamp", 0)
        dt = datetime.fromtimestamp(ts).strftime("%b %d %H:%M") if ts else ""

        feedback_line = f'<div style="color:#f59e0b;font-size:12px;margin-top:6px">Feedback: {feedback_text}</div>' if feedback_text else ""

        cards.append(f"""
        <div class="card">
            <div class="card-meta">{dt} — {_badge(status, color)}</div>
            <div style="font-size:12px;color:#64748b;margin-bottom:6px">Request: {request}</div>
            <div class="card-content">{caption}</div>
            {feedback_line}
        </div>
        """)

    sections = f"""
    <div class="stats-row">{''.join(pills)}</div>

    <div class="section">
        <h2>Recent Feedback</h2>
        {''.join(cards)}
    </div>
    """
    return _wrap_html(title, subtitle, sections)


def _build_custom_report(title: str, subtitle: str, sections: list[dict]) -> str:
    """Build a report from freeform sections provided by the agent.

    Each section dict: {"heading": "...", "content": "...", "type": "text|table|stats"}
    - text: rendered as preformatted text block
    - table: content is a JSON string of {"headers": [...], "rows": [[...], ...]}
    - stats: content is a JSON string of [{"label": "...", "value": "..."}]
    """
    html_sections = []
    for sec in sections:
        heading = _esc(sec.get("heading", ""))
        content = sec.get("content", "")
        sec_type = sec.get("type", "text")

        if sec_type == "stats":
            try:
                items = json.loads(content) if isinstance(content, str) else content
                pills = [_stat_pill(item["value"], item["label"]) for item in items]
                html_sections.append(f'<div class="stats-row">{"".join(pills)}</div>')
            except (json.JSONDecodeError, KeyError):
                html_sections.append(f'<div class="section"><h2>{heading}</h2><div class="custom-section">{_esc(content)}</div></div>')

        elif sec_type == "table":
            try:
                table_data = json.loads(content) if isinstance(content, str) else content
                headers = table_data.get("headers", [])
                rows = table_data.get("rows", [])
                th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
                trs = []
                for row in rows:
                    tds = "".join(f"<td>{_esc(str(cell))}</td>" for cell in row)
                    trs.append(f"<tr>{tds}</tr>")
                html_sections.append(
                    f'<div class="section"><h2>{heading}</h2>'
                    f'<table class="data-table"><thead><tr>{th}</tr></thead>'
                    f'<tbody>{"".join(trs)}</tbody></table></div>'
                )
            except (json.JSONDecodeError, KeyError):
                html_sections.append(f'<div class="section"><h2>{heading}</h2><div class="custom-section">{_esc(content)}</div></div>')

        else:
            html_sections.append(
                f'<div class="section"><h2>{heading}</h2>'
                f'<div class="custom-section">{_esc(content)}</div></div>'
            )

    return _wrap_html(title, subtitle, "\n".join(html_sections))


def _wrap_html(title: str, subtitle: str, body_sections: str) -> str:
    """Wrap report sections in the FOID-branded HTML template.

    Uses .EXE window chrome, frosted glass panels, iridescent blue background,
    Orbitron headings — matching the foid.fun visual language.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_title = _esc(title).upper().replace(" ", "_")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)} — FOID Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
    <div class="exe-window">
        <div class="exe-titlebar">
            <div class="dots">
                <div class="exe-dot exe-dot-red"></div>
                <div class="exe-dot exe-dot-yellow"></div>
                <div class="exe-dot exe-dot-green"></div>
            </div>
            {safe_title}.EXE
        </div>
        <div class="exe-body">
            <div class="report-header">
                <h1>{_esc(title)}</h1>
                <div class="report-subtitle">{_esc(subtitle)}</div>
            </div>

            {body_sections}
        </div>
    </div>

    <div class="footer">
        generated {now} — foid foundation
    </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    report_type: str = "performance",
    title: str = "",
    subtitle: str = "",
    campaign_name: str = "",
    sections: list[dict] | None = None,
) -> str | None:
    """Generate a branded HTML report.

    Args:
        report_type: "performance", "campaign", "feedback", or "custom"
        title: Report title (auto-generated if empty)
        subtitle: Report subtitle
        campaign_name: Required for campaign reports
        sections: Required for custom reports — list of {"heading", "content", "type"}

    Returns:
        Path to the generated HTML file, or None on error.
    """
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now().strftime("%Y-%m-%d")

    if report_type == "performance":
        title = title or "Performance Report"
        subtitle = subtitle or f"Content generation analytics — {now_str}"
        html_content = _build_performance_report(title, subtitle)

    elif report_type == "campaign":
        if not campaign_name:
            logger.error("Campaign name required for campaign report")
            return None
        title = title or f"{campaign_name} — Campaign Report"
        subtitle = subtitle or f"Campaign progress and post breakdown — {now_str}"
        html_content = _build_campaign_report(title, subtitle, campaign_name)
        if html_content is None:
            return None

    elif report_type == "feedback":
        title = title or "Feedback Report"
        subtitle = subtitle or f"Approval patterns and preferences — {now_str}"
        html_content = _build_feedback_report(title, subtitle)

    elif report_type == "custom":
        if not sections:
            logger.error("Sections required for custom report")
            return None
        title = title or "Report"
        subtitle = subtitle or now_str
        html_content = _build_custom_report(title, subtitle, sections)

    else:
        logger.error("Unknown report type: %s", report_type)
        return None

    # Write file
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title).strip().replace(" ", "_")[:50]
    filename = f"{safe_title}_{int(time.time())}.html"
    output_path = str(_REPORT_DIR / filename)
    Path(output_path).write_text(html_content, encoding="utf-8")
    logger.info("Report generated: %s", output_path)
    return output_path
