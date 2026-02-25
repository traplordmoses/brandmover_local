"""Web dashboard for evaluation results.

Usage:
    python -m eval.dashboard
    python eval/dashboard.py

Opens on http://localhost:5001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from flask import Flask, abort

RESULTS_DIR = Path(__file__).resolve().parent / "results"

app = Flask(__name__)


def _load_latest_results() -> dict[str, dict]:
    """Load the most recent result file per task_id."""
    results = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == ".gitkeep":
            continue
        try:
            data = json.loads(f.read_text())
            task_id = data.get("scenario", {}).get("task_id", f.stem)
            results[task_id] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return results


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BrandMover Eval Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0a0a0a; color: #e0e0e0; padding: 2rem; }
    h1 { color: #ff8800; margin-bottom: 1.5rem; }
    h2 { color: #ff8800; margin: 1.5rem 0 0.75rem; }
    a { color: #ff8800; text-decoration: none; }
    a:hover { text-decoration: underline; }
    table { width: 100%%; border-collapse: collapse; margin-bottom: 2rem; }
    th, td { padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #222; }
    th { color: #ff8800; font-weight: 600; }
    .pass { color: #4caf50; font-weight: bold; }
    .fail { color: #f44336; font-weight: bold; }
    .chart-wrap { max-width: 700px; margin: 1rem 0 2rem; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
             font-size: 0.8rem; font-weight: 600; }
    .badge-pass { background: #1b5e20; color: #a5d6a7; }
    .badge-fail { background: #b71c1c; color: #ef9a9a; }
    pre { background: #111; padding: 1rem; border-radius: 6px; overflow-x: auto;
          font-size: 0.85rem; margin: 0.5rem 0; }
    .card { background: #111; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>BrandMover Agent Eval</h1>
  <table>
    <thead>
      <tr>
        <th>Task ID</th><th>Status</th><th>Tool Correctness</th>
        <th>Turns</th><th>Latency</th><th>Violations</th>
      </tr>
    </thead>
    <tbody>
      %(rows)s
    </tbody>
  </table>
  <h2>Tool Correctness by Scenario</h2>
  <div class="chart-wrap"><canvas id="tcChart"></canvas></div>
  <script>
    const labels = %(labels_json)s;
    const data = %(tc_json)s;
    new Chart(document.getElementById('tcChart'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Tool Correctness',
          data: data,
          backgroundColor: data.map(v => v >= 0.7 ? '#4caf50' : '#f44336'),
          borderRadius: 4
        }]
      },
      options: {
        scales: { y: { beginAtZero: true, max: 1, ticks: { color: '#aaa' },
                        grid: { color: '#222' } },
                  x: { ticks: { color: '#aaa' }, grid: { color: '#222' } } },
        plugins: { legend: { display: false } }
      }
    });
  </script>
</body></html>"""

DETAIL_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>%(task_id)s — Eval Detail</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0a0a0a; color: #e0e0e0; padding: 2rem; }
    h1 { color: #ff8800; margin-bottom: 0.5rem; }
    h2 { color: #ff8800; margin: 1.5rem 0 0.75rem; }
    a { color: #ff8800; }
    .card { background: #111; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
    pre { background: #0d0d0d; padding: 1rem; border-radius: 6px; overflow-x: auto;
          font-size: 0.85rem; margin: 0.5rem 0; }
    table { width: 100%%; border-collapse: collapse; }
    th, td { padding: 0.4rem 0.8rem; text-align: left; border-bottom: 1px solid #222; }
    th { color: #ff8800; }
    .pass { color: #4caf50; } .fail { color: #f44336; }
    img { max-width: 400px; border-radius: 8px; margin: 0.5rem 0; }
    .timeline { list-style: none; padding: 0; }
    .timeline li { padding: 0.3rem 0; border-left: 2px solid #ff8800; padding-left: 1rem;
                   margin-left: 0.5rem; }
  </style>
</head>
<body>
  <a href="/">&larr; Back</a>
  <h1>%(task_id)s</h1>
  <p><em>%(category)s</em></p>

  <h2>Scores</h2>
  <div class="card">
    <table>%(score_rows)s</table>
  </div>

  <h2>Tool Timeline</h2>
  <div class="card">
    <ul class="timeline">%(timeline_items)s</ul>
  </div>

  <h2>Draft</h2>
  <div class="card"><pre>%(draft_json)s</pre></div>

  %(image_section)s

  <h2>Final Text</h2>
  <div class="card"><pre>%(final_text)s</pre></div>
</body></html>"""


@app.route("/")
def index():
    results = _load_latest_results()
    rows = ""
    labels = []
    tc_values = []

    for task_id, data in sorted(results.items()):
        s = data.get("scores", {})
        success = s.get("success", False)
        cls = "pass" if success else "fail"
        badge = f'<span class="badge badge-{cls}">{"PASS" if success else "FAIL"}</span>'
        violations = ", ".join(s.get("forbidden_term_violations", [])) or "none"
        rows += (
            f"<tr>"
            f'<td><a href="/scenario/{task_id}">{task_id}</a></td>'
            f"<td>{badge}</td>"
            f"<td>{s.get('tool_correctness', 0):.1%}</td>"
            f"<td>{s.get('turns_used', 0)}/{s.get('max_rounds', '?')}</td>"
            f"<td>{s.get('latency_seconds', 0):.1f}s</td>"
            f"<td>{violations}</td>"
            f"</tr>\n"
        )
        labels.append(task_id)
        tc_values.append(s.get("tool_correctness", 0))

    html = INDEX_TEMPLATE % {
        "rows": rows or "<tr><td colspan='6'>No results yet. Run eval/runner.py first.</td></tr>",
        "labels_json": json.dumps(labels),
        "tc_json": json.dumps(tc_values),
    }
    return html


@app.route("/scenario/<task_id>")
def scenario_detail(task_id):
    results = _load_latest_results()
    data = results.get(task_id)
    if not data:
        abort(404)

    scores = data.get("scores", {})
    trace = data.get("trace", {})
    scenario = data.get("scenario", {})

    # Score table
    score_rows = ""
    for key, val in sorted(scores.items()):
        if key == "task_id":
            continue
        display = val
        cls = ""
        if isinstance(val, bool):
            cls = ' class="pass"' if val else ' class="fail"'
            display = "Yes" if val else "No"
        elif isinstance(val, float):
            display = f"{val:.3f}"
        score_rows += f"<tr><th>{key}</th><td{cls}>{display}</td></tr>\n"

    # Tool timeline
    tool_trace = trace.get("tool_trace", [])
    timeline_items = ""
    t0 = tool_trace[0]["timestamp"] if tool_trace else 0
    for entry in tool_trace:
        offset = entry["timestamp"] - t0
        timeline_items += (
            f'<li><strong>{entry["tool"]}</strong> '
            f'— {entry.get("description", "")} '
            f"<small>(+{offset:.1f}s)</small></li>\n"
        )
    if not timeline_items:
        timeline_items = "<li>No tool calls recorded</li>"

    # Draft JSON
    draft_json = json.dumps(trace.get("draft", {}), indent=2, default=str)

    # Image preview
    image_url = trace.get("image_url") or ""
    image_urls = trace.get("image_urls", [])
    all_images = ([image_url] if image_url else []) + image_urls
    image_section = ""
    if all_images:
        image_section = "<h2>Generated Images</h2><div class='card'>"
        for url in all_images:
            image_section += f'<img src="{url}" alt="generated">\n'
        image_section += "</div>"

    html = DETAIL_TEMPLATE % {
        "task_id": task_id,
        "category": scenario.get("category", "unknown"),
        "score_rows": score_rows,
        "timeline_items": timeline_items,
        "draft_json": draft_json,
        "image_section": image_section,
        "final_text": trace.get("final_text", "(empty)"),
    }
    return html


def main():
    print("BrandMover Eval Dashboard")
    print("http://localhost:5001")
    import os
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5001, debug=debug)


if __name__ == "__main__":
    main()
