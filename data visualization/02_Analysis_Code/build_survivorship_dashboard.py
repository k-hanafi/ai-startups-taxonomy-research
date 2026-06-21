#!/usr/bin/env python3
"""Build the survivorship-bias Wayback findings dashboard.

Same house style as build_tavily_dashboard.py (fixed left nav, serif headers,
metric cards, Plotly charts). The data is produced by
wayback_machine/scripts/summarize_death_coverage.py so the HTML and the in-Cursor
canvas share one source of truth and cannot drift. That summarizer lives under a
non-package `scripts/` dir, so it is imported by file path.

Reads (via the summarizer): wayback_machine/data/death_coverage.csv (live, tolerant
parse) + not_found_cohort.csv, and writes:
    data visualization/01_Presentation_Materials/survivorship_wayback_cohort.html
"""

from __future__ import annotations

import argparse
import datetime
import html
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUMMARIZE_PATH = _PROJECT_ROOT / "wayback_machine" / "scripts" / "summarize_death_coverage.py"
OUTPUT_PATH = (
    _PROJECT_ROOT / "data visualization" / "01_Presentation_Materials" / "survivorship_wayback_cohort.html"
)

# Shared chart palette (mirrors the Tavily dashboard's semantic colors).
PALETTE = {
    "emerald": "#059669",
    "indigo": "#4f46e5",
    "amber": "#d97706",
    "gray": "#94a3b8",
    "rose": "#e11d48",
    "slate": "#64748b",
    "cyan": "#0891b2",
}


def load_summarizer() -> ModuleType:
    """Import summarize_death_coverage.py by path (it self-registers PROJECT_ROOT)."""
    spec = importlib.util.spec_from_file_location("summarize_death_coverage", _SUMMARIZE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load summarizer at {_SUMMARIZE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shape_metrics(summary: dict) -> dict:
    """Reshape the summarizer dict into the flat object the HTML/JS consumes."""
    meta, rec, q = summary["meta"], summary["recovery"], summary["quality"]
    temp, comp = summary["temporal"], summary["composition"]

    failures = [
        {"label": s["status"].replace("error:", ""), "count": s["count"]}
        for s in summary["status_breakdown"]
        if s["status"] != "ok"
    ]
    monthly = temp["monthly"]
    wa = {seg["label"]: seg["count"] for seg in comp["website_alive"]}

    return {
        "meta": {
            "cohort": meta["cohort_n"],
            "probed": meta["probed_n"],
            "probedPct": meta["probed_pct"],
            "lookback": meta["lookback_days"],
            "gpt4": meta["gpt4_launch"],
        },
        "recovery": {
            "preDeath": rec["pre_death"],
            "thin": rec["thin"],
            "noSnapshots": rec["no_snapshots"],
            "noHost": rec["no_host"],
            "error": rec["error"],
            "ok": rec["ok"],
            "recoveryPct": rec["recovery_pct"],
        },
        "failures": failures,
        "quality": {
            "drift": q["drift_buckets"],
            "captures": q["capture_buckets"],
            "buffer": q["buffer_buckets"],
            "medianDrift": q["median_drift_days"],
            "medianBuffer": q["median_buffer_days"],
            "medianCaptures": q["median_captures"],
            "within30Pct": q["within_30d_of_target_pct"],
            "thinPct": q["thin_pct"],
        },
        "temporal": {
            "months": [d["month"] for d in monthly],
            "death": [d["death"] for d in monthly],
            "closest": [d["closest"] for d in monthly],
            "preGenai": temp["closest_pre_genai"],
            "postGenai": temp["closest_post_genai"],
            "gpt4": meta["gpt4_launch"],
        },
        "composition": {
            "founded": comp["founded_year"],
            "aliveCount": wa.get("flagged alive", 0),
            "deadCount": wa.get("flagged dead", 0),
        },
        "examples": summary["examples"],
    }


def examples_rows_html(examples: list[dict]) -> str:
    rows = []
    for e in examples:
        cls = ' class="thin-row"' if e["thin"] else ""
        kind = "thin" if e["thin"] else "pre-death"
        caps = f'{e["n_captures"]:,}' if e["n_captures"] is not None else "&mdash;"
        url = html.escape(e["url"], quote=True)
        rows.append(
            f"<tr{cls}>"
            f'<td>{html.escape(e["name"])}</td>'
            f'<td>{html.escape(e["founded"])}</td>'
            f'<td>{html.escape(e["snapshot_date"])}</td>'
            f'<td class="num">{caps}</td>'
            f"<td>{kind}</td>"
            f'<td><a href="{url}" target="_blank" rel="noopener">view</a></td>'
            f"</tr>"
        )
    return "\n".join(rows)


STYLE = """
:root {
  --bg: #ffffff; --bg2: #ffffff; --bg3: #f8f9fb;
  --border: #e5e7eb; --border2: #d1d5db;
  --text: #1a1a1a; --text2: #4a4a4a; --muted: #8a8a8a;
  --navy: #1e2a4a; --indigo: #4f46e5; --indigo-light: #eef2ff; --indigo-border: #c7d2fe;
  --emerald: #059669; --emerald-light: #ecfdf5;
  --amber: #d97706; --amber-light: #fffbeb;
  --rose: #e11d48; --rose-light: #fff1f2;
  --serif: 'Cormorant Garamond', Georgia, serif;
  --sans: 'Inter', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: var(--sans); background: var(--bg); color: var(--text); line-height: 1.7; font-size: 15px; }
::selection { background: var(--navy); color: white; }

nav {
  position: fixed; top: 0; left: 0; height: 100vh; width: 216px;
  padding: 2.25rem 1.75rem; background: #000000; border-right: 1px solid rgba(255,255,255,0.08);
  z-index: 100; display: flex; flex-direction: column; overflow-y: auto;
}
.nav-brand { font-family: var(--serif); font-size: 1rem; font-weight: 600; color: #ffffff; margin-bottom: 0.2rem; }
.nav-sub { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.5); margin-bottom: 2.5rem; }
.nav-section { margin-bottom: 1.5rem; }
.nav-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255,255,255,0.4); margin-bottom: 0.55rem; }
nav ul { list-style: none; }
nav ul li { margin-bottom: 0.3rem; }
nav ul a { color: rgba(255,255,255,0.65); text-decoration: none; font-size: 0.8rem; display: block; padding: 0.15rem 0; transition: color 0.15s; }
nav ul a:hover, nav ul a.active { color: #ffffff; font-weight: 500; }
.nav-meta { margin-top: auto; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); }
.nav-meta p { font-size: 0.7rem; color: rgba(255,255,255,0.45); line-height: 1.6; }
.nav-meta strong { color: rgba(255,255,255,0.7); }

main { margin-left: 216px; }
section {
  padding: 5rem 4.5rem; max-width: 1100px;
  border-bottom: 1px solid var(--border);
  opacity: 0; transform: translateY(20px);
  transition: opacity 0.65s ease, transform 0.65s ease;
}
section.visible { opacity: 1; transform: translateY(0); }
section:last-of-type { border-bottom: none; }

.section-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.16em; color: var(--navy); font-weight: 600; margin-bottom: 0.85rem; display: block; }
h1 { font-family: var(--serif); font-size: clamp(2.2rem, 4vw, 3rem); font-weight: 400; letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.4rem; color: var(--navy); }
h2 { font-family: var(--serif); font-size: clamp(1.6rem, 2.8vw, 2rem); font-weight: 400; line-height: 1.2; margin-bottom: 0.85rem; color: var(--navy); }
p { color: var(--text2); font-size: 0.9rem; max-width: 720px; margin-bottom: 1.1rem; line-height: 1.75; }
p:last-child { margin-bottom: 0; }

.tags { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1.25rem 0 0.5rem; }
.tag { font-family: var(--mono); font-size: 0.68rem; padding: 0.3rem 0.7rem; border: 1px solid var(--border); border-radius: 999px; color: var(--text2); background: var(--bg3); }

.hero-metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin: 2rem 0 1.5rem; }
.hero-metrics.cols-4 { grid-template-columns: repeat(4, 1fr); }
.metric-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem 1.3rem; text-align: center; transition: box-shadow 0.2s; }
.metric-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.metric-card.hl { border-color: var(--indigo-border); background: var(--indigo-light); }
.metric-card.hl .mc-val { color: var(--indigo); }
.metric-card.ok { border-color: #a7f3d0; background: var(--emerald-light); }
.metric-card.ok .mc-val { color: var(--emerald); }
.mc-val { font-family: var(--serif); font-size: 2.2rem; line-height: 1; margin-bottom: 0.25rem; color: var(--navy); }
.mc-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.3rem; }
.mc-ctx { font-size: 0.75rem; color: var(--muted); }

.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0; }
.chart-row.single { grid-template-columns: 1fr; }
.chart-box { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.chart-box-header { padding: 1.1rem 1.4rem 0.75rem; border-bottom: 1px solid var(--border); }
.chart-box-title { font-family: var(--serif); font-size: 1.05rem; font-weight: 600; color: var(--navy); margin-bottom: 0.3rem; }
.chart-box-desc { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }
.chart-body { padding: 0.5rem 0.5rem; }

.insight { padding: 1.1rem 1.4rem; border-radius: 8px; margin: 1.5rem 0; font-size: 0.85rem; line-height: 1.7; }
.insight p { font-size: 0.85rem; max-width: none; margin-bottom: 0.35rem; }
.insight p:last-child { margin-bottom: 0; }
.insight-blue { background: var(--indigo-light); border: 1px solid var(--indigo-border); color: var(--text2); }
.insight-blue strong { color: #3730a3; }
.insight-green { background: var(--emerald-light); border: 1px solid #a7f3d0; color: var(--text2); }
.insight-green strong { color: #047857; }
.insight-amber { background: var(--amber-light); border: 1px solid #fde68a; color: var(--text2); }
.insight-amber strong { color: #b45309; }

.data-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.data-table th { text-align: left; font-weight: 600; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--border2); }
.data-table td { padding: 0.55rem 0.9rem; border-bottom: 1px solid var(--border); color: var(--text2); }
.data-table td.num { font-family: var(--mono); text-align: right; }
.data-table tbody tr:hover { background: var(--bg3); }
.data-table tr.thin-row { background: var(--amber-light); }
.data-table tr.thin-row:hover { background: #fef3c7; }
.data-table a { color: var(--indigo); text-decoration: none; font-weight: 500; }
.data-table a:hover { text-decoration: underline; }

footer { padding: 2.5rem 4.5rem; text-align: center; color: var(--muted); font-size: 0.75rem; border-top: 1px solid var(--border); margin-left: 216px; line-height: 1.8; }
footer strong { color: var(--text2); }

@media (max-width: 1100px) {
  nav { display: none; } main { margin-left: 0; } section { padding: 3rem 1.5rem; }
  .hero-metrics, .hero-metrics.cols-4 { grid-template-columns: repeat(2, 1fr); }
  .chart-row { grid-template-columns: 1fr; }
  footer { margin-left: 0; padding: 2rem 1.5rem; }
}
@media print {
  section { opacity: 1 !important; transform: none !important; }
  nav { display: none; } main { margin-left: 0; }
}
"""


SCRIPT_TEMPLATE = """
const M = __M_JSON__;
const C = __PALETTE__;

const plotlyConfig = {displayModeBar: false, responsive: true};
const axisFont = {family: 'Inter, sans-serif', size: 11, color: '#4a4a4a'};
const titleFont = {family: 'Inter, sans-serif', size: 12, color: '#1e2a4a'};
const monoFont = {family: 'JetBrains Mono', size: 10};

function plotLayout(extra) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {l: 60, r: 24, t: 24, b: 48}, font: axisFont,
    xaxis: {gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'},
    yaxis: {gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'},
  }, extra || {});
}

function barV(id, arr, color, xTitle) {
  const labels = arr.map(d => d.label);
  const vals = arr.map(d => d.count);
  Plotly.newPlot(id, [{
    type: 'bar', x: labels, y: vals, marker: {color: color},
    text: vals.map(v => v.toLocaleString()), textposition: 'outside', textfont: monoFont,
    hovertemplate: '%{x}<br>%{y:,} companies<extra></extra>',
  }], plotLayout({
    yaxis: {title: {text: 'Companies', font: titleFont}},
    xaxis: {title: {text: xTitle, font: titleFont}, tickfont: {family: 'JetBrains Mono', size: 11}},
    margin: {l: 64, r: 24, t: 24, b: 50},
  }), plotlyConfig);
}

function donut(id, labels, values, colors) {
  Plotly.newPlot(id, [{
    type: 'pie', hole: 0.55, sort: false,
    labels: labels, values: values, marker: {colors: colors},
    textinfo: 'percent', textfont: {family: 'JetBrains Mono', size: 12},
    hovertemplate: '%{label}<br>%{value:,} (%{percent})<extra></extra>',
  }], plotLayout({
    margin: {l: 8, r: 8, t: 8, b: 8}, showlegend: true,
    legend: {orientation: 'h', y: -0.05, x: 0.5, xanchor: 'center', font: {size: 10}},
  }), plotlyConfig);
}

function renderRecovery() {
  const r = M.recovery;
  const segs = [
    {name: 'Pre-death ready', val: r.preDeath, color: C.emerald},
    {name: 'Thin history', val: r.thin, color: C.cyan},
    {name: 'Errors (retryable)', val: r.error, color: C.amber},
    {name: 'Never archived', val: r.noSnapshots, color: C.gray},
    {name: 'Invalid homepage', val: r.noHost, color: C.rose},
  ].filter(s => s.val > 0);
  const traces = segs.map(s => ({
    type: 'bar', orientation: 'h', name: s.name,
    y: ['Probed'], x: [s.val], marker: {color: s.color},
    text: [s.val.toLocaleString()], textposition: 'inside', insidetextanchor: 'middle',
    textfont: {family: 'JetBrains Mono', size: 11, color: '#ffffff'},
    hovertemplate: s.name + ': %{x:,}<extra></extra>',
  }));
  Plotly.newPlot('chart-recovery', traces, plotLayout({
    barmode: 'stack',
    xaxis: {visible: false}, yaxis: {visible: false},
    legend: {orientation: 'h', y: -0.35, x: 0.5, xanchor: 'center', font: {size: 11}},
    margin: {l: 8, r: 8, t: 8, b: 8},
  }), plotlyConfig);
}

function renderFailures() {
  const f = M.failures.slice().reverse();
  Plotly.newPlot('chart-failures', [{
    type: 'bar', orientation: 'h',
    y: f.map(d => d.label), x: f.map(d => d.count),
    marker: {color: C.amber},
    text: f.map(d => d.count.toLocaleString()), textposition: 'outside', textfont: monoFont,
    hovertemplate: '%{y}<br>%{x:,} companies<extra></extra>',
  }], plotLayout({
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}},
    xaxis: {title: {text: 'Companies', font: titleFont}},
    margin: {l: 150, r: 64, t: 16, b: 44},
  }), plotlyConfig);
}

function renderTemporal() {
  const t = M.temporal;
  Plotly.newPlot('chart-temporal', [
    {type: 'scatter', mode: 'lines', name: 'Last capture (death anchor)', x: t.months, y: t.death,
     line: {color: C.slate, width: 2}, hovertemplate: '%{x}<br>Death anchor: %{y:,}<extra></extra>'},
    {type: 'scatter', mode: 'lines', name: 'Chosen pre-death snapshot', x: t.months, y: t.closest,
     line: {color: C.emerald, width: 2}, hovertemplate: '%{x}<br>Snapshot: %{y:,}<extra></extra>'},
  ], plotLayout({
    yaxis: {title: {text: 'Captures per month', font: titleFont}},
    xaxis: {tickfont: {size: 10}, nticks: 14},
    legend: {orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center', font: {size: 11}},
    margin: {l: 60, r: 24, t: 44, b: 44},
    shapes: [{type: 'line', x0: '2023-03', x1: '2023-03', y0: 0, y1: 1, yref: 'paper',
              line: {color: C.rose, width: 1, dash: 'dot'}}],
    annotations: [{x: '2023-03', y: 1, yref: 'paper', yanchor: 'bottom',
                   text: 'GPT-4 launch', showarrow: false,
                   font: {size: 10, color: C.rose, family: 'JetBrains Mono'}}],
  }), plotlyConfig);
}

renderRecovery();
renderFailures();
barV('chart-drift', M.quality.drift, C.indigo, 'Days from ideal 6-month target');
barV('chart-captures', M.quality.captures, C.slate, 'Wayback captures per site');
barV('chart-buffer', M.quality.buffer, C.emerald, 'Days between snapshot and death');
renderTemporal();
donut('chart-era', ['GenAI era (on/after GPT-4)', 'Pre-GPT-4'], [M.temporal.postGenai, M.temporal.preGenai], [C.indigo, C.slate]);
barV('chart-founded', M.composition.founded, C.indigo, 'Founding year');
donut('chart-website', ['Flagged live', 'Flagged dead'], [M.composition.aliveCount, M.composition.deadCount], [C.emerald, C.rose]);

const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, {threshold: 0.06, rootMargin: '0px 0px -30px 0px'});
document.querySelectorAll('section').forEach(s => observer.observe(s));
document.getElementById('overview').classList.add('visible');

const navLinks = document.querySelectorAll('nav ul a');
const navObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(a => a.classList.remove('active'));
      const link = document.querySelector('nav ul a[href="#' + e.target.id + '"]');
      if (link) link.classList.add('active');
    }
  });
}, {threshold: 0.25});
document.querySelectorAll('section').forEach(s => navObs.observe(s));
"""


def build_html(summary: dict) -> str:
    m = shape_metrics(summary)
    meta, rec, q, temp, comp = m["meta"], m["recovery"], m["quality"], m["temporal"], m["composition"]

    cohort, probed, probed_pct, lookback = meta["cohort"], meta["probed"], meta["probedPct"], meta["lookback"]
    pre_death, errors, no_snap, ok = rec["preDeath"], rec["error"], rec["noSnapshots"], rec["ok"]
    recovery_pct = rec["recoveryPct"]

    genai_total = temp["postGenai"] + temp["preGenai"]
    genai_pct = round(temp["postGenai"] / genai_total * 100, 1) if genai_total else 0.0
    nosnap_pct = round(no_snap / probed * 100, 1) if probed else 0.0
    alive_total = comp["aliveCount"] + comp["deadCount"]
    alive_pct = round(comp["aliveCount"] / alive_total * 100, 1) if alive_total else 0.0
    n_examples = len(m["examples"])
    today = datetime.date.today().strftime("%b %d, %Y")

    rows_html = examples_rows_html(m["examples"])
    script = SCRIPT_TEMPLATE.replace("__M_JSON__", json.dumps(m)).replace("__PALETTE__", json.dumps(PALETTE))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Survivorship recovery &mdash; Wayback findings</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{STYLE}</style>
</head>
<body>

<nav>
  <div class="nav-brand">Survivorship recovery</div>
  <div class="nav-sub">Wayback findings</div>
  <div class="nav-section">
    <div class="nav-label">Sections</div>
    <ul>
      <li><a href="#overview">Overview</a></li>
      <li><a href="#recovery">Recoverability</a></li>
      <li><a href="#quality">Snapshot Quality</a></li>
      <li><a href="#temporal">Temporal</a></li>
      <li><a href="#composition">Composition</a></li>
      <li><a href="#audit">Audit Sample</a></li>
    </ul>
  </div>
  <div class="nav-meta">
    <p><strong>Method</strong><br>Death-anchored CDX probe</p>
    <p style="margin-top:0.75rem;"><strong>Source</strong><br>Internet Archive<br>(Wayback Machine)</p>
    <p style="margin-top:0.75rem;"><strong>Lookback</strong><br>{lookback} days pre-death</p>
    <p style="margin-top:0.75rem;"><strong>Cohort</strong><br>{cohort:,} not-found<br>{probed:,} probed ({probed_pct}%)</p>
  </div>
</nav>

<main>

<section id="overview">
  <span class="section-label">Survivorship-Bias Recovery</span>
  <h1>Recovering the Companies<br>the Live Crawl Never Saw</h1>
  <p>
    The live pipeline can only classify companies whose sites Tavily can still read today, which silently
    biases the dataset toward survivors. This <strong>death-anchored Wayback probe</strong> recovers a pre-death
    snapshot from the Internet Archive for the {cohort:,} companies Tavily could not extract, so they can run
    through the <strong>unchanged classifier</strong>. For each company it takes the full capture history, treats the
    most recent capture as the site&apos;s death, and picks the capture closest to <strong>{lookback} days
    (~6 months)</strong> before death &mdash; staying out of the parked / dead final-page tail.
  </p>
  <div class="tags">
    <span class="tag">Death-anchored CDX probe</span>
    <span class="tag">{lookback}-day pre-death lookback</span>
    <span class="tag">Internet Archive</span>
    <span class="tag">Findings only</span>
  </div>

  <div class="hero-metrics">
    <div class="metric-card hl">
      <div class="mc-label">Lost cohort</div>
      <div class="mc-val">{cohort:,}</div>
      <div class="mc-ctx">Tavily could not extract</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">Probed so far</div>
      <div class="mc-val">{probed_pct}%</div>
      <div class="mc-ctx">{probed:,} companies</div>
    </div>
    <div class="metric-card ok">
      <div class="mc-label">Pre-death ready</div>
      <div class="mc-val">{pre_death:,}</div>
      <div class="mc-ctx">clean snapshot recovered</div>
    </div>
    <div class="metric-card ok">
      <div class="mc-label">Recovery rate</div>
      <div class="mc-val">{recovery_pct}%</div>
      <div class="mc-ctx">of probed companies</div>
    </div>
    <div class="metric-card" style="border-color:#fecdd3;background:var(--rose-light);">
      <div class="mc-label">Retryable errors</div>
      <div class="mc-val" style="color:var(--rose);">{errors:,}</div>
      <div class="mc-ctx">transient, re-probe recovers</div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p><strong>In-progress snapshot.</strong> The overnight CDX sweep is still running ({probed_pct}% of the
    {cohort:,}-company cohort probed). These are point-in-time numbers; re-running the aggregator refreshes the
    dashboard once the full run completes.</p>
  </div>
</section>

<section id="recovery">
  <span class="section-label">01. Recoverability</span>
  <h2>How Much of the Lost Cohort Is Recoverable</h2>
  <p>
    {pre_death:,} of {probed:,} probed companies (<strong>{recovery_pct}%</strong>) yield a clean pre-death snapshot.
    Only {no_snap:,} were never archived; the {errors:,} errors are transient network / CDX throttles that a re-run
    recovers.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Recovery composition</div>
        <div class="chart-box-desc">Outcome share across all {probed:,} probed companies</div>
      </div>
      <div class="chart-body"><div id="chart-recovery" style="height:170px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Unresolved outcomes by type</div>
        <div class="chart-box-desc">The {ok:,} &ldquo;ok&rdquo; rows are excluded to show the failure tail</div>
      </div>
      <div class="chart-body"><div id="chart-failures" style="height:280px;"></div></div>
    </div>
  </div>

  <div class="insight insight-green">
    <p><strong>Coverage is high and mostly floor-limited by transient errors.</strong> Recovery sits at
    {recovery_pct}% and will rise once the {errors:,} retryable errors are re-probed. Genuine dead-ends &mdash; sites
    never archived at all &mdash; are just {no_snap:,} ({nosnap_pct}% of probed).</p>
  </div>
</section>

<section id="quality">
  <span class="section-label">02. Snapshot Quality</span>
  <h2>Are the Recovered Snapshots Good Enough to Classify</h2>
  <p>
    The probe targets a capture ~6 months before death. These distributions show how close it lands, how richly the
    sites are archived, and how large a safety buffer sits between the chosen snapshot and the parked-page tail.
  </p>

  <div class="hero-metrics cols-4">
    <div class="metric-card">
      <div class="mc-label">Median drift</div>
      <div class="mc-val">{q["medianDrift"]}d</div>
      <div class="mc-ctx">from the 6-month anchor</div>
    </div>
    <div class="metric-card ok">
      <div class="mc-label">Within 30 days</div>
      <div class="mc-val">{q["within30Pct"]}%</div>
      <div class="mc-ctx">of the ideal target</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">Median captures</div>
      <div class="mc-val">{q["medianCaptures"]}</div>
      <div class="mc-ctx">per recovered site</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">Thin history</div>
      <div class="mc-val">{q["thinPct"]}%</div>
      <div class="mc-ctx">fallback captures</div>
    </div>
  </div>

  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Anchor drift</div>
        <div class="chart-box-desc">Companies by days from the ideal 6-month target</div>
      </div>
      <div class="chart-body"><div id="chart-drift" style="height:300px;"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Archival richness</div>
        <div class="chart-box-desc">Companies by total Wayback capture count</div>
      </div>
      <div class="chart-body"><div id="chart-captures" style="height:300px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Pre-death buffer</div>
        <div class="chart-box-desc">Days between the chosen snapshot and death &mdash; bounded below by the {lookback}-day lookback</div>
      </div>
      <div class="chart-body"><div id="chart-buffer" style="height:300px;"></div></div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p>Every pre-death snapshot sits at least {lookback} days before the death anchor (median {q["medianBuffer"]} days),
    so the classifier never reads a dead or parked final page. Median drift from the ideal 6-month target is only
    {q["medianDrift"]} days, and {q["within30Pct"]}% land within a month. Sites are richly archived (median
    {q["medianCaptures"]} captures), and only {q["thinPct"]}% fall back to a thin history.</p>
  </div>
</section>

<section id="temporal">
  <span class="section-label">03. Temporal</span>
  <h2>When the Cohort Was Last Seen, and What Evidence Era We Recovered</h2>
  <p>
    The death anchor (most recent Wayback capture) clusters heavily in 2025&ndash;2026: many of these companies were
    still being archived recently, so they are unextractable rather than long-dead. The chosen snapshot necessarily
    lags it by ~6 months.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Monthly Wayback activity</div>
        <div class="chart-box-desc">Captures per month: death anchor vs chosen snapshot (GPT-4 launch marked)</div>
      </div>
      <div class="chart-body"><div id="chart-temporal" style="height:360px;"></div></div>
    </div>
  </div>

  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Evidence era vs GPT-4 launch</div>
        <div class="chart-box-desc">Chosen-snapshot date relative to {meta["gpt4"]}</div>
      </div>
      <div class="chart-body"><div id="chart-era" style="height:300px;"></div></div>
    </div>
    <div class="insight insight-blue" style="margin:0; align-self:center;">
      <p><strong>{genai_pct}% of recovered snapshots are GenAI-era.</strong> {temp["postGenai"]:,} postdate
      GPT-4&apos;s launch versus {temp["preGenai"]:,} before it. The pre-death evidence we feed the classifier mostly
      reflects the same GenAI era the live-web survivors were seen in &mdash; keeping the survivorship comparison fair
      rather than skewing the dead cohort toward older, pre-AI messaging.</p>
    </div>
  </div>
</section>

<section id="composition">
  <span class="section-label">04. Composition</span>
  <h2>Who Is in the Lost Cohort</h2>
  <p>
    The not-found cohort skews to recent founders, and a striking share were still flagged live at crawl time &mdash;
    a reminder that &ldquo;Tavily could not extract&rdquo; is an extraction failure, not a death certificate.
  </p>

  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Founding year</div>
        <div class="chart-box-desc">Company count by founding year</div>
      </div>
      <div class="chart-body"><div id="chart-founded" style="height:300px;"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Live-site status</div>
        <div class="chart-box-desc">website_alive flag among Tavily-not-found companies</div>
      </div>
      <div class="chart-body"><div id="chart-website" style="height:300px;"></div></div>
    </div>
  </div>

  <div class="insight insight-amber">
    <p><strong>&ldquo;Not found&rdquo; does not mean &ldquo;dead&rdquo;.</strong> {comp["aliveCount"]:,} of these
    companies ({alive_pct}%) are still flagged live, yet Tavily could not extract them &mdash; typically parked,
    JS-only, or bot-blocked pages. That is precisely the evidence the Wayback recovery restores.</p>
  </div>
</section>

<section id="audit">
  <span class="section-label">05. Spot-Check</span>
  <h2>Recovered Snapshots: Audit Sample</h2>
  <p>
    A sample of recovered companies &mdash; clean pre-death snapshots plus a few thin-history cases (highlighted).
    Open the Wayback link to inspect the exact page the classifier would read.
  </p>
  <div class="chart-box">
    <div class="chart-box-header">
      <div class="chart-box-title">Audit sample</div>
      <div class="chart-box-desc">{n_examples} of {pre_death:,} recovered shown</div>
    </div>
    <div style="padding:0.4rem 0.6rem;">
      <table class="data-table">
        <thead>
          <tr><th>Company</th><th>Founded</th><th>Snapshot</th><th class="num">Captures</th><th>Type</th><th>Wayback</th></tr>
        </thead>
        <tbody>
{rows_html}
        </tbody>
      </table>
    </div>
  </div>
</section>

</main>

<footer>
  <strong>Method:</strong> Death-anchored Wayback CDX probe &nbsp;&middot;&nbsp;
  <strong>Lookback:</strong> {lookback} days pre-death &nbsp;&middot;&nbsp;
  <strong>Source:</strong> Internet Archive &nbsp;&middot;&nbsp;
  <strong>Generated:</strong> {today}
  <br>
  Point-in-time snapshot of death_coverage.csv &mdash; {probed:,} / {cohort:,} probed ({probed_pct}%).
  Regenerate with build_survivorship_dashboard.py once the full sweep completes.
</footer>

<script>
{script}
</script>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="death_coverage.csv (default: summarizer's path)")
    parser.add_argument("--cohort", type=Path, default=None, help="not_found_cohort.csv (default: summarizer's path)")
    parser.add_argument("-o", "--output", type=Path, default=None, help=f"Output HTML (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    sdc = load_summarizer()
    input_path = (args.input or sdc.DEATH_CSV).resolve()
    cohort_path = (args.cohort or sdc.COHORT_CSV).resolve()
    out_path = (args.output or OUTPUT_PATH).resolve()

    if not input_path.is_file():
        print(f"Missing input CSV: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path} ...")
    df = sdc.load(input_path)
    summary = sdc.summarize(df, sdc.cohort_size(cohort_path))
    print(f"  {summary['meta']['probed_n']:,} probed / {summary['meta']['cohort_n']:,} cohort")

    print("Building HTML ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(summary), encoding="utf-8")
    print(f"Dashboard written to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
