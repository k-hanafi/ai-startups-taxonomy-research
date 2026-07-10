#!/usr/bin/env python3
"""Build the Stage 9 golden-set eval dashboard (LangSmith-inspired light UX).

Thin viewer skeleton: experiments table + summary line chart, Pareto,
confidence, latency, with a client-side config filter. Defaults to the
synthetic Stage 8 matrix fixture; pass --runs / --scored for real scored.json.

Writes:
    data visualization/01_Presentation_Materials/eval_dashboard.html

Visual language follows LangSmith's light-mode eval kit: pure white canvas,
generous whitespace, blue tab underlines, score bars, soft latency chips.
Deliberately does NOT reuse the survivorship navy/Cormorant house STYLE.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evals.dashboard_metrics import (  # noqa: E402
    DEFAULT_FIXTURE,
    load_fixture,
    load_from_scored_paths,
)

OUTPUT_PATH = (
    _PROJECT_ROOT
    / "data visualization"
    / "01_Presentation_Materials"
    / "eval_dashboard.html"
)

# Soft accents for model groups on charts (not the dense house palette).
GROUP_COLORS = {
    "nano": "#3b82f6",
    "mini": "#14b8a6",
    "luna": "#f59e0b",
}

STYLE = """
:root {
  --bg: #ffffff;
  --surface: #ffffff;
  --surface-muted: #fafafa;
  --border: #eceef2;
  --border-strong: #d8dce3;
  --text: #0f172a;
  --text2: #475569;
  --muted: #94a3b8;
  --accent: #2563eb;
  --accent-soft: #eff6ff;
  --accent-border: #bfdbfe;
  --green-bar: #86efac;
  --line: #4ade80;
  --amber-pill: #fff7ed;
  --amber-text: #c2410c;
  --rose-pill: #fff1f2;
  --rose-text: #be123c;
  --radius: 8px;
  --sans: "Plus Jakarta Sans", "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.55;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  font-family: var(--mono);
  font-size: 0.92em;
  background: var(--surface-muted);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.app {
  max-width: 1120px;
  margin: 0 auto;
  padding: 40px 40px 80px;
}

/* Header: folder + title + subtitle (LangSmith dataset page rhythm) */
.header {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  margin-bottom: 8px;
}
.header-icon {
  flex: 0 0 auto;
  width: 36px;
  height: 36px;
  margin-top: 2px;
  color: var(--muted);
}
.header-copy { min-width: 0; }
.header-title {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.15;
  color: var(--text);
}
.header-sub {
  margin-top: 6px;
  font-size: 14px;
  color: var(--text2);
}

/* Tabs */
.tabs {
  display: flex;
  gap: 28px;
  border-bottom: 1px solid var(--border);
  margin: 28px 0 0;
}
.tab {
  appearance: none;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  padding: 12px 0 14px;
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  color: var(--text2);
  cursor: pointer;
}
.tab:hover { color: var(--text); }
.tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

/* Panels */
.panel { display: none; padding-top: 28px; }
.panel.active { display: block; }

/* Banner */
.banner {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 14px 16px;
  margin-bottom: 24px;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: var(--radius);
  color: #92400e;
  font-size: 13px;
  line-height: 1.5;
}
.banner strong { font-weight: 600; }

/* Toolbar */
.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px 16px;
  padding: 0 0 20px;
  margin-bottom: 8px;
}
.toolbar-left, .toolbar-right {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.toolbar-right { margin-left: auto; }
.toolbar-label {
  font-size: 12px;
  color: var(--muted);
  font-weight: 500;
  margin-right: 2px;
}
.toolbar-count {
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
}
.search {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 200px;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--muted);
}
.search:focus-within {
  border-color: var(--accent-border);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.search svg { flex: 0 0 auto; }
.search input {
  border: none;
  outline: none;
  background: transparent;
  font: inherit;
  font-size: 13px;
  color: var(--text);
  width: 100%;
}
.search input::placeholder { color: var(--muted); }

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  width: 100%;
  padding-top: 4px;
}
.chip {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text2);
  border-radius: 6px;
  padding: 6px 12px;
  font: inherit;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.chip:hover { border-color: var(--border-strong); color: var(--text); }
.chip.active {
  background: var(--accent-soft);
  border-color: var(--accent-border);
  color: var(--accent);
}
.chip.group { font-weight: 600; }
.chip.hidden-chip { display: none; }
.btn-ghost {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text2);
  border-radius: 6px;
  padding: 7px 12px;
  font: inherit;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
}
.btn-ghost:hover { background: var(--surface-muted); color: var(--text); }

/* Chart surface (interaction host for Plotly) */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px 12px;
  margin-bottom: 28px;
}
.card-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
  margin-bottom: 4px;
}
.card-desc {
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 12px;
}
.chart { width: 100%; height: 300px; }
.chart.hero { height: 280px; }
.chart.short { height: 240px; }
.empty {
  padding: 48px 20px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  background: var(--surface-muted);
}

/* Table: hairline rules, airy rows (LangSmith experiments table) */
.table-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 20px;
}
table.ls {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.ls th {
  text-align: left;
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  white-space: nowrap;
}
table.ls th.num, table.ls td.num { text-align: right; }
table.ls td {
  padding: 16px 18px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  vertical-align: middle;
}
table.ls tr:last-child td { border-bottom: none; }
table.ls tbody tr:hover { background: var(--surface-muted); }
table.ls td.mono { font-family: var(--mono); font-size: 12px; color: var(--text2); }
.name-cell { font-weight: 600; letter-spacing: -0.01em; }
.sub-cell { font-size: 12px; color: var(--muted); margin-top: 3px; }

.score-cell {
  display: flex;
  align-items: center;
  gap: 12px;
  justify-content: flex-end;
}
.score-val {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  min-width: 3rem;
}
.score-bar {
  width: 88px;
  height: 6px;
  border-radius: 3px;
  background: #f1f5f9;
  overflow: hidden;
}
.score-bar > span {
  display: block;
  height: 100%;
  border-radius: 3px;
  background: var(--green-bar);
}

.latency-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  background: var(--amber-pill);
  color: var(--amber-text);
}
.latency-pill svg { width: 12px; height: 12px; opacity: 0.85; }
.latency-pill.slow {
  background: var(--rose-pill);
  color: var(--rose-text);
}

.stub-note {
  font-size: 13px;
  color: var(--muted);
  padding: 8px 4px 0;
  line-height: 1.6;
}

footer {
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  line-height: 1.7;
}

@media (max-width: 720px) {
  .app { padding: 24px 18px 56px; }
  .header-title { font-size: 22px; }
  .toolbar-right { width: 100%; margin-left: 0; }
  .search { width: 100%; }
  .chart, .chart.hero { height: 240px; }
}
"""

SCRIPT = r"""
const M = __M_JSON__;
const COLORS = __GROUP_COLORS__;
const CLOCK_SVG = '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="5.5" stroke="currentColor" stroke-width="1.4"/><path d="M8 5v3.2l2 1.3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

const cfg = {displayModeBar: false, responsive: true};
const axisFont = {family: 'Plus Jakarta Sans, Segoe UI, sans-serif', size: 11, color: '#94a3b8'};

function layout(extra) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {l: 52, r: 24, t: 16, b: 52},
    font: axisFont,
    xaxis: {gridcolor: '#f1f5f9', zeroline: false, linecolor: '#eceef2'},
    yaxis: {gridcolor: '#f1f5f9', zeroline: false, linecolor: '#eceef2'},
    legend: {orientation: 'h', y: 1.14, x: 0, font: {size: 11, color: '#64748b'}},
  }, extra || {});
}

function colorFor(c) {
  return COLORS[c.model_group] || '#2563eb';
}

function pct(x) {
  if (x === null || x === undefined) return 'n/a';
  return (100 * x).toFixed(1) + '%';
}

function money(x) {
  if (x === null || x === undefined) return 'n/a';
  return '$' + Number(x).toLocaleString(undefined, {maximumFractionDigits: 0});
}

function sec(x) {
  if (x === null || x === undefined) return 'n/a';
  return Number(x).toFixed(2) + 's';
}

let visible = new Set(M.configs.map(c => c.id));
let query = '';

function matchesQuery(c) {
  if (!query) return true;
  const hay = (c.label + ' ' + c.model + ' ' + c.effort_b + ' ' + c.model_group).toLowerCase();
  return hay.includes(query);
}

function visibleConfigs() {
  return M.configs.filter(c => visible.has(c.id) && matchesQuery(c));
}

function updateCount() {
  const el = document.getElementById('filter-count');
  if (el) el.textContent = visibleConfigs().length + ' of ' + M.configs.length + ' visible';
}

function syncChips() {
  document.querySelectorAll('[data-config]').forEach(btn => {
    const c = M.configs.find(x => x.id === btn.dataset.config);
    const show = !c || matchesQuery(c);
    btn.classList.toggle('hidden-chip', !show);
    btn.classList.toggle('active', visible.has(btn.dataset.config));
  });
  document.querySelectorAll('[data-group]').forEach(btn => {
    const ids = (M.model_groups[btn.dataset.group] || {}).ids || [];
    const allOn = ids.length > 0 && ids.every(id => visible.has(id));
    btn.classList.toggle('active', allOn);
  });
  updateCount();
}

function toggleConfig(id) {
  if (visible.has(id)) visible.delete(id);
  else visible.add(id);
  syncChips();
  renderAll();
}

function toggleGroup(group) {
  const ids = (M.model_groups[group] || {}).ids || [];
  const allOn = ids.length > 0 && ids.every(id => visible.has(id));
  if (allOn) ids.forEach(id => visible.delete(id));
  else ids.forEach(id => visible.add(id));
  syncChips();
  renderAll();
}

function showAll() {
  visible = new Set(M.configs.map(c => c.id));
  syncChips();
  renderAll();
}

function clearAll() {
  visible = new Set();
  syncChips();
  renderAll();
}

function emptyChart(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '<div class="empty">' + msg + '</div>';
}

function renderSummaryLine() {
  const runs = visibleConfigs();
  const host = document.getElementById('chart-summary');
  if (!host) return;
  if (!runs.length) {
    emptyChart('chart-summary', 'Turn on a config above to show subclass accuracy.');
    return;
  }
  host.innerHTML = '';
  const xs = runs.map(c => c.label);
  const ys = runs.map(c => c.subclass_acc);
  Plotly.newPlot('chart-summary', [{
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Subclass accuracy',
    x: xs,
    y: ys,
    line: {color: '#4ade80', width: 2.5, shape: 'linear'},
    marker: {
      size: 9,
      color: '#ffffff',
      line: {width: 2.5, color: '#4ade80'},
    },
    hovertemplate: '%{x}<br>subclass %{y:.1%}<extra></extra>',
  }], layout({
    yaxis: {
      title: {text: 'Subclass accuracy', font: {size: 11, color: '#94a3b8'}},
      tickformat: '.0%',
      range: [0.55, 0.95],
      gridcolor: '#f1f5f9',
      zeroline: false,
    },
    xaxis: {
      tickangle: -28,
      gridcolor: 'rgba(0,0,0,0)',
      zeroline: false,
    },
    showlegend: false,
    margin: {l: 56, r: 20, t: 12, b: 72},
  }), cfg);
}

function renderPareto() {
  const runs = visibleConfigs();
  const host = document.getElementById('chart-pareto');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-pareto', 'Turn on a config above to show the Pareto chart.'); return; }
  const plottable = runs.filter(c => Number.isFinite(c.projected_usd) && c.projected_usd > 0);
  if (!plottable.length) {
    emptyChart('chart-pareto', 'No configs with a positive projected cost to plot.');
    return;
  }
  host.innerHTML = '';
  const traces = plottable.map(c => ({
    type: 'scatter',
    mode: 'markers+text',
    name: c.label,
    x: [c.projected_usd],
    y: [c.subclass_acc],
    text: [c.label],
    textposition: 'top center',
    textfont: {size: 10, color: '#64748b'},
    marker: {
      size: 12,
      color: colorFor(c),
      line: {width: 1, color: '#fff'},
    },
    error_y: c.subclass_ci == null ? undefined : {
      type: 'data', array: [c.subclass_ci], visible: true,
      color: '#d8dce3', thickness: 1, width: 3,
    },
    hovertemplate: c.label + '<br>subclass %{y:.1%}<br>cost %{x:$,.0f}<extra></extra>',
  }));
  Plotly.newPlot('chart-pareto', traces, layout({
    xaxis: {
      title: {text: 'Projected production $ (41k)', font: {size: 11, color: '#94a3b8'}},
      type: 'log', gridcolor: '#f1f5f9', zeroline: false,
    },
    yaxis: {
      title: {text: 'Subclass accuracy', font: {size: 11, color: '#94a3b8'}},
      tickformat: '.0%', range: [0.6, 0.95], gridcolor: '#f1f5f9', zeroline: false,
    },
    showlegend: false,
    margin: {l: 56, r: 16, t: 24, b: 52},
  }), cfg);
}

function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-body');
  if (!tbody) return;
  const runs = visibleConfigs().slice().sort((a, b) => b.subclass_acc - a.subclass_acc);
  if (!runs.length) {
    tbody.innerHTML = '<tr><td colspan="7"><div class="empty">No configs visible.</div></td></tr>';
    return;
  }
  tbody.innerHTML = runs.map((c, i) => {
    const bar = Math.max(0, Math.min(1, c.subclass_acc));
    const lat = c.latency_p50;
    const slow = lat != null && lat >= 8;
    const pillCls = slow ? 'latency-pill slow' : 'latency-pill';
    return '<tr>' +
      '<td class="mono">#' + (i + 1) + '</td>' +
      '<td><div class="name-cell">' + c.label + '</div>' +
        '<div class="sub-cell">' + c.model + ' · Pass B ' + c.effort_b + '</div></td>' +
      '<td class="num"><div class="score-cell"><span class="score-val">' + pct(c.subclass_acc) + '</span>' +
        '<div class="score-bar"><span style="width:' + (100 * bar).toFixed(1) + '%"></span></div></div></td>' +
      '<td class="num mono">' + pct(c.ai_native_acc) + '</td>' +
      '<td class="num mono">' + pct(c.rad_acc) + '</td>' +
      '<td class="num mono">' + money(c.projected_usd) + '</td>' +
      '<td class="num"><span class="' + pillCls + '">' + CLOCK_SVG + sec(lat) + '</span></td>' +
      '</tr>';
  }).join('');
}

function renderConfidence() {
  const runs = visibleConfigs();
  const host = document.getElementById('chart-confidence');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-confidence', 'Turn on a config above to show confidence.'); return; }
  host.innerHTML = '';
  Plotly.newPlot('chart-confidence', [{
    type: 'bar',
    x: runs.map(c => c.label),
    y: runs.map(c => c.share_above_90 == null ? null : 100 * c.share_above_90),
    marker: {color: runs.map(c => colorFor(c)), opacity: 0.88},
    hovertemplate: '%{x}<br>≥90% conf: %{y:.1f}%<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'Share of rows ≥ 90% confidence (%)', font: {size: 11, color: '#94a3b8'}}, gridcolor: '#f1f5f9'},
    xaxis: {tickangle: -30, gridcolor: 'rgba(0,0,0,0)'},
    margin: {l: 52, r: 12, t: 12, b: 80},
    showlegend: false,
  }), cfg);
}

function renderLatency() {
  const runs = visibleConfigs();
  const host = document.getElementById('chart-latency');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-latency', 'Turn on a config above to show latency.'); return; }
  host.innerHTML = '';
  Plotly.newPlot('chart-latency', [
    {
      type: 'bar', name: 'p50',
      x: runs.map(c => c.label),
      y: runs.map(c => c.latency_p50),
      marker: {color: '#86efac'},
      hovertemplate: '%{x}<br>p50 %{y:.2f}s<extra></extra>',
    },
    {
      type: 'bar', name: 'p95',
      x: runs.map(c => c.label),
      y: runs.map(c => c.latency_p95),
      marker: {color: '#fda4af'},
      hovertemplate: '%{x}<br>p95 %{y:.2f}s<extra></extra>',
    },
  ], layout({
    barmode: 'group',
    yaxis: {title: {text: 'Seconds', font: {size: 11, color: '#94a3b8'}}, gridcolor: '#f1f5f9'},
    xaxis: {tickangle: -30, gridcolor: 'rgba(0,0,0,0)'},
    margin: {l: 48, r: 12, t: 28, b: 80},
  }), cfg);
}

function renderAll() {
  renderSummaryLine();
  renderPareto();
  renderLeaderboard();
  renderConfidence();
  renderLatency();
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.toggle('active', p.id === 'panel-' + name);
  });
  if (name === 'experiments') {
    setTimeout(() => { renderSummaryLine(); }, 30);
  }
  if (name === 'charts') {
    setTimeout(() => { renderPareto(); renderConfidence(); renderLatency(); }, 30);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-config]').forEach(btn => {
    btn.addEventListener('click', () => toggleConfig(btn.dataset.config));
  });
  document.querySelectorAll('[data-group]').forEach(btn => {
    btn.addEventListener('click', () => toggleGroup(btn.dataset.group));
  });
  const allBtn = document.getElementById('btn-show-all');
  const clearBtn = document.getElementById('btn-clear');
  if (allBtn) allBtn.addEventListener('click', showAll);
  if (clearBtn) clearBtn.addEventListener('click', clearAll);
  const search = document.getElementById('config-search');
  if (search) {
    search.addEventListener('input', () => {
      query = search.value.trim().toLowerCase();
      syncChips();
      renderAll();
    });
  }
  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => showTab(t.dataset.tab));
  });
  syncChips();
  renderAll();
});
"""


def _filter_toolbar_html(metrics: dict) -> str:
    groups = metrics.get("model_group_order") or list(metrics.get("model_groups", {}))
    group_btns = []
    for g in groups:
        label = (metrics["model_groups"].get(g) or {}).get("label", g)
        group_btns.append(
            f'<button type="button" class="chip group active" data-group="{g}">{label}</button>'
        )
    config_btns = []
    for c in metrics["configs"]:
        config_btns.append(
            f'<button type="button" class="chip active" data-config="{c["id"]}">{c["label"]}</button>'
        )
    return f"""
<div class="toolbar" id="config-filter">
  <div class="toolbar-left">
    <span class="toolbar-label">Model family</span>
    {"".join(group_btns)}
    <button type="button" class="btn-ghost" id="btn-show-all">Show all</button>
    <button type="button" class="btn-ghost" id="btn-clear">Clear</button>
  </div>
  <div class="toolbar-right">
    <label class="search" for="config-search">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <circle cx="7" cy="7" r="4.5" stroke="currentColor" stroke-width="1.4"/>
        <path d="M10.5 10.5L14 14" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
      </svg>
      <input id="config-search" type="search" placeholder="Search by name..." autocomplete="off"/>
    </label>
    <span class="toolbar-count" id="filter-count"></span>
  </div>
  <div class="chip-row">{"".join(config_btns)}</div>
</div>
"""


def build_html(metrics: dict) -> str:
    today = datetime.date.today().isoformat()
    synthetic = bool(metrics.get("synthetic"))
    banner = ""
    if synthetic:
        banner = """
<div class="banner" id="synthetic-banner">
  <div><strong>SYNTHETIC data.</strong> Numbers are design placeholders for the
  Stage 8 matrix (nano / mini / luna × Pass B low / medium / high). Replace by
  loading real <code>evals/runs/*/scored.json</code> after the paid sweep.</div>
</div>
"""
    toolbar = _filter_toolbar_html(metrics)
    script = (
        SCRIPT.replace("__M_JSON__", json.dumps(metrics, ensure_ascii=False))
        .replace("__GROUP_COLORS__", json.dumps(GROUP_COLORS))
    )
    source = metrics.get("source") or "fixture"
    n = metrics.get("n_configs", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Eval dashboard · golden-set screen</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{STYLE}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <svg class="header-icon" viewBox="0 0 36 36" fill="none" aria-hidden="true">
      <path d="M6 12.5h9.2l2.2-2.4H30a2 2 0 0 1 2 2V27a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V14.5a2 2 0 0 1 2-2Z"
            stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
    </svg>
    <div class="header-copy">
      <div class="header-title">Golden-set eval screen</div>
      <div class="header-sub">Stage 8 model × Pass B effort matrix · {n} configs · source: {source}</div>
    </div>
  </header>

  <nav class="tabs" aria-label="Dashboard sections">
    <button type="button" class="tab active" data-tab="experiments">Experiments</button>
    <button type="button" class="tab" data-tab="charts">Charts</button>
    <button type="button" class="tab" data-tab="stub">Confusion</button>
  </nav>

  {banner}
  {toolbar}

  <section class="panel active" id="panel-experiments">
    <div class="card">
      <div class="card-title">Subclass_accuracy</div>
      <div class="card-desc">Primary screen metric across visible configs (LangSmith-style summary line).</div>
      <div id="chart-summary" class="chart hero"></div>
    </div>
    <div class="table-wrap">
      <table class="ls" id="leaderboard">
        <thead>
          <tr>
            <th></th>
            <th>Experiment Name</th>
            <th class="num">Subclass_accuracy</th>
            <th class="num">AI-native</th>
            <th class="num">RAD</th>
            <th class="num">Total Cost</th>
            <th class="num">P50 Latency</th>
          </tr>
        </thead>
        <tbody id="leaderboard-body"></tbody>
      </table>
    </div>
    <p class="stub-note">Filter and search update this table and the Charts tab without a reload.
    Confusion / disagreement film-room is deferred until Stage 8 banked runs exist.</p>
  </section>

  <section class="panel" id="panel-charts">
    <div class="card">
      <div class="card-title">Cost × subclass accuracy</div>
      <div class="card-desc">Pareto view of the Stage 8 screen matrix. Log x-axis on projected production cost.</div>
      <div id="chart-pareto" class="chart"></div>
    </div>
    <div class="card">
      <div class="card-title">High-confidence share</div>
      <div class="card-desc">Share of rows with binary confidence ≥ 90%.</div>
      <div id="chart-confidence" class="chart short"></div>
    </div>
    <div class="card">
      <div class="card-title">Latency</div>
      <div class="card-desc">Wall-clock p50 and p95 per config.</div>
      <div id="chart-latency" class="chart short"></div>
    </div>
  </section>

  <section class="panel" id="panel-stub">
    <div class="empty">Confusion matrices and disagreement rows land after Stage 8 scored runs.
    This tab is a placeholder so the tab chrome matches the target UX.</div>
  </section>

  <footer>
    Generated {today} · regenerate with
    <code>python "data visualization/02_Analysis_Code/build_eval_dashboard.py"</code>
    or <code>python -m evals dashboard</code> (mock fixture by default).
    Config filter is required for multi-run charts.
  </footer>
</div>
<script>
{script}
</script>
</body>
</html>
"""


def resolve_metrics(args: argparse.Namespace) -> dict:
    if args.fixture:
        return load_fixture(Path(args.fixture) if args.fixture is not True else None)
    if args.scored:
        return load_from_scored_paths(Path(p) for p in args.scored)
    if args.runs:
        from evals.dashboard_metrics import load_from_run_ids

        return load_from_run_ids(args.runs)
    # Default: mock fixture. Do not auto-load every evals/runs/*/scored.json
    # (banked single-pass baselines would mix architectures and hide the banner).
    # --force-fixture remains accepted for CLI compatibility (same outcome).
    return load_fixture(DEFAULT_FIXTURE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        nargs="?",
        const=True,
        default=None,
        help="Use the synthetic mock fixture (optional path). Same as the default when --runs/--scored are omitted.",
    )
    parser.add_argument(
        "--force-fixture",
        action="store_true",
        help="Explicitly use the mock fixture (same as default when --runs/--scored are omitted).",
    )
    parser.add_argument(
        "--scored",
        nargs="+",
        default=None,
        help="One or more scored.json paths (required to load real runs; no auto-discovery)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Run ids under evals/runs/ (loads each scored.json; required for real runs)",
    )
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    metrics = resolve_metrics(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(metrics), encoding="utf-8")
    mode = "SYNTHETIC" if metrics.get("synthetic") else "scored"
    print(
        f"[{mode}] {metrics['n_configs']} configs → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
