#!/usr/bin/env python3
"""Build the Stage 9 golden-set eval dashboard (LangSmith-light UX).

Thin viewer skeleton: Pareto, leaderboard, confidence, latency, with a
client-side config filter (model groups + per-config pills). Defaults to the
synthetic Stage 8 matrix fixture; pass --runs / --scored for real scored.json.

Writes:
    data visualization/01_Presentation_Materials/eval_dashboard.html

Style is eval-specific (white / light gray, tab underlines, toolbar filters).
It deliberately does NOT reuse the survivorship navy/Cormorant house STYLE.
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
    "nano": "#6366f1",
    "mini": "#0d9488",
    "luna": "#d97706",
}

STYLE = """
:root {
  --bg: #f7f7f8;
  --surface: #ffffff;
  --border: #e5e7eb;
  --border-strong: #d1d5db;
  --text: #111827;
  --text2: #4b5563;
  --muted: #9ca3af;
  --accent: #2563eb;
  --accent-soft: #eff6ff;
  --green: #16a34a;
  --green-soft: #dcfce7;
  --green-bar: #86efac;
  --amber-pill: #fef3c7;
  --amber-text: #92400e;
  --rose-pill: #ffe4e6;
  --rose-text: #9f1239;
  --radius: 8px;
  --sans: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
  min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.app {
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 32px 64px;
}

/* Header */
.header { margin-bottom: 8px; }
.header-title {
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text);
}
.header-sub {
  margin-top: 4px;
  font-size: 13px;
  color: var(--text2);
}

/* Tabs */
.tabs {
  display: flex;
  gap: 20px;
  border-bottom: 1px solid var(--border);
  margin: 20px 0 0;
}
.tab {
  appearance: none;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  padding: 10px 2px 12px;
  font: inherit;
  font-size: 13px;
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
.panel { display: none; padding-top: 20px; }
.panel.active { display: block; }

/* Banner */
.banner {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 12px 14px;
  margin-bottom: 16px;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: var(--radius);
  color: #92400e;
  font-size: 13px;
}
.banner strong { font-weight: 600; }
.banner.hidden { display: none; }

/* Toolbar / filter */
.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 12px;
  padding: 12px 0 16px;
  margin-bottom: 8px;
}
.toolbar-label {
  font-size: 12px;
  color: var(--muted);
  font-weight: 500;
  margin-right: 4px;
}
.toolbar-count {
  margin-left: auto;
  font-size: 12px;
  color: var(--muted);
}
.pill-row { display: flex; flex-wrap: wrap; gap: 6px; width: 100%; }
.pill {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text2);
  border-radius: 999px;
  padding: 5px 11px;
  font: inherit;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.pill:hover { border-color: var(--border-strong); color: var(--text); }
.pill.active {
  background: var(--accent-soft);
  border-color: #bfdbfe;
  color: var(--accent);
}
.pill.group { font-weight: 600; }
.btn-ghost {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text2);
  border-radius: 6px;
  padding: 5px 10px;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.btn-ghost:hover { background: #f3f4f6; color: var(--text); }

/* Cards / charts */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px 8px;
  margin-bottom: 16px;
}
.card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 2px;
}
.card-desc {
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 8px;
}
.chart { width: 100%; height: 320px; }
.chart.short { height: 260px; }
.empty {
  padding: 36px 16px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

/* Table */
.table-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 16px;
}
table.ls {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.ls th {
  text-align: left;
  font-size: 11px;
  font-weight: 600;
  text-transform: none;
  letter-spacing: 0;
  color: var(--muted);
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  background: #fafafa;
  white-space: nowrap;
}
table.ls th.num, table.ls td.num { text-align: right; }
table.ls td {
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  vertical-align: middle;
}
table.ls tr:last-child td { border-bottom: none; }
table.ls tbody tr:hover { background: #fafafa; }
table.ls td.mono { font-family: var(--mono); font-size: 12px; }
.name-cell { font-weight: 500; }
.sub-cell { font-size: 11px; color: var(--muted); margin-top: 2px; }

.score-cell { display: flex; align-items: center; gap: 10px; justify-content: flex-end; }
.score-val { font-variant-numeric: tabular-nums; font-weight: 500; min-width: 2.6rem; }
.score-bar {
  width: 72px; height: 6px; border-radius: 999px;
  background: #f3f4f6; overflow: hidden;
}
.score-bar > span {
  display: block; height: 100%; border-radius: 999px;
  background: var(--green-bar);
}

.latency-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  font-weight: 500;
  background: var(--amber-pill);
  color: var(--amber-text);
}
.latency-pill.slow {
  background: var(--rose-pill);
  color: var(--rose-text);
}

.partial-badge {
  display: inline-flex;
  align-items: center;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.02em;
  background: var(--amber-pill);
  color: var(--amber-text);
  vertical-align: middle;
}

.stub-note {
  font-size: 12px;
  color: var(--muted);
  padding: 24px 8px;
}

footer {
  margin-top: 28px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  line-height: 1.7;
}

@media (max-width: 720px) {
  .app { padding: 16px; }
  .toolbar-count { width: 100%; margin-left: 0; }
  .chart { height: 260px; }
}
"""

SCRIPT = r"""
const M = __M_JSON__;
const COLORS = __GROUP_COLORS__;

const cfg = {displayModeBar: false, responsive: true};
const axisFont = {family: 'Inter, ui-sans-serif, system-ui, sans-serif', size: 11, color: '#6b7280'};

function layout(extra) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {l: 48, r: 20, t: 12, b: 48},
    font: axisFont,
    xaxis: {gridcolor: '#f3f4f6', zeroline: false, linecolor: '#e5e7eb'},
    yaxis: {gridcolor: '#f3f4f6', zeroline: false, linecolor: '#e5e7eb'},
    legend: {orientation: 'h', y: 1.12, x: 0, font: {size: 11, color: '#6b7280'}},
  }, extra || {});
}

function colorFor(c) {
  return COLORS[c.model_group] || '#6366f1';
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

function visibleConfigs() {
  return M.configs.filter(c => visible.has(c.id));
}

function updateCount() {
  // Count visible *rows*, not Set size: ids are unique per run_id, but
  // row count is the source of truth if a bad payload ever reuses an id.
  const el = document.getElementById('filter-count');
  if (el) el.textContent = visibleConfigs().length + ' of ' + M.configs.length + ' visible';
}

function syncPills() {
  document.querySelectorAll('[data-config]').forEach(btn => {
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
  syncPills();
  renderAll();
}

function toggleGroup(group) {
  const ids = (M.model_groups[group] || {}).ids || [];
  const allOn = ids.length > 0 && ids.every(id => visible.has(id));
  if (allOn) ids.forEach(id => visible.delete(id));
  else ids.forEach(id => visible.add(id));
  syncPills();
  renderAll();
}

function showAll() {
  visible = new Set(M.configs.map(c => c.id));
  syncPills();
  renderAll();
}

function clearAll() {
  visible = new Set();
  syncPills();
  renderAll();
}

function emptyChart(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '<div class="empty">' + msg + '</div>';
}

function renderPareto() {
  const runs = visibleConfigs();
  const host = document.getElementById('chart-pareto');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-pareto', 'Turn on a config above to show the Pareto chart.'); return; }
  // Log x-axis cannot place null/NaN/<=0 cost (legacy scored.json). Omit
  // those points; leaderboard still shows money() "n/a" for the same configs.
  const plottable = runs.filter(c => Number.isFinite(c.projected_usd) && c.projected_usd > 0);
  if (!plottable.length) {
    emptyChart('chart-pareto', 'No configs with a positive projected cost to plot.');
    return;
  }
  host.innerHTML = '';
  // Fit y to visible points (± CI), not the mock fixture's 60–95% band.
  // Banked / single-pass real runs can land near ~41% subclass accuracy.
  let yLo = Infinity;
  let yHi = -Infinity;
  for (const c of plottable) {
    const ci = c.subclass_ci == null ? 0 : c.subclass_ci;
    yLo = Math.min(yLo, c.subclass_acc - ci);
    yHi = Math.max(yHi, c.subclass_acc + ci);
  }
  const pad = Math.max(0.03, (yHi - yLo) * 0.12);
  const yRange = [Math.max(0, yLo - pad), Math.min(1, yHi + pad)];
  const traces = plottable.map(c => {
    const partial = isPartial(c);
    const nBit = sampleCaption(c);
    const hoverExtra = (partial ? '<br>partial screen' : '') +
      (nBit ? '<br>' + nBit : '');
    return {
      type: 'scatter',
      mode: 'markers+text',
      name: c.label,
      x: [c.projected_usd],
      y: [c.subclass_acc],
      text: [partial ? c.label + ' · partial' : c.label],
      textposition: 'top center',
      textfont: {size: 10, color: partial ? '#b45309' : '#6b7280'},
      marker: {
        size: 12,
        color: colorFor(c),
        symbol: partial ? 'circle-open' : 'circle',
        line: {width: partial ? 2 : 1, color: partial ? '#b45309' : '#fff'},
      },
      error_y: c.subclass_ci == null ? undefined : {
        type: 'data', array: [c.subclass_ci], visible: true,
        color: '#d1d5db', thickness: 1, width: 3,
      },
      hovertemplate: c.label + '<br>subclass %{y:.1%}<br>cost %{x:$,.0f}' +
        hoverExtra + '<extra></extra>',
    };
  });
  Plotly.newPlot('chart-pareto', traces, layout({
    xaxis: {
      title: {text: 'Projected production $ (41k)', font: {size: 11, color: '#9ca3af'}},
      type: 'log', gridcolor: '#f3f4f6', zeroline: false,
    },
    yaxis: {
      title: {text: 'Subclass accuracy', font: {size: 11, color: '#9ca3af'}},
      tickformat: '.0%', range: yRange, gridcolor: '#f3f4f6', zeroline: false,
    },
    showlegend: false,
    margin: {l: 56, r: 16, t: 24, b: 52},
  }), cfg);
}

function effortCaption(c) {
  // Two-pass Stage 8 uses Pass B effort; banked single-pass refs use reasoning effort.
  // Do not hard-code "Pass B" when kind is single_pass or effort is none.
  // Missing effort must read as unknown, never as a fabricated medium.
  const e = (c.effort_b == null || c.effort_b === '' || c.effort_b === 'unknown')
    ? 'unknown'
    : String(c.effort_b);
  if (e === 'unknown') return 'effort unknown';
  if (c.kind === 'two_pass') return 'Pass B ' + e;
  if (c.kind === 'single_pass' || e === 'none') return 'effort ' + e;
  return 'effort ' + e;
}

function isPartial(c) {
  if (c.is_partial === true) return true;
  return c.n_scored != null && c.n_expected != null && c.n_scored < c.n_expected;
}

function sampleCaption(c) {
  if (c.n_scored == null && c.n_expected == null) return '';
  if (c.n_expected != null) return String(c.n_scored ?? '?') + '/' + c.n_expected + ' scored';
  return String(c.n_scored) + ' scored';
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
    const partial = isPartial(c);
    const badge = partial ? '<span class="partial-badge">partial</span>' : '';
    const nLine = sampleCaption(c);
    const sub = [c.model + ' · ' + effortCaption(c), nLine].filter(Boolean).join(' · ');
    return '<tr>' +
      '<td class="mono">#' + (i + 1) + '</td>' +
      '<td><div class="name-cell">' + c.label + badge + '</div>' +
        '<div class="sub-cell">' + sub + '</div></td>' +
      '<td class="num"><div class="score-cell"><span class="score-val">' + pct(c.subclass_acc) + '</span>' +
        '<div class="score-bar"><span style="width:' + (100 * bar).toFixed(1) + '%"></span></div></div></td>' +
      '<td class="num mono">' + pct(c.ai_native_acc) + '</td>' +
      '<td class="num mono">' + pct(c.rad_acc) + '</td>' +
      '<td class="num mono">' + money(c.projected_usd) + '</td>' +
      '<td class="num"><span class="' + pillCls + '">' + sec(lat) + '</span></td>' +
      '</tr>';
  }).join('');
}

function renderConfidence() {
  const runs = visibleConfigs().filter(c => !c.is_aggregate);
  const host = document.getElementById('chart-confidence');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-confidence', 'Turn on a config above to show calibration.'); return; }
  host.innerHTML = '';
  const eceTrace = {
    type: 'bar',
    name: 'ECE (lower better)',
    x: runs.map(c => c.label),
    y: runs.map(c => c.ece == null ? null : 100 * c.ece),
    marker: {color: runs.map(c => colorFor(c)), opacity: 0.85},
    hovertemplate: '%{x}<br>ECE %{y:.2f}%<extra></extra>',
  };
  const selTrace = {
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Selective @50% cov',
    x: runs.map(c => c.label),
    y: runs.map(c => c.selective_acc_50 == null ? null : 100 * c.selective_acc_50),
    yaxis: 'y2',
    line: {color: '#64748b', width: 2},
    marker: {size: 7, color: '#334155'},
    hovertemplate: '%{x}<br>selective@50 %{y:.1f}%<extra></extra>',
  };
  Plotly.newPlot('chart-confidence', [eceTrace, selTrace], layout({
    yaxis: {title: {text: 'ECE (%)', font: {size: 11, color: '#9ca3af'}}, gridcolor: '#f3f4f6'},
    yaxis2: {
      title: {text: 'Selective accuracy @50% coverage (%)', font: {size: 11, color: '#9ca3af'}},
      overlaying: 'y',
      side: 'right',
      showgrid: false,
      range: [50, 105],
    },
    xaxis: {tickangle: -30, gridcolor: 'rgba(0,0,0,0)'},
    margin: {l: 52, r: 56, t: 28, b: 80},
    legend: {orientation: 'h', y: 1.15},
  }), cfg);
}

function renderReliability() {
  const runs = visibleConfigs().filter(c => !c.is_aggregate && c.reliability_bins && c.reliability_bins.length);
  const host = document.getElementById('chart-reliability');
  if (!host) return;
  if (!runs.length) {
    emptyChart('chart-reliability', 'Reliability bins appear when scored.json carries calibration.reliability.bins.');
    return;
  }
  host.innerHTML = '';
  // Show the first visible config with bins (one diagram; filter to compare).
  const c = runs[0];
  const bins = c.reliability_bins.filter(b => b.count > 0 && b.mean_confidence != null);
  Plotly.newPlot('chart-reliability', [
    {
      type: 'scatter',
      mode: 'lines',
      name: 'perfect',
      x: [0, 1],
      y: [0, 1],
      line: {dash: 'dash', color: '#d1d5db', width: 1},
      hoverinfo: 'skip',
    },
    {
      type: 'scatter',
      mode: 'markers+lines',
      name: c.label,
      x: bins.map(b => b.mean_confidence),
      y: bins.map(b => b.accuracy),
      marker: {size: bins.map(b => 6 + Math.sqrt(b.count)), color: colorFor(c)},
      line: {color: colorFor(c), width: 1.5},
      text: bins.map(b => 'n=' + b.count),
      hovertemplate: 'conf %{x:.2f}<br>acc %{y:.2f}<br>%{text}<extra></extra>',
    },
  ], layout({
    xaxis: {title: {text: 'Mean confidence', font: {size: 11, color: '#9ca3af'}}, range: [0, 1], gridcolor: '#f3f4f6'},
    yaxis: {title: {text: 'Accuracy', font: {size: 11, color: '#9ca3af'}}, range: [0, 1], gridcolor: '#f3f4f6'},
    margin: {l: 52, r: 12, t: 12, b: 48},
    showlegend: true,
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
    yaxis: {title: {text: 'Seconds', font: {size: 11, color: '#9ca3af'}}, gridcolor: '#f3f4f6'},
    xaxis: {tickangle: -30, gridcolor: 'rgba(0,0,0,0)'},
    margin: {l: 48, r: 12, t: 28, b: 80},
  }), cfg);
}

function renderAll() {
  renderPareto();
  renderLeaderboard();
  renderConfidence();
  renderReliability();
  renderLatency();
  renderBaseline();
}

function renderBaseline() {
  const host = document.getElementById('baseline-table');
  if (!host) return;
  const runs = visibleConfigs().filter(c => c.vs_baseline);
  if (!runs.length) {
    host.innerHTML = '<p class="stub-note">No vs_baseline blocks yet. Score with <code>--baseline &lt;run_id&gt;</code> for paired bootstrap deltas.</p>';
    return;
  }
  host.innerHTML = '<table class="ls"><thead><tr><th>Config</th><th>Baseline</th><th class="num">Δ subclass</th><th class="num">95% CI</th><th>Sig?</th></tr></thead><tbody>' +
    runs.map(c => {
      const v = c.vs_baseline;
      const d = v.delta_accuracy;
      const ci = v.ci95 || [];
      return '<tr><td>' + c.label + '</td><td class="mono">' + (v.baseline_run_id || '') +
        '</td><td class="num mono">' + (d == null ? '—' : ((d >= 0 ? '+' : '') + (100 * d).toFixed(1) + '%')) +
        '</td><td class="num mono">' + (ci.length === 2 ? ((100 * ci[0]).toFixed(1) + ' … ' + (100 * ci[1]).toFixed(1) + '%') : '—') +
        '</td><td>' + (v.significant ? 'yes' : 'no') + '</td></tr>';
    }).join('') + '</tbody></table>';
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.toggle('active', p.id === 'panel-' + name);
  });
  // Re-layout Plotly when a chart panel becomes visible.
  if (name === 'charts') {
    setTimeout(() => { renderPareto(); renderConfidence(); renderReliability(); renderLatency(); renderBaseline(); }, 30);
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
  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => showTab(t.dataset.tab));
  });
  syncPills();
  renderAll();
});
"""


def _filter_toolbar_html(metrics: dict) -> str:
    groups = metrics.get("model_group_order") or list(metrics.get("model_groups", {}))
    group_btns = []
    for g in groups:
        label = (metrics["model_groups"].get(g) or {}).get("label", g)
        group_btns.append(
            f'<button type="button" class="pill group active" data-group="{g}">{label}</button>'
        )
    config_btns = []
    for c in metrics["configs"]:
        config_btns.append(
            f'<button type="button" class="pill active" data-config="{c["id"]}">{c["label"]}</button>'
        )
    return f"""
<div class="toolbar" id="config-filter">
  <span class="toolbar-label">Show configs</span>
  {"".join(group_btns)}
  <button type="button" class="btn-ghost" id="btn-show-all">Show all</button>
  <button type="button" class="btn-ghost" id="btn-clear">Clear</button>
  <span class="toolbar-count" id="filter-count"></span>
  <div class="pill-row">{"".join(config_btns)}</div>
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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{STYLE}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <div class="header-title">Golden-set eval screen</div>
    <div class="header-sub">Stage 9 viewer · {n} configs · source: {source}</div>
  </header>

  <nav class="tabs" aria-label="Dashboard sections">
    <button type="button" class="tab active" data-tab="experiments">Experiments</button>
    <button type="button" class="tab" data-tab="charts">Charts</button>
    <button type="button" class="tab" data-tab="stub">Confusion</button>
  </nav>

  {banner}
  {toolbar}

  <section class="panel active" id="panel-experiments">
    <div class="table-wrap">
      <table class="ls" id="leaderboard">
        <thead>
          <tr>
            <th></th>
            <th>Config</th>
            <th class="num">Subclass acc</th>
            <th class="num">AI-native</th>
            <th class="num">RAD</th>
            <th class="num">Proj. $ (41k)</th>
            <th class="num">P50 latency</th>
          </tr>
        </thead>
        <tbody id="leaderboard-body"></tbody>
      </table>
    </div>
    <p class="stub-note">Filter updates this table and the Charts tab without a reload.
    Confusion / disagreement film-room is deferred until Stage 8 banked runs exist.</p>
  </section>

  <section class="panel" id="panel-charts">
    <div class="card">
      <div class="card-title">Cost × subclass accuracy</div>
      <div class="card-desc">Pareto view of the Stage 8 screen matrix. Log x-axis on projected production cost.</div>
      <div id="chart-pareto" class="chart"></div>
    </div>
    <div class="card">
      <div class="card-title">Calibration (ECE + selective)</div>
      <div class="card-desc">Expected Calibration Error (lower is better) and accuracy when answering only on the top 50% most confident rows. Share ≥90% is secondary.</div>
      <div id="chart-confidence" class="chart short"></div>
    </div>
    <div class="card">
      <div class="card-title">Reliability curve</div>
      <div class="card-desc">Mean confidence vs accuracy per bin for the first visible config that carries bins. Filter to one config to inspect it.</div>
      <div id="chart-reliability" class="chart short"></div>
    </div>
    <div class="card">
      <div class="card-title">Paired vs baseline</div>
      <div class="card-desc">Paired-bootstrap subclass deltas when scored.json includes vs_baseline (score --baseline).</div>
      <div id="baseline-table"></div>
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
