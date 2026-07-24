#!/usr/bin/env python3
"""Build the Classifier Eval Suite (three-tab eval dashboard).

Tabs, one per evaluation question:

1. Pipeline robustness: do logprob extraction and Batch API parity behave
   as intended, will this classifier survive production?
2. Model benchmarks: which GPT-family model should production use?
3. Confidence correctness correlation: how correlated is logprob confidence
   with actual correctness?

Defaults to the synthetic locked-matrix fixture; pass --runs / --scored for
real scored.json files. Robustness checks render "pending" for anything a
run has not recorded; nothing is fabricated at render time.

Writes a single self-contained HTML file (Plotly inlined, no CDN). Open in any
browser offline, or email as one attachment.

Writes:
    data visualization/01_Presentation_Materials/eval_dashboard.html
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import sys
from functools import lru_cache
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
from evals.instances import (  # noqa: E402
    archive_instance,
    format_run_headline,
    run_meta_bits,
)
from evals.paths import EVAL_DASHBOARD_HTML  # noqa: E402

OUTPUT_PATH = EVAL_DASHBOARD_HTML
PLOTLY_VENDOR = _HERE / "vendor" / "plotly-2.35.2.min.js"
# build_html stays small for tests; write_dashboard swaps this for the real library.
PLOTLY_PLACEHOLDER = "/*__PLOTLY_INLINE__*/"

# Muted categorical palette for model-group chart series (lifted for dark bg).
GROUP_COLORS = {
    "nano": "#5b8fc4",
    "mini": "#9a78a8",
    "luna": "#c9944f",
}

STYLE = """
:root {
  --bg: #0a0a0a;
  --surface: #111111;
  --surface-muted: #161616;
  --border: #2a2a2a;
  --border-strong: #333333;
  --text: #e8e8ea;
  --text2: #a8abb0;
  --muted: #7a7e85;
  --accent: #5b8fc4;
  --accent-bg: #152033;
  --pass: #3fb950;
  --pass-bg: #0f1f14;
  --fail: #f85149;
  --fail-bg: #2a1210;
  --pending: #8b9096;
  --pending-bg: #1a1a1a;
  --sans: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
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
  border: 1px solid var(--border);
  padding: 0.05em 0.35em;
}

/* App bar */
.appbar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 24px;
  border-bottom: 1px solid var(--border);
  padding: 18px 36px;
}
.brand {
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--text);
}
.brand small {
  margin-left: 12px;
  font-size: 12px;
  font-weight: 400;
  color: var(--muted);
}
.appbar-meta {
  font-size: 12px;
  font-family: var(--mono);
  color: var(--muted);
  white-space: nowrap;
}

/* Tab bar */
.tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  padding: 0 36px;
}
.tab {
  appearance: none;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  padding: 13px 14px;
  font: inherit;
  font-size: 13.5px;
  font-weight: 500;
  color: var(--text2);
  cursor: pointer;
}
.tab:hover { color: var(--text); background: var(--surface-muted); }
.tab.active {
  color: var(--text);
  border-bottom-color: var(--accent);
}

/* Content */
.content {
  max-width: 1160px;
  margin: 0 auto;
  padding: 28px 36px 72px;
}
.panel { display: none; }
.panel.active { display: block; }
.tab-lead {
  margin-bottom: 24px;
  max-width: 860px;
}
.tab-lead h2 {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.015em;
  margin-bottom: 4px;
}
.tab-lead p { font-size: 13.5px; color: var(--text2); }

/* Data notice */
.notice {
  display: flex;
  align-items: baseline;
  gap: 12px;
  border: 1px solid var(--border);
  border-left: 2px solid var(--border-strong);
  background: var(--surface-muted);
  padding: 10px 14px;
  margin: 16px 36px 0;
  font-size: 12.5px;
  color: var(--text2);
  line-height: 1.5;
}
.notice .tag {
  flex: 0 0 auto;
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--pending);
  border: 1px solid var(--border-strong);
  padding: 2px 7px;
  background: var(--surface);
}
.notice.scored .tag { color: var(--accent); }
.notice .run-headline {
  color: var(--text);
  font-size: 13px;
  font-weight: 500;
}
.notice .run-meta {
  display: block;
  margin-top: 3px;
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--muted);
}

/* Robustness checks panel */
.check {
  border: 1px solid var(--border);
  margin-bottom: 16px;
  background: var(--surface);
}
.check-head {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
}
.check-head h3 {
  font-size: 14.5px;
  font-weight: 600;
  letter-spacing: -0.01em;
}
.badge {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  padding: 3px 9px;
  border: 1px solid;
}
.badge.pass { color: var(--pass); border-color: var(--pass); background: var(--pass-bg); }
.badge.fail { color: var(--fail); border-color: var(--fail); background: var(--fail-bg); }
.badge.pending { color: var(--pending); border-color: var(--border-strong); background: var(--pending-bg); }
.check-body { padding: 14px 18px 16px; }
.check-meaning {
  font-size: 13px;
  color: var(--text2);
  max-width: 820px;
  margin-bottom: 14px;
}
.check-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  border: 1px solid var(--border);
  margin-bottom: 14px;
  width: fit-content;
}
.stat {
  padding: 8px 18px;
  border-right: 1px solid var(--border);
  min-width: 130px;
}
.stat:last-child { border-right: none; }
.stat-label {
  display: block;
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 1px;
}
.stat-value {
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
}
.pending-note {
  font-size: 12.5px;
  color: var(--muted);
  border: 1px dashed var(--border-strong);
  padding: 10px 14px;
  max-width: 720px;
}
.check-footnote {
  font-size: 12px;
  color: var(--muted);
  max-width: 720px;
  margin-top: 12px;
}
table.mini {
  border-collapse: collapse;
  font-size: 12.5px;
  min-width: 420px;
}
table.mini th, table.mini td {
  text-align: left;
  padding: 6px 16px 6px 0;
  border-bottom: 1px solid var(--border);
}
table.mini th {
  font-size: 11px;
  font-weight: 500;
  color: var(--muted);
}
table.mini tr:last-child td { border-bottom: none; }
table.mini td.mono { font-family: var(--mono); font-size: 12px; }
.mini-status { font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
.mini-status.pass { color: var(--pass); }
.mini-status.fail { color: var(--fail); }
.mini-status.pending { color: var(--pending); }

/* Toolbar */
.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 14px;
  padding-bottom: 18px;
}
.toolbar-left, .toolbar-right {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.toolbar-right { margin-left: auto; }
.toolbar-label {
  font-size: 11px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  margin-right: 2px;
}
.toolbar-count {
  font-size: 12px;
  font-family: var(--mono);
  color: var(--muted);
  white-space: nowrap;
}
.search {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 210px;
  padding: 6px 10px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
}
.search:focus-within { border-color: var(--accent); }
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
  gap: 6px;
  width: 100%;
  padding-top: 2px;
}
.chip {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text2);
  padding: 5px 11px;
  font: inherit;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
}
.chip:hover { border-color: var(--border-strong); color: var(--text); }
.chip.active {
  background: var(--accent-bg);
  border-color: var(--accent);
  color: var(--accent);
}
.chip.group { font-weight: 600; }
.chip.hidden-chip { display: none; }
.btn-ghost {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text2);
  padding: 5px 11px;
  font: inherit;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
}
.btn-ghost:hover { background: var(--surface-muted); color: var(--text); }

/* Chart cards */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 18px 20px 10px;
  margin-bottom: 20px;
}
.card-title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
  margin-bottom: 3px;
}
.card-desc {
  font-size: 12.5px;
  color: var(--muted);
  margin-bottom: 10px;
  max-width: 860px;
}
.chart { width: 100%; height: 300px; }
.chart.short { height: 250px; }
.empty {
  padding: 44px 20px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
  border: 1px dashed var(--border);
  background: var(--surface-muted);
}

/* Leaderboard table */
.table-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  overflow-x: auto;
  margin-bottom: 20px;
}
table.grid {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.grid th {
  text-align: left;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-strong);
  background: var(--surface-muted);
  white-space: nowrap;
}
table.grid th.num, table.grid td.num { text-align: right; }
table.grid td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  vertical-align: middle;
}
table.grid tr:last-child td { border-bottom: none; }
table.grid tbody tr:hover { background: var(--surface-muted); }
table.grid td.mono { font-family: var(--mono); font-size: 12.5px; }
.name-cell { font-weight: 600; letter-spacing: -0.01em; }
.sub-cell { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
.score-cell {
  display: flex;
  align-items: center;
  gap: 10px;
  justify-content: flex-end;
}
.score-val {
  font-family: var(--mono);
  font-size: 12.5px;
  font-weight: 500;
  min-width: 3.2rem;
  text-align: right;
}
.score-bar {
  width: 84px;
  height: 5px;
  background: var(--surface-muted);
  border: 1px solid var(--border);
}
.score-bar > span {
  display: block;
  height: 100%;
  background: var(--accent);
}
.partial-badge {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  font-family: var(--mono);
  font-size: 9.5px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--fail);
  border: 1px solid var(--fail);
  background: var(--fail-bg);
  vertical-align: middle;
}

/* Cost-breakdown popover (info icon beside each projected-cost value) */
.cost-cell {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.cost-info {
  appearance: none;
  border: none;
  background: none;
  padding: 0;
  display: inline-flex;
  align-items: center;
  cursor: pointer;
  color: var(--muted);
  line-height: 0;
}
.cost-info:hover, .cost-info.open { color: var(--accent); }
.cost-info svg { width: 13px; height: 13px; }
.cost-popover {
  position: absolute;
  z-index: 40;
  width: 400px;
  max-width: calc(100vw - 32px);
  background: var(--surface);
  border: 1px solid var(--border-strong);
  padding: 14px 16px 12px;
  font-size: 12px;
  color: var(--text2);
  text-align: left;
}
.cost-popover h4 {
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 2px;
}
.cost-popover .pop-sub { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
.cost-popover .pop-section {
  margin: 8px 0 2px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.cost-popover table { width: 100%; border-collapse: collapse; }
.cost-popover td {
  padding: 3px 0;
  vertical-align: top;
  border: none;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text2);
}
.cost-popover td.val {
  color: var(--text);
  text-align: right;
  white-space: nowrap;
  padding-left: 10px;
}
.cost-popover tr.total td { font-weight: 600; color: var(--text); }
.cost-popover .pop-note {
  margin-top: 8px;
  font-size: 10.5px;
  color: var(--muted);
  line-height: 1.5;
}
.cost-popover .nr { color: var(--muted); font-style: italic; }

footer {
  margin-top: 44px;
  padding-top: 18px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  line-height: 1.7;
}

@media (max-width: 720px) {
  .appbar, .tabs { padding-left: 18px; padding-right: 18px; }
  .notice { margin: 14px 18px 0; }
  .content { padding: 20px 18px 56px; }
  .toolbar-right { width: 100%; margin-left: 0; }
  .search { width: 100%; }
  .chart, .chart.short { height: 240px; }
}
"""

SCRIPT = r"""
const M = __M_JSON__;
const COLORS = __GROUP_COLORS__;
const INFO_SVG = '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="6.2" stroke="currentColor" stroke-width="1.3"/><circle cx="8" cy="5.1" r="0.9" fill="currentColor"/><path d="M8 7.4v3.8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';

const cfg = {displayModeBar: false, responsive: true};
const axisFont = {family: 'Segoe UI, -apple-system, sans-serif', size: 11, color: '#7a7e85'};
const numFont = {family: 'ui-monospace, SF Mono, Menlo, Consolas, monospace', size: 10.5, color: '#7a7e85'};

function layout(extra) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {l: 54, r: 24, t: 16, b: 52},
    font: axisFont,
    xaxis: {gridcolor: '#222222', zeroline: false, linecolor: '#2a2a2a', tickfont: numFont},
    yaxis: {gridcolor: '#222222', zeroline: false, linecolor: '#2a2a2a', tickfont: numFont},
    legend: {orientation: 'h', y: 1.14, x: 0, font: {size: 11, color: '#a8abb0'}},
    // Plotly defaults the hover box to the trace color, which is unreadable
    // on the dark theme; pin it to a dark panel with light text everywhere.
    hoverlabel: {
      bgcolor: '#1c1f26',
      bordercolor: '#3a3f4b',
      font: {family: 'Segoe UI, -apple-system, sans-serif', size: 12, color: '#e8e8ea'},
    },
  }, extra || {});
}

function colorFor(c) {
  return COLORS[c.model_group] || '#5b8fc4';
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

// --- Cost breakdown popover ------------------------------------------------
// Renders only recorded fields; anything missing shows "not recorded".
const NOT_RECORDED = '<span class="nr">not recorded</span>';

function tok(x) {
  if (x === null || x === undefined) return NOT_RECORDED;
  return Number(x).toLocaleString();
}

function usd(x, digits) {
  if (x === null || x === undefined) return NOT_RECORDED;
  return '$' + Number(x).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
}

function rate(x) {
  if (x === null || x === undefined) return NOT_RECORDED;
  return (100 * x).toFixed(1) + '%';
}

function passRow(label, p) {
  return '<tr><td>' + label + ' in ' + tok(p.input) + ' (cached ' + tok(p.cached) + ')</td>' +
    '<td class="val">out ' + tok(p.output) + ' (reason ' + tok(p.reasoning) + ')</td></tr>';
}

function costBreakdownHtml(c) {
  const b = c.cost_breakdown;
  const p = b.pricing_per_mtok || null;
  const priceLine = p
    ? 'input $' + p.input + ' / output $' + p.output + ' per 1M tokens'
    : 'per-1M prices not recorded';
  let html = '<h4>' + c.label + ' &middot; cost breakdown</h4>' +
    '<div class="pop-sub">' + (b.model || 'model not recorded') + ' &middot; ' + priceLine + '</div>';

  html += '<div class="pop-section">Measured tokens (' +
    (b.n_golden != null ? b.n_golden + '-row golden run' : 'golden run, n not recorded') + ')</div>';
  html += '<table>';
  if (b.per_pass && b.per_pass.pass_a && b.per_pass.pass_b) {
    html += passRow('Pass A', b.per_pass.pass_a);
    html += passRow('Pass B', b.per_pass.pass_b);
  } else {
    html += '<tr><td colspan="2">per-pass split ' + NOT_RECORDED + '</td></tr>';
  }
  const hitBit = b.cache_hit_rate != null ? ' = ' + rate(b.cache_hit_rate) + ' hit' : '';
  html += '<tr><td>Total in ' + tok(b.total_input_tokens) +
    ' (cached ' + tok(b.total_cached_tokens) + hitBit + ')</td>' +
    '<td class="val">out ' + tok(b.total_output_tokens) + '</td></tr>';
  html += '</table>';

  html += '<div class="pop-section">Cost ladder</div><table>';
  const canSync = p && b.total_input_tokens != null && b.total_output_tokens != null;
  html += '<tr><td>1 &middot; Sync list: ' + (canSync
    ? tok(b.total_input_tokens) + ' &times; $' + p.input + '/1M + ' +
      tok(b.total_output_tokens) + ' &times; $' + p.output + '/1M'
    : 'tokens or prices not recorded') +
    '</td><td class="val">' + usd(b.golden_sync_usd, 4) + '</td></tr>';

  if (b.golden_after_cache_usd != null && p && b.total_cached_tokens != null && b.cache_discount != null) {
    const uncached = b.total_input_tokens - b.total_cached_tokens;
    const cachedPrice = p.input * b.cache_discount;
    html += '<tr><td>2 &middot; Cache: ' + tok(uncached) + ' &times; $' + p.input + '/1M + ' +
      tok(b.total_cached_tokens) + ' &times; $' + cachedPrice + '/1M + output' +
      '</td><td class="val">' + usd(b.golden_after_cache_usd, 4) + '</td></tr>';
  } else {
    html += '<tr><td>2 &middot; Cache adjustment</td><td class="val">' + NOT_RECORDED + '</td></tr>';
  }

  const scaleBit = (b.n_prod != null && b.n_golden != null)
    ? '&times; (' + tok(b.n_prod) + ' / ' + tok(b.n_golden) + ')'
    : 'scale ' + (b.scale_factor != null ? '&times; ' + b.scale_factor : NOT_RECORDED);
  html += '<tr class="total"><td>3 &middot; Scale to production ' + scaleBit +
    '</td><td class="val">' + usd(b.estimated_production_usd, 2) + '</td></tr>';
  html += '<tr><td>&nbsp;&nbsp;&nbsp; per company</td><td class="val">' +
    usd(b.estimated_usd_per_company, 4) + '</td></tr>';
  html += '</table>';

  const notes = [];
  if (b.n_prod_label) notes.push('N = ' + tok(b.n_prod) + ' (' + b.n_prod_label + ').');
  notes.push('Sync Responses API pricing; production runs sync, so no Batch API discount is assumed.');
  notes.push('Reasoning tokens are billed inside output.');
  if (b.cache_source) notes.push('Cache rate: ' + b.cache_source.replaceAll('_', ' ') + '.');
  if (b.cache_step_reason) notes.push('Cache step unavailable: ' + b.cache_step_reason);
  else if (!b.available && b.reason) notes.push('Ladder unavailable: ' + b.reason.replaceAll('_', ' ') + '.');
  html += '<div class="pop-note">' + notes.join(' ') + '</div>';
  return html;
}

let openCostId = null;

function closeCostPopover() {
  const pop = document.getElementById('cost-popover');
  if (pop) pop.remove();
  document.querySelectorAll('.cost-info.open').forEach(el => el.classList.remove('open'));
  openCostId = null;
}

function openCostPopover(btn, c) {
  closeCostPopover();
  const pop = document.createElement('div');
  pop.id = 'cost-popover';
  pop.className = 'cost-popover';
  pop.innerHTML = costBreakdownHtml(c);
  document.body.appendChild(pop);
  const r = btn.getBoundingClientRect();
  pop.style.top = (window.scrollY + r.bottom + 8) + 'px';
  const left = Math.max(8, window.scrollX + r.right - pop.offsetWidth);
  pop.style.left = left + 'px';
  btn.classList.add('open');
  openCostId = c.id;
}

function toggleCostPopover(btn) {
  const id = btn.dataset.costInfo;
  if (openCostId === id) { closeCostPopover(); return; }
  const c = M.configs.find(x => x.id === id);
  if (c && c.cost_breakdown) openCostPopover(btn, c);
}

// --- Config filter (chips + search); state is global across tabs -----------
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

// --- Row captions -----------------------------------------------------------

function effortCaption(c) {
  // Classification matrix cells use Pass B effort; banked single-pass refs
  // use reasoning effort. Missing effort must read as unknown, never as a
  // fabricated medium.
  const e = (c.effort_b == null || c.effort_b === '' || c.effort_b === 'unknown')
    ? 'unknown'
    : String(c.effort_b);
  if (e === 'unknown') return 'effort unknown';
  if (c.kind === 'classification' || c.kind === 'two_pass') return 'Pass B ' + e;
  if (c.kind === 'single_pass' || e === 'none') return 'effort ' + e;
  return 'effort ' + e;
}

function isPartial(c) {
  if (c.is_partial === true) return true;
  return c.n_scored != null && c.n_expected != null && c.n_scored < c.n_expected;
}

function sampleCaption(c) {
  // "100 of 100 rows scored" = golden-set coverage, never an accuracy;
  // the old "100/100 scored" read as a score out of 100.
  if (c.n_scored == null && c.n_expected == null) return '';
  if (c.n_expected != null) return String(c.n_scored ?? '?') + ' of ' + c.n_expected + ' rows scored';
  return String(c.n_scored) + ' rows scored';
}

// --- Tab 2: model benchmarks -------------------------------------------------

function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-body');
  if (!tbody) return;
  closeCostPopover();
  const runs = visibleConfigs().slice().sort((a, b) => b.subclass_acc - a.subclass_acc);
  if (!runs.length) {
    tbody.innerHTML = '<tr><td colspan="8"><div class="empty">No configurations selected.</div></td></tr>';
    return;
  }
  tbody.innerHTML = runs.map((c, i) => {
    const bar = Math.max(0, Math.min(1, c.subclass_acc));
    const partial = isPartial(c);
    const badge = partial ? '<span class="partial-badge">partial</span>' : '';
    const nLine = sampleCaption(c);
    const sub = [c.model + ' &middot; ' + effortCaption(c), nLine].filter(Boolean).join(' &middot; ');
    return '<tr>' +
      '<td class="mono">' + (i + 1) + '</td>' +
      '<td><div class="name-cell">' + c.label + badge + '</div>' +
        '<div class="sub-cell">' + sub + '</div></td>' +
      '<td class="num"><div class="score-cell"><span class="score-val">' + pct(c.subclass_acc) + '</span>' +
        '<div class="score-bar"><span style="width:' + (100 * bar).toFixed(1) + '%"></span></div></div></td>' +
      '<td class="num mono">' + pct(c.ai_native_acc) + '</td>' +
      '<td class="num mono">' + pct(c.rad_acc) + '</td>' +
      '<td class="num mono">' + pct(c.mean_confidence) + '</td>' +
      '<td class="num mono"><span class="cost-cell">' + money(c.projected_usd) +
        (c.cost_breakdown
          ? '<button type="button" class="cost-info" data-cost-info="' + c.id +
            '" aria-label="How this cost estimate was computed" title="Cost breakdown">' + INFO_SVG + '</button>'
          : '') +
        '</span></td>' +
      '<td class="num mono">' + sec(c.latency_p50) + '</td>' +
      '</tr>';
  }).join('');
}

function renderPareto() {
  const runs = visibleConfigs();
  const host = document.getElementById('chart-pareto');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-pareto', 'Select a configuration above to plot cost against accuracy.'); return; }
  const plottable = runs.filter(c => Number.isFinite(c.projected_usd) && c.projected_usd > 0);
  if (!plottable.length) {
    emptyChart('chart-pareto', 'No configurations with a positive projected cost to plot.');
    return;
  }
  host.innerHTML = '';
  // Fit y to visible points (with CI), never a hard-coded band: real runs
  // can land far below the mock's range.
  let yLo = Infinity;
  let yHi = -Infinity;
  for (const c of plottable) {
    const ci = c.subclass_ci == null ? 0 : c.subclass_ci;
    yLo = Math.min(yLo, c.subclass_acc - ci);
    yHi = Math.max(yHi, c.subclass_acc + ci);
  }
  const pad = Math.max(0.03, (yHi - yLo) * 0.12);
  const yRange = [Math.max(0, yLo - pad), Math.min(1, yHi + pad)];
  // Explicit dollar ticks on the log axis: Plotly's default log labels render
  // minor ticks as bare digits (2, 3, 4...), which reads as broken. A 1-2-5
  // ladder covers any cost range, mock or real.
  const xs = plottable.map(c => c.projected_usd);
  const xLo = Math.min(...xs);
  const xHi = Math.max(...xs);
  const tickVals = [];
  for (let e = Math.floor(Math.log10(xLo)) - 1; e <= Math.ceil(Math.log10(xHi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= xLo / 1.4 && v <= xHi * 1.4) tickVals.push(v);
    }
  }
  const xRange = [Math.log10(xLo) - 0.12, Math.log10(xHi) + 0.14];
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
      text: [partial ? c.label + ' (partial)' : c.label],
      textposition: 'top center',
      textfont: {size: 10, color: partial ? '#f85149' : '#a8abb0'},
      marker: {
        size: 11,
        color: colorFor(c),
        symbol: partial ? 'circle-open' : 'circle',
        line: {width: partial ? 2 : 1, color: partial ? '#f85149' : '#e8e8ea'},
      },
      error_y: c.subclass_ci == null ? undefined : {
        type: 'data', array: [c.subclass_ci], visible: true,
        color: '#7a7e85', thickness: 1, width: 3,
      },
      hovertemplate: c.label + '<br>subclass %{y:.1%}<br>cost %{x:$,.0f}' +
        hoverExtra + '<extra></extra>',
    };
  });
  Plotly.newPlot('chart-pareto', traces, layout({
    xaxis: {
      title: {text: 'Projected production cost, USD (log scale)', font: axisFont},
      type: 'log', range: xRange,
      tickvals: tickVals, ticktext: tickVals.map(money),
      gridcolor: '#222222', zeroline: false, tickfont: numFont,
    },
    yaxis: {
      title: {text: 'Subclass accuracy', font: axisFont},
      tickformat: '.0%', range: yRange, gridcolor: '#222222', zeroline: false, tickfont: numFont,
    },
    showlegend: false,
    margin: {l: 56, r: 16, t: 24, b: 52},
  }), cfg);
}

function renderLatency() {
  const runs = visibleConfigs();
  const host = document.getElementById('chart-latency');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-latency', 'Select a configuration above to show latency.'); return; }
  host.innerHTML = '';
  Plotly.newPlot('chart-latency', [
    {
      type: 'bar', name: 'p50',
      x: runs.map(c => c.label),
      y: runs.map(c => c.latency_p50),
      marker: {color: '#5b8fc4'},
      hovertemplate: '%{x}<br>p50 %{y:.2f}s<extra></extra>',
    },
    {
      type: 'bar', name: 'p95',
      x: runs.map(c => c.label),
      y: runs.map(c => c.latency_p95),
      marker: {color: '#3d5568'},
      hovertemplate: '%{x}<br>p95 %{y:.2f}s<extra></extra>',
    },
  ], layout({
    barmode: 'group',
    yaxis: {title: {text: 'Seconds', font: axisFont}, gridcolor: '#222222', tickfont: numFont},
    xaxis: {tickangle: -30, gridcolor: 'rgba(0,0,0,0)', tickfont: numFont},
    margin: {l: 48, r: 12, t: 28, b: 80},
  }), cfg);
}

// --- Tab 3: confidence correctness correlation --------------------------------
// Pass A confidence is banked once per model, so calibration is a model
// property: one trace per visible model group.

function calibrationGroups() {
  const seen = new Set();
  const out = [];
  for (const c of visibleConfigs()) {
    if (c.is_aggregate || seen.has(c.model_group)) continue;
    if (!c.reliability_bins || !c.reliability_bins.length) continue;
    seen.add(c.model_group);
    out.push(c);
  }
  return out;
}

function renderReliability() {
  const host = document.getElementById('chart-reliability');
  if (!host) return;
  const groups = calibrationGroups();
  if (!groups.length) {
    emptyChart('chart-reliability', 'Reliability bins appear when a scored run carries calibration.reliability.bins (score with --confidence-from-raw).');
    return;
  }
  host.innerHTML = '';
  const traces = [{
    type: 'scatter',
    mode: 'lines',
    name: 'perfect calibration',
    x: [0, 1],
    y: [0, 1],
    line: {dash: 'dash', color: '#444444', width: 1},
    hoverinfo: 'skip',
  }];
  for (const c of groups) {
    const bins = c.reliability_bins.filter(b => b.count > 0 && b.mean_confidence != null);
    traces.push({
      type: 'scatter',
      mode: 'markers+lines',
      name: c.model_group,
      x: bins.map(b => b.mean_confidence),
      y: bins.map(b => b.accuracy),
      marker: {size: bins.map(b => 5 + Math.sqrt(b.count)), color: colorFor(c)},
      line: {color: colorFor(c), width: 1.5},
      text: bins.map(b => 'n=' + b.count),
      hovertemplate: c.model_group + '<br>conf %{x:.2f}<br>acc %{y:.2f}<br>%{text}<extra></extra>',
    });
  }
  Plotly.newPlot('chart-reliability', traces, layout({
    xaxis: {title: {text: 'Mean confidence in bin', font: axisFont}, range: [0.35, 1.02], gridcolor: '#222222', tickfont: numFont},
    yaxis: {title: {text: 'Observed accuracy', font: axisFont}, range: [0.35, 1.02], gridcolor: '#222222', tickfont: numFont},
    margin: {l: 54, r: 12, t: 28, b: 48},
    showlegend: true,
  }), cfg);
}

function renderEce() {
  const runs = visibleConfigs().filter(c => !c.is_aggregate && c.ece != null);
  const host = document.getElementById('chart-ece');
  if (!host) return;
  if (!runs.length) { emptyChart('chart-ece', 'ECE appears when a scored run carries calibration (score with --confidence-from-raw).'); return; }
  host.innerHTML = '';
  Plotly.newPlot('chart-ece', [{
    type: 'bar',
    x: runs.map(c => c.label),
    y: runs.map(c => 100 * c.ece),
    marker: {color: runs.map(c => colorFor(c)), opacity: 0.9},
    hovertemplate: '%{x}<br>ECE %{y:.2f} pp<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'ECE, percentage points (lower is better)', font: axisFont}, gridcolor: '#222222', tickfont: numFont},
    xaxis: {tickangle: -30, gridcolor: 'rgba(0,0,0,0)', tickfont: numFont},
    showlegend: false,
    margin: {l: 54, r: 12, t: 16, b: 80},
  }), cfg);
}

function renderSelective() {
  const host = document.getElementById('chart-selective');
  if (!host) return;
  const groups = [];
  const seen = new Set();
  for (const c of visibleConfigs()) {
    if (c.is_aggregate || seen.has(c.model_group)) continue;
    if (!c.selective_curve || !c.selective_curve.length) continue;
    seen.add(c.model_group);
    groups.push(c);
  }
  if (!groups.length) {
    emptyChart('chart-selective', 'Selective-prediction curves appear when a scored run carries calibration.selective_prediction.');
    return;
  }
  host.innerHTML = '';
  const traces = groups.map(c => ({
    type: 'scatter',
    mode: 'lines+markers',
    name: c.model_group,
    x: c.selective_curve.map(p => p.coverage),
    y: c.selective_curve.map(p => p.accuracy),
    line: {color: colorFor(c), width: 2},
    marker: {size: 6, color: colorFor(c)},
    hovertemplate: c.model_group + '<br>coverage %{x:.0%}<br>accuracy %{y:.1%}<extra></extra>',
  }));
  Plotly.newPlot('chart-selective', traces, layout({
    xaxis: {title: {text: 'Coverage (share of rows answered, most confident first)', font: axisFont}, tickformat: '.0%', gridcolor: '#222222', tickfont: numFont},
    yaxis: {title: {text: 'Accuracy on answered rows', font: axisFont}, tickformat: '.0%', gridcolor: '#222222', tickfont: numFont},
    margin: {l: 58, r: 12, t: 28, b: 52},
    showlegend: true,
  }), cfg);
}

function renderAll() {
  renderLeaderboard();
  renderPareto();
  renderLatency();
  renderReliability();
  renderEce();
  renderSelective();
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.toggle('active', p.id === 'panel-' + name);
  });
  closeCostPopover();
  if (name === 'benchmarks') {
    setTimeout(() => { renderPareto(); renderLatency(); }, 30);
  }
  if (name === 'confidence') {
    setTimeout(() => { renderReliability(); renderEce(); renderSelective(); }, 30);
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
  // Cost-info icons live inside a re-rendered tbody, so delegate from the
  // document: toggle on icon click, close on outside click or Escape.
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.cost-info');
    if (btn) { toggleCostPopover(btn); return; }
    if (!e.target.closest('.cost-popover')) closeCostPopover();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeCostPopover();
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
      <input id="config-search" type="search" placeholder="Search configurations..." autocomplete="off"/>
    </label>
    <span class="toolbar-count" id="filter-count"></span>
  </div>
  <div class="chip-row">{"".join(config_btns)}</div>
</div>
"""


def _robustness_panel_html(metrics: dict) -> str:
    """Server-rendered checks panel: badge + meaning + numbers per check."""
    checks = (metrics.get("robustness") or {}).get("checks") or []
    if not checks:
        return '<div class="empty">No robustness data available for this load.</div>'
    parts = []
    for check in checks:
        status = html.escape(str(check.get("status") or "pending"))
        title = html.escape(str(check.get("title") or check.get("id") or ""))
        meaning = html.escape(str(check.get("meaning") or ""))
        stats_html = ""
        if check.get("stats"):
            cells = "".join(
                '<div class="stat"><span class="stat-label">'
                + html.escape(str(s["label"]))
                + '</span><span class="stat-value">'
                + html.escape(str(s["value"]))
                + "</span></div>"
                for s in check["stats"]
            )
            stats_html = f'<div class="check-stats">{cells}</div>'
        rows_html = ""
        if check.get("per_model"):
            rows = "".join(
                "<tr><td class=\"mono\">"
                + html.escape(str(r["model"]))
                + '</td><td><span class="mini-status '
                + html.escape(str(r["status"]))
                + '">'
                + html.escape(str(r["status"]))
                + "</span></td><td>"
                + html.escape(str(r["detail"]))
                + "</td></tr>"
                for r in check["per_model"]
            )
            rows_html = (
                '<table class="mini"><thead><tr><th>Model</th><th>Status</th>'
                f"<th>Evidence</th></tr></thead><tbody>{rows}</tbody></table>"
            )
        note_html = ""
        if check.get("pending_note"):
            note_html = (
                '<p class="pending-note">'
                + html.escape(str(check["pending_note"]))
                + "</p>"
            )
        footnote_html = ""
        if check.get("footnote"):
            footnote_html = (
                '<p class="check-footnote">'
                + html.escape(str(check["footnote"]))
                + "</p>"
            )
        parts.append(f"""
<div class="check" id="check-{html.escape(str(check.get('id') or ''))}">
  <div class="check-head">
    <span class="badge {status}">{status}</span>
    <h3>{title}</h3>
  </div>
  <div class="check-body">
    <p class="check-meaning">{meaning}</p>
    {stats_html}
    {rows_html}
    {note_html}
    {footnote_html}
  </div>
</div>
""")
    return "".join(parts)


@lru_cache(maxsize=1)
def _plotly_inline_js() -> str:
    """Vendored Plotly, escaped so a literal </script> cannot break the page."""
    if not PLOTLY_VENDOR.is_file():
        raise FileNotFoundError(
            f"Missing vendored Plotly at {PLOTLY_VENDOR}. "
            "Restore data visualization/02_Analysis_Code/vendor/plotly-2.35.2.min.js"
        )
    # A raw </script> inside the library would close the HTML script tag early.
    return PLOTLY_VENDOR.read_text(encoding="utf-8").replace("</", "<\\/")


def _run_instance_card_html(metrics: dict) -> str:
    """Header card naming the run behind the page, synthetic or scored."""
    run = metrics.get("run_instance") or {}
    if metrics.get("synthetic") or run.get("synthetic"):
        return """
<div class="notice" id="run-instance" data-mode="synthetic">
  <span class="tag">Synthetic data</span>
  <div>All numbers on this page are internally consistent placeholders for the
  locked evaluation matrix (3 models &times; 3 Pass B reasoning efforts).
  Live results replace them once scored runs are loaded with
  <code>--runs</code> or <code>--scored</code>.</div>
</div>
"""
    recorded = [s for s in (run.get("started_first"), run.get("started_last")) if s]
    utc_span = " to ".join(dict.fromkeys(recorded))
    title = f' title="Run start recorded as {html.escape(utc_span)}"' if utc_span else ""
    meta = " &middot; ".join(html.escape(b) for b in run_meta_bits(run))
    meta_html = f'\n    <span class="run-meta">{meta}</span>' if meta else ""
    return f"""
<div class="notice scored" id="run-instance" data-mode="scored">
  <span class="tag">Run</span>
  <div{title}>
    <span class="run-headline">{html.escape(format_run_headline(run))}</span>{meta_html}
  </div>
</div>
"""


def build_html(metrics: dict) -> str:
    today = datetime.date.today().isoformat()
    notice = _run_instance_card_html(metrics)
    toolbar = _filter_toolbar_html(metrics)
    robustness_panel = _robustness_panel_html(metrics)
    script = (
        SCRIPT.replace("__M_JSON__", json.dumps(metrics, ensure_ascii=False))
        .replace("__GROUP_COLORS__", json.dumps(GROUP_COLORS))
    )
    source = html.escape(str(metrics.get("source") or "fixture"))
    n = metrics.get("n_configs", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Classifier Eval Suite</title>
<script>
{PLOTLY_PLACEHOLDER}
</script>
<style>
{STYLE}
</style>
</head>
<body>
<header class="appbar">
  <div class="brand">Classifier Eval Suite<small>AI-native startup classifier</small></div>
  <div class="appbar-meta">{n} configurations &middot; {source}</div>
</header>

<nav class="tabs" aria-label="Suite sections">
  <button type="button" class="tab active" data-tab="robustness">Pipeline robustness</button>
  <button type="button" class="tab" data-tab="benchmarks">Model benchmarks</button>
  <button type="button" class="tab" data-tab="confidence">Confidence correctness correlation</button>
</nav>

{notice}

<main class="content">
  <section class="panel active" id="panel-robustness">
    <div class="tab-lead">
      <h2>Will this pipeline survive production?</h2>
      <p>Structural checks on the confidence pipeline: whether a confidence
      score was recovered for every golden-set company, the probability
      mass behind each confidence value, and the transfer from the sync API
      used in evaluation to the Batch API used at scale.</p>
    </div>
    {robustness_panel}
  </section>

  <section class="panel" id="panel-benchmarks">
    <div class="tab-lead">
      <h2>Which model should production use?</h2>
      <p>The locked model &times; effort matrix compared on accuracy,
      confidence quality, projected cost, and latency. Latency is a
      production-practice metric (sync API wall-clock), not a model-quality
      score.</p>
    </div>
    {toolbar}
    <div class="table-wrap">
      <table class="grid" id="leaderboard">
        <thead>
          <tr>
            <th>#</th>
            <th>Configuration</th>
            <th class="num">Subclass accuracy</th>
            <th class="num">AI-native</th>
            <th class="num">RAD</th>
            <th class="num">Mean confidence</th>
            <th class="num">Projected cost</th>
            <th class="num">Latency p50</th>
          </tr>
        </thead>
        <tbody id="leaderboard-body"></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Cost against subclass accuracy</div>
      <div class="card-desc">Each point is one configuration; whiskers are the
      95% CI on subclass accuracy. The cost axis is the projected production
      spend from the per-row cost ladder (open any cost value in the table
      above for the full arithmetic).</div>
      <div id="chart-pareto" class="chart"></div>
    </div>
    <div class="card">
      <div class="card-title">Latency, p50 and p95</div>
      <div class="card-desc">Wall-clock seconds per row over the sync API.
      A production-practice consideration for throughput planning; it does
      not measure output quality.</div>
      <div id="chart-latency" class="chart short"></div>
    </div>
  </section>

  <section class="panel" id="panel-confidence">
    <div class="tab-lead">
      <h2>Does confidence track correctness?</h2>
      <p>Per-row confidence comes from the log-probabilities on the binary
      decision token. These views test whether that number behaves like a
      probability: high-confidence answers should be right more often, and
      abstaining on low confidence should raise accuracy on what remains.
      Confidence is banked once per model, so curves are per model, not per
      effort.</p>
    </div>
    <div class="card">
      <div class="card-title">Reliability diagram</div>
      <div class="card-desc">Rows are grouped into confidence bins; each
      marker plots the bin's mean confidence against its observed accuracy
      (marker size tracks bin population). A well-calibrated model hugs the
      dashed diagonal.</div>
      <div id="chart-reliability" class="chart"></div>
    </div>
    <div class="card">
      <div class="card-title">Expected calibration error</div>
      <div class="card-desc">The population-weighted gap between confidence
      and accuracy across bins, per configuration. Lower is better; zero
      means confidence values can be read as probabilities.</div>
      <div id="chart-ece" class="chart short"></div>
    </div>
    <div class="card">
      <div class="card-title">Selective prediction</div>
      <div class="card-desc">Accuracy when the classifier only answers its
      most-confident rows and abstains below the cutoff. A useful confidence
      signal slopes down: the confident head outperforms full coverage.</div>
      <div id="chart-selective" class="chart short"></div>
    </div>
  </section>

  <footer>
    Page built {today} (run times are on the card above) &middot; regenerate with
    <code>python -m evals dashboard</code> (synthetic fixture by default) or
    pass <code>--runs</code> / <code>--scored</code> to load real scored runs.
  </footer>
</main>
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
    # (banked single-pass baselines would mix architectures and hide the notice).
    # --force-fixture remains accepted for CLI compatibility (same outcome).
    return load_fixture(DEFAULT_FIXTURE)


def finalize_html(page: str) -> str:
    """Inline vendored Plotly so the file opens offline and can be emailed alone."""
    if PLOTLY_PLACEHOLDER not in page:
        raise ValueError("HTML is missing the Plotly placeholder; cannot finalize")
    return page.replace(PLOTLY_PLACEHOLDER, _plotly_inline_js(), 1)


def write_dashboard(
    metrics: dict, output: Path, *, save_instance: bool = False
) -> Path | None:
    """Write the suite page, archiving the build when it is worth keeping.

    Real scored runs are archived every time; mock builds only on request, so
    development rebuilds do not bury them. Returns the archived path, or None.
    The written HTML is self-contained (Plotly inlined, no CDN).
    """
    page = finalize_html(build_html(metrics))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    mode = "SYNTHETIC" if metrics.get("synthetic") else "scored"
    print(f"[{mode}] {metrics['n_configs']} configs → {output}", file=sys.stderr)

    if metrics.get("synthetic") and not save_instance:
        return None
    archived = archive_instance(page, metrics)
    verb = "replaced" if archived.replaced else "archived"
    print(
        f"[instance {archived.number:02d}] {verb} → {archived.path}\n"
        f"[index] {archived.index_path}",
        file=sys.stderr,
    )
    return archived.path


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
    parser.add_argument(
        "--save-instance",
        action="store_true",
        help="Also archive this build under eval_instances/ (automatic for real runs)",
    )
    args = parser.parse_args()

    write_dashboard(
        resolve_metrics(args), args.output, save_instance=args.save_instance
    )


if __name__ == "__main__":
    main()
