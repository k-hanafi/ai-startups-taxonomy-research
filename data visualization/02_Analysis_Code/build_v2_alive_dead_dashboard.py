#!/usr/bin/env python3
"""Build the V2 alive-vs-dead presentation dashboard.

Reads the V2 professor CSV (18 columns, evidence-only alive + dead). Confidence
is three fields with measurement explainers. Chart inventory follows the
consolidation cuts in .cursor/plans/v2_production_dashboard.plan.md.

Writes:
    data visualization/01_Presentation_Materials/v2_alive_dead_cohort.html
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

_HERE = Path(__file__).resolve().parent
# Same idiom as build_classification_dashboard.py: file -> Analysis_Code ->
# data visualization -> repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    _PROJECT_ROOT / "data visualization" / "01_Presentation_Materials" / "v2_alive_dead_cohort.html"
)
PLOTLY_VENDOR = _HERE / "vendor" / "plotly-2.35.2.min.js"
PLOTLY_PLACEHOLDER = "<!--PLOTLY_INLINE-->"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # let dataclasses resolve string annotations
    spec.loader.exec_module(mod)
    return mod


_ANALYSIS = _load("survivorship_analysis", _HERE / "survivorship_analysis.py")
_BASE = _load("build_classification_dashboard", _HERE / "build_classification_dashboard.py")

SUBCLASS_ORDER = _BASE.SUBCLASS_ORDER
SUBCLASS_LABELS = _BASE.SUBCLASS_LABELS
SUBCLASS_COLORS = _BASE.SUBCLASS_COLORS
RAD_ORDER = _BASE.RAD_ORDER
RAD_COLORS = _BASE.RAD_COLORS
COHORT_COLORS = _BASE.COHORT_COLORS
RAD_NA_CLASSES = _BASE.RAD_NA_CLASSES

# Survivor / dead series colors used across the flagship section.
ALIVE_COLOR = "#059669"
DEAD_COLOR = "#e11d48"
DEAD_STRICT_COLOR = "#9f1239"


def typed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast string columns for base metrics and V2 confidence charts."""
    t = df.copy()
    t["ai_native"] = pd.to_numeric(t["ai_native"], errors="coerce").fillna(0).astype(int)
    t["conf_classification"] = pd.to_numeric(t["conf_classification"], errors="coerce")
    t["conf_rad"] = pd.to_numeric(t["conf_rad"], errors="coerce")
    if "ai_native_confidence" in t.columns:
        t["ai_native_confidence"] = pd.to_numeric(t["ai_native_confidence"], errors="coerce")
    if "subclass_confidence" in t.columns:
        t["subclass_confidence"] = pd.to_numeric(t["subclass_confidence"], errors="coerce")
    if "rad_confidence" in t.columns:
        t["rad_confidence"] = pd.to_numeric(t["rad_confidence"], errors="coerce")
    return t


def build_filter_data(t: pd.DataFrame) -> dict:
    """Cohort x min subclass-confidence grid for Landscape filters."""
    out = {}
    for cohort in ["ALL", "PRE-GENAI", "GENAI-ERA"]:
        for min_conf in range(1, 6):
            sub = t
            if cohort != "ALL":
                sub = sub[sub.cohort == cohort]
            sub = sub[sub.conf_classification >= min_conf]
            out[f"{cohort}_{min_conf}"] = {
                "subclass": {s: int((sub.subclass == s).sum()) for s in SUBCLASS_ORDER},
                "total": int(len(sub)),
                "ai_native": int((sub.ai_native == 1).sum()),
            }
    return out


def _clean_term(name: str) -> str:
    """Turn a statsmodels design term into a readable label."""
    if ":" in name:  # interaction terms (Model 3)
        return " x ".join(_clean_term(part) for part in name.split(":"))
    name = name.replace("ai_native_int", "AI-native")
    name = name.replace("log_funding", "Funding (per 10x)")
    m = re.search(r"rad_score.*\[T\.(RAD-[HML])\]", name)
    if m:
        return f"{m.group(1)} (vs RAD-L)"
    m = re.search(r"subclass_group\)\[T\.(.+?)\]", name)
    if m:
        return f"{m.group(1)} (vs Commoditizable)"
    m = re.search(r"founding_era\)\[T\.(.+?)\]", name)
    if m:
        return f"Founded {m.group(1)}"
    m = re.search(r"cat_grp\)\[T\.(.+?)\]", name)
    if m:
        return f"Vertical: {m.group(1)}"
    return name


def _forest(model: dict) -> list[dict]:
    """Readable, ordered odds-ratio rows for a forest plot."""
    if not model.get("available"):
        return []
    rows = [{
        "label": _clean_term(t["term"]),
        "or": t["odds_ratio"], "lo": t["ci_low"], "hi": t["ci_high"],
        "sig": t["pvalue"] < 0.05, "risk": t["odds_ratio"] > 1,
    } for t in model["terms"]]
    rows.sort(key=lambda r: r["or"])
    return rows


STYLE = """
:root {
  --bg: #ffffff;
  --bg2: #ffffff;
  --bg3: #f8f9fb;
  --border: #e5e7eb;
  --border2: #d1d5db;
  --text: #1a1a1a;
  --text2: #4a4a4a;
  --muted: #8a8a8a;
  --navy: #1e2a4a;
  --navy-light: #f0f2f7;
  --indigo: #4f46e5;
  --indigo-light: #eef2ff;
  --indigo-border: #c7d2fe;
  --emerald: #059669;
  --emerald-light: #ecfdf5;
  --amber: #d97706;
  --amber-light: #fffbeb;
  --rose: #e11d48;
  --rose-light: #fff1f2;
  --cyan: #0891b2;
  --violet: #7c3aed;
  --slate: #475569;
  --serif: 'Cormorant Garamond', Georgia, serif;
  --sans: 'Inter', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: var(--sans); background: var(--bg); color: var(--text); line-height: 1.7; font-size: 15px; }
::selection { background: var(--navy); color: white; }

.preview-banner {
  /* Sit in the main column only so the fixed left nav stays fully clickable. */
  position: sticky; top: 0; z-index: 50;
  margin-left: 216px;
  background: #b45309; color: #ffffff;
  padding: 0.8rem 1.5rem; text-align: center;
  font-size: 0.85rem; font-weight: 600; letter-spacing: 0.02em;
  border-bottom: 3px solid #92400e;
}
.preview-banner code { font-family: var(--mono); font-size: 0.8rem; background: rgba(0,0,0,0.18); padding: 0.05rem 0.35rem; border-radius: 4px; }

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
h3 { font-family: var(--serif); font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--navy); }
p { color: var(--text2); font-size: 0.9rem; max-width: 720px; margin-bottom: 1.1rem; line-height: 1.75; }
p:last-child { margin-bottom: 0; }

.hero-metrics {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem;
  margin: 2rem 0 1.5rem;
}
.hero-metrics.cols4 { grid-template-columns: repeat(4, 1fr); }
.hero-metrics.cols3 { grid-template-columns: repeat(3, 1fr); }
.metric-card {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  padding: 1.25rem 1.3rem; text-align: center;
  transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.metric-card.hl { border-color: var(--indigo-border); background: var(--indigo-light); }
.metric-card.ok { border-color: #a7f3d0; background: var(--emerald-light); }
.metric-card.bad { border-color: #fecdd3; background: var(--rose-light); }
.mc-val { font-family: var(--serif); font-size: 2.2rem; line-height: 1; margin-bottom: 0.25rem; color: var(--navy); }
.metric-card.hl .mc-val { color: var(--indigo); }
.metric-card.ok .mc-val { color: var(--emerald); }
.metric-card.bad .mc-val { color: var(--rose); }
.mc-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.3rem; }
.mc-ctx { font-size: 0.75rem; color: var(--muted); }

.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0; }
.chart-row.single { grid-template-columns: 1fr; }
.chart-box {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  overflow: hidden;
}
.chart-box-header { padding: 1.1rem 1.4rem 0.75rem; border-bottom: 1px solid var(--border); }
.chart-box-title { font-family: var(--serif); font-size: 1.05rem; font-weight: 600; color: var(--navy); margin-bottom: 0.3rem; }
.chart-box-desc { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }
.chart-body { padding: 0.5rem 0.5rem; }

.insight { padding: 1.1rem 1.4rem; border-radius: 8px; margin: 1.5rem 0; font-size: 0.85rem; line-height: 1.7; }
.insight p { font-size: 0.85rem; max-width: none; margin-bottom: 0.35rem; }
.insight p:last-child { margin-bottom: 0; }
.insight-blue { background: var(--indigo-light); border: 1px solid var(--indigo-border); color: var(--text2); }
.insight-blue strong { color: #3730a3; }
.insight-amber { background: var(--amber-light); border: 1px solid #fde68a; color: var(--text2); }
.insight-amber strong { color: #92400e; }
.insight-green { background: var(--emerald-light); border: 1px solid #a7f3d0; color: var(--text2); }
.insight-green strong { color: #065f46; }

.filter-bar {
  display: flex; align-items: center; gap: 1.5rem;
  padding: 0.85rem 1.4rem; background: var(--bg3); border: 1px solid var(--border);
  border-radius: 8px; margin: 1.5rem 0; flex-wrap: wrap;
}
.filter-bar label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--navy); }
.filter-bar .pill-group { display: flex; gap: 0; }
.pill-btn {
  font-family: var(--mono); font-size: 0.7rem; padding: 0.35rem 0.85rem;
  border: 1px solid var(--border); background: var(--bg2); color: var(--text2);
  cursor: pointer; transition: all 0.15s;
}
.pill-btn:first-child { border-radius: 5px 0 0 5px; }
.pill-btn:last-child { border-radius: 0 5px 5px 0; }
.pill-btn:not(:last-child) { border-right: none; }
.pill-btn.active { background: var(--navy); color: white; border-color: var(--navy); }

.slider-group { display: flex; align-items: center; gap: 0.6rem; }
.slider-group input[type=range] { width: 120px; accent-color: var(--navy); }
.slider-val { font-family: var(--mono); font-size: 0.75rem; color: var(--navy); font-weight: 600; min-width: 20px; }

.methods-box {
  background: var(--bg3); border: 1px solid var(--border2); border-radius: 8px;
  padding: 1.5rem 1.75rem; margin: 2rem 0;
}
.explainer {
  background: var(--bg3); border: 1px solid var(--border); border-radius: 8px;
  padding: 1.1rem 1.4rem; margin: 1.25rem 0 1.5rem; font-size: 0.85rem; color: var(--text2);
  line-height: 1.7; max-width: 820px;
}
.explainer p { font-size: 0.85rem; margin-bottom: 0.45rem; max-width: none; }
.explainer p:last-child { margin-bottom: 0; }
.explainer ol { margin: 0.4rem 0 0 1.2rem; }
.explainer li { margin-bottom: 0.35rem; }
.subsec { margin-top: 2.5rem; padding-top: 0.5rem; }
.rob-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.rob-table th, .rob-table td { text-align: left; padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border); }
.rob-table th { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.rob-table td:nth-child(2), .rob-table td:nth-child(3) { font-family: var(--mono); }

.methods-box h3 { margin-bottom: 0.8rem; }
.methods-box ol { margin-left: 1.2rem; }
.methods-box li { color: var(--text2); font-size: 0.85rem; line-height: 1.7; margin-bottom: 0.55rem; }

footer {
  padding: 2.5rem 4.5rem; text-align: center; color: var(--muted);
  font-size: 0.75rem; border-top: 1px solid var(--border); margin-left: 216px;
  line-height: 1.8;
}
footer strong { color: var(--text2); }

@media (max-width: 1100px) {
  nav { display: none; } main { margin-left: 0; } section { padding: 3rem 1.5rem; }
  .preview-banner { margin-left: 0; }
  .hero-metrics, .hero-metrics.cols4, .hero-metrics.cols3 { grid-template-columns: repeat(2, 1fr); }
  .chart-row { grid-template-columns: 1fr; }
  footer { margin-left: 0; padding: 2rem 1.5rem; }
}
@media print {
  section { opacity: 1 !important; transform: none !important; }
  nav { display: none; } main { margin-left: 0; }
  .preview-banner { position: static; margin-left: 0; }
}
"""


SCRIPT_TEMPLATE = """
const MB = __BASE_JSON__;
const MS = __SURV_JSON__;
const FILTER_DATA = __FILTER_JSON__;
const F1 = __FOREST1__;
const F2 = __FOREST2__;
const F3 = __FOREST3__;

const SUBCLASS_ORDER = __SUBCLASS_ORDER__;
const SUBCLASS_LABELS = __SUBCLASS_LABELS__;
const SUBCLASS_COLORS = __SUBCLASS_COLORS__;
const RAD_ORDER = __RAD_ORDER__;
const RAD_COLORS = __RAD_COLORS__;
const COHORT_COLORS = __COHORT_COLORS__;
const RAD_NA_CLASSES = __RAD_NA_CLASSES__;
const ALIVE_COLOR = '__ALIVE_COLOR__';
const DEAD_COLOR = '__DEAD_COLOR__';
const DEAD_STRICT_COLOR = '__DEAD_STRICT_COLOR__';

const cfg = {displayModeBar: false, responsive: true};
const axisFont = {family: 'Inter, sans-serif', size: 11, color: '#4a4a4a'};
const titleFont = {family: 'Inter, sans-serif', size: 12, color: '#1e2a4a'};
const mono = {family: 'JetBrains Mono', size: 10};

function layout(extra) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {l: 60, r: 24, t: 24, b: 48}, font: axisFont,
    xaxis: {gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'},
    yaxis: {gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'},
  }, extra || {});
}

/* ------------------------- base sections (forked) ------------------------ */

function renderSubclass() {
  const labels = SUBCLASS_ORDER.slice().reverse();
  const vals = labels.map(s => MB.subclass_counts[s]);
  Plotly.newPlot('chart-subclass', [{
    type: 'bar', orientation: 'h',
    y: labels.map(s => SUBCLASS_LABELS[s]), x: vals,
    marker: {color: labels.map(s => SUBCLASS_COLORS[s])},
    text: vals.map(v => v.toLocaleString()), textposition: 'outside', textfont: mono,
    hovertemplate: '%{y}<br>%{x:,} companies<extra></extra>',
  }], layout({
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}},
    xaxis: {title: {text: 'Number of startups', font: titleFont}},
    margin: {l: 200, r: 60, t: 16, b: 44},
  }), cfg);
}

function renderAiRateCohort() {
  const cohorts = ['PRE-GENAI', 'GENAI-ERA'];
  Plotly.newPlot('chart-ai-rate-cohort', [
    {type: 'bar', name: 'AI-Native', x: cohorts, y: cohorts.map(c => MB.ai_rate_by_cohort[c]),
     marker: {color: '#4f46e5'},
     text: cohorts.map(c => MB.ai_rate_by_cohort[c] + '%'), textposition: 'outside',
     textfont: {family: 'JetBrains Mono', size: 12, color: '#4f46e5'},
     hovertemplate: '%{x}<br>AI-Native: %{y:.2f}%<extra></extra>'},
    {type: 'bar', name: 'Not AI-Native', x: cohorts,
     y: cohorts.map(c => (100 - MB.ai_rate_by_cohort[c]).toFixed(2)),
     marker: {color: '#e2e8f0'},
     hovertemplate: '%{x}<br>Not AI-Native: %{y}%<extra></extra>'},
  ], layout({
    barmode: 'group',
    yaxis: {title: {text: 'Percentage', font: titleFont}, range: [0, 105]},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 12}},
    legend: {orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center', font: {size: 11}},
    margin: {l: 55, r: 24, t: 40, b: 44},
  }), cfg);
}

function renderRad() {
  const radActive = RAD_ORDER.filter(r => r !== 'RAD-NA');
  Plotly.newPlot('chart-rad', [{
    type: 'bar', x: radActive, y: radActive.map(r => MB.rad_counts[r]),
    marker: {color: radActive.map(r => RAD_COLORS[r])},
    text: radActive.map(r => MB.rad_counts[r].toLocaleString()),
    textposition: 'outside', textfont: mono,
    hovertemplate: '%{x}<br>%{y:,} companies<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'Number of startups', font: titleFont}},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 12}},
    margin: {l: 65, r: 24, t: 16, b: 44},
    annotations: [{
      x: 0.98, y: 0.95, xref: 'paper', yref: 'paper',
      text: 'Excluding ' + MB.rad_counts['RAD-NA'].toLocaleString() + ' RAD-NA',
      showarrow: false, font: {size: 10, color: '#8a8a8a', family: 'Inter'}, xanchor: 'right',
    }],
  }), cfg);
}

function renderHeatmap() {
  const radNA_classes = new Set(RAD_NA_CLASSES);
  const yLabels = SUBCLASS_ORDER.map(s => SUBCLASS_LABELS[s]).concat(['Total']);
  const cols = RAD_ORDER.concat(['Total']);
  const radTotals = {};
  RAD_ORDER.forEach(r => radTotals[r] = 0);
  const rawGrid = [];
  const rowTotals = [];
  SUBCLASS_ORDER.forEach(s => {
    const row = RAD_ORDER.map(r => MB.heatmap[s][r]);
    rawGrid.push(row);
    const total = radNA_classes.has(s) ? null : row.reduce((a, b) => a + b, 0);
    rowTotals.push(total);
    if (!radNA_classes.has(s)) RAD_ORDER.forEach((r, i) => radTotals[r] += row[i]);
  });
  const bottomRow = RAD_ORDER.map(r => (r === 'RAD-NA') ? null : radTotals[r]);
  const bottomTotal = bottomRow.filter(v => v !== null).reduce((a, b) => a + b, 0);
  bottomRow.push(bottomTotal);
  const zHeat = rawGrid.map(row => row.slice());
  zHeat.push(RAD_ORDER.map(() => null));
  const zLog = zHeat.map(row => row.map(v => (v != null && v > 0) ? Math.log10(v) : null));
  const textHeat = zHeat.map(row => row.map(v => (v != null && v > 0) ? v.toLocaleString() : ''));
  const zFull = zLog.map(row => row.concat([null]));
  const textFull = textHeat.map(row => row.concat(['']));
  const annotations = [];
  const boldFont = {family: 'JetBrains Mono', size: 11, color: '#1e2a4a'};
  rowTotals.forEach((t, i) => annotations.push({
    x: 'Total', y: yLabels[i],
    text: (t !== null && t > 0) ? '<b>' + t.toLocaleString() + '</b>' : '',
    showarrow: false, font: boldFont,
  }));
  RAD_ORDER.forEach((r, i) => annotations.push({
    x: r, y: 'Total',
    text: (bottomRow[i] !== null && bottomRow[i] > 0) ? '<b>' + bottomRow[i].toLocaleString() + '</b>' : '',
    showarrow: false, font: boldFont,
  }));
  annotations.push({x: 'Total', y: 'Total', text: '<b>' + bottomTotal.toLocaleString() + '</b>',
                    showarrow: false, font: boldFont});
  Plotly.newPlot('chart-heatmap', [{
    type: 'heatmap', z: zFull, x: cols, y: yLabels,
    text: textFull, texttemplate: '%{text}',
    colorscale: [[0, '#f8f9fb'], [0.2, '#eef2ff'], [0.4, '#c7d2fe'], [0.65, '#818cf8'], [1, '#3730a3']],
    showscale: false, hovertemplate: '%{y}<br>%{x}: %{text}<extra></extra>', xgap: 2, ygap: 2,
  }], layout({
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 10}, autorange: 'reversed'},
    xaxis: {side: 'top', tickfont: {family: 'JetBrains Mono', size: 11}},
    margin: {l: 190, r: 16, t: 40, b: 16},
    annotations: annotations,
  }), cfg);
}

function renderSubclassCohort() {
  const labels = SUBCLASS_ORDER.slice().reverse().map(s => SUBCLASS_LABELS[s]);
  const traces = ['PRE-GENAI', 'GENAI-ERA'].map(c => ({
    type: 'bar', orientation: 'h', name: c, y: labels,
    x: SUBCLASS_ORDER.slice().reverse().map(s => MB.subclass_by_cohort[c][s]),
    marker: {color: COHORT_COLORS[c]},
    hovertemplate: c + '<br>%{y}: %{x:,}<extra></extra>',
  }));
  Plotly.newPlot('chart-subclass-cohort', traces, layout({
    barmode: 'group',
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 10}},
    xaxis: {title: {text: 'Number of startups', font: titleFont}},
    legend: {orientation: 'h', y: 1.06, x: 0.5, xanchor: 'center', font: {size: 11}},
    margin: {l: 200, r: 24, t: 40, b: 44},
  }), cfg);
}

function renderRadCohort() {
  const radActive = RAD_ORDER.filter(r => r !== 'RAD-NA');
  const traces = ['PRE-GENAI', 'GENAI-ERA'].map(c => ({
    type: 'bar', name: c, x: radActive, y: radActive.map(r => MB.rad_by_cohort[c][r]),
    marker: {color: COHORT_COLORS[c]},
    text: radActive.map(r => MB.rad_by_cohort[c][r].toLocaleString()),
    textposition: 'outside', textfont: mono,
    hovertemplate: c + '<br>%{x}: %{y:,}<extra></extra>',
  }));
  Plotly.newPlot('chart-rad-cohort', traces, layout({
    barmode: 'group',
    yaxis: {title: {text: 'Number of startups', font: titleFont}},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 12}},
    legend: {orientation: 'h', y: 1.06, x: 0.5, xanchor: 'center', font: {size: 11}},
    margin: {l: 65, r: 24, t: 40, b: 44},
  }), cfg);
}

function renderConfDist(id, dist) {
  const levels = Object.keys(dist).map(Number).sort();
  const vals = levels.map(l => dist[l]);
  const colors = levels.map(l => l <= 2 ? '#e11d48' : l === 3 ? '#d97706' : '#059669');
  Plotly.newPlot(id, [{
    type: 'bar', x: levels, y: vals, marker: {color: colors},
    text: vals.map(v => v.toLocaleString()), textposition: 'outside', textfont: mono,
    hovertemplate: 'Confidence %{x}<br>%{y:,} companies<extra></extra>',
  }], layout({
    xaxis: {title: {text: 'Confidence Level', font: titleFont}, dtick: 1,
            tickfont: {family: 'JetBrains Mono', size: 13}},
    yaxis: {title: {text: 'Count', font: titleFont}},
    margin: {l: 70, r: 24, t: 16, b: 50},
  }), cfg);
}

function renderConfSubclass() {
  const labels = SUBCLASS_ORDER.map(s => SUBCLASS_LABELS[s]);
  const means = SUBCLASS_ORDER.map(s => MB.conf_by_subclass[s].mean);
  Plotly.newPlot('chart-conf-subclass', [{
    type: 'bar', name: 'Mean confidence', x: labels, y: means,
    marker: {color: SUBCLASS_ORDER.map(s => SUBCLASS_COLORS[s])},
    text: means.map(v => v.toFixed(1)), textposition: 'outside', textfont: mono,
    hovertemplate: '%{x}<br>Mean: %{y:.2f}<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'Mean Confidence', font: titleFont}, range: [0, 5.5]},
    xaxis: {tickangle: -35, tickfont: {family: 'JetBrains Mono', size: 9}},
    margin: {l: 55, r: 24, t: 16, b: 120},
    shapes: [{type: 'line', x0: -0.5, x1: SUBCLASS_ORDER.length - 0.5, y0: 3, y1: 3,
              line: {color: '#d97706', width: 1, dash: 'dot'}}],
    annotations: [{x: SUBCLASS_ORDER.length - 0.7, y: 3.15, text: 'Threshold = 3',
                   showarrow: false, font: {size: 10, color: '#d97706', family: 'JetBrains Mono'}}],
  }), cfg);
}

/* ------------------------ flagship: shared helpers ----------------------- */

function grouped(id, x, a, b, na, nb, ca, cb, yTitle, annos) {
  Plotly.newPlot(id, [
    {type: 'bar', name: na, x: x, y: a, marker: {color: ca},
     hovertemplate: na + '<br>%{x}: %{y}<extra></extra>'},
    {type: 'bar', name: nb, x: x, y: b, marker: {color: cb},
     hovertemplate: nb + '<br>%{x}: %{y}<extra></extra>'},
  ], layout({
    barmode: 'group',
    yaxis: {title: {text: yTitle, font: titleFont}},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 11}},
    legend: {orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center', font: {size: 11}},
    margin: {l: 60, r: 16, t: 36, b: 44},
    annotations: annos || [],
  }), cfg);
}

function mortBars(id, items, colors) {
  const it = items.slice().reverse();
  Plotly.newPlot(id, [{
    type: 'bar', orientation: 'h',
    y: it.map(d => d.label), x: it.map(d => d.mortality),
    marker: {color: colors ? it.map(d => colors[d.label] || '#475569') : '#4f46e5'},
    text: it.map(d => d.mortality.toFixed(1) + '%  (n=' + d.n.toLocaleString() + ')'),
    textposition: 'outside', textfont: mono,
    hovertemplate: '%{y}<br>Mortality %{x:.1f}%<extra></extra>',
  }], layout({
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}},
    xaxis: {title: {text: 'Mortality rate (dead / dead+alive)', font: titleFont}, ticksuffix: '%'},
    margin: {l: 150, r: 100, t: 16, b: 44},
  }), cfg);
}

function forest(id, rows) {
  if (!rows.length) {
    document.getElementById(id).innerHTML =
      '<p style="padding:1rem;color:#8a8a8a;font-size:0.8rem;">Model not available for this run.</p>';
    return;
  }
  Plotly.newPlot(id, [{
    type: 'scatter', mode: 'markers', x: rows.map(r => r.or), y: rows.map(r => r.label),
    marker: {color: rows.map(r => !r.sig ? '#94a3b8' : (r.risk ? DEAD_COLOR : ALIVE_COLOR)), size: 9},
    error_x: {type: 'data', symmetric: false,
              array: rows.map(r => r.hi - r.or), arrayminus: rows.map(r => r.or - r.lo),
              color: '#9aa3b2', thickness: 1.4, width: 4},
    hovertemplate: '%{y}<br>OR %{x:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>',
    customdata: rows.map(r => [r.lo, r.hi]),
  }], layout({
    xaxis: {type: 'log', title: {text: 'Odds ratio of death (log scale)', font: titleFont}},
    yaxis: {automargin: true, tickfont: {size: 11}},
    shapes: [{type: 'line', x0: 1, x1: 1, y0: -0.5, y1: rows.length - 0.5,
              line: {color: '#1e2a4a', width: 1, dash: 'dot'}}],
    margin: {l: 210, r: 24, t: 16, b: 44},
  }), cfg);
}

/* ------------------------------- Act 1 ----------------------------------- */

function renderCorrectionRate() {
  const b = MS.correction.biased.ai_native, c = MS.correction.corrected.ai_native;
  Plotly.newPlot('chart-correction-rate', [{
    type: 'bar', x: ['Alive only (biased)', 'Alive + dead (corrected)'],
    y: [b.rate, c.rate],
    marker: {color: ['#94a3b8', '#4f46e5']},
    error_y: {type: 'data', symmetric: false,
              array: [b.ci.hi - b.rate, c.ci.hi - c.rate],
              arrayminus: [b.rate - b.ci.lo, c.rate - c.ci.lo],
              color: '#475569', thickness: 1.4, width: 6},
    text: [b.rate + '%', c.rate + '%'], textposition: 'outside', textfont: mono,
    customdata: [[b.n, b.ci.lo, b.ci.hi], [c.n, c.ci.lo, c.ci.hi]],
    hovertemplate: '%{x}<br>%{y:.1f}% AI-native (95% CI %{customdata[1]}-%{customdata[2]})' +
                   '<br>n=%{customdata[0]:,}<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'AI-native rate', font: titleFont}, ticksuffix: '%'},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 11}},
    margin: {l: 60, r: 16, t: 24, b: 44},
  }), cfg);
}

function renderComposition() {
  const views = [['biased', 'Alive only (biased)'], ['corrected', 'Alive + dead (corrected)']];
  const traces = SUBCLASS_ORDER.map(s => ({
    type: 'bar', orientation: 'h', name: s,
    y: views.map(v => v[1]),
    x: views.map(v => MS.correction[v[0]].subclass.share[s]),
    marker: {color: SUBCLASS_COLORS[s]},
    hovertemplate: s + ': %{x:.1f}%<extra>%{y}</extra>',
  }));
  Plotly.newPlot('chart-composition', traces, layout({
    barmode: 'stack',
    xaxis: {title: {text: 'Share of universe', font: titleFont}, ticksuffix: '%', range: [0, 100]},
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}},
    legend: {orientation: 'h', y: 1.25, x: 0.5, xanchor: 'center',
             font: {family: 'JetBrains Mono', size: 9}},
    margin: {l: 170, r: 16, t: 52, b: 44},
  }), cfg);
}

function renderShareShift() {
  const rows = SUBCLASS_ORDER.map(s => ({
    s: s,
    d: Math.round((MS.correction.corrected.subclass.share[s] - MS.correction.biased.subclass.share[s]) * 100) / 100,
  })).reverse();
  Plotly.newPlot('chart-share-shift', [{
    type: 'bar', orientation: 'h',
    y: rows.map(r => SUBCLASS_LABELS[r.s]), x: rows.map(r => r.d),
    marker: {color: rows.map(r => r.d >= 0 ? ALIVE_COLOR : DEAD_COLOR)},
    text: rows.map(r => (r.d >= 0 ? '+' : '') + r.d.toFixed(2) + ' pp'),
    textposition: 'outside', textfont: mono,
    hovertemplate: '%{y}<br>%{x:+.2f} pp after correction<extra></extra>',
  }], layout({
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 10}},
    xaxis: {title: {text: 'Change in ecosystem share (percentage points)', font: titleFont}},
    margin: {l: 200, r: 70, t: 16, b: 44},
  }), cfg);
}

/* ------------------------------- Act 2 ----------------------------------- */

function renderSubclassSurvival() {
  const t = MS.subclass_tests;
  const sigSet = new Set(t.available ? t.tests.filter(r => r.significant).map(r => r.subclass) : []);
  const surv = MS.ai_vs_survival.subclass_survivor.share;
  const dead = MS.ai_vs_survival.subclass_dead.share;
  const annos = SUBCLASS_ORDER.filter(s => sigSet.has(s)).map(s => ({
    x: s, y: Math.max(surv[s], dead[s]), yshift: 14, text: '<b>*</b>',
    showarrow: false, font: {size: 15, color: '#1e2a4a'},
  }));
  grouped('chart-subclass-survival', SUBCLASS_ORDER,
    SUBCLASS_ORDER.map(s => surv[s]), SUBCLASS_ORDER.map(s => dead[s]),
    'Alive', 'Dead', ALIVE_COLOR, DEAD_COLOR, 'Share of cohort (%)', annos);
}

function renderLift() {
  const r = MS.ai_vs_survival.subclass_lift.filter(d => d.lift !== null);
  Plotly.newPlot('chart-lift', [{
    type: 'bar', x: r.map(d => d.subclass), y: r.map(d => d.lift),
    marker: {color: r.map(d => d.lift >= 1 ? DEAD_COLOR : ALIVE_COLOR)},
    text: r.map(d => d.lift.toFixed(2) + 'x'), textposition: 'outside', textfont: mono,
    hovertemplate: '%{x}<br>Dead share %{customdata[0]}% vs alive %{customdata[1]}%<extra></extra>',
    customdata: r.map(d => [d.dead, d.survivor]),
  }], layout({
    yaxis: {title: {text: 'Over-representation among the dead', font: titleFont}},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 11}},
    shapes: [{type: 'line', x0: -0.5, x1: r.length - 0.5, y0: 1, y1: 1,
              line: {color: '#475569', width: 1, dash: 'dot'}}],
    margin: {l: 60, r: 16, t: 16, b: 44},
  }), cfg);
}

function renderEraMortality() {
  const rows = MS.rad_survival.by_era;
  Plotly.newPlot('chart-mort-era', [{
    type: 'bar', x: rows.map(r => r.label), y: rows.map(r => r.mortality),
    marker: {color: rows.map(r => COHORT_COLORS[r.label] || '#475569')},
    text: rows.map(r => r.mortality.toFixed(1) + '%  (n=' + r.n.toLocaleString() + ')'),
    textposition: 'outside', textfont: mono,
    hovertemplate: '%{x}<br>Mortality %{y:.1f}%<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'Mortality rate', font: titleFont}, ticksuffix: '%'},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 12}},
    margin: {l: 60, r: 16, t: 24, b: 44},
  }), cfg);
}

function renderFundingShare() {
  const rows = MS.funding_survival.shares;
  grouped('chart-funding-share', rows.map(r => r.bucket),
    rows.map(r => r.survivor_share), rows.map(r => r.dead_share),
    'Alive', 'Dead', ALIVE_COLOR, DEAD_COLOR, 'Share of cohort (%)');
}

function renderFundingMort() {
  mortBars('chart-funding-mort',
    MS.funding_survival.mortality.map(r => ({label: r.label, mortality: r.mortality, n: r.n})), null);
}

/* ------------------------------- Act 3 ----------------------------------- */

function renderTrap() {
  const grid = MS.dependency_trap;
  const z = grid.cells.map(row => row.map(c => c.mortality));
  const txt = grid.cells.map(row => row.map(c =>
    c.mortality === null ? '' : c.mortality.toFixed(0) + '%<br>n=' + c.n));
  Plotly.newPlot('chart-trap', [{
    type: 'heatmap', x: grid.rad_order, y: grid.funding_order, z: z,
    text: txt, texttemplate: '%{text}', textfont: {family: 'JetBrains Mono', size: 11},
    colorscale: [[0, '#ecfdf5'], [0.5, '#fde68a'], [1, '#e11d48']],
    hovertemplate: 'Funding %{y}<br>%{x}<br>Mortality %{z:.1f}%<extra></extra>',
    colorbar: {title: {text: 'Mortality %', font: {size: 10}}, thickness: 12, len: 0.9},
    xgap: 3, ygap: 3,
  }], layout({
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 11}, side: 'top'},
    yaxis: {tickfont: {family: 'JetBrains Mono', size: 11}, autorange: 'reversed'},
    margin: {l: 80, r: 16, t: 36, b: 16},
  }), cfg);
}

function renderTemporal() {
  const t = MS.temporal;
  const shapes = t.releases.filter(r => t.months.includes(r.month)).map(r => ({
    type: 'line', x0: r.month, x1: r.month, y0: 0, y1: 1, yref: 'paper',
    line: {color: '#475569', width: 1, dash: 'dot'},
  }));
  const annos = t.releases.filter(r => t.months.includes(r.month)).map(r => ({
    x: r.month, y: 1, yref: 'paper', yanchor: 'bottom', text: r.label, showarrow: false,
    font: {size: 9, color: '#475569', family: 'JetBrains Mono'},
  }));
  Plotly.newPlot('chart-temporal', [
    {type: 'bar', name: 'All dead', x: t.months, y: t.total, marker: {color: '#cbd5e1'},
     hovertemplate: '%{x}<br>%{y} deaths<extra></extra>'},
    {type: 'scatter', mode: 'lines', name: 'Commoditizable (1C,1G)', x: t.months, y: t.commoditizable,
     line: {color: DEAD_COLOR, width: 2}},
    {type: 'scatter', mode: 'lines', name: 'Defensible (1A,1B,1E)', x: t.months, y: t.defensible,
     line: {color: ALIVE_COLOR, width: 2}},
  ], layout({
    yaxis: {title: {text: 'Last-seen (death) count', font: titleFont}},
    xaxis: {tickfont: {size: 9}, nticks: 16},
    legend: {orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center', font: {size: 10}},
    shapes: shapes, annotations: annos,
    margin: {l: 60, r: 24, t: 44, b: 44},
  }), cfg);
}

/* ------------------------------- Act 4 ----------------------------------- */

function renderAiNativeConfHist(id, block) {
  if (!block || !block.n) {
    document.getElementById(id).innerHTML =
      '<p style="padding:1rem;color:#8a8a8a;font-size:0.8rem;">No confidence values.</p>';
    return;
  }
  Plotly.newPlot(id, [{
    type: 'bar', x: block.bins, y: block.counts,
    marker: {color: '#4f46e5'},
    text: block.counts.map(v => v.toLocaleString()), textposition: 'outside', textfont: mono,
    hovertemplate: '%{x}<br>%{y:,} companies<extra></extra>',
  }], layout({
    xaxis: {title: {text: 'ai_native_confidence bin', font: titleFont}, tickangle: -20,
            tickfont: {family: 'JetBrains Mono', size: 10}},
    yaxis: {title: {text: 'Count', font: titleFont}},
    margin: {l: 70, r: 24, t: 16, b: 70},
  }), cfg);
}

function renderConfSurvival() {
  const a = MS.v2_confidence.ai_native.survivor;
  const d = MS.v2_confidence.ai_native.dead;
  Plotly.newPlot('chart-conf-survival', [
    {type: 'bar', name: 'Alive', x: a.bins, y: a.counts.map((c,i) => a.n ? c/a.n*100 : 0),
     marker: {color: ALIVE_COLOR},
     hovertemplate: 'Alive<br>%{x}: %{y:.1f}%<extra></extra>'},
    {type: 'bar', name: 'Dead', x: d.bins, y: d.counts.map((c,i) => d.n ? c/d.n*100 : 0),
     marker: {color: DEAD_COLOR},
     hovertemplate: 'Dead<br>%{x}: %{y:.1f}%<extra></extra>'},
  ], layout({
    barmode: 'group',
    yaxis: {title: {text: 'Share of cohort (%)', font: titleFont}, ticksuffix: '%'},
    xaxis: {tickangle: -20, tickfont: {family: 'JetBrains Mono', size: 10}},
    legend: {orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center', font: {size: 11}},
    margin: {l: 60, r: 16, t: 36, b: 70},
  }), cfg);
}

function renderRobustnessTable() {
  const el = document.getElementById('robustness-table');
  if (!el) return;
  const thin = MS.sensitivity.by_thin_history || [];
  const age = MS.sensitivity.by_snapshot_age || [];
  const ar = MS.ai_vs_survival.ai_rate;
  const rows = [];
  rows.push(['Alive', ar.survivor.rate.toFixed(1) + '%', ar.survivor.n.toLocaleString()]);
  rows.push(['Dead (evidence)', ar.dead.rate.toFixed(1) + '%', ar.dead.n.toLocaleString()]);
  rows.push(['Dead (strict subset)', ar.dead_strict.rate.toFixed(1) + '%', ar.dead_strict.n.toLocaleString()]);
  thin.forEach(r => rows.push(['Dead / ' + r.label, r.rate.toFixed(1) + '%', r.n.toLocaleString()]));
  age.forEach(r => rows.push(['Dead / snapshot ' + r.label, r.rate.toFixed(1) + '%', r.n.toLocaleString()]));
  el.innerHTML = '<table class="rob-table"><thead><tr><th>Cut</th><th>AI-native rate</th><th>n</th></tr></thead><tbody>' +
    rows.map(r => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td><td>' + r[2] + '</td></tr>').join('') +
    '</tbody></table>';
}

function aiRateBars(id, rows, colors) {
  Plotly.newPlot(id, [{
    type: 'bar', x: rows.map(r => r.label), y: rows.map(r => r.rate),
    marker: {color: colors},
    error_y: {type: 'data', symmetric: false,
              array: rows.map(r => r.ci.hi - r.rate), arrayminus: rows.map(r => r.rate - r.ci.lo),
              color: '#475569', thickness: 1.2, width: 5},
    text: rows.map(r => r.rate.toFixed(1) + '%'), textposition: 'outside', textfont: mono,
    customdata: rows.map(r => r.n),
    hovertemplate: '%{x}<br>%{y:.1f}% AI-native<br>n=%{customdata:,}<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'AI-native rate', font: titleFont}, ticksuffix: '%'},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 10}},
    margin: {l: 60, r: 16, t: 24, b: 44},
  }), cfg);
}

function renderThin() {
  aiRateBars('chart-thin', MS.sensitivity.by_thin_history, ['#d97706', '#4f46e5']);
}

function renderAge() {
  aiRateBars('chart-age', MS.sensitivity.by_snapshot_age,
    MS.sensitivity.by_snapshot_age.map(() => '#4f46e5'));
}

function renderStrict() {
  const ar = MS.ai_vs_survival.ai_rate;
  const deadLabel = MS.meta.preview ? 'Dead (metadata preview)' : 'Dead (evidence)';
  aiRateBars('chart-strict', [
    {label: 'Alive', ...ar.survivor},
    {label: deadLabel, ...ar.dead},
    {label: 'Dead (strict subset)', ...ar.dead_strict},
  ], [ALIVE_COLOR, DEAD_COLOR, DEAD_STRICT_COLOR]);
}

function renderFunnel() {
  const st = MS.funnel.stages;
  Plotly.newPlot('chart-funnel', [{
    type: 'funnel', y: st.map(s => s.label), x: st.map(s => s.n),
    marker: {color: ['#94a3b8', '#4f46e5', DEAD_COLOR]},
    textinfo: 'value+percent initial',
    textfont: {family: 'JetBrains Mono', size: 12},
    hovertemplate: '%{y}<br>%{x:,} companies<extra></extra>',
  }], layout({
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}},
    margin: {l: 210, r: 24, t: 16, b: 24},
  }), cfg);
}

/* ------------------------------- render all ------------------------------ */

renderSubclass();
renderAiRateCohort();
renderRad();
renderHeatmap();
renderSubclassCohort();
renderRadCohort();

renderAiNativeConfHist('chart-conf-ainative', MS.v2_confidence.ai_native.all);
renderConfDist('chart-conf-subclass-dist', MS.v2_confidence.subclass.all.dist);
(function() {
  const labels = SUBCLASS_ORDER.map(s => SUBCLASS_LABELS[s]);
  const means = SUBCLASS_ORDER.map(s => MS.v2_confidence.subclass.by_subclass[s].mean);
  Plotly.newPlot('chart-conf-subclass', [{
    type: 'bar', name: 'Mean subclass_confidence', x: labels, y: means,
    marker: {color: SUBCLASS_ORDER.map(s => SUBCLASS_COLORS[s])},
    text: means.map(v => v.toFixed(1)), textposition: 'outside', textfont: mono,
    hovertemplate: '%{x}<br>Mean: %{y:.2f}<extra></extra>',
  }], layout({
    yaxis: {title: {text: 'Mean subclass confidence', font: titleFont}, range: [0, 5.5]},
    xaxis: {tickangle: -35, tickfont: {family: 'JetBrains Mono', size: 9}},
    margin: {l: 55, r: 24, t: 16, b: 120},
    shapes: [{type: 'line', x0: -0.5, x1: SUBCLASS_ORDER.length - 0.5, y0: 3, y1: 3,
              line: {color: '#d97706', width: 1, dash: 'dot'}}],
  }), cfg);
})();
renderConfDist('chart-conf-rad', MS.v2_confidence.rad.all.dist);

renderCorrectionRate();
renderShareShift();
renderSubclassSurvival();
// Merged mortality: subclass bars colored by defensibility group
(function() {
  const groupOf = { '1A':'Defensible AI','1B':'Defensible AI','1E':'Defensible AI',
    '1C':'Commoditizable AI','1G':'Commoditizable AI','1D':'Other AI','1F':'Other AI',
    '0A':'Not AI','0B':'Not AI','0C':'Not AI' };
  const groupColor = {
    'Commoditizable AI': '#e11d48', 'Other AI': '#d97706',
    'Defensible AI': '#059669', 'Not AI': '#94a3b8'
  };
  const items = MS.rad_survival.by_subclass.slice();
  mortBars('chart-mort-subclass', items, Object.fromEntries(
    items.map(d => [d.label, groupColor[groupOf[d.label]] || '#475569'])
  ));
})();
mortBars('chart-mort-rad', MS.rad_survival.by_rad,
  {'RAD-H': '#e11d48', 'RAD-M': '#d97706', 'RAD-L': '#059669'});
mortBars('chart-vertical', MS.vertical.map(v => ({label: v.group, mortality: v.mortality, n: v.n})), null);
renderFundingShare();
forest('chart-forest1', F1);
forest('chart-forest2', F2);
renderTrap();
renderTemporal();
renderConfSurvival();
renderRobustnessTable();
renderFunnel();

/* ------------------------- scroll + nav wiring --------------------------- */

const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.06, rootMargin: '0px 0px -30px 0px' });
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
}, { threshold: 0.25 });
document.querySelectorAll('section').forEach(s => navObs.observe(s));

/* --------------------------- Landscape filters --------------------------- */

(function() {
  let currentCohort = 'ALL';
  let currentMinConf = 1;

  function updateSubclassChart() {
    const fd = FILTER_DATA[currentCohort + '_' + currentMinConf];
    if (!fd) return;
    const labels = SUBCLASS_ORDER.slice().reverse();
    const vals = labels.map(s => fd.subclass[s]);
    Plotly.react('chart-subclass', [{
      type: 'bar', orientation: 'h',
      y: labels.map(s => SUBCLASS_LABELS[s]), x: vals,
      marker: {color: labels.map(s => SUBCLASS_COLORS[s])},
      text: vals.map(v => v.toLocaleString()), textposition: 'outside', textfont: mono,
      hovertemplate: '%{y}<br>%{x:,} companies<extra></extra>',
    }], layout({
      yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}},
      xaxis: {title: {text: 'Number of startups (' + fd.total.toLocaleString() + ' shown)', font: titleFont}},
      margin: {l: 200, r: 60, t: 16, b: 44},
    }), cfg);
  }

  function wirePills(groupId, setter) {
    document.querySelectorAll('#' + groupId + ' .pill-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#' + groupId + ' .pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        setter(btn.dataset.val);
        updateSubclassChart();
      });
    });
  }
  wirePills('cohort-filter', v => currentCohort = v);

  const slider = document.getElementById('conf-slider');
  const sliderVal = document.getElementById('conf-slider-val');
  slider.addEventListener('input', () => {
    currentMinConf = parseInt(slider.value);
    sliderVal.textContent = currentMinConf;
    updateSubclassChart();
  });
})();
"""


def build_html(mb: dict, ms: dict, filter_data: dict) -> str:
    meta = ms["meta"]
    preview = meta["preview"]
    corr = ms["correction"]
    biased_ai = corr["biased"]["ai_native"]["rate"]
    corrected_ai = corr["corrected"]["ai_native"]["rate"]
    delta = round(corrected_ai - biased_ai, 1)
    ar = ms["ai_vs_survival"]["ai_rate"]
    rad = {r["label"]: r for r in ms["rad_survival"]["by_rad"]}
    radh = rad.get("RAD-H", {}).get("mortality")
    radl = rad.get("RAD-L", {}).get("mortality")
    tests = ms["subclass_tests"]
    sig_subclasses = ([t["subclass"] for t in tests.get("tests", []) if t.get("significant")]
                      if tests.get("available") else [])
    funnel = ms["funnel"]
    n_not_found = next(
        (s["n"] for s in funnel.get("stages", []) if s["label"] == "Not found by Tavily"),
        0,
    )
    today = datetime.date.today().strftime("%b %d, %Y")

    v2c = ms.get("v2_confidence") or {}
    mean_ai_conf = (v2c.get("ai_native") or {}).get("all", {}).get("mean")
    if mean_ai_conf is None:
        conf_dist = mb["conf_class_dist"]
        mean_ai_conf = sum(k * v for k, v in conf_dist.items()) / max(sum(conf_dist.values()), 1)
    mean_conf = mean_ai_conf

    f1 = _forest(ms["regression"]["model1"])
    f2 = _forest(ms["regression"]["model2"])
    f3 = _forest(ms["regression"].get("model3", {}))

    script = (SCRIPT_TEMPLATE
              .replace("__BASE_JSON__", json.dumps(mb, default=str))
              .replace("__SURV_JSON__", json.dumps(ms, default=str))
              .replace("__FILTER_JSON__", json.dumps(filter_data))
              .replace("__FOREST1__", json.dumps(f1))
              .replace("__FOREST2__", json.dumps(f2))
              .replace("__FOREST3__", json.dumps(f3))
              .replace("__SUBCLASS_ORDER__", json.dumps(SUBCLASS_ORDER))
              .replace("__SUBCLASS_LABELS__", json.dumps(SUBCLASS_LABELS))
              .replace("__SUBCLASS_COLORS__", json.dumps(SUBCLASS_COLORS))
              .replace("__RAD_ORDER__", json.dumps(RAD_ORDER))
              .replace("__RAD_COLORS__", json.dumps(RAD_COLORS))
              .replace("__COHORT_COLORS__", json.dumps(COHORT_COLORS))
              .replace("__RAD_NA_CLASSES__", json.dumps(RAD_NA_CLASSES))
              .replace("__ALIVE_COLOR__", ALIVE_COLOR)
              .replace("__DEAD_COLOR__", DEAD_COLOR)
              .replace("__DEAD_STRICT_COLOR__", DEAD_STRICT_COLOR))

    preview_banner = ""
    if meta.get("partial"):
        done = meta.get("n_completed") or meta.get("n_universe")
        expected = meta.get("n_expected") or "?"
        pct = (100.0 * done / expected) if isinstance(expected, int) and expected else None
        pct_txt = f"{pct:.1f}%" if pct is not None else "partial"
        preview_banner = (
            f'<div class="preview-banner">PARTIAL V2 RUN: {done:,} of {expected:,} companies '
            f'classified ({pct_txt}). Numbers refresh when you re-run this builder on the final CSV.</div>'
        )
    elif preview:
        preview_banner = (
            '<div class="preview-banner">PREVIEW BUILD: dead-cohort verdicts are still '
            'metadata-only stand-ins. Re-run once evidence-based dead classifications land.</div>'
        )

    flips = ms["flips"]
    if flips.get("available"):
        flip_line = (
            f'Adding recovered evidence flipped the AI-native verdict for '
            f'<strong>{flips.get("ai_native", {}).get("pct", 0)}%</strong> of the dead cohort, '
            f'the subclass for {flips.get("subclass", {}).get("pct", 0)}%, and the RAD score for '
            f'{flips.get("rad_score", {}).get("pct", 0)}%. Metadata alone is not enough.'
        )
    else:
        flip_line = ("Flip analysis runs once the evidence-based verdicts land: it will quantify "
                     "how often recovered evidence overturns a metadata-only label.")

    if tests.get("available"):
        chi_p = tests["chi2"]["pvalue"]
        if sig_subclasses:
            sig_line = (
                f'BH-significant subclasses at alpha {tests["alpha"]}: '
                f'<strong>{", ".join(sig_subclasses)}</strong> '
                f'(omnibus chi-square p = {chi_p:.2e}).'
            )
        else:
            sig_line = (
                f'No subclass passed BH at alpha {tests["alpha"]} '
                f'(omnibus chi-square p = {chi_p:.2e}).'
            )
    else:
        sig_line = "Significance annotations unavailable for this run."

    rad_headline = ""
    rad_line = ""
    if radh is not None and radl is not None:
        rad_line = (f"RAD-H firms die at <strong>{radh}%</strong> versus "
                    f"<strong>{radl}%</strong> for RAD-L.")
        if radh > radl:
            rad_headline = "Dependency tracks death."
        elif radh < radl:
            rad_headline = "Raw RAD mortality does not track dependency."
            rad_line += (" Higher dependency is not deadlier in the uncontrolled "
                         "rates; Act 3 checks whether controls change the story.")
        else:
            rad_headline = "RAD mortality is flat in the raw rates."

    mode_label = (
        "PARTIAL" if meta.get("partial")
        else ("PREVIEW" if preview else "evidence-based")
    )

    if preview:
        overview_p = (
            f"Alive companies ({meta['n_survivor']:,}) use live website evidence. "
            f"Dead companies ({meta['n_dead']:,}) are still a PREVIEW stand-in "
            f"(metadata-only). Re-run after evidence-based dead verdicts land."
        )
    else:
        overview_p = (
            f"Every company here was classified from real website evidence by the "
            f"<strong>V2 two-pass classifier</strong>. "
            f"Alive ({meta['n_survivor']:,}): live crawl. "
            f"Dead ({meta['n_dead']:,}): pre-death archive snapshot. "
            f"Only the evidence source differs."
        )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alive vs Dead: survivorship-corrected V2 classification</title>
{PLOTLY_PLACEHOLDER}
<style>{STYLE}</style>
</head>
<body>

{preview_banner}

<nav>
  <div class="nav-brand">Alive vs Dead</div>
  <div class="nav-sub">Survivorship-corrected V2</div>
  <div class="nav-section">
    <div class="nav-label">Corrected baseline</div>
    <ul>
      <li><a href="#overview">Overview</a></li>
      <li><a href="#landscape">AI-Native Landscape</a></li>
      <li><a href="#rad">RAD Score Analysis</a></li>
      <li><a href="#cohorts">Cohort Dynamics</a></li>
      <li><a href="#confidence">Confidence Audit</a></li>
    </ul>
  </div>
  <div class="nav-section">
    <div class="nav-label">Survivorship</div>
    <ul>
      <li><a href="#act1">1. The Bias Exists</a></li>
      <li><a href="#act2">2. Who Dies</a></li>
      <li><a href="#act3">3. Why They Die</a></li>
      <li><a href="#act4">4. Robustness</a></li>
    </ul>
  </div>
</nav>

<main>

<section id="overview">
  <span class="section-label">Evidence-Only Universe</span>
  <h1>Alive vs Dead: Survivorship-Corrected V2</h1>
  <p>
    {overview_p}
  </p>

  <div class="hero-metrics">
    <div class="metric-card hl">
      <div class="mc-label">Universe</div>
      <div class="mc-val">{meta["n_universe"]:,}</div>
      <div class="mc-ctx">evidence-classified companies</div>
    </div>
    <div class="metric-card ok">
      <div class="mc-label">Alive</div>
      <div class="mc-val">{meta["n_survivor"]:,}</div>
      <div class="mc-ctx">live website evidence</div>
    </div>
    <div class="metric-card bad">
      <div class="mc-label">Dead</div>
      <div class="mc-val">{meta["n_dead"]:,}</div>
      <div class="mc-ctx">{"metadata stand-in (preview)" if preview else "pre-death archive evidence"}</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">Corrected AI-native rate</div>
      <div class="mc-val">{corrected_ai}%</div>
      <div class="mc-ctx">{"+" if delta >= 0 else ""}{delta} pts vs alive-only</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">Mean AI-native confidence</div>
      <div class="mc-val">{mean_conf:.3f}</div>
      <div class="mc-ctx">logprob P(sampled 0/1)</div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p><strong>Dead</strong> means the live site could not be extracted. It is a failure proxy, not a death certificate. Act 4 checks a stricter offline subset and lists the limits.</p>
  </div>
</section>

<section id="landscape">
  <span class="section-label">01. Landscape</span>
  <h2>Corrected AI-Native Mix</h2>
  <p>
    <strong>{mb["ai_native_pct"]}%</strong> ({mb["ai_native_count"]:,}) of
    {mb["total"]:,} companies are AI-native after adding recovered dead firms.
    Filters use Pass B <code>subclass_confidence</code> (1 to 5).
  </p>

  <div class="filter-bar">
    <label>Cohort</label>
    <div class="pill-group" id="cohort-filter">
      <button class="pill-btn active" data-val="ALL">All</button>
      <button class="pill-btn" data-val="PRE-GENAI">PRE-GENAI</button>
      <button class="pill-btn" data-val="GENAI-ERA">GENAI-ERA</button>
    </div>
    <label style="margin-left:auto;">Min subclass confidence</label>
    <div class="slider-group">
      <input type="range" id="conf-slider" min="1" max="5" value="1" step="1">
      <span class="slider-val" id="conf-slider-val">1</span>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Subclass Distribution</div>
        <div class="chart-box-desc">All 10 subclasses by count, filterable by evidence source, founding cohort, and confidence</div>
      </div>
      <div class="chart-body"><div id="chart-subclass" style="height:480px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">AI-Native Rate by Cohort</div>
        <div class="chart-box-desc">PRE-GENAI vs GENAI-ERA, corrected universe</div>
      </div>
      <div class="chart-body"><div id="chart-ai-rate-cohort" style="height:360px;"></div></div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p><strong>GENAI-ERA AI-native rate: {mb["ai_rate_by_cohort"]["GENAI-ERA"]}%.</strong>
    PRE-GENAI: {mb["ai_rate_by_cohort"]["PRE-GENAI"]}%. Post-GPT-4 founding years stay more AI-native after the dead are added.</p>
  </div>
</section>

<section id="rad">
  <span class="section-label">02. Dependency</span>
  <h2>RAD Score Analysis</h2>
  <p>
    {mb["rad_counts"]["RAD-H"] + mb["rad_counts"]["RAD-M"] + mb["rad_counts"]["RAD-L"]:,} companies in the
    corrected universe carry a meaningful RAD score (excluding RAD-NA). Most are <strong>RAD-H</strong>:
    structurally dependent on third-party GenAI APIs.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">RAD Score Distribution</div>
        <div class="chart-box-desc">Structural dependency on third-party GenAI, corrected universe</div>
      </div>
      <div class="chart-body"><div id="chart-rad" style="height:360px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Subclass &times; RAD Heatmap</div>
        <div class="chart-box-desc">Which subclasses are most API-dependent?</div>
      </div>
      <div class="chart-body"><div id="chart-heatmap" style="height:520px;"></div></div>
    </div>
  </div>
</section>

<section id="cohorts">
  <span class="section-label">03. Temporal</span>
  <h2>Cohort Dynamics</h2>
  <p>
    GPT-4 launched March 2023; that is the cohort boundary. {mb["cohort_counts"]["GENAI-ERA"]:,}
    companies ({mb["genai_era_pct"]}%) in the corrected universe were founded after it.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Subclass Distribution by Cohort</div>
        <div class="chart-box-desc">Which subclasses are new vs established?</div>
      </div>
      <div class="chart-body"><div id="chart-subclass-cohort" style="height:480px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">RAD Score by Cohort</div>
        <div class="chart-box-desc">Are GENAI-ERA startups more API-dependent?</div>
      </div>
      <div class="chart-body"><div id="chart-rad-cohort" style="height:360px;"></div></div>
    </div>
  </div>
</section>

<section id="confidence">
  <span class="section-label">04. Confidence</span>
  <h2>How Confidence Is Measured</h2>
  <p>
    Three fields, three methods. Read the explainer, then the chart. Alive-vs-dead comparison for the measured logprob field is in Act 4.
  </p>

  <div class="subsec" id="conf-ainative">
    <h3>5.1 AI-native confidence (logprob)</h3>
    <div class="explainer">
      <p>Pass A answers only <code>ai_native</code> as <code>0</code> or <code>1</code>. It does not write a confidence score.</p>
      <p>The API returns log-probabilities on the sampled tokens: how likely each generated token was, plus nearby alternatives.</p>
      <p>We find the decision digit, convert those log-probabilities into ordinary probabilities on {{0, 1}}, and take the mass on the digit the model sampled. That value is <code>ai_native_confidence</code>.</p>
      <p>If the opposing digit is missing from the alternatives, production uses a validated midpoint bound instead of pretending the opponent had zero mass.</p>
    </div>
    <div class="chart-row single">
      <div class="chart-box">
        <div class="chart-box-header">
          <div class="chart-box-title">AI-native confidence distribution</div>
          <div class="chart-box-desc">Continuous P(sampled digit) over the evidence-only universe. Mean {mean_conf:.3f}</div>
        </div>
        <div class="chart-body"><div id="chart-conf-ainative" style="height:380px;"></div></div>
      </div>
    </div>
  </div>

  <div class="subsec" id="conf-subclass">
    <h3>5.2 Subclass confidence (prompt 1 to 5)</h3>
    <div class="explainer">
      <p>Pass B writes this score. It is not extracted from token probabilities. The prompt scale:</p>
      <ol>
        <li><strong>5:</strong> Substantial, consistent evidence. Decision is unambiguous.</li>
        <li><strong>4:</strong> Strong evidence, one plausible adjacent choice.</li>
        <li><strong>3:</strong> Incomplete evidence. Judgment between alternatives.</li>
        <li><strong>2:</strong> Thin, stale, or contradictory evidence.</li>
        <li><strong>1:</strong> Insufficient evidence. Family fallback rule required.</li>
      </ol>
      <p>Critiques must explain any score below 3.</p>
    </div>
    <div class="chart-row">
      <div class="chart-box">
        <div class="chart-box-header">
          <div class="chart-box-title">Subclass confidence distribution</div>
          <div class="chart-box-desc">subclass_confidence values 1 to 5</div>
        </div>
        <div class="chart-body"><div id="chart-conf-subclass-dist" style="height:380px;"></div></div>
      </div>
      <div class="chart-box">
        <div class="chart-box-header">
          <div class="chart-box-title">Mean subclass confidence by subclass</div>
          <div class="chart-box-desc">Which genres are hardest to place?</div>
        </div>
        <div class="chart-body"><div id="chart-conf-subclass" style="height:380px;"></div></div>
      </div>
    </div>
  </div>

  <div class="subsec" id="conf-rad">
    <h3>5.3 RAD confidence (prompt 1 to 5)</h3>
    <div class="explainer">
      <p>Same Pass B self-report scale as subclass confidence. AI-native rows emit <code>rad_confidence</code> with <code>rad_score</code>. Not-AI-native rows are <code>RAD-NA</code> with a blank confidence. Blanks are excluded from the chart denominator.</p>
    </div>
    <div class="chart-row single">
      <div class="chart-box">
        <div class="chart-box-header">
          <div class="chart-box-title">RAD confidence distribution</div>
          <div class="chart-box-desc">Non-blank rad_confidence only (RAD-NA rows excluded)</div>
        </div>
        <div class="chart-body"><div id="chart-conf-rad" style="height:380px;"></div></div>
      </div>
    </div>
  </div>
</section>

<section id="act1">
  <span class="section-label">Survivorship &middot; Act 1</span>
  <h2>The Bias Exists</h2>
  <p>
    Live-site samples miss the dead. If the dead differ from the living, every headline inherits that bias. Measure it: AI-native rate and subclass share, with and without recovered dead firms.
  </p>

  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Headline correction: AI-native rate</div>
        <div class="chart-box-desc">Biased = evidence-based alive only. Corrected = alive + evidence-based dead. Wilson 95% CIs</div>
      </div>
      <div class="chart-body"><div id="chart-correction-rate" style="height:320px;"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Subclass share shift after correction</div>
        <div class="chart-box-desc">Corrected share minus biased share (percentage points)</div>
      </div>
      <div class="chart-body"><div id="chart-share-shift" style="height:320px;"></div></div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p><strong>The survivor lens misstates the AI-native rate by {abs(delta)} points</strong>
    ({biased_ai}% biased vs {corrected_ai}% corrected). {flip_line}</p>
  </div>
</section>

<section id="act2">
  <span class="section-label">Survivorship &middot; Act 2</span>
  <h2>Who Dies</h2>
  <p>
    Act 1's composition shift has a source: some genres die more. Compare alive vs dead by subclass, RAD, vertical, and funding.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Subclass distribution: alive vs dead</div>
        <div class="chart-box-desc">Share of each cohort. Stars mark BH-adjusted significant differences (alpha 0.05)</div>
      </div>
      <div class="chart-body"><div id="chart-subclass-survival" style="height:380px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Mortality by subclass</div>
        <div class="chart-box-desc">Dead / (dead + alive). Bar color = defensibility group (commoditizable / other / defensible / not AI)</div>
      </div>
      <div class="chart-body"><div id="chart-mort-subclass" style="height:380px;"></div></div>
    </div>
  </div>

  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Mortality by RAD score</div>
        <div class="chart-box-desc">AI-native firms only</div>
      </div>
      <div class="chart-body"><div id="chart-mort-rad" style="height:260px;"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">AI-native mortality by vertical</div>
        <div class="chart-box-desc">Top category groups by n. Descriptive only</div>
      </div>
      <div class="chart-body"><div id="chart-vertical" style="height:260px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Funding distribution by survival status</div>
        <div class="chart-box-desc">Share of each cohort per funding bucket. Unknown kept explicit</div>
      </div>
      <div class="chart-body"><div id="chart-funding-share" style="height:320px;"></div></div>
    </div>
  </div>

  <div class="insight insight-green">
    <p>{f"<strong>{rad_headline}</strong> {rad_line}" if rad_headline else rad_line}
    AI-native rate among the dead: {ar["dead"]["rate"]}% vs {ar["survivor"]["rate"]}% among the living.
    {sig_line}</p>
  </div>
</section>

<section id="act3">
  <span class="section-label">Survivorship &middot; Act 3</span>
  <h2>Why They Die</h2>
  <p>
    Raw mortality can hide confounders. Two logistic models predict death. Markers are odds ratios with 95% CIs. Red = risk, green = protective, gray = not significant. Dotted line at 1.0 = no effect.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Model 1: death across the full universe</div>
        <div class="chart-box-desc">Outcome death=1. Predictors: AI-native, log funding, founding era, top verticals</div>
      </div>
      <div class="chart-body"><div id="chart-forest1" style="height:340px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Model 2: death among AI-native firms</div>
        <div class="chart-box-desc">RAD level (vs RAD-L) and defensibility group (vs commoditizable), controlling for funding and era</div>
      </div>
      <div class="chart-body"><div id="chart-forest2" style="height:340px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Funding &times; RAD mortality</div>
        <div class="chart-box-desc">Darker red is deadlier. Cell labels show mortality and n</div>
      </div>
      <div class="chart-body"><div id="chart-trap" style="height:360px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Deaths over time vs model releases</div>
        <div class="chart-box-desc">Last Wayback capture as the death anchor; exploratory and bounded by archive coverage</div>
      </div>
      <div class="chart-body"><div id="chart-temporal" style="height:360px;"></div></div>
    </div>
  </div>
</section>

<section id="act4">
  <span class="section-label">Survivorship &middot; Act 4</span>
  <h2>Robustness and Limits</h2>
  <p>
    Stress-test the headline: logprob confidence by cohort, sensitivity cuts, and who is still missing from the archive.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">AI-native confidence: alive vs dead</div>
        <div class="chart-box-desc">Logprob P(sampled digit). Caveat: dead evidence is one archived homepage vs a multi-page live crawl</div>
      </div>
      <div class="chart-body"><div id="chart-conf-survival" style="height:320px;"></div></div>
    </div>
  </div>

  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Robustness cuts</div>
        <div class="chart-box-desc">AI-native rate under thin history, snapshot age, and strict-dead definitions</div>
      </div>
      <div class="chart-body" style="padding:1rem 1.25rem;"><div id="robustness-table"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Coverage funnel</div>
        <div class="chart-box-desc">From not-found companies to the evidence-classified dead cohort</div>
      </div>
      <div class="chart-body"><div id="chart-funnel" style="height:300px;"></div></div>
    </div>
  </div>

</section>

</main>

<footer>
  <strong>Classifier:</strong> V2 two-pass (identical on both cohorts) &nbsp;&middot;&nbsp;
  <strong>Universe:</strong> evidence-only ({meta["n_universe"]:,} companies) &nbsp;&middot;&nbsp;
  <strong>Source:</strong> {meta.get("v2_csv", "V2 professor CSV")} &nbsp;&middot;&nbsp;
  <strong>Generated:</strong> {today}
  <br>
    Rebuild with build_v2_alive_dead_dashboard.py.
</footer>

<script>
{script}
</script>

</body>
</html>'''


DEFAULT_V2_CSV = (
    _PROJECT_ROOT / "outputs" / "two_pass_classifier" / "production_classifications.csv"
)


def _plotly_inline_script() -> str:
    """Vendored Plotly, escaped so a literal </script> cannot break the page."""
    if not PLOTLY_VENDOR.is_file():
        raise FileNotFoundError(
            f"Missing vendored Plotly at {PLOTLY_VENDOR}. "
            "Restore data visualization/02_Analysis_Code/vendor/plotly-2.35.2.min.js"
        )
    body = PLOTLY_VENDOR.read_text(encoding="utf-8").replace("</", "<\\/")
    return f"<script>{body}</script>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-csv",
        type=Path,
        default=DEFAULT_V2_CSV,
        help="V2 professor CSV (default: outputs/two_pass_classifier/production_classifications.csv)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Optional manifest size. If set and CSV is smaller, shows a PARTIAL banner.",
    )
    parser.add_argument("--production", type=Path, default=None,
                        help="V1 production CSV used only for Act 1 flip contrast.")
    parser.add_argument("--master", type=Path, default=None)
    parser.add_argument("--classifier-input", type=Path, default=None)
    parser.add_argument("--targets-dead", type=Path, default=None)
    parser.add_argument("--not-found", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    p = _ANALYSIS.Paths()
    for attr, val in [("production", args.production),
                      ("master", args.master), ("classifier_input", args.classifier_input),
                      ("targets_dead", args.targets_dead), ("not_found", args.not_found)]:
        if val is not None:
            setattr(p, attr, val)

    df, meta = _ANALYSIS.load_v2_frame(
        p, args.v2_csv, expected_rows=args.expected_rows,
    )
    mode = "PARTIAL" if meta.get("partial") else "COMPLETE"
    print(
        f"[{mode}] v2={args.v2_csv} universe={meta['n_universe']:,} "
        f"alive={meta['n_survivor']:,} dead={meta['n_dead']:,} "
        f"strict={meta['n_dead_strict']:,} completed={meta['n_completed']:,}"
        + (f"/{meta['n_expected']:,}" if meta.get("n_expected") else ""),
        file=sys.stderr,
    )

    t = typed_frame(df)
    universe = t[t["is_survivor"] | t["is_dead"]]
    base_metrics = _BASE.compute_metrics(universe)
    filter_data = build_filter_data(universe)
    surv_metrics = _ANALYSIS.compute_metrics(df, meta, p)
    if "v2_confidence" not in surv_metrics:
        raise SystemExit("V2 confidence metrics missing; check load_v2_frame columns.")

    page = build_html(base_metrics, surv_metrics, filter_data)
    if PLOTLY_PLACEHOLDER not in page:
        raise SystemExit("Plotly placeholder missing from HTML template.")
    page = page.replace(PLOTLY_PLACEHOLDER, _plotly_inline_script(), 1)
    # Refuse CDN <script>/<link> tags. Vendored Plotly itself contains URL
    # strings, so a raw "https://" scan is too blunt.
    if re.search(
        r"""<(?:script[^>]+src|link[^>]+href)\s*=\s*["']https?://""",
        page,
        flags=re.IGNORECASE,
    ):
        raise SystemExit("HTML still loads external script/link URLs; refuse to write.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"Dashboard written to {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1024:.0f} KB (self-contained)")


if __name__ == "__main__":
    main()
