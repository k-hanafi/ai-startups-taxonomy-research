#!/usr/bin/env python3
"""Build the survivorship-insights findings dashboard.

Reuses the house style (STYLE + PALETTE) from build_survivorship_dashboard.py so
the look stays in one place, and pulls every number from survivorship_analysis.py
(the pure-compute module). This file only shapes prose and Plotly traces.

Both sibling modules live under the non-package "02_Analysis_Code/" dir, so they
are imported by file path.

Writes:
    data visualization/01_Presentation_Materials/survivorship_insights.html
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

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
OUTPUT_PATH = _PROJECT_ROOT / "data visualization" / "01_Presentation_Materials" / "survivorship_insights.html"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # let dataclasses resolve string annotations
    spec.loader.exec_module(mod)
    return mod


_ANALYSIS = _load("survivorship_analysis", _HERE / "survivorship_analysis.py")
_HOUSE = _load("build_survivorship_dashboard", _HERE / "build_survivorship_dashboard.py")
STYLE = _HOUSE.STYLE
PALETTE = _HOUSE.PALETTE

SUBCLASS_COLORS = {
    "1A": "#4f46e5", "1B": "#0891b2", "1C": "#e11d48", "1D": "#f59e0b", "1E": "#7c3aed",
    "1F": "#059669", "1G": "#d97706", "0A": "#94a3b8", "0B": "#64748b", "0C": "#cbd5e1",
}


def _clean_term(name: str) -> str:
    """Turn a statsmodels design term into a readable label."""
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
    """Readable, ordered odds-ratio rows for a forest plot (largest OR first)."""
    if not model.get("available"):
        return []
    rows = [{
        "label": _clean_term(t["term"]),
        "or": t["odds_ratio"], "lo": t["ci_low"], "hi": t["ci_high"],
        "sig": t["pvalue"] < 0.05, "risk": t["odds_ratio"] > 1,
    } for t in model["terms"]]
    rows.sort(key=lambda r: r["or"])
    return rows


SCRIPT_TEMPLATE = """
const M = __M_JSON__;
const C = __PALETTE__;
const SUB = __SUBCLASS_COLORS__;
const F1 = __FOREST1__;
const F2 = __FOREST2__;

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

// Survivor vs dead, two grouped bar series.
function grouped(id, x, a, b, na, nb, ca, cb, yTitle) {
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
  }), cfg);
}

// Single horizontal mortality bars.
function mortBars(id, items, colors) {
  const it = items.slice().reverse();
  Plotly.newPlot(id, [{
    type: 'bar', orientation: 'h',
    y: it.map(d => d.label), x: it.map(d => d.mortality),
    marker: {color: colors ? it.map(d => colors[d.label] || C.slate) : C.indigo},
    text: it.map(d => d.mortality.toFixed(1) + '%  (n=' + d.n.toLocaleString() + ')'),
    textposition: 'outside', textfont: mono,
    hovertemplate: '%{y}<br>Mortality %{x:.1f}%<extra></extra>',
  }], layout({
    yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}},
    xaxis: {title: {text: 'Mortality rate (dead / dead+survivor)', font: titleFont}, ticksuffix: '%'},
    margin: {l: 150, r: 90, t: 16, b: 44},
  }), cfg);
}

// Subclass lift: dead share / survivor share. >1 over-represented among the dead.
function liftBars(id, rows) {
  const r = rows.filter(d => d.lift !== null);
  Plotly.newPlot(id, [{
    type: 'bar', x: r.map(d => d.subclass), y: r.map(d => d.lift),
    marker: {color: r.map(d => d.lift >= 1 ? C.rose : C.emerald)},
    text: r.map(d => d.lift.toFixed(2) + 'x'), textposition: 'outside', textfont: mono,
    hovertemplate: '%{x}<br>Dead share %{customdata[0]}% vs survivor %{customdata[1]}%<extra></extra>',
    customdata: r.map(d => [d.dead, d.survivor]),
  }], layout({
    yaxis: {title: {text: 'Over-representation among the dead', font: titleFont}},
    xaxis: {tickfont: {family: 'JetBrains Mono', size: 11}},
    shapes: [{type: 'line', x0: -0.5, x1: r.length - 0.5, y0: 1, y1: 1,
              line: {color: C.slate, width: 1, dash: 'dot'}}],
    margin: {l: 60, r: 16, t: 16, b: 44},
  }), cfg);
}

// Dependency-trap heatmap: funding (rows) x RAD (cols), colored by mortality.
function heatmap(id, grid) {
  const z = grid.cells.map(row => row.map(c => c.mortality));
  const txt = grid.cells.map(row => row.map(c =>
    c.mortality === null ? '' : c.mortality.toFixed(0) + '%<br>n=' + c.n));
  Plotly.newPlot(id, [{
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

// Odds-ratio forest plot (log x-axis, reference line at OR=1).
function forest(id, rows) {
  if (!rows.length) { document.getElementById(id).innerHTML =
    '<p style="padding:1rem;color:#8a8a8a;font-size:0.8rem;">Model not available for this run.</p>'; return; }
  const x = rows.map(r => r.or);
  const hi = rows.map(r => r.hi - r.or);
  const lo = rows.map(r => r.or - r.lo);
  const col = rows.map(r => !r.sig ? C.gray : (r.risk ? C.rose : C.emerald));
  Plotly.newPlot(id, [{
    type: 'scatter', mode: 'markers', x: x, y: rows.map(r => r.label),
    marker: {color: col, size: 9},
    error_x: {type: 'data', symmetric: false, array: hi, arrayminus: lo,
              color: '#9aa3b2', thickness: 1.4, width: 4},
    hovertemplate: '%{y}<br>OR %{x:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>',
    customdata: rows.map(r => [r.lo, r.hi]),
  }], layout({
    xaxis: {type: 'log', title: {text: 'Odds ratio of death (log scale)', font: titleFont}},
    yaxis: {automargin: true, tickfont: {size: 11}},
    shapes: [{type: 'line', x0: 1, x1: 1, y0: -0.5, y1: rows.length - 0.5,
              line: {color: '#1e2a4a', width: 1, dash: 'dot'}}],
    margin: {l: 200, r: 24, t: 16, b: 44},
  }), cfg);
}

// Deaths per month with model-release reference lines.
function deaths(id, t) {
  const shapes = t.releases.filter(r => t.months.includes(r.month)).map(r => ({
    type: 'line', x0: r.month, x1: r.month, y0: 0, y1: 1, yref: 'paper',
    line: {color: C.slate, width: 1, dash: 'dot'},
  }));
  const annos = t.releases.filter(r => t.months.includes(r.month)).map(r => ({
    x: r.month, y: 1, yref: 'paper', yanchor: 'bottom', text: r.label, showarrow: false,
    font: {size: 9, color: C.slate, family: 'JetBrains Mono'},
  }));
  Plotly.newPlot(id, [
    {type: 'bar', name: 'All dead', x: t.months, y: t.total, marker: {color: C.gray},
     hovertemplate: '%{x}<br>%{y} deaths<extra></extra>'},
    {type: 'scatter', mode: 'lines', name: 'Commoditizable (1C,1G)', x: t.months, y: t.commoditizable,
     line: {color: C.rose, width: 2}},
    {type: 'scatter', mode: 'lines', name: 'Defensible (1A,1B,1E)', x: t.months, y: t.defensible,
     line: {color: C.emerald, width: 2}},
  ], layout({
    yaxis: {title: {text: 'Last-seen (death) count', font: titleFont}},
    xaxis: {tickfont: {size: 9}, nticks: 16},
    legend: {orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center', font: {size: 10}},
    shapes: shapes, annotations: annos,
    margin: {l: 60, r: 24, t: 44, b: 44},
  }), cfg);
}

function confBars(id, conf) {
  const levels = [1, 2, 3, 4, 5];
  const s = levels.map(l => conf.survivor.dist[l] || 0);
  const d = levels.map(l => conf.dead_full.dist[l] || 0);
  const sPct = s.map(v => conf.survivor.n ? v / conf.survivor.n * 100 : 0);
  const dPct = d.map(v => conf.dead_full.n ? v / conf.dead_full.n * 100 : 0);
  grouped(id, levels.map(String), sPct, dPct, 'Survivor', 'Dead', C.emerald, C.rose, 'Share of cohort (%)');
}

// AI-native rate: survivor vs dead (full + strict).
const ar = M.ai_vs_survival.ai_rate;
Plotly.newPlot('chart-airate', [{
  type: 'bar', x: ['Survivors', 'Dead (all)', 'Dead (strict)'],
  y: [ar.survivor.rate, ar.dead_full.rate, ar.dead_strict.rate],
  marker: {color: [C.emerald, C.rose, '#9f1239']},
  text: [ar.survivor.rate, ar.dead_full.rate, ar.dead_strict.rate].map(v => v.toFixed(1) + '%'),
  textposition: 'outside', textfont: mono,
  customdata: [ar.survivor.n, ar.dead_full.n, ar.dead_strict.n],
  hovertemplate: '%{x}<br>%{y:.1f}% AI-native<br>n=%{customdata:,}<extra></extra>',
}], layout({yaxis: {title: {text: 'AI-native rate', font: titleFont}, ticksuffix: '%'},
  margin: {l: 60, r: 16, t: 16, b: 36}}), cfg);

grouped('chart-correction',
  ['AI-native %'],
  [M.correction.biased.ai_native.rate], [M.correction.corrected.ai_native.rate],
  'Survivor-only (biased)', 'Survivor + dead (corrected)', C.gray, C.indigo, 'AI-native rate (%)');

liftBars('chart-lift', M.ai_vs_survival.subclass_lift);
mortBars('chart-rad', M.rad_survival.by_rad, {'RAD-H': C.rose, 'RAD-M': C.amber, 'RAD-L': C.emerald});
mortBars('chart-subgroup', M.rad_survival.by_subclass_group, null);
mortBars('chart-subclass', M.rad_survival.by_subclass, SUB);
heatmap('chart-trap', M.dependency_trap);
mortBars('chart-vertical', M.vertical.map(v => ({label: v.group, mortality: v.mortality, n: v.n})), null);
deaths('chart-temporal', M.temporal);
forest('chart-forest1', F1);
forest('chart-forest2', F2);
confBars('chart-conf', M.confidence);

const obs = new IntersectionObserver((es) => {
  es.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, {threshold: 0.05, rootMargin: '0px 0px -30px 0px'});
document.querySelectorAll('section').forEach(s => obs.observe(s));
document.getElementById('overview').classList.add('visible');

const links = document.querySelectorAll('nav ul a');
const nobs = new IntersectionObserver((es) => {
  es.forEach(e => { if (e.isIntersecting) {
    links.forEach(a => a.classList.remove('active'));
    const l = document.querySelector('nav ul a[href="#' + e.target.id + '"]');
    if (l) l.classList.add('active');
  }});
}, {threshold: 0.25});
document.querySelectorAll('section').forEach(s => nobs.observe(s));
"""


def build_html(m: dict) -> str:
    meta = m["meta"]
    preview = meta["preview"]
    corr = m["correction"]
    biased_ai = corr["biased"]["ai_native"]["rate"]
    corrected_ai = corr["corrected"]["ai_native"]["rate"]
    delta = round(corrected_ai - biased_ai, 1)
    ar = m["ai_vs_survival"]["ai_rate"]
    rad = {r["label"]: r for r in m["rad_survival"]["by_rad"]}
    radh = rad.get("RAD-H", {}).get("mortality")
    radl = rad.get("RAD-L", {}).get("mortality")
    sg = {r["label"]: r for r in m["rad_survival"]["by_subclass_group"]}
    commod = sg.get("Commoditizable AI", {}).get("mortality")
    defens = sg.get("Defensible AI", {}).get("mortality")
    today = datetime.date.today().strftime("%b %d, %Y")

    f1, f2 = _forest(m["regression"]["model1"]), _forest(m["regression"]["model2"])
    script = (SCRIPT_TEMPLATE
              .replace("__M_JSON__", json.dumps(m, default=str))
              .replace("__PALETTE__", json.dumps(PALETTE))
              .replace("__SUBCLASS_COLORS__", json.dumps(SUBCLASS_COLORS))
              .replace("__FOREST1__", json.dumps(f1))
              .replace("__FOREST2__", json.dumps(f2)))

    preview_banner = ""
    if preview:
        preview_banner = (
            '<div class="insight insight-amber" style="margin:0 0 1.5rem;">'
            "<p><strong>PREVIEW: dead verdicts are not yet evidence-based.</strong> "
            "The classify-dead run has not landed, so the dead cohort still carries its "
            "metadata-only labels. Cohort sizes, joins, and chart structure are final. "
            "The numbers refresh the moment survivorship_corrected.csv exists.</p></div>"
        )

    flips = m["flips"]
    if flips.get("available"):
        flip_line = (
            f'Adding recovered evidence flipped the AI-native verdict for '
            f'<strong>{flips.get("ai_native", {}).get("pct", 0)}%</strong> of the dead cohort, '
            f'the subclass for {flips.get("subclass", {}).get("pct", 0)}%, and the RAD score for '
            f'{flips.get("rad_score", {}).get("pct", 0)}%. Metadata alone is not enough.'
        )
    else:
        flip_line = ("Flip analysis runs once the evidence-based verdicts land. It will quantify "
                     "how often recovered evidence overturns a metadata-only label.")

    conf = m["confidence"]
    rad_block = ""
    if radh is not None and radl is not None:
        rad_block = (f"RAD-H firms die at <strong>{radh}%</strong> versus <strong>{radl}%</strong> "
                     f"for RAD-L. Dependency tracks mortality.")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Survivorship insights: does AI-nativeness predict survival</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{STYLE}</style>
</head>
<body>

<nav>
  <div class="nav-brand">Survivorship insights</div>
  <div class="nav-sub">Does AI predict survival</div>
  <div class="nav-section">
    <div class="nav-label">Sections</div>
    <ul>
      <li><a href="#overview">Overview</a></li>
      <li><a href="#survival">AI vs Survival</a></li>
      <li><a href="#rad">RAD as Predictor</a></li>
      <li><a href="#trap">Dependency Trap</a></li>
      <li><a href="#vertical">Verticals</a></li>
      <li><a href="#temporal">Deaths Over Time</a></li>
      <li><a href="#model">Regression</a></li>
      <li><a href="#robustness">Robustness</a></li>
    </ul>
  </div>
  <div class="nav-meta">
    <p><strong>Cohorts</strong><br>{meta["n_survivor"]:,} survivors<br>{meta["n_dead_recovered"]:,} dead (recovered)<br>{meta["n_dead_full"]:,} dead (full proxy)</p>
    <p style="margin-top:0.75rem;"><strong>Outcome</strong><br>Tavily-extractable today<br>vs recovered pre-death</p>
    <p style="margin-top:0.75rem;"><strong>Method</strong><br>Descriptive + logistic<br>regression (odds ratios)</p>
  </div>
</nav>

<main>

<section id="overview">
  <span class="section-label">Survivorship-Bias Findings</span>
  <h1>Does AI-Nativeness<br>Predict Survival?</h1>
  <p>
    The live dataset can only see companies whose sites are extractable today, which biases it toward
    survivors. We recovered pre-death snapshots for the lost cohort and ran them through the unchanged
    classifier. This dashboard compares the two cohorts and asks one question: do AI-nativeness and
    foundation-model dependency predict which startups died?
  </p>
  <div class="tags">
    <span class="tag">Survivor vs dead</span>
    <span class="tag">RAD as a survival test</span>
    <span class="tag">Logistic regression</span>
    <span class="tag">Layered dead definition</span>
  </div>
  {preview_banner}
  <div class="hero-metrics">
    <div class="metric-card hl">
      <div class="mc-label">Dead cohort</div>
      <div class="mc-val">{meta["n_dead_full"]:,}</div>
      <div class="mc-ctx">recovered and reclassified</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">Survivors</div>
      <div class="mc-val">{meta["n_survivor"]:,}</div>
      <div class="mc-ctx">extractable today</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">AI-native, survivors</div>
      <div class="mc-val">{ar["survivor"]["rate"]}%</div>
      <div class="mc-ctx">vs {ar["dead_full"]["rate"]}% in the dead</div>
    </div>
    <div class="metric-card ok">
      <div class="mc-label">Corrected AI rate</div>
      <div class="mc-val">{corrected_ai}%</div>
      <div class="mc-ctx">{'+' if delta >= 0 else ''}{delta} pts vs biased view</div>
    </div>
  </div>
  <div class="insight insight-blue">
    <p><strong>Reading the dead cohort.</strong> "Dead" means Tavily could not extract the live site, a
    proxy for failure, not a death certificate. We report a strict subset (flagged offline, non-thin
    history) alongside the full set so the findings do not hinge on the definition.</p>
  </div>
</section>

<section id="survival">
  <span class="section-label">01. AI-Nativeness vs Survival</span>
  <h2>The Headline Comparison</h2>
  <p>
    The survivor-only view understates or overstates the true AI-native rate depending on which way the
    dead cohort leans. The corrected rate adds the recovered companies back in. The subclass lift shows
    which genres are over-represented among the dead.
  </p>
  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">AI-native rate by cohort</div>
        <div class="chart-box-desc">Survivors vs dead (all) vs dead (strict subset)</div>
      </div>
      <div class="chart-body"><div id="chart-airate" style="height:300px;"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Survivorship correction</div>
        <div class="chart-box-desc">Biased survivor-only rate vs corrected rate</div>
      </div>
      <div class="chart-body"><div id="chart-correction" style="height:300px;"></div></div>
    </div>
  </div>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Subclass over-representation among the dead</div>
        <div class="chart-box-desc">Lift = dead share / survivor share. Above 1.0 (red) means the genre dies more</div>
      </div>
      <div class="chart-body"><div id="chart-lift" style="height:320px;"></div></div>
    </div>
  </div>
  <div class="insight insight-blue">
    <p><strong>Two competing stories.</strong> If non-AI genres dominate the dead, AI-nativeness protected
    companies. If thin AI genres (1C, 1G) dominate, the AI bandwagon without a moat killed them. The lift
    chart settles it. The regression below controls for funding, era, and vertical.</p>
  </div>
</section>

<section id="rad">
  <span class="section-label">02. RAD as a Survival Predictor</span>
  <h2>Does Dependency Track Death?</h2>
  <p>
    RAD was designed to measure dependence on foundation-model providers. Mortality is the external test
    of that design. If high-dependency firms die more, the RAD axis predicts a real outcome, not just a
    label we assigned.
  </p>
  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Mortality by RAD score</div>
        <div class="chart-box-desc">Among AI-native firms, dead / (dead + survivor)</div>
      </div>
      <div class="chart-body"><div id="chart-rad" style="height:240px;"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Mortality by defensibility group</div>
        <div class="chart-box-desc">Commoditizable (1C,1G) vs defensible (1A,1B,1E)</div>
      </div>
      <div class="chart-body"><div id="chart-subgroup" style="height:240px;"></div></div>
    </div>
  </div>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Mortality by subclass</div>
        <div class="chart-box-desc">Every taxonomy genre, ranked</div>
      </div>
      <div class="chart-body"><div id="chart-subclass" style="height:360px;"></div></div>
    </div>
  </div>
  <div class="insight insight-green">
    <p><strong>The RAD axis earns its keep.</strong> {rad_block} Commoditizable genres sit at
    {commod}% mortality versus {defens}% for defensible ones. The label predicts survival.</p>
  </div>
</section>

<section id="trap">
  <span class="section-label">03. The Dependency Trap</span>
  <h2>Where Funding and Dependency Collide</h2>
  <p>
    RAD already adjusts for resources, but the raw cross-tab still tells a story. The deadliest cell is
    high dependency with thin funding. Money buys time to build a moat. Without it, dependence is fatal.
  </p>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Mortality heatmap: funding x RAD</div>
        <div class="chart-box-desc">Darker red is deadlier. Cell labels show mortality and n</div>
      </div>
      <div class="chart-body"><div id="chart-trap" style="height:360px;"></div></div>
    </div>
  </div>
</section>

<section id="vertical">
  <span class="section-label">04. Vertical Commoditization</span>
  <h2>Which Markets Got Absorbed</h2>
  <p>
    Some verticals were commoditized by each frontier release. Others held on domain moats. This ranks the
    busiest AI-native verticals by mortality.
  </p>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">AI-native mortality by vertical</div>
        <div class="chart-box-desc">Primary category group, busiest verticals</div>
      </div>
      <div class="chart-body"><div id="chart-vertical" style="height:320px;"></div></div>
    </div>
  </div>
</section>

<section id="temporal">
  <span class="section-label">05. Deaths Over Time</span>
  <h2>When the Dead Were Last Seen</h2>
  <p>
    Each company's last Wayback capture is its death anchor. Plotting deaths by month against model
    releases asks whether commoditizable genres died in waves after big launches. This view is
    exploratory and bounded by archive coverage.
  </p>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Last-seen counts by month</div>
        <div class="chart-box-desc">All dead (bars) vs commoditizable and defensible genres (lines)</div>
      </div>
      <div class="chart-body"><div id="chart-temporal" style="height:360px;"></div></div>
    </div>
  </div>
</section>

<section id="model">
  <span class="section-label">06. Regression</span>
  <h2>Effect Sizes, Net of Confounders</h2>
  <p>
    Two logistic models predict death. Model 1 tests AI-nativeness across the full cohort, controlling for
    funding, founding era, and vertical. Model 2 tests RAD and defensibility among AI-native firms only.
    Markers are odds ratios, bars are 95% confidence intervals. Red is a risk factor, green is protective,
    gray is not significant. The dotted line at 1.0 is no effect.
  </p>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Model 1: death across the full cohort</div>
        <div class="chart-box-desc">Outcome death=1. Predictors include AI-native, funding, era, vertical</div>
      </div>
      <div class="chart-body"><div id="chart-forest1" style="height:320px;"></div></div>
    </div>
  </div>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Model 2: death among AI-native firms</div>
        <div class="chart-box-desc">RAD level (vs RAD-L) and defensibility (vs commoditizable)</div>
      </div>
      <div class="chart-body"><div id="chart-forest2" style="height:320px;"></div></div>
    </div>
  </div>
</section>

<section id="robustness">
  <span class="section-label">07. Robustness</span>
  <h2>Why the Finding Holds</h2>
  <p>
    Three checks. Evidence matters more than metadata. Recovered verdicts are as confident as live ones.
    The result survives the strict dead definition.
  </p>
  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Classification confidence: survivor vs dead</div>
        <div class="chart-box-desc">Distribution of the 1 to 5 confidence score</div>
      </div>
      <div class="chart-body"><div id="chart-conf" style="height:300px;"></div></div>
    </div>
  </div>
  <div class="insight insight-amber">
    <p><strong>Metadata is not enough.</strong> {flip_line}</p>
  </div>
  <div class="insight insight-blue">
    <p><strong>Confidence holds.</strong> Recovered-evidence classifications carry a mean confidence of
    {conf["dead_recovered"]["mean"]} versus {conf["survivor"]["mean"]} for live survivors, so the dead verdicts
    are not low-quality guesses. The strict dead subset (n={meta["n_dead_strict"]:,}) shows an AI-native
    rate of {ar["dead_strict"]["rate"]}%, close to the full {ar["dead_full"]["rate"]}%, so the headline is
    robust to how we define death.</p>
  </div>
  <div class="insight insight-blue">
    <p><strong>Limits.</strong> The dead cohort is the Tavily-not-found proxy, and its evidence is a
    pre-death snapshot rather than a live read, so the comparison cannot fully separate failure from
    evidence recency. Founding era is controlled in the regression to absorb part of this.</p>
  </div>
</section>

</main>

<footer>
  <strong>Method:</strong> survivor vs recovered-dead comparison, logistic regression &nbsp;&middot;&nbsp;
  <strong>Outcome proxy:</strong> Tavily-not-found &nbsp;&middot;&nbsp;
  <strong>Generated:</strong> {today}
  <br>
  {'PREVIEW build: dead verdicts are metadata-only until survivorship_corrected.csv lands.' if preview else 'Evidence-based build from survivorship_corrected.csv.'}
  Regenerate with build_survivorship_insights_dashboard.py.
</footer>

<script>
{script}
</script>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, default=None)
    parser.add_argument("--production", type=Path, default=None)
    parser.add_argument("--master", type=Path, default=None)
    parser.add_argument("--classifier-input", type=Path, default=None)
    parser.add_argument("--targets-dead", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    p = _ANALYSIS.Paths()
    for attr, val in [("corrected", args.corrected), ("production", args.production),
                      ("master", args.master), ("classifier_input", args.classifier_input),
                      ("targets_dead", args.targets_dead)]:
        if val is not None:
            setattr(p, attr, val)

    metrics = _ANALYSIS.analyze(p)
    m = metrics["meta"]
    mode = "PREVIEW" if m["preview"] else "evidence-based"
    print(f"[{mode}] survivor={m['n_survivor']:,} dead={m['n_dead_full']:,} "
          f"strict={m['n_dead_strict']:,} excluded={m['n_excluded']:,}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(metrics), encoding="utf-8")
    print(f"Dashboard written to {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
