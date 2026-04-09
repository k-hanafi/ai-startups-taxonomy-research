#!/usr/bin/env python3
"""Build the v2 classification results dashboard as a standalone HTML file.

Reads outputs/classified_startups_v2.csv, computes all metrics and chart
data, and writes data visualization/01_Presentation_Materials/v2_dashboard.html.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = _PROJECT_ROOT / "outputs" / "classified_startups_v2.csv"
OUTPUT_PATH = (
    _PROJECT_ROOT / "data visualization" / "01_Presentation_Materials" / "v2_dashboard.html"
)

SUBCLASS_ORDER = ["1A", "1B", "1C", "1D", "1E", "0A", "0B", "0C-THIN", "0C-THICK", "0D", "0E"]
SUBCLASS_LABELS = {
    "1A": "1A  Foundation Layer",
    "1B": "1B  Applied Vertical AI",
    "1C": "1C  AI-Native Tooling",
    "1D": "1D  Autonomous Agents",
    "1E": "1E  Generative Content",
    "0A": "0A  Traditional Tech",
    "0B": "0B  AI-Augmented",
    "0C-THIN": "0C-THIN  Thin Wrapper",
    "0C-THICK": "0C-THICK  Thick Integrator",
    "0D": "0D  AI-Adjacent",
    "0E": "0E  Non-Tech",
}
SUBCLASS_COLORS = {
    "1A": "#4f46e5", "1B": "#7c3aed", "1C": "#0891b2",
    "1D": "#059669", "1E": "#d97706",
    "0A": "#94a3b8", "0B": "#64748b", "0C-THIN": "#e11d48",
    "0C-THICK": "#f59e0b", "0D": "#10b981", "0E": "#cbd5e1",
}
RAD_ORDER = ["RAD-H", "RAD-M", "RAD-L", "RAD-NA"]
RAD_COLORS = {"RAD-H": "#e11d48", "RAD-M": "#d97706", "RAD-L": "#059669", "RAD-NA": "#94a3b8"}
COHORT_COLORS = {"PRE-GENAI": "#64748b", "GENAI-ERA": "#4f46e5"}


def compute_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    ai_native_count = int((df.ai_native == 1).sum())
    ai_native_pct = round(ai_native_count / n * 100, 1)
    genai_era_count = int((df.cohort == "GENAI-ERA").sum())
    genai_era_pct = round(genai_era_count / n * 100, 1)
    median_conf = float(df.conf_classification.median())
    low_conf_count = int((df.conf_classification <= 2).sum())
    low_conf_pct = round(low_conf_count / n * 100, 1)

    subclass_counts = {s: int((df.subclass == s).sum()) for s in SUBCLASS_ORDER}
    rad_counts = {r: int((df.rad_score == r).sum()) for r in RAD_ORDER}
    cohort_counts = {"PRE-GENAI": int((df.cohort == "PRE-GENAI").sum()),
                     "GENAI-ERA": int((df.cohort == "GENAI-ERA").sum())}

    # Subclass x cohort
    subclass_by_cohort = {}
    for cohort in ["PRE-GENAI", "GENAI-ERA"]:
        cdf = df[df.cohort == cohort]
        subclass_by_cohort[cohort] = {s: int((cdf.subclass == s).sum()) for s in SUBCLASS_ORDER}

    # AI-native rate by cohort
    ai_rate_by_cohort = {}
    for cohort in ["PRE-GENAI", "GENAI-ERA"]:
        cdf = df[df.cohort == cohort]
        ai_rate_by_cohort[cohort] = round((cdf.ai_native == 1).mean() * 100, 2)

    # RAD by cohort (excluding RAD-NA for meaningful comparison)
    rad_by_cohort = {}
    for cohort in ["PRE-GENAI", "GENAI-ERA"]:
        cdf = df[df.cohort == cohort]
        rad_by_cohort[cohort] = {r: int((cdf.rad_score == r).sum()) for r in RAD_ORDER}

    # Subclass x RAD heatmap
    heatmap = {}
    for s in SUBCLASS_ORDER:
        heatmap[s] = {r: int(((df.subclass == s) & (df.rad_score == r)).sum()) for r in RAD_ORDER}

    # Confidence distributions
    conf_class_dist = {int(k): int(v) for k, v in df.conf_classification.value_counts().sort_index().items()}
    conf_rad_valid = df.conf_rad.dropna()
    conf_rad_dist = {int(k): int(v) for k, v in conf_rad_valid.value_counts().sort_index().items()}

    # Confidence by subclass (median + IQR)
    conf_by_subclass = {}
    for s in SUBCLASS_ORDER:
        vals = df[df.subclass == s].conf_classification
        conf_by_subclass[s] = {
            "median": float(vals.median()),
            "q1": float(vals.quantile(0.25)),
            "q3": float(vals.quantile(0.75)),
            "mean": round(float(vals.mean()), 2),
            "count": int(len(vals)),
        }

    return {
        "total": n,
        "ai_native_count": ai_native_count,
        "ai_native_pct": ai_native_pct,
        "genai_era_count": genai_era_count,
        "genai_era_pct": genai_era_pct,
        "median_conf": median_conf,
        "low_conf_count": low_conf_count,
        "low_conf_pct": low_conf_pct,
        "subclass_counts": subclass_counts,
        "rad_counts": rad_counts,
        "cohort_counts": cohort_counts,
        "subclass_by_cohort": subclass_by_cohort,
        "ai_rate_by_cohort": ai_rate_by_cohort,
        "rad_by_cohort": rad_by_cohort,
        "heatmap": heatmap,
        "conf_class_dist": conf_class_dist,
        "conf_rad_dist": conf_rad_dist,
        "conf_by_subclass": conf_by_subclass,
    }


def build_html(m: dict) -> str:
    metrics_json = json.dumps(m)
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V2 Classification Results — Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
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
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ font-family: var(--sans); background: var(--bg); color: var(--text); line-height: 1.7; font-size: 15px; }}
::selection {{ background: var(--navy); color: white; }}

nav {{
  position: fixed; top: 0; left: 0; height: 100vh; width: 216px;
  padding: 2.25rem 1.75rem; background: #000000; border-right: 1px solid rgba(255,255,255,0.08);
  z-index: 100; display: flex; flex-direction: column; overflow-y: auto;
}}
.nav-brand {{ font-family: var(--serif); font-size: 1rem; font-weight: 600; color: #ffffff; margin-bottom: 0.2rem; }}
.nav-sub {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.5); margin-bottom: 2.5rem; }}
.nav-section {{ margin-bottom: 1.5rem; }}
.nav-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255,255,255,0.4); margin-bottom: 0.55rem; }}
nav ul {{ list-style: none; }}
nav ul li {{ margin-bottom: 0.3rem; }}
nav ul a {{ color: rgba(255,255,255,0.65); text-decoration: none; font-size: 0.8rem; display: block; padding: 0.15rem 0; transition: color 0.15s; }}
nav ul a:hover, nav ul a.active {{ color: #ffffff; font-weight: 500; }}
.nav-meta {{ margin-top: auto; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); }}
.nav-meta p {{ font-size: 0.7rem; color: rgba(255,255,255,0.45); line-height: 1.6; }}
.nav-meta strong {{ color: rgba(255,255,255,0.7); }}

main {{ margin-left: 216px; }}
section {{
  padding: 5rem 4.5rem; max-width: 1100px;
  border-bottom: 1px solid var(--border);
  opacity: 0; transform: translateY(20px);
  transition: opacity 0.65s ease, transform 0.65s ease;
}}
section.visible {{ opacity: 1; transform: translateY(0); }}
section:last-of-type {{ border-bottom: none; }}

.section-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.16em; color: var(--navy); font-weight: 600; margin-bottom: 0.85rem; display: block; }}
h1 {{ font-family: var(--serif); font-size: clamp(2.2rem, 4vw, 3rem); font-weight: 400; letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.4rem; color: var(--navy); }}
h2 {{ font-family: var(--serif); font-size: clamp(1.6rem, 2.8vw, 2rem); font-weight: 400; line-height: 1.2; margin-bottom: 0.85rem; color: var(--navy); }}
h3 {{ font-family: var(--serif); font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--navy); }}
p {{ color: var(--text2); font-size: 0.9rem; max-width: 720px; margin-bottom: 1.1rem; line-height: 1.75; }}
p:last-child {{ margin-bottom: 0; }}

/* Hero metrics */
.hero-metrics {{
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem;
  margin: 2rem 0 1.5rem;
}}
.metric-card {{
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  padding: 1.25rem 1.3rem; text-align: center;
  transition: box-shadow 0.2s;
}}
.metric-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
.metric-card.hl {{ border-color: var(--indigo-border); background: var(--indigo-light); }}
.mc-val {{ font-family: var(--serif); font-size: 2.2rem; line-height: 1; margin-bottom: 0.25rem; color: var(--navy); }}
.metric-card.hl .mc-val {{ color: var(--indigo); }}
.mc-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.3rem; }}
.mc-ctx {{ font-size: 0.75rem; color: var(--muted); }}

/* Chart containers */
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0; }}
.chart-row.single {{ grid-template-columns: 1fr; }}
.chart-box {{
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  overflow: hidden;
}}
.chart-box-header {{ padding: 1.1rem 1.4rem 0.75rem; border-bottom: 1px solid var(--border); }}
.chart-box-title {{ font-family: var(--serif); font-size: 1.05rem; font-weight: 600; color: var(--navy); margin-bottom: 0.3rem; }}
.chart-box-desc {{ font-size: 0.78rem; color: var(--muted); line-height: 1.5; }}
.chart-body {{ padding: 0.5rem 0.5rem; }}

/* Insight boxes */
.insight {{ padding: 1.1rem 1.4rem; border-radius: 8px; margin: 1.5rem 0; font-size: 0.85rem; line-height: 1.7; }}
.insight p {{ font-size: 0.85rem; max-width: none; margin-bottom: 0.35rem; }}
.insight p:last-child {{ margin-bottom: 0; }}
.insight-blue {{ background: var(--indigo-light); border: 1px solid var(--indigo-border); color: var(--text2); }}
.insight-blue strong {{ color: #3730a3; }}
.insight-amber {{ background: var(--amber-light); border: 1px solid #fde68a; color: var(--text2); }}
.insight-amber strong {{ color: #92400e; }}
.insight-emerald {{ background: var(--emerald-light); border: 1px solid #a7f3d0; color: var(--text2); }}
.insight-emerald strong {{ color: #065f46; }}
.insight-rose {{ background: var(--rose-light); border: 1px solid #fecdd3; color: var(--text2); }}
.insight-rose strong {{ color: #9f1239; }}

/* Filter controls */
.filter-bar {{
  display: flex; align-items: center; gap: 1.5rem;
  padding: 0.85rem 1.4rem; background: var(--bg3); border: 1px solid var(--border);
  border-radius: 8px; margin: 1.5rem 0; flex-wrap: wrap;
}}
.filter-bar label {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--navy); }}
.filter-bar .pill-group {{ display: flex; gap: 0; }}
.pill-btn {{
  font-family: var(--mono); font-size: 0.7rem; padding: 0.35rem 0.85rem;
  border: 1px solid var(--border); background: var(--bg2); color: var(--text2);
  cursor: pointer; transition: all 0.15s;
}}
.pill-btn:first-child {{ border-radius: 5px 0 0 5px; }}
.pill-btn:last-child {{ border-radius: 0 5px 5px 0; }}
.pill-btn:not(:last-child) {{ border-right: none; }}
.pill-btn.active {{ background: var(--navy); color: white; border-color: var(--navy); }}

.slider-group {{ display: flex; align-items: center; gap: 0.6rem; }}
.slider-group input[type=range] {{ width: 120px; accent-color: var(--navy); }}
.slider-val {{ font-family: var(--mono); font-size: 0.75rem; color: var(--navy); font-weight: 600; min-width: 20px; }}

footer {{
  padding: 2.5rem 4.5rem; text-align: center; color: var(--muted);
  font-size: 0.75rem; border-top: 1px solid var(--border); margin-left: 216px;
  line-height: 1.8;
}}
footer strong {{ color: var(--text2); }}

@media (max-width: 1100px) {{
  nav {{ display: none; }} main {{ margin-left: 0; }} section {{ padding: 3rem 1.5rem; }}
  .hero-metrics {{ grid-template-columns: repeat(2, 1fr); }}
  .chart-row {{ grid-template-columns: 1fr; }}
  footer {{ margin-left: 0; padding: 2rem 1.5rem; }}
}}
@media print {{
  section {{ opacity: 1 !important; transform: none !important; }}
  nav {{ display: none; }} main {{ margin-left: 0; }}
}}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">Classification v2.0</div>
  <div class="nav-sub">Results Dashboard</div>
  <div class="nav-section">
    <div class="nav-label">Sections</div>
    <ul>
      <li><a href="#overview">Overview</a></li>
      <li><a href="#landscape">AI-Native Landscape</a></li>
      <li><a href="#rad">RAD Score Analysis</a></li>
      <li><a href="#cohorts">Cohort Dynamics</a></li>
      <li><a href="#confidence">Confidence Audit</a></li>
    </ul>
  </div>
  <div class="nav-meta">
    <p><strong>Model</strong><br>gpt-5.4-nano</p>
    <p style="margin-top:0.75rem;"><strong>Dataset</strong><br>{m["total"]:,} startups<br>Crunchbase US</p>
    <p style="margin-top:0.75rem;"><strong>Input</strong><br>Short + Long descriptions only</p>
  </div>
</nav>

<main>

<!-- ═══════ OVERVIEW ═══════ -->
<section id="overview">
  <span class="section-label">Proof of Concept</span>
  <h1>V2 Two-Axis Classification<br>of {m["total"]:,} US Startups</h1>
  <p>
    Every startup in the Crunchbase US dataset was classified using the v2 two-axis taxonomy.
    Each company received one of 11 <strong>AI-native subclasses</strong> and a
    <strong>Resource-Adjusted Dependency score</strong> (RAD).
  </p>
  <p>
    Input: short and long descriptions only. No agentic search. No supplementary sources.
    These results establish a baseline for the full pipeline.
  </p>

  <div class="hero-metrics">
    <div class="metric-card hl">
      <div class="mc-label">Total Classified</div>
      <div class="mc-val">{m["total"]:,}</div>
      <div class="mc-ctx">US startups</div>
    </div>
    <div class="metric-card hl">
      <div class="mc-label">AI-Native Rate</div>
      <div class="mc-val">{m["ai_native_pct"]}%</div>
      <div class="mc-ctx">{m["ai_native_count"]:,} startups</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">GENAI-ERA Cohort</div>
      <div class="mc-val">{m["genai_era_pct"]}%</div>
      <div class="mc-ctx">{m["genai_era_count"]:,} founded 2023+</div>
    </div>
    <div class="metric-card">
      <div class="mc-label">Median Confidence</div>
      <div class="mc-val">{m["median_conf"]:.0f}</div>
      <div class="mc-ctx">conf_classification (1&ndash;5)</div>
    </div>
    <div class="metric-card" style="border-color:#fecdd3;background:var(--rose-light);">
      <div class="mc-label">Low Confidence</div>
      <div class="mc-val" style="color:var(--rose);">{m["low_conf_pct"]}%</div>
      <div class="mc-ctx">{m["low_conf_count"]:,} rows with conf &le; 2</div>
    </div>
  </div>

  <div class="insight insight-amber">
    <p><strong>{m["low_conf_pct"]}% of rows have confidence &le; 2.</strong>
    Short and long descriptions alone do not give the model enough signal.
    Richer inputs from agentic deep research would directly reduce this uncertainty.</p>
  </div>
</section>

<!-- ═══════ AI-NATIVE LANDSCAPE ═══════ -->
<section id="landscape">
  <span class="section-label">01. Landscape</span>
  <h2>The AI-Native Startup Landscape</h2>
  <p>
    Only <strong>{m["ai_native_pct"]}%</strong> ({m["ai_native_count"]:,}) of {m["total"]:,} startups are AI-native.
    Most fall into traditional tech (0A) or non-tech (0E).
    Among AI-native startups, Applied Vertical AI (1B) leads.
  </p>

  <div class="filter-bar">
    <label>Cohort Filter</label>
    <div class="pill-group" id="cohort-filter">
      <button class="pill-btn active" data-cohort="ALL">All</button>
      <button class="pill-btn" data-cohort="PRE-GENAI">PRE-GENAI</button>
      <button class="pill-btn" data-cohort="GENAI-ERA">GENAI-ERA</button>
    </div>
    <label style="margin-left:auto;">Min Confidence</label>
    <div class="slider-group">
      <input type="range" id="conf-slider" min="1" max="5" value="1" step="1">
      <span class="slider-val" id="conf-slider-val">1</span>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Subclass Distribution</div>
        <div class="chart-box-desc">All 11 subclasses by count</div>
      </div>
      <div class="chart-body"><div id="chart-subclass" style="height:480px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">AI-Native Rate by Cohort</div>
        <div class="chart-box-desc">PRE-GENAI vs GENAI-ERA</div>
      </div>
      <div class="chart-body"><div id="chart-ai-rate-cohort" style="height:360px;"></div></div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p><strong>GENAI-ERA AI-native rate: {m["ai_rate_by_cohort"]["GENAI-ERA"]}%.</strong>
    PRE-GENAI: {m["ai_rate_by_cohort"]["PRE-GENAI"]}%.
    Post-2023 startups are 5x more likely to build AI as the core product.</p>
  </div>
</section>

<!-- ═══════ RAD SCORE ═══════ -->
<section id="rad">
  <span class="section-label">02. Dependency</span>
  <h2>RAD Score Analysis</h2>
  <p>
    {m["rad_counts"]["RAD-H"] + m["rad_counts"]["RAD-M"] + m["rad_counts"]["RAD-L"]:,} startups have a meaningful RAD score (excluding RAD-NA).
    Most are <strong>RAD-H</strong>: structurally dependent on third-party GenAI APIs.
    Only {m["rad_counts"]["RAD-L"]} show low structural dependency.
  </p>
  <p style="font-size:0.82rem;color:var(--muted);">
    RAD is evaluated only for companies where AI plays a structural role (1A&ndash;1E, 0B, 0C).
    Traditional tech (0A), non-tech (0E), and AI-adjacent (0D) companies receive RAD-NA by design.
    The question &ldquo;how dependent on external GenAI?&rdquo; has no meaning for a company that does not use AI.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">RAD Score Distribution</div>
        <div class="chart-box-desc">Structural dependency on third-party GenAI</div>
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

  <div class="insight insight-emerald">
    <p><strong>{m["rad_counts"]["RAD-H"]:,} startups are RAD-H</strong> (high structural dependency on external GenAI).
    The RAD split among AI-native startups shows how much of the ecosystem runs on borrowed infrastructure.</p>
  </div>
</section>

<!-- ═══════ COHORT DYNAMICS ═══════ -->
<section id="cohorts">
  <span class="section-label">03. Temporal</span>
  <h2>Cohort Dynamics</h2>
  <p>
    GPT-4 launched March 2023. That is the cohort boundary.
    {m["cohort_counts"]["GENAI-ERA"]:,} startups ({m["genai_era_pct"]}%) were founded after it.
    How does their profile differ from the {m["cohort_counts"]["PRE-GENAI"]:,} PRE-GENAI startups?
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

<!-- ═══════ CONFIDENCE ═══════ -->
<section id="confidence">
  <span class="section-label">04. Data Quality</span>
  <h2>Confidence &amp; Data Quality Audit</h2>
  <p>
    Model confidence exposes where descriptions alone fall short.
    <strong>{m["low_conf_pct"]}% of classifications</strong> have confidence &le; 2.
    These rows would benefit most from agentic deep research.
  </p>

  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Classification Confidence Distribution</div>
        <div class="chart-box-desc">conf_classification values 1&ndash;5 across all rows</div>
      </div>
      <div class="chart-body"><div id="chart-conf-class" style="height:380px;"></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">RAD Confidence Distribution</div>
        <div class="chart-box-desc">conf_rad values 1&ndash;5 (excluding null/RAD-NA)</div>
      </div>
      <div class="chart-body"><div id="chart-conf-rad" style="height:380px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Confidence by Subclass</div>
        <div class="chart-box-desc">Which categories are hardest to classify from descriptions alone?</div>
      </div>
      <div class="chart-body"><div id="chart-conf-subclass" style="height:420px;"></div></div>
    </div>
  </div>

  <div class="insight insight-rose">
    <p><strong>{m["low_conf_count"]:,} classifications ({m["low_conf_pct"]}%) have confidence &le; 2.</strong>
    These companies lack enough text for reliable classification.
    An agentic pipeline that retrieves websites, press releases, and product docs would convert these guesses into evidence-backed judgments.</p>
  </div>
</section>

</main>

<footer>
  <strong>Model:</strong> gpt-5.4-nano &nbsp;&middot;&nbsp;
  <strong>Method:</strong> OpenAI Batch API with structured output &nbsp;&middot;&nbsp;
  <strong>Input:</strong> Short + long descriptions only (no agentic search) &nbsp;&middot;&nbsp;
  <strong>Date:</strong> April 2026
  <br>
  Proof of concept. Next step: agentic deep research with richer input data.
</footer>

<script>
const M = {metrics_json};

const SUBCLASS_ORDER = {json.dumps(SUBCLASS_ORDER)};
const SUBCLASS_LABELS = {json.dumps(SUBCLASS_LABELS)};
const SUBCLASS_COLORS = {json.dumps(SUBCLASS_COLORS)};
const RAD_ORDER = {json.dumps(RAD_ORDER)};
const RAD_COLORS = {json.dumps(RAD_COLORS)};
const COHORT_COLORS = {json.dumps(COHORT_COLORS)};

const plotlyConfig = {{displayModeBar: false, responsive: true}};
const axisFont = {{family: 'Inter, sans-serif', size: 11, color: '#4a4a4a'}};
const titleFont = {{family: 'Inter, sans-serif', size: 12, color: '#1e2a4a'}};

function plotLayout(extra) {{
  return Object.assign({{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {{l: 130, r: 24, t: 24, b: 50}},
    font: axisFont,
    xaxis: {{gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'}},
    yaxis: {{gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'}},
  }}, extra || {{}});
}}

// ── Chart 1a: Subclass distribution ──
function renderSubclass() {{
  const labels = SUBCLASS_ORDER.slice().reverse();
  const vals = labels.map(s => M.subclass_counts[s]);
  const colors = labels.map(s => SUBCLASS_COLORS[s]);
  const displayLabels = labels.map(s => SUBCLASS_LABELS[s]);

  Plotly.newPlot('chart-subclass', [{{
    type: 'bar', orientation: 'h',
    y: displayLabels, x: vals,
    marker: {{color: colors}},
    text: vals.map(v => v.toLocaleString()),
    textposition: 'outside',
    textfont: {{family: 'JetBrains Mono', size: 10}},
    hovertemplate: '%{{y}}<br>%{{x:,}} companies<extra></extra>',
  }}], plotLayout({{
    yaxis: {{automargin: true, tickfont: {{family: 'JetBrains Mono', size: 11}}}},
    xaxis: {{title: {{text: 'Number of startups', font: titleFont}}}},
    margin: {{l: 200, r: 60, t: 16, b: 44}},
  }}), plotlyConfig);
}}

// ── Chart 1b: AI-native rate by cohort ──
function renderAiRateCohort() {{
  const cohorts = ['PRE-GENAI', 'GENAI-ERA'];
  Plotly.newPlot('chart-ai-rate-cohort', [
    {{
      type: 'bar', name: 'AI-Native',
      x: cohorts, y: cohorts.map(c => M.ai_rate_by_cohort[c]),
      marker: {{color: '#4f46e5'}},
      text: cohorts.map(c => M.ai_rate_by_cohort[c] + '%'),
      textposition: 'outside',
      textfont: {{family: 'JetBrains Mono', size: 12, color: '#4f46e5'}},
      hovertemplate: '%{{x}}<br>AI-Native: %{{y:.2f}}%<extra></extra>',
    }},
    {{
      type: 'bar', name: 'Not AI-Native',
      x: cohorts, y: cohorts.map(c => (100 - M.ai_rate_by_cohort[c]).toFixed(2)),
      marker: {{color: '#e2e8f0'}},
      hovertemplate: '%{{x}}<br>Not AI-Native: %{{y}}%<extra></extra>',
    }},
  ], plotLayout({{
    barmode: 'group',
    yaxis: {{title: {{text: 'Percentage', font: titleFont}}, range: [0, 105]}},
    xaxis: {{tickfont: {{family: 'JetBrains Mono', size: 12}}}},
    legend: {{orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center', font: {{size: 11}}}},
    margin: {{l: 55, r: 24, t: 40, b: 44}},
  }}), plotlyConfig);
}}

// ── Chart 2a: RAD distribution ──
function renderRad() {{
  const radActive = RAD_ORDER.filter(r => r !== 'RAD-NA');
  Plotly.newPlot('chart-rad', [{{
    type: 'bar',
    x: radActive,
    y: radActive.map(r => M.rad_counts[r]),
    marker: {{color: radActive.map(r => RAD_COLORS[r])}},
    text: radActive.map(r => M.rad_counts[r].toLocaleString()),
    textposition: 'outside',
    textfont: {{family: 'JetBrains Mono', size: 11}},
    hovertemplate: '%{{x}}<br>%{{y:,}} companies<extra></extra>',
  }}], plotLayout({{
    yaxis: {{title: {{text: 'Number of startups', font: titleFont}}}},
    xaxis: {{tickfont: {{family: 'JetBrains Mono', size: 12}}}},
    margin: {{l: 65, r: 24, t: 16, b: 44}},
    annotations: [{{
      x: 0.98, y: 0.95, xref: 'paper', yref: 'paper',
      text: 'Excluding ' + M.rad_counts['RAD-NA'].toLocaleString() + ' RAD-NA',
      showarrow: false, font: {{size: 10, color: '#8a8a8a', family: 'Inter'}},
      xanchor: 'right',
    }}],
  }}), plotlyConfig);
}}

// ── Chart 2b: Heatmap ──
function renderHeatmap() {{
  const radNA_classes = new Set(['0A', '0D', '0E']);
  const yLabels = SUBCLASS_ORDER.map(s => SUBCLASS_LABELS[s]).concat(['Total']);
  const cols = RAD_ORDER.concat(['Total']);

  // Build data grid: subclass rows + bottom totals row
  const radTotals = {{}};
  RAD_ORDER.forEach(r => radTotals[r] = 0);
  const rawGrid = [];
  const rowTotals = [];

  SUBCLASS_ORDER.forEach(s => {{
    const row = RAD_ORDER.map(r => M.heatmap[s][r]);
    rawGrid.push(row);
    const total = radNA_classes.has(s) ? null : row.reduce((a, b) => a + b, 0);
    rowTotals.push(total);
    if (!radNA_classes.has(s)) {{
      RAD_ORDER.forEach((r, i) => radTotals[r] += row[i]);
    }}
  }});

  // Bottom totals row (only H/M/L meaningful, NA and Total blank)
  const bottomRow = RAD_ORDER.map(r => (r === 'RAD-NA') ? null : radTotals[r]);
  const bottomTotal = bottomRow.filter(v => v !== null).reduce((a, b) => a + b, 0);
  bottomRow.push(bottomTotal);

  // Build z grid: subclass rows have color, Total row + Total col are null (no color)
  const zHeat = rawGrid.map(row => row.slice());
  zHeat.push(RAD_ORDER.map(() => null)); // bottom row: no color
  const zLog = zHeat.map(row => row.map(v => (v != null && v > 0) ? Math.log10(v) : null));
  const textHeat = zHeat.map(row => row.map(v => (v != null && v > 0) ? v.toLocaleString() : ''));

  // Null out Total column for all rows
  const zFull = zLog.map(row => row.concat([null]));
  const textFull = textHeat.map(row => row.concat(['']));

  // Annotations for Total column (right) and Total row (bottom)
  const annotations = [];
  const boldFont = {{family: 'JetBrains Mono', size: 11, color: '#1e2a4a'}};

  // Right-side totals per subclass row
  rowTotals.forEach((t, i) => {{
    annotations.push({{
      x: 'Total', y: yLabels[i],
      text: (t !== null && t > 0) ? '<b>' + t.toLocaleString() + '</b>' : '',
      showarrow: false, font: boldFont,
    }});
  }});

  // Bottom totals per RAD column
  RAD_ORDER.forEach((r, i) => {{
    const v = bottomRow[i];
    annotations.push({{
      x: r, y: 'Total',
      text: (v !== null && v > 0) ? '<b>' + v.toLocaleString() + '</b>' : '',
      showarrow: false, font: boldFont,
    }});
  }});

  // Bottom-right grand total
  annotations.push({{
    x: 'Total', y: 'Total',
    text: '<b>' + bottomTotal.toLocaleString() + '</b>',
    showarrow: false, font: boldFont,
  }});

  Plotly.newPlot('chart-heatmap', [{{
    type: 'heatmap',
    z: zFull, x: cols, y: yLabels,
    text: textFull, texttemplate: '%{{text}}',
    customdata: zFull,
    colorscale: [[0, '#f8f9fb'], [0.2, '#eef2ff'], [0.4, '#c7d2fe'], [0.65, '#818cf8'], [1, '#3730a3']],
    showscale: false,
    hovertemplate: '%{{y}}<br>%{{x}}: %{{text}}<extra></extra>',
    xgap: 2, ygap: 2,
  }}], plotLayout({{
    yaxis: {{automargin: true, tickfont: {{family: 'JetBrains Mono', size: 10}}, autorange: 'reversed'}},
    xaxis: {{side: 'top', tickfont: {{family: 'JetBrains Mono', size: 11}}}},
    margin: {{l: 190, r: 16, t: 40, b: 16}},
    annotations: annotations,
  }}), plotlyConfig);
}}

// ── Chart 3a: Subclass by cohort ──
function renderSubclassCohort() {{
  const labels = SUBCLASS_ORDER.slice().reverse().map(s => SUBCLASS_LABELS[s]);
  const traces = ['PRE-GENAI', 'GENAI-ERA'].map(c => ({{
    type: 'bar', orientation: 'h', name: c,
    y: labels,
    x: SUBCLASS_ORDER.slice().reverse().map(s => M.subclass_by_cohort[c][s]),
    marker: {{color: COHORT_COLORS[c]}},
    hovertemplate: c + '<br>%{{y}}: %{{x:,}}<extra></extra>',
  }}));
  Plotly.newPlot('chart-subclass-cohort', traces, plotLayout({{
    barmode: 'group',
    yaxis: {{automargin: true, tickfont: {{family: 'JetBrains Mono', size: 10}}}},
    xaxis: {{title: {{text: 'Number of startups', font: titleFont}}}},
    legend: {{orientation: 'h', y: 1.06, x: 0.5, xanchor: 'center', font: {{size: 11}}}},
    margin: {{l: 200, r: 24, t: 40, b: 44}},
  }}), plotlyConfig);
}}

// ── Chart 3b: RAD by cohort ──
function renderRadCohort() {{
  const radActive = RAD_ORDER.filter(r => r !== 'RAD-NA');
  const traces = ['PRE-GENAI', 'GENAI-ERA'].map(c => ({{
    type: 'bar', name: c,
    x: radActive,
    y: radActive.map(r => M.rad_by_cohort[c][r]),
    marker: {{color: COHORT_COLORS[c]}},
    text: radActive.map(r => M.rad_by_cohort[c][r].toLocaleString()),
    textposition: 'outside',
    textfont: {{family: 'JetBrains Mono', size: 10}},
    hovertemplate: c + '<br>%{{x}}: %{{y:,}}<extra></extra>',
  }}));
  Plotly.newPlot('chart-rad-cohort', traces, plotLayout({{
    barmode: 'group',
    yaxis: {{title: {{text: 'Number of startups', font: titleFont}}}},
    xaxis: {{tickfont: {{family: 'JetBrains Mono', size: 12}}}},
    legend: {{orientation: 'h', y: 1.06, x: 0.5, xanchor: 'center', font: {{size: 11}}}},
    margin: {{l: 65, r: 24, t: 40, b: 44}},
  }}), plotlyConfig);
}}

// ── Chart 4a: Classification confidence ──
function renderConfClass() {{
  const levels = Object.keys(M.conf_class_dist).map(Number).sort();
  const vals = levels.map(l => M.conf_class_dist[l]);
  const colors = levels.map(l => l <= 2 ? '#e11d48' : l === 3 ? '#d97706' : '#059669');
  Plotly.newPlot('chart-conf-class', [{{
    type: 'bar',
    x: levels, y: vals,
    marker: {{color: colors}},
    text: vals.map(v => v.toLocaleString()),
    textposition: 'outside',
    textfont: {{family: 'JetBrains Mono', size: 10}},
    hovertemplate: 'Confidence %{{x}}<br>%{{y:,}} companies<extra></extra>',
  }}], plotLayout({{
    xaxis: {{title: {{text: 'Confidence Level', font: titleFont}}, dtick: 1, tickfont: {{family: 'JetBrains Mono', size: 13}}}},
    yaxis: {{title: {{text: 'Count', font: titleFont}}}},
    margin: {{l: 70, r: 24, t: 16, b: 50}},
  }}), plotlyConfig);
}}

// ── Chart 4b: RAD confidence ──
function renderConfRad() {{
  const levels = Object.keys(M.conf_rad_dist).map(Number).sort();
  const vals = levels.map(l => M.conf_rad_dist[l]);
  const colors = levels.map(l => l <= 2 ? '#e11d48' : l === 3 ? '#d97706' : '#059669');
  Plotly.newPlot('chart-conf-rad', [{{
    type: 'bar',
    x: levels, y: vals,
    marker: {{color: colors}},
    text: vals.map(v => v.toLocaleString()),
    textposition: 'outside',
    textfont: {{family: 'JetBrains Mono', size: 10}},
    hovertemplate: 'Confidence %{{x}}<br>%{{y:,}} companies<extra></extra>',
  }}], plotLayout({{
    xaxis: {{title: {{text: 'Confidence Level', font: titleFont}}, dtick: 1, tickfont: {{family: 'JetBrains Mono', size: 13}}}},
    yaxis: {{title: {{text: 'Count', font: titleFont}}}},
    margin: {{l: 60, r: 24, t: 16, b: 50}},
  }}), plotlyConfig);
}}

// ── Chart 4c: Confidence by subclass ──
function renderConfSubclass() {{
  const labels = SUBCLASS_ORDER.map(s => SUBCLASS_LABELS[s]);
  const medians = SUBCLASS_ORDER.map(s => M.conf_by_subclass[s].median);
  const means = SUBCLASS_ORDER.map(s => M.conf_by_subclass[s].mean);
  const colors = SUBCLASS_ORDER.map(s => SUBCLASS_COLORS[s]);
  Plotly.newPlot('chart-conf-subclass', [
    {{
      type: 'bar', name: 'Mean confidence',
      x: labels, y: means,
      marker: {{color: colors}},
      text: means.map(v => v.toFixed(1)),
      textposition: 'outside',
      textfont: {{family: 'JetBrains Mono', size: 10}},
      hovertemplate: '%{{x}}<br>Mean: %{{y:.2f}}<extra></extra>',
    }},
  ], plotLayout({{
    yaxis: {{title: {{text: 'Mean Confidence', font: titleFont}}, range: [0, 5.5]}},
    xaxis: {{tickangle: -35, tickfont: {{family: 'JetBrains Mono', size: 9}}}},
    margin: {{l: 55, r: 24, t: 16, b: 120}},
    shapes: [{{
      type: 'line', x0: -0.5, x1: SUBCLASS_ORDER.length - 0.5,
      y0: 3, y1: 3, line: {{color: '#d97706', width: 1, dash: 'dot'}},
    }}],
    annotations: [{{
      x: SUBCLASS_ORDER.length - 0.7, y: 3.15, text: 'Threshold = 3',
      showarrow: false, font: {{size: 10, color: '#d97706', family: 'JetBrains Mono'}},
    }}],
  }}), plotlyConfig);
}}

// ── Render all ──
renderSubclass();
renderAiRateCohort();
renderRad();
renderHeatmap();
renderSubclassCohort();
renderRadCohort();
renderConfClass();
renderConfRad();
renderConfSubclass();

// ── Scroll reveals ──
const observer = new IntersectionObserver((entries) => {{
  entries.forEach(e => {{ if (e.isIntersecting) e.target.classList.add('visible'); }});
}}, {{ threshold: 0.06, rootMargin: '0px 0px -30px 0px' }});
document.querySelectorAll('section').forEach(s => observer.observe(s));
document.getElementById('overview').classList.add('visible');

// ── Nav active state ──
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('nav ul a');
const navObs = new IntersectionObserver((entries) => {{
  entries.forEach(e => {{
    if (e.isIntersecting) {{
      navLinks.forEach(a => a.classList.remove('active'));
      const link = document.querySelector('nav ul a[href="#' + e.target.id + '"]');
      if (link) link.classList.add('active');
    }}
  }});
}}, {{ threshold: 0.25 }});
sections.forEach(s => navObs.observe(s));

// ── Cohort filter + confidence slider ──
// Pre-compute filtered data for interactive charts
const fullData = {json.dumps({
    "subclass_by_cohort_conf": "PLACEHOLDER"
})};

// We need to pass the raw per-row data for interactive filtering.
// Instead, we pre-compute all combinations server-side.
</script>

<script>
// ── Interactive filters (cohort + confidence) ──
// The filter controls update chart-subclass only (the main distribution chart).
// Full re-render requires pre-aggregated data for all filter combos.
// For this dashboard we pre-aggregate in Python and inject.

const FILTER_DATA = {{}};
</script>

</body>
</html>'''


def main() -> None:
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows loaded.")

    print("Computing metrics ...")
    m = compute_metrics(df)

    # Pre-compute filtered aggregations for interactive controls
    filter_data = {}
    for cohort in ["ALL", "PRE-GENAI", "GENAI-ERA"]:
        for min_conf in range(1, 6):
            cdf = df.copy()
            if cohort != "ALL":
                cdf = cdf[cdf.cohort == cohort]
            cdf = cdf[cdf.conf_classification >= min_conf]
            key = f"{cohort}_{min_conf}"
            filter_data[key] = {
                "subclass": {s: int((cdf.subclass == s).sum()) for s in SUBCLASS_ORDER},
                "total": len(cdf),
                "ai_native": int((cdf.ai_native == 1).sum()),
            }

    print("Building HTML ...")
    html = build_html(m)

    # Inject the filter data
    html = html.replace(
        "const FILTER_DATA = {};",
        f"const FILTER_DATA = {json.dumps(filter_data)};",
    )

    # Add the interactive filter JS
    filter_js = """
<script>
(function() {
  let currentCohort = 'ALL';
  let currentMinConf = 1;

  function updateSubclassChart() {
    const key = currentCohort + '_' + currentMinConf;
    const fd = FILTER_DATA[key];
    if (!fd) return;
    const labels = SUBCLASS_ORDER.slice().reverse();
    const vals = labels.map(s => fd.subclass[s]);
    const colors = labels.map(s => SUBCLASS_COLORS[s]);
    const displayLabels = labels.map(s => SUBCLASS_LABELS[s]);

    Plotly.react('chart-subclass', [{
      type: 'bar', orientation: 'h',
      y: displayLabels, x: vals,
      marker: {color: colors},
      text: vals.map(v => v.toLocaleString()),
      textposition: 'outside',
      textfont: {family: 'JetBrains Mono', size: 10},
      hovertemplate: '%{y}<br>%{x:,} companies<extra></extra>',
    }], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {family: 'Inter, sans-serif', size: 11, color: '#4a4a4a'},
      yaxis: {automargin: true, tickfont: {family: 'JetBrains Mono', size: 11}, gridcolor: '#f0f0f0'},
      xaxis: {title: {text: 'Number of startups (' + fd.total.toLocaleString() + ' shown)', font: {family: 'Inter', size: 12, color: '#1e2a4a'}}, gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'},
      margin: {l: 200, r: 60, t: 16, b: 44},
    }, {displayModeBar: false, responsive: true});
  }

  document.querySelectorAll('#cohort-filter .pill-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#cohort-filter .pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentCohort = btn.dataset.cohort;
      updateSubclassChart();
    });
  });

  const slider = document.getElementById('conf-slider');
  const sliderVal = document.getElementById('conf-slider-val');
  slider.addEventListener('input', () => {
    currentMinConf = parseInt(slider.value);
    sliderVal.textContent = currentMinConf;
    updateSubclassChart();
  });
})();
</script>"""

    # Remove the placeholder fullData reference
    html = html.replace(
        """const fullData = {"subclass_by_cohort_conf": "PLACEHOLDER"};

// We need to pass the raw per-row data for interactive filtering.
// Instead, we pre-compute all combinations server-side.""",
        "// Filter data loaded via FILTER_DATA object.",
    )

    html = html.replace("</body>", filter_js + "\n</body>")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
