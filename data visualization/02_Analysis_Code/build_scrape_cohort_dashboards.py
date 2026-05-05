#!/usr/bin/env python3
"""Build two classification dashboards (same layout as full_baseline_cohort) for scrape comparison.

**Default cohort** — companies where Tavily crawl text was actually present in the classifier prompt:
rows in ``classifier_input.csv`` with non-empty ``website_evidence`` (same ``org_uuid`` as
``CompanyID`` in the classification CSVs). In typical runs this is ~20k of ~44k batch rows; the
rest had no usable crawl text, so comparing “with vs without scraping” should use this subset.

Pass ``--all-classifier-rows`` to instead use every ``org_uuid`` in ``classifier_input.csv`` (full
batch input, including rows without evidence).

Writes:
    tavily_scrape_cohort.html — production rows for that cohort (Tavily-enriched where evidence existed).
    baseline_scrape_cohort_crunchbase_only.html — migrated baseline, same IDs (Crunchbase-only labels).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASELINE = (
    _PROJECT_ROOT / "outputs" / "legacy_csv" / "classified_startups_v21_migrated.csv"
)
_DEFAULT_PRODUCTION = _PROJECT_ROOT / "outputs" / "production_csvs" / "production_classifications.csv"
_DEFAULT_CLASSIFIER_INPUT = (
    _PROJECT_ROOT / "outputs" / "tavilycrawl" / "processed" / "classifier_input.csv"
)
_DASHBOARD_BUILDER = (
    _PROJECT_ROOT / "data visualization" / "02_Analysis_Code" / "build_classification_dashboard.py"
)
_OUTPUT_DIR = _PROJECT_ROOT / "data visualization" / "01_Presentation_Materials"
_OUT_BASELINE = _OUTPUT_DIR / "baseline_scrape_cohort_crunchbase_only.html"
_OUT_TAVILY = _OUTPUT_DIR / "tavily_scrape_cohort.html"

_DEFAULT_OVERVIEW_P1 = """  <p>
    Full US Crunchbase cohort: short and long descriptions only at classification time.
    The model uses a <strong>10-class taxonomy</strong> (1A&ndash;1G for AI-native patterns, 0A&ndash;0C for non-native buckets):
    thin and thick LLM wrappers are 1C and 1D, AI-augmented companies sit in 0B,
    and the AI-native subclasses are ordered from foundation (1A) outward.
  </p>"""

_DEFAULT_FOOTER_TAIL = """  <br>
  Full US baseline run: Crunchbase descriptions only; batch API with structured output.
</footer>"""


def _load_dashboard_builder():
    spec = importlib.util.spec_from_file_location(
        "build_classification_dashboard_module", _DASHBOARD_BUILDER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {_DASHBOARD_BUILDER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dedupe_ids(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n = len(df)
    out = df.drop_duplicates(subset=["CompanyID"], keep="last")
    if len(out) < n:
        print(f"  {label}: dropped {n - len(out):,} duplicate CompanyID rows (kept last).")
    return out


def load_scrape_company_ids(
    classifier_input_csv: Path,
    *,
    require_website_evidence: bool,
) -> set[str]:
    """Unique org_uuid values from classifier_input.

    When ``require_website_evidence`` is True, keep only rows whose ``website_evidence`` column is
    non-empty after stripping — i.e. Tavily crawl text was included in the enriched prompt.
    """
    if not classifier_input_csv.is_file():
        raise FileNotFoundError(f"Missing classifier_input: {classifier_input_csv}")
    cols = ["org_uuid", "website_evidence"] if require_website_evidence else ["org_uuid"]
    df = pd.read_csv(classifier_input_csv, usecols=cols, dtype=str, keep_default_na=False)
    u = df["org_uuid"].dropna().astype(str).str.strip()
    mask = u != ""
    if require_website_evidence:
        ev = df["website_evidence"].fillna("").astype(str).str.strip()
        mask = mask & (ev.str.len() > 0)
    return set(u[mask])


def patch_baseline_scrape_html(html: str, n_input_ids: int, evidence_only: bool) -> str:
    nav_sub = (
        "Evidence cohort · baseline (Crunchbase-only)"
        if evidence_only
        else "Full batch input · baseline (Crunchbase-only)"
    )
    section = (
        "Evidence cohort (Crunchbase-only)"
        if evidence_only
        else "Full batch input (Crunchbase-only)"
    )
    html = html.replace('<div class="nav-sub">Results Dashboard</div>', f'<div class="nav-sub">{nav_sub}</div>')
    html = html.replace(
        '<span class="section-label">Taxonomy</span>',
        f'<span class="section-label">{section}</span>',
    )
    if evidence_only:
        cohort_sentence = (
            f"restricted to the <strong>{n_input_ids:,}</strong> companies that have non-empty "
            "<strong>website_evidence</strong> in <code>classifier_input.csv</code> "
            "<em>and</em> appear in both the migrated baseline export and production export "
            "(same paired IDs as the Tavily page). This isolates rows where crawl text was available "
            "for a fair before/after comparison."
        )
    else:
        cohort_sentence = (
            f"restricted to the <strong>{n_input_ids:,}</strong> companies in "
            "<code>classifier_input.csv</code> that appear in both classification exports "
            "(including rows without crawl text)."
        )
    p1 = f"""  <p>
    Historical <strong>full-corpus</strong> classifications (Crunchbase short + long description only)
    from the migrated US baseline, {cohort_sentence}
    Same charts as <code>full_baseline_cohort.html</code>.
  </p>"""
    html = html.replace(_DEFAULT_OVERVIEW_P1, p1)
    html = html.replace(
        _DEFAULT_FOOTER_TAIL,
        """  <br>
  Cohort: full baseline labels for the same company IDs as the paired Tavily page (no website text in that baseline prompt).
</footer>""",
    )
    return html


def patch_tavily_scrape_html(html: str, n_input_ids: int, evidence_only: bool) -> str:
    nav_sub = (
        "Evidence cohort · Tavily-enriched"
        if evidence_only
        else "Full batch input · Tavily-enriched"
    )
    section = (
        "Evidence cohort (Tavily-enriched)"
        if evidence_only
        else "Full batch input (Tavily-enriched)"
    )
    html = html.replace('<div class="nav-sub">Results Dashboard</div>', f'<div class="nav-sub">{nav_sub}</div>')
    html = html.replace(
        '<span class="section-label">Taxonomy</span>',
        f'<span class="section-label">{section}</span>',
    )
    if evidence_only:
        cohort_sentence = (
            f"same <strong>{n_input_ids:,}</strong> paired companies as the baseline page: "
            "non-empty <strong>website_evidence</strong> in <code>classifier_input.csv</code> "
            "and a row in both exports. Side-by-side with the baseline view shows the effect of "
            "including Tavily crawl text in the prompt."
        )
    else:
        cohort_sentence = (
            f"restricted to the <strong>{n_input_ids:,}</strong> companies in the batch input file "
            "that appear in both exports (including rows where crawls did not return text)."
        )
    p1 = f"""  <p>
    <strong>Tavily-enriched</strong> classifications from <code>production_classifications.csv</code>,
    {cohort_sentence}
  </p>"""
    html = html.replace(_DEFAULT_OVERVIEW_P1, p1)
    html = html.replace(
        _DEFAULT_FOOTER_TAIL,
        """  <br>
  Cohort: production classifications for the same company IDs as the paired baseline page.
</footer>""",
    )
    return html


FILTER_JS = """
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


def build_classification_dashboard_html(
    mod,
    df: pd.DataFrame,
    output_path: Path,
    patch_fn,
    n_input_ids: int,
    evidence_only: bool,
) -> None:
    m = mod.compute_metrics(df)
    filter_data: dict = {}
    for cohort in ["ALL", "PRE-GENAI", "GENAI-ERA"]:
        for min_conf in range(1, 6):
            cdf = df.copy()
            if cohort != "ALL":
                cdf = cdf[cdf.cohort == cohort]
            cdf = cdf[cdf.conf_classification >= min_conf]
            key = f"{cohort}_{min_conf}"
            filter_data[key] = {
                "subclass": {
                    s: int((cdf.subclass == s).sum()) for s in mod.SUBCLASS_ORDER
                },
                "total": len(cdf),
                "ai_native": int((cdf.ai_native == 1).sum()),
            }

    html = mod.build_html(m)
    html = html.replace(
        "const FILTER_DATA = {};",
        f"const FILTER_DATA = {json.dumps(filter_data)};",
    )
    html = patch_fn(html, n_input_ids, evidence_only)
    html = html.replace("</body>", FILTER_JS + "\n</body>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build classification dashboards for classifier_input scrape cohort (baseline vs Tavily)."
    )
    parser.add_argument(
        "--classifier-input",
        type=Path,
        default=_DEFAULT_CLASSIFIER_INPUT,
        metavar="PATH",
        help=f"classifier_input.csv (default: {_DEFAULT_CLASSIFIER_INPUT})",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_DEFAULT_BASELINE,
        metavar="PATH",
        help=f"Migrated US baseline CSV (default: {_DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--production",
        type=Path,
        default=_DEFAULT_PRODUCTION,
        metavar="PATH",
        help=f"production_classifications.csv (default: {_DEFAULT_PRODUCTION})",
    )
    parser.add_argument(
        "-o1",
        "--output-baseline",
        type=Path,
        default=_OUT_BASELINE,
        help=f"Baseline scrape cohort HTML (default: {_OUT_BASELINE})",
    )
    parser.add_argument(
        "-o2",
        "--output-tavily",
        type=Path,
        default=_OUT_TAVILY,
        help=f"Tavily scrape cohort HTML (default: {_OUT_TAVILY})",
    )
    parser.add_argument(
        "--all-classifier-rows",
        action="store_true",
        help=(
            "Use every org_uuid in classifier_input.csv. Default: only rows with non-empty "
            "website_evidence (~20k when the rest of the batch had no crawl text)."
        ),
    )
    args = parser.parse_args()
    ci = args.classifier_input.expanduser().resolve()
    bp = args.baseline.expanduser().resolve()
    pp = args.production.expanduser().resolve()

    if not bp.is_file():
        print(f"Missing baseline CSV: {bp}", file=sys.stderr)
        sys.exit(1)
    if not pp.is_file():
        print(f"Missing production CSV: {pp}", file=sys.stderr)
        sys.exit(1)
    if not ci.is_file():
        print(f"Missing classifier_input CSV: {ci}", file=sys.stderr)
        sys.exit(1)

    evidence_only = not args.all_classifier_rows
    scrape_ids = load_scrape_company_ids(ci, require_website_evidence=evidence_only)
    n_input = len(scrape_ids)
    mode = "Evidence (non-empty website_evidence)" if evidence_only else "All classifier_input rows"
    print(f"{mode}: {n_input:,} unique org_uuid ← {ci}")

    baseline = _dedupe_ids(pd.read_csv(bp), "Baseline")
    production = _dedupe_ids(pd.read_csv(pp), "Production")

    df_baseline = baseline[baseline["CompanyID"].isin(scrape_ids)].copy()
    df_tavily = production[production["CompanyID"].isin(scrape_ids)].copy()

    missing_base = len(scrape_ids - set(df_baseline["CompanyID"]))
    missing_prod = len(scrape_ids - set(df_tavily["CompanyID"]))
    if missing_base > 0:
        print(f"  Note: {missing_base:,} cohort IDs missing from baseline CSV (dropped for pairing).")
    if missing_prod > 0:
        print(f"  Note: {missing_prod:,} cohort IDs missing from production CSV (dropped for pairing).")

    paired_ids = set(df_baseline["CompanyID"]) & set(df_tavily["CompanyID"])
    n_paired = len(paired_ids)
    if n_paired == 0:
        print("No CompanyIDs overlap between baseline and production for this cohort.", file=sys.stderr)
        sys.exit(1)
    df_baseline = df_baseline[df_baseline["CompanyID"].isin(paired_ids)].copy()
    df_tavily = df_tavily[df_tavily["CompanyID"].isin(paired_ids)].copy()
    if n_paired != len(scrape_ids):
        print(f"  Paired cohort (baseline ∩ production ∩ classifier filter): {n_paired:,} companies")

    print(f"  Baseline dashboard rows: {len(df_baseline):,}")
    print(f"  Tavily dashboard rows: {len(df_tavily):,}")

    mod = _load_dashboard_builder()

    print(f"Building baseline scrape cohort → {args.output_baseline}")
    build_classification_dashboard_html(
        mod,
        df_baseline,
        args.output_baseline.expanduser().resolve(),
        patch_baseline_scrape_html,
        n_paired,
        evidence_only,
    )

    print(f"Building Tavily scrape cohort → {args.output_tavily}")
    build_classification_dashboard_html(
        mod,
        df_tavily,
        args.output_tavily.expanduser().resolve(),
        patch_tavily_scrape_html,
        n_paired,
        evidence_only,
    )


if __name__ == "__main__":
    main()