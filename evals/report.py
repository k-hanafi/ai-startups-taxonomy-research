"""Focused cost-extrapolation report for one scored run.

Stage 9 will own the full dashboard. Until then, ``python -m evals report``
renders the production-cost ladder from ``scored.json`` as a small HTML
snippet (and prints the same ladder to the terminal). The HTML is a pure
renderer: all math lives in ``evals.cost_extrapolate`` so numbers stay
reproducible without re-deriving JS.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

from evals.cost_extrapolate import format_cost_ladder
from evals.paths import RUNS_DIR, run_dir, run_scored_path

logger = logging.getLogger(__name__)


def cost_report_path(run_id: str) -> Path:
    return run_dir(run_id) / "cost_report.html"


def _step_row(label: str, value: str, note: str = "") -> str:
    note_html = (
        f'<div class="note">{html.escape(note)}</div>' if note else ""
    )
    return (
        f"<tr><td>{html.escape(label)}{note_html}</td>"
        f"<td class=\"num\">{html.escape(value)}</td></tr>\n"
    )


def render_cost_html(run_id: str, estimate: dict) -> str:
    """Minimal static HTML for the sync-priced cost ladder (embeddable later)."""
    assumptions = estimate.get("assumptions") or {}
    steps = estimate.get("steps") or {}
    rows: list[str] = []

    s1 = steps.get("1_golden_sync")
    if s1:
        rows.append(
            _step_row(
                "1. Golden sync list",
                f"${s1['total_usd']:,.4f}",
                s1.get("note", ""),
            )
        )
    s2 = steps.get("2_cache")
    if s2:
        if s2.get("available"):
            rows.append(
                _step_row(
                    f"2. After cache ({s2['cache_hit_rate']:.1%} hit)",
                    f"${s2['total_usd_after_cache']:,.4f}",
                    s2.get("note", ""),
                )
            )
        else:
            rows.append(
                _step_row("2. Cache adjustment", "UNAVAILABLE", s2.get("reason", ""))
            )
    s3 = steps.get("3_scale")
    if s3:
        if s3.get("available"):
            rows.append(
                _step_row(
                    f"3. Scaled to N={s3['n_prod']:,}",
                    f"${s3['estimated_production_usd']:,.2f}",
                    s3.get("note", ""),
                )
            )
        else:
            rows.append(
                _step_row("3. Scale to production", "UNAVAILABLE", s3.get("reason", ""))
            )

    avail = "available" if estimate.get("available") else "unavailable"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Cost extrapolation — {html.escape(run_id)}</title>
<style>
  body {{ font-family: "IBM Plex Sans", system-ui, sans-serif; margin: 2rem;
         background: #f7f5f0; color: #1a1a1a; max-width: 42rem; }}
  h1 {{ font-size: 1.25rem; font-weight: 600; }}
  .meta {{ font-size: 0.9rem; color: #444; margin-bottom: 1.25rem; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 0.45rem 0.25rem; border-bottom: 1px solid #ddd; vertical-align: top; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .note {{ font-size: 0.8rem; color: #555; margin: 0 0 0.6rem 0; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; font-size: 0.75rem;
            background: #e8e4dc; border-radius: 3px; }}
  .assumptions {{ margin-top: 1.5rem; font-size: 0.85rem; }}
  .assumptions li {{ margin: 0.25rem 0; }}
</style>
</head>
<body>
<h1>Production cost extrapolation</h1>
<p class="meta">
  Run <strong>{html.escape(run_id)}</strong>
  · <span class="badge">{avail}</span>
  · {html.escape(str(assumptions.get("architecture", "?")))}
  · model {html.escape(str(assumptions.get("model", "?")))}
</p>
<table>
{"".join(rows)}
</table>
<div class="assumptions">
  <strong>Assumptions</strong>
  <ul>
    <li>N_prod = {html.escape(str(assumptions.get("n_prod")))}
        ({html.escape(str(assumptions.get("n_prod_label")))})</li>
    <li>n_golden = {html.escape(str(assumptions.get("n_golden")))}</li>
    <li>Cache source: {html.escape(str(assumptions.get("cache_source")))}</li>
    <li>Sync Responses API pricing (no Batch API discount),
        cache discount = {html.escape(str(assumptions.get("cache_discount")))}</li>
    <li>Reasoning tokens billed inside output</li>
    <li>Do not use historical production cache rate
        ({html.escape(str(assumptions.get("do_not_use_historical_production_cache_rate")))})</li>
  </ul>
</div>
</body>
</html>
"""


def write_cost_report(run_id: str) -> Path:
    """Load scored.json, write cost_report.html, print the ladder. Returns path."""
    scored_path = run_scored_path(run_id)
    if not scored_path.exists():
        raise SystemExit(
            f"No scored.json for run {run_id} at {scored_path}. "
            f"Run: python -m evals score {run_id}"
        )
    scored = json.loads(scored_path.read_text(encoding="utf-8"))
    estimate = scored.get("production_cost_estimate")
    if estimate is None:
        raise SystemExit(
            f"scored.json for {run_id} has no production_cost_estimate. "
            "Re-score the run with the current scorer."
        )

    text = format_cost_ladder(estimate)
    for line in text.splitlines():
        logger.info("%s", line)

    out = cost_report_path(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_cost_html(run_id, estimate), encoding="utf-8")
    logger.info("Wrote %s", out)
    return out


def report_cli(run_id: str | None = None) -> None:
    """CLI: report one run, or the most recently modified scored run."""
    if run_id is None:
        candidates = sorted(
            RUNS_DIR.glob("*/scored.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(f"No scored.json under {RUNS_DIR}")
        run_id = candidates[0].parent.name
        logger.info("No run_id given; using most recent scored run: %s", run_id)
    write_cost_report(run_id)
