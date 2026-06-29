#!/usr/bin/env python3
"""Stage F (survivorship): overlay dead-cohort verdicts onto the modern dataset.

All 44,387 companies already sit in ``production_classifications.csv`` — but the
~19k dead ones were labelled on metadata alone (empty website evidence). This
step overlays the evidence-based verdict we just recovered for each dead company,
tags every row with its ``evidence_source``, and writes the survivorship-corrected
dataset plus the before/after distribution — the headline survivorship-bias result.

It is a pure overlay (read-only on both inputs): the modern CSV is never mutated;
a new corrected CSV is written under ``outputs/wayback_dead/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS = PROJECT_ROOT / "outputs"
PRODUCTION_CSV = OUTPUTS / "production_csvs" / "production_classifications.csv"
DEAD_CSV = OUTPUTS / "wayback_dead" / "wayback_dead_classifications.csv"
TARGETS_DEAD_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "scrape_targets_dead.csv"
CORRECTED_CSV = OUTPUTS / "wayback_dead" / "survivorship_corrected.csv"

# Verdict fields overlaid from the dead run; CompanyID/CompanyName stay from the
# modern row (they are copied verbatim, so the join key is stable).
VERDICT_COLS = [
    "ai_native", "subclass", "rad_score", "cohort", "conf_classification",
    "conf_rad", "reasons_3_points", "sources_used", "verification_critique",
]


def _dist(series: pd.Series) -> dict[str, int]:
    return series.value_counts(dropna=False).to_dict()


def _print_before_after(console: Console, before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Compare metadata-only (before) vs recovered-evidence (after) on the dead cohort."""
    n = len(after)
    ai_before = pd.to_numeric(before["ai_native"], errors="coerce").fillna(0)
    ai_after = pd.to_numeric(after["ai_native"], errors="coerce").fillna(0)

    table = Table(title=f"Dead cohort: metadata-only -> recovered evidence (n={n:,})")
    table.add_column("Metric")
    table.add_column("Before (metadata)", justify="right")
    table.add_column("After (evidence)", justify="right")
    table.add_row(
        "AI-native rate",
        f"{ai_before.mean() * 100:.1f}%",
        f"{ai_after.mean() * 100:.1f}%",
    )
    table.add_row("AI-native count", f"{int(ai_before.sum()):,}", f"{int(ai_after.sum()):,}")
    console.print(table)

    flips = Table(title="Verdict changes vs the metadata-only baseline")
    flips.add_column("Field")
    flips.add_column("Changed", justify="right")
    flips.add_column("% of cohort", justify="right")
    for col in ("ai_native", "subclass", "rad_score"):
        changed = int((before[col].astype(str).values != after[col].astype(str).values).sum())
        flips.add_row(col, f"{changed:,}", f"{changed / n * 100:.1f}%" if n else "n/a")
    console.print(flips)


def merge(production_csv: Path, dead_csv: Path, targets_csv: Path, output_csv: Path) -> None:
    console = Console()
    if not production_csv.exists():
        raise SystemExit(f"Modern classifications not found: {production_csv}")
    if not dead_csv.exists():
        raise SystemExit(f"Dead classifications not found: {dead_csv}. Run classify_dead.py first.")

    prod = pd.read_csv(production_csv, dtype=str, keep_default_na=False)
    dead = pd.read_csv(dead_csv, dtype=str, keep_default_na=False)
    dead = dead.drop_duplicates(subset="CompanyID", keep="last").set_index("CompanyID")

    provenance: dict[str, dict[str, str]] = {}
    if targets_csv.exists():
        tgt = pd.read_csv(targets_csv, dtype=str, keep_default_na=False)
        provenance = {
            str(r["org_uuid"]): {"snapshot_ts": r.get("closest_ts", ""),
                                 "thin_history": r.get("thin_history", "")}
            for _, r in tgt.iterrows()
        }

    corrected = prod.copy()
    corrected["evidence_source"] = "live"
    corrected["snapshot_ts"] = ""
    corrected["thin_history"] = ""

    dead_ids = set(dead.index)
    is_dead = corrected["CompanyID"].isin(dead_ids)
    baseline = corrected[is_dead].copy()  # metadata-only "before" for the recovered set

    overlaid = 0
    for idx in corrected.index[is_dead]:
        cid = corrected.at[idx, "CompanyID"]
        drow = dead.loc[cid]
        for col in VERDICT_COLS:
            if col in drow:
                corrected.at[idx, col] = drow[col]
        corrected.at[idx, "evidence_source"] = "wayback_dead"
        prov = provenance.get(cid, {})
        corrected.at[idx, "snapshot_ts"] = prov.get("snapshot_ts", "")
        corrected.at[idx, "thin_history"] = prov.get("thin_history", "")
        overlaid += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_csv, index=False)

    after = corrected[corrected["CompanyID"].isin(dead_ids)].sort_values("CompanyID")
    baseline = baseline.sort_values("CompanyID")
    console.print(f"\n[bold]Overlaid {overlaid:,} recovered dead verdicts onto "
                  f"{len(prod):,} modern rows.[/]  -> {output_csv}\n")
    if overlaid:
        _print_before_after(console, baseline, after)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=PRODUCTION_CSV)
    parser.add_argument("--dead", type=Path, default=DEAD_CSV)
    parser.add_argument("--targets", type=Path, default=TARGETS_DEAD_CSV)
    parser.add_argument("--output", type=Path, default=CORRECTED_CSV)
    args = parser.parse_args()
    merge(args.production, args.dead, args.targets, args.output)


if __name__ == "__main__":
    main()
