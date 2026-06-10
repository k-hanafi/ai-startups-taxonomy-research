#!/usr/bin/env python3
"""Freeze the longitudinal cohort: companies the live Tavily crawl returned.

Streams the large classifier_input.csv, keeps only rows with non-empty
website_evidence (the companies we actually have current evidence for), and
writes a lean metadata CSV. The bulky evidence text itself is dropped; we keep
its length as `live_evidence_chars` so the dashboard can show how rich today's
evidence is without carrying ~200MB of text into the discovery phase.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = PROJECT_ROOT / "outputs" / "tavilycrawl" / "processed" / "classifier_input.csv"
OUTPUT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "wayback_cohort.csv"

KEEP_COLUMNS = [
    "org_uuid",
    "name",
    "homepage_url",
    "short_description",
    "category_list",
    "category_groups_list",
    "founded_date",
    "employee_count",
    "total_funding_usd",
    "website_pages_used",
]
OUTPUT_COLUMNS = KEEP_COLUMNS + ["live_evidence_chars"]

# Evidence fields can be tens of thousands of chars; the default csv field cap is too small.
csv.field_size_limit(1_000_000_000)


def main() -> None:
    if not SOURCE_CSV.exists():
        raise SystemExit(f"Source not found: {SOURCE_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    with SOURCE_CSV.open(encoding="utf-8", newline="") as src, \
            OUTPUT_CSV.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in reader:
            total += 1
            evidence = (row.get("website_evidence") or "").strip()
            if not evidence:
                continue
            out = {col: row.get(col, "") for col in KEEP_COLUMNS}
            out["live_evidence_chars"] = len(evidence)
            writer.writerow(out)
            kept += 1

    pct = (kept / total * 100) if total else 0.0
    print(f"Scanned rows:            {total:,}")
    print(f"Tavily-returned cohort:  {kept:,} ({pct:.1f}% of scanned)")
    print(f"Wrote:                   {OUTPUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
