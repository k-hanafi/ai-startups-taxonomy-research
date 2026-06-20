"""Stage B: freeze the retrievable companies into a scrape work list.

Reads the coverage probe output (``coverage_full.csv``), keeps only companies
with a usable March-2023 capture, and writes one row per company with the exact
archive snapshot URL Tavily will fetch. This is the frozen contract the paid
scrape runs against — small, deterministic, and re-runnable.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .cohort import build_snapshot_url, existed_by, is_retrievable
from .config import COHORT_FOUNDED_CUTOFF
from .paths import COVERAGE_FULL_CSV, SCRAPE_TARGETS_CSV

# Large evidence-free rows, but bump the cap so any wide field never trips csv.
csv.field_size_limit(1_000_000_000)

TARGET_FIELDS = [
    "org_uuid", "name", "homepage_url", "founded_date",
    "closest_ts", "days_from_target", "snapshot_url",
]


def _dedupe(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """One row per org_uuid, preferring a resolved (status ok) capture."""
    best: dict[str, dict[str, str]] = {}
    for r in rows:
        oid = (r.get("org_uuid") or "").strip()
        if not oid:
            continue
        prev = best.get(oid)
        if prev is None or (r.get("status") == "ok" and prev.get("status") != "ok"):
            best[oid] = r
    return best


def build_targets(
    coverage_csv: str | Path = COVERAGE_FULL_CSV,
    output_csv: str | Path = SCRAPE_TARGETS_CSV,
    founded_cutoff: str = COHORT_FOUNDED_CUTOFF,
) -> dict[str, int]:
    """Write the scrape target list. Returns counts for the caller to print.

    A company makes the list only if it is BOTH retrievable (has a usable
    March-2023 capture) AND existed by ``founded_cutoff`` — so we never scrape a
    later snapshot of a company that did not exist at GPT-4 launch.
    """
    coverage_path = Path(coverage_csv)
    if not coverage_path.exists():
        raise SystemExit(f"Coverage file not found: {coverage_path}")

    with coverage_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    deduped = _dedupe(rows)
    retrievable = [r for r in deduped.values() if is_retrievable(r)]
    final = [r for r in retrievable if existed_by(r.get("founded_date", ""), founded_cutoff)]
    dropped_not_existed = len(retrievable) - len(final)
    # Stable order: by org_uuid so the frozen list is reproducible run to run.
    final.sort(key=lambda r: r.get("org_uuid", ""))

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        for r in final:
            writer.writerow({
                "org_uuid": r.get("org_uuid", ""),
                "name": r.get("name", ""),
                "homepage_url": r.get("homepage_url", ""),
                "founded_date": r.get("founded_date", ""),
                "closest_ts": r.get("closest_ts", ""),
                "days_from_target": r.get("days_from_target", ""),
                "snapshot_url": build_snapshot_url(r.get("closest_ts", ""), r.get("homepage_url", "")),
            })

    counts = {
        "coverage_rows": len(rows),
        "unique_companies": len(deduped),
        "retrievable": len(retrievable),
        "dropped_not_existed": dropped_not_existed,
        "final": len(final),
    }
    print(
        f"Coverage rows:          {counts['coverage_rows']:,}\n"
        f"Unique companies:       {counts['unique_companies']:,}\n"
        f"Retrievable:            {counts['retrievable']:,}\n"
        f"Dropped (founded >{founded_cutoff}): {counts['dropped_not_existed']:,}\n"
        f"Final targets (out):    {counts['final']:,}\n"
        f"Wrote:                  {out_path}",
        file=sys.stderr,
    )
    return counts
