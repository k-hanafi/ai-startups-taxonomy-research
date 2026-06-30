"""Stage B (survivorship): freeze the dead cohort into a crawl work list.

Reads the death-anchored probe output (``death_coverage.csv``) and writes one row
per resolvable company with the exact pre-death archive URL the crawler will hit.
The GO methodology matches the live cohort: a 5-page Tavily crawl on the Wayback
``if_`` snapshot (iframe mode keeps links inside the archive), scoped to the
company's own archived pages.

Unlike the historical Stage B there is no founded-date cutoff: every dead company
that resolved a usable capture is in scope. ``thin_history`` is carried as a flag
(not a filter) so the merge can mark companies whose archive was shorter than the
6-month pre-death buffer.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .paths import DEATH_COVERAGE_CSV, SCRAPE_TARGETS_DEAD_CSV
from .tavily_archive_lab import _scope_for, archive_url

# death_coverage rows carry full-page evidence-free metadata, but bump the cap so
# any wide field never trips csv.
csv.field_size_limit(1_000_000_000)

TARGET_FIELDS = [
    "org_uuid", "name", "homepage_url", "founded_date", "closest_ts",
    "snapshot_url", "select_paths", "website_alive", "thin_history",
    "death_ts", "days_before_death",
]


def _is_resolvable(row: dict[str, str]) -> bool:
    """True if a probe row has a usable pre-death capture to crawl."""
    return row.get("status") == "ok" and bool((row.get("closest_ts") or "").strip())


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


def build_targets_dead(
    coverage_csv: str | Path = DEATH_COVERAGE_CSV,
    output_csv: str | Path = SCRAPE_TARGETS_DEAD_CSV,
) -> dict[str, int]:
    """Write the dead-cohort crawl target list. Returns counts for the caller."""
    coverage_path = Path(coverage_csv)
    if not coverage_path.exists():
        raise SystemExit(f"Death coverage file not found: {coverage_path}")

    with coverage_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    deduped = _dedupe(rows)
    final = [r for r in deduped.values() if _is_resolvable(r)]
    final.sort(key=lambda r: r.get("org_uuid", ""))

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thin = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        for r in final:
            homepage = r.get("homepage_url", "")
            closest_ts = r.get("closest_ts", "")
            select_paths, _ = _scope_for(homepage)
            if r.get("thin_history") == "True":
                thin += 1
            writer.writerow({
                "org_uuid": r.get("org_uuid", ""),
                "name": r.get("name", ""),
                "homepage_url": homepage,
                "founded_date": r.get("founded_date", ""),
                "closest_ts": closest_ts,
                "snapshot_url": archive_url(homepage, closest_ts, "if_"),
                "select_paths": select_paths[0] if select_paths else "",
                "website_alive": r.get("website_alive", ""),
                "thin_history": r.get("thin_history", ""),
                "death_ts": r.get("death_ts", ""),
                "days_before_death": r.get("days_before_death", ""),
            })

    counts = {
        "coverage_rows": len(rows),
        "unique_companies": len(deduped),
        "final": len(final),
        "thin_history": thin,
    }
    print(
        f"Coverage rows:        {counts['coverage_rows']:,}\n"
        f"Unique companies:     {counts['unique_companies']:,}\n"
        f"Final targets (out):  {counts['final']:,}\n"
        f"  of which thin_history: {counts['thin_history']:,}\n"
        f"Wrote:                {out_path}",
        file=sys.stderr,
    )
    return counts
