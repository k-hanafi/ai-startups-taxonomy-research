#!/usr/bin/env python3
"""Aggregate coverage_sample.csv into the numbers the dashboard embeds.

Prints a JSON blob to stdout. Run after probe_coverage.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "coverage_sample.csv"
COHORT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "wayback_cohort.csv"
TARGET = datetime(2023, 3, 14)


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse to one row per org_uuid, preferring resolved (status ok), then latest."""
    best: dict[str, dict[str, str]] = {}
    for r in rows:
        oid = r.get("org_uuid", "")
        if not oid:
            continue
        prev = best.get(oid)
        if prev is None or (r.get("status") == "ok") or (prev.get("status") != "ok"):
            best[oid] = r
    return list(best.values())


def year_of(founded: str) -> int | None:
    s = (founded or "").strip()[:4]
    return int(s) if s.isdigit() else None


def founded_bucket(year: int | None) -> str:
    if year is None:
        return "unknown"
    if year <= 2018:
        return "≤2018"
    if year >= 2024:
        return "2024+"
    return str(year)


BUCKET_ORDER = ["≤2018", "2019", "2020", "2021", "2022", "2023", "2024+", "unknown"]
DRIFT_BUCKETS = [("≤7d", 0, 7), ("8–30d", 8, 30), ("31–90d", 31, 90),
                 ("91–180d", 91, 180), (">180d", 181, 10**9)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SAMPLE_CSV)
    args = parser.parse_args()

    rows = dedupe(list(csv.DictReader(args.input.open(encoding="utf-8", newline=""))))
    cohort_n = sum(1 for _ in csv.DictReader(COHORT_CSV.open(encoding="utf-8", newline="")))

    resolved = [r for r in rows if r.get("status") == "ok"]
    has = [r for r in resolved if r.get("has_2023") == "True"]
    not_2023_ever = [r for r in resolved if r.get("has_2023") == "False" and r.get("has_any_ever") == "True"]
    never = [r for r in resolved if r.get("has_2023") == "False" and r.get("has_any_ever") == "False"]
    errors = [r for r in rows if r.get("status") != "ok"]

    n_res = len(resolved)
    p = len(has) / n_res if n_res else 0.0
    ci = 1.96 * math.sqrt(p * (1 - p) / n_res) if n_res else 0.0

    # Coverage by founding-year bucket (resolved only).
    by_bucket: dict[str, dict[str, int]] = {b: {"resolved": 0, "has": 0} for b in BUCKET_ORDER}
    for r in resolved:
        b = founded_bucket(year_of(r.get("founded_date", "")))
        by_bucket[b]["resolved"] += 1
        if r.get("has_2023") == "True":
            by_bucket[b]["has"] += 1

    # Founded <= 2022 (fair denominator for a 2023 comparison).
    pre = [r for r in resolved if (y := year_of(r.get("founded_date", ""))) is not None and y <= 2022]
    pre_has = [r for r in pre if r.get("has_2023") == "True"]
    p_pre = len(pre_has) / len(pre) if pre else 0.0
    ci_pre = 1.96 * math.sqrt(p_pre * (1 - p_pre) / len(pre)) if pre else 0.0

    # Date drift among companies WITH a 2023 capture.
    days = [int(r["days_from_target"]) for r in has if r.get("days_from_target", "").lstrip("-").isdigit()]
    drift = {label: sum(1 for d in days if lo <= d <= hi) for label, lo, hi in DRIFT_BUCKETS}
    median_days = statistics.median(days) if days else None

    # Concrete examples: closest captures to March 14.
    def ts_to_date(ts: str) -> str:
        try:
            return datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d")
        except ValueError:
            return ts

    examples = sorted(
        [r for r in has if r.get("closest_ts")],
        key=lambda r: int(r["days_from_target"]) if r["days_from_target"].lstrip("-").isdigit() else 10**9,
    )[:8]
    example_rows = [
        {
            "name": r.get("name", ""),
            "founded": (r.get("founded_date", "") or "")[:4],
            "snapshot_date": ts_to_date(r.get("closest_ts", "")),
            "days_from_target": int(r["days_from_target"]) if r["days_from_target"].lstrip("-").isdigit() else None,
            "wayback_url": f"https://web.archive.org/web/{r.get('closest_ts','')}/{r.get('homepage_url','')}",
        }
        for r in examples
    ]

    out = {
        "cohort_n": cohort_n,
        "sample_n": len(rows),
        "resolved_n": n_res,
        "errors_n": len(errors),
        "has_2023_n": len(has),
        "coverage_pct": round(p * 100, 1),
        "coverage_ci": round(ci * 100, 1),
        "composition": {
            "has_2023": len(has),
            "archived_not_2023": len(not_2023_ever),
            "never_archived": len(never),
            # When the 'ever' call is skipped on big runs, misses aren't split.
            "miss_unknown": len(resolved) - len(has) - len(not_2023_ever) - len(never),
        },
        "pre2022": {
            "resolved_n": len(pre),
            "has_2023_n": len(pre_has),
            "coverage_pct": round(p_pre * 100, 1),
            "coverage_ci": round(ci_pre * 100, 1),
        },
        "by_founding_year": [
            {
                "bucket": b,
                "resolved": by_bucket[b]["resolved"],
                "has": by_bucket[b]["has"],
                "pct": round(by_bucket[b]["has"] / by_bucket[b]["resolved"] * 100, 1)
                if by_bucket[b]["resolved"] else 0.0,
            }
            for b in BUCKET_ORDER if by_bucket[b]["resolved"] > 0
        ],
        "drift": drift,
        "median_days_from_target": median_days,
        "examples": example_rows,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
