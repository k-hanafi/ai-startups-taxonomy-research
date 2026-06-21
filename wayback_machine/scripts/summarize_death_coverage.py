#!/usr/bin/env python3
"""Aggregate death_coverage.csv into the compact JSON the survivorship canvas embeds.

Read-only. Prints one JSON blob to stdout (optionally also writes it with --out).
Mirrors summarize_coverage.py's reporting patterns (founded/drift buckets,
timestamp->date, examples with a Wayback URL) but uses pandas for the histogram
and monthly-series work. The death probe still appends live, so the CSV is read
with a tolerant parser and treated as a point-in-time snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.config import DEATH_LOOKBACK_DAYS  # noqa: E402

DEATH_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "death_coverage.csv"
COHORT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "not_found_cohort.csv"

GPT4_LAUNCH = pd.Timestamp("2023-03-14")
TEMPORAL_START = pd.Timestamp("2021-01-01")  # keep the time-series readable
TS_COLS = ["first_ts", "death_ts", "target_ts", "closest_ts"]
NUM_COLS = ["n_captures", "days_from_target", "days_before_death", "lifespan_days"]

FOUNDED_ORDER = ["<=2018", "2019", "2020", "2021", "2022", "2023", "2024+", "unknown"]
DRIFT_BUCKETS = [("<=7d", 0, 7), ("8-30d", 8, 30), ("31-90d", 31, 90),
                 ("91-180d", 91, 180), (">180d", 181, 10**18)]
CAPTURE_BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20),
                   ("21-100", 21, 100), ("100+", 101, 10**18)]
# days_before_death is bounded below by the lookback for pre-death snapshots, so
# bins above the floor show how much *older* than the 6-month anchor we landed.
BUFFER_BUCKETS = [("<182d", 0, 181), ("182-270d", 182, 270), ("271-365d", 271, 365),
                  ("1-1.5yr", 366, 545), ("1.5-2yr", 546, 730), ("2yr+", 731, 10**18)]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    for c in TS_COLS:
        df[c] = pd.to_datetime(df[c], format="%Y%m%d%H%M%S", errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def cohort_size(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_csv(path, usecols=[0], dtype=str))


def bucketize(series: pd.Series, buckets: list[tuple[str, int, int]]) -> list[dict]:
    """Count a numeric series into labeled [lo, hi] ranges; drop empty buckets."""
    vals = series.dropna()
    out = []
    for label, lo, hi in buckets:
        n = int(((vals >= lo) & (vals <= hi)).sum())
        if n:
            out.append({"label": label, "count": n})
    return out


def founded_bucket(year: float) -> str:
    if pd.isna(year):
        return "unknown"
    y = int(year)
    if y <= 2018:
        return "<=2018"
    if y >= 2024:
        return "2024+"
    return str(y)


def monthly_series(death: pd.Series, closest: pd.Series) -> list[dict]:
    death = death.dropna()
    death = death[death >= TEMPORAL_START]
    closest = closest.dropna()
    closest = closest[closest >= TEMPORAL_START]
    dc = death.dt.to_period("M").astype(str).value_counts().to_dict()
    cc = closest.dt.to_period("M").astype(str).value_counts().to_dict()
    months = sorted(set(dc) | set(cc))
    return [{"month": m, "death": int(dc.get(m, 0)), "closest": int(cc.get(m, 0))}
            for m in months]


def example_row(r: pd.Series, thin: bool) -> dict:
    snap = r["closest_ts"]
    return {
        "name": r["name"],
        "founded": (r["founded_date"] or "")[:7],
        "snapshot_date": snap.strftime("%Y-%m-%d") if pd.notna(snap) else "",
        "days_before_death": int(r["days_before_death"]) if pd.notna(r["days_before_death"]) else None,
        "days_from_target": int(r["days_from_target"]) if pd.notna(r["days_from_target"]) else None,
        "n_captures": int(r["n_captures"]) if pd.notna(r["n_captures"]) else None,
        "url": r["target_url"],
        "thin": thin,
    }


def summarize(df: pd.DataFrame, cohort_n: int) -> dict:
    status = df["status"]
    ok = status == "ok"
    has_pre = ok & (df["has_pre_death_snapshot"] == "True")
    thin = ok & (df["thin_history"] == "True")

    probed_n = len(df)
    ok_n, pre_n, thin_n = int(ok.sum()), int(has_pre.sum()), int(thin.sum())
    no_snap_n = int((status == "no_snapshots").sum())
    no_host_n = int((status == "no_host").sum())
    err_n = int(status.str.startswith("error").sum())

    closest_pre = df.loc[has_pre, "closest_ts"].dropna()
    pre_genai = int((closest_pre < GPT4_LAUNCH).sum())
    post_genai = int((closest_pre >= GPT4_LAUNCH).sum())

    years = pd.to_numeric(df["founded_date"].str[:4], errors="coerce")
    fb = years.apply(founded_bucket).value_counts().to_dict()
    founded = [{"label": b, "count": int(fb[b])} for b in FOUNDED_ORDER if fb.get(b, 0)]

    wa = df["website_alive"].str.lower()
    alive_n = int((wa == "true").sum())
    dead_n = int((wa == "false").sum())
    unknown_n = probed_n - alive_n - dead_n
    website = [seg for seg in (
        {"label": "flagged alive", "count": alive_n},
        {"label": "flagged dead", "count": dead_n},
        {"label": "unknown", "count": unknown_n},
    ) if seg["count"]]

    pre_df = df[has_pre]
    best = pre_df.sort_values("days_from_target").head(6)
    thin_df = df[thin].sort_values("lifespan_days", ascending=False).head(3)
    examples = ([example_row(r, False) for _, r in best.iterrows()]
                + [example_row(r, True) for _, r in thin_df.iterrows()])

    drift = pre_df["days_from_target"]
    dbd = pre_df["days_before_death"]
    return {
        "meta": {
            "cohort_n": cohort_n,
            "probed_n": probed_n,
            "probed_pct": round(probed_n / cohort_n * 100, 1) if cohort_n else None,
            "lookback_days": DEATH_LOOKBACK_DAYS,
            "gpt4_launch": GPT4_LAUNCH.strftime("%Y-%m-%d"),
        },
        "recovery": {
            "pre_death": pre_n,
            "thin": thin_n,
            "no_snapshots": no_snap_n,
            "no_host": no_host_n,
            "error": err_n,
            "ok": ok_n,
            "recovery_pct": round(pre_n / probed_n * 100, 1) if probed_n else None,
        },
        "status_breakdown": [
            {"status": k, "count": int(v)}
            for k, v in status.value_counts().items()
        ],
        "quality": {
            "drift_buckets": bucketize(drift, DRIFT_BUCKETS),
            "capture_buckets": bucketize(df.loc[ok, "n_captures"], CAPTURE_BUCKETS),
            "buffer_buckets": bucketize(dbd, BUFFER_BUCKETS),
            "median_drift_days": int(drift.median()) if drift.notna().any() else None,
            "median_buffer_days": int(dbd.median()) if dbd.notna().any() else None,
            "median_captures": int(df.loc[ok, "n_captures"].median()) if ok_n else None,
            "median_lifespan_days": int(df.loc[ok, "lifespan_days"].median()) if ok_n else None,
            "within_30d_of_target_pct": round(float((drift <= 30).mean()) * 100, 1) if pre_n else None,
            "thin_pct": round(thin_n / ok_n * 100, 1) if ok_n else None,
        },
        "temporal": {
            "monthly": monthly_series(df.loc[ok, "death_ts"], df.loc[has_pre, "closest_ts"]),
            "closest_pre_genai": pre_genai,
            "closest_post_genai": post_genai,
            "window_start": TEMPORAL_START.strftime("%Y-%m"),
        },
        "composition": {"founded_year": founded, "website_alive": website},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEATH_CSV)
    parser.add_argument("--cohort", type=Path, default=COHORT_CSV)
    parser.add_argument("--out", type=Path, default=None, help="Also write JSON here.")
    args = parser.parse_args()

    df = load(args.input)
    out = summarize(df, cohort_size(args.cohort))
    blob = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(blob + "\n", encoding="utf-8")
    print(blob)


if __name__ == "__main__":
    main()
