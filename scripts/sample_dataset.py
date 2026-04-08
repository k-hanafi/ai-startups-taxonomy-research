#!/usr/bin/env python3
"""Draw a reproducible random sample from the full Crunchbase CSV.

Usage:
    python scripts/sample_dataset.py [--n 1000] [--seed 42] [--output data/sample_1000.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = _PROJECT_ROOT / "data" / "company_us_short_long_desc_.csv"
DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "sample_1000.csv"


def _year_from_founded(s: str) -> int | None:
    """Extract a 4-digit year from strings like '01nov2016' or '2020-03-15'."""
    import re

    if not isinstance(s, str):
        return None
    m = re.search(r"((?:19|20)\d{2})", s)
    return int(m.group(1)) if m else None


def sample(source: Path, n: int, seed: int, output: Path) -> None:
    df = pd.read_csv(source)
    print(f"Full dataset: {len(df):,} rows")

    sample_df = df.sample(n=n, random_state=seed)
    sample_df.to_csv(output, index=False)
    print(f"Sample written: {len(sample_df):,} rows → {output}")

    # --- representativeness stats ---
    years = sample_df["founded_date"].apply(_year_from_founded)
    has_long = sample_df["Long description"].notna() & (sample_df["Long description"].str.strip() != "")
    has_keywords = sample_df["category_list"].notna() & (sample_df["category_list"].str.strip() != "")

    print(f"\n--- Sample stats ---")
    print(f"Founding year range : {int(years.min())} – {int(years.max())}")
    print(f"Median founding year: {int(years.median())}")
    print(f"With long description: {has_long.sum()} / {n} ({has_long.mean():.1%})")
    print(f"With keywords        : {has_keywords.sum()} / {n} ({has_keywords.mean():.1%})")

    decade_bins = pd.cut(years, bins=[1990, 2000, 2010, 2015, 2020, 2023, 2030],
                         labels=["<2000", "2000s", "2010-14", "2015-19", "2020-22", "2023+"])
    print(f"\nFounding decade distribution:")
    for label, count in decade_bins.value_counts().sort_index().items():
        print(f"  {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample N random rows from the dataset")
    parser.add_argument("--n", type=int, default=1000, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    sample(args.source, args.n, args.seed, args.output)


if __name__ == "__main__":
    main()
