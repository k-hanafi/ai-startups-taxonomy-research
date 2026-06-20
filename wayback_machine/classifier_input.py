"""Stage D: assemble ``classifier_input_2023.csv`` for the existing classifier.

The research design hinges on one thing: the only difference between the live
input and the 2023 input is the website evidence. So we take the SAME static
metadata base the live crawl used (``master_csv.csv``) and swap in the 2023
evidence for the companies we scraped. Companies without 2023 evidence are
dropped (the panel is retrievable-only), so the file is ready for ``classify.py``
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .cohort import CLASSIFIER_INPUT_COLUMNS
from .paths import CLASSIFIER_INPUT_2023_CSV, MASTER_CSV, SCRAPE_PROCESSED_CSV


def build_classifier_input_2023(
    master_csv: str | Path = MASTER_CSV,
    processed_csv: str | Path = SCRAPE_PROCESSED_CSV,
    output_csv: str | Path = CLASSIFIER_INPUT_2023_CSV,
) -> int:
    """Join master metadata + 2023 evidence. Returns the output row count."""
    master_path = Path(master_csv)
    processed_path = Path(processed_csv)
    if not master_path.exists():
        raise SystemExit(f"Master CSV not found: {master_path}")
    if not processed_path.exists() or processed_path.stat().st_size == 0:
        raise SystemExit(f"No scrape output yet: {processed_path}. Run the extract first.")

    master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)

    ev_cols = ["org_uuid", "website_pages_used", "website_evidence"]
    processed = processed[ev_cols].drop_duplicates(subset=["org_uuid"], keep="last")
    # Retrievable-only panel: keep companies that actually produced evidence.
    processed = processed[processed["website_evidence"].str.strip() != ""]

    output = master.merge(processed, on="org_uuid", how="inner")
    for col in CLASSIFIER_INPUT_COLUMNS:
        if col not in output.columns:
            output[col] = ""
    output = output[list(CLASSIFIER_INPUT_COLUMNS)].copy()

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"Wrote classifier_input_2023.csv: {len(output):,} rows -> {out_path}", file=sys.stderr)
    return len(output)
