#!/usr/bin/env python3
"""Build the 'Tavily not found' cohort for death-anchored Wayback discovery.

The survivorship-bias correction targets every company that did NOT yield usable
live website text: rows with an empty ``website_evidence`` in classifier_input.csv.
That is the union of dead/parked domains (``website_alive=false``) and live sites
Tavily could not extract (``website_alive=true`` but empty evidence). We keep only
the columns the probe needs and drop rows with no usable host (nothing to query).

Free and offline. Mirrors ``extract_cohort.py`` but selects the complement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.cdx import to_host  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "tavilycrawl" / "processed" / "classifier_input.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "wayback_machine" / "data" / "not_found_cohort.csv"

COHORT_FIELDS = ["org_uuid", "name", "homepage_url", "website_alive", "founded_date"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(
        args.input,
        usecols=["org_uuid", "name", "homepage_url", "website_alive",
                 "founded_date", "website_evidence"],
        dtype=str,
    ).fillna("")

    complement = df[df["website_evidence"].str.strip() == ""].copy()
    complement["host"] = complement["homepage_url"].map(to_host)
    feedable = complement[complement["host"] != ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feedable[COHORT_FIELDS].to_csv(args.output, index=False)

    print(
        f"complement(empty evidence)={len(complement):,}  "
        f"feedable(has host)={len(feedable):,}  "
        f"skipped(no host)={len(complement) - len(feedable):,}  -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
