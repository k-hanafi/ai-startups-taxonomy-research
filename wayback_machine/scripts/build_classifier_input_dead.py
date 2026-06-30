#!/usr/bin/env python3
"""Stage D CLI (survivorship): master metadata + dead evidence -> classifier_input_dead.csv.

Free, deterministic. Reuses the historical Stage-D join verbatim — it inner-joins
the recovered evidence onto master_csv.csv, so the only thing that differs from
the live + 2023 classifier inputs is the website evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.classifier_input import build_classifier_input_2023  # noqa: E402
from wayback_machine.paths import (  # noqa: E402
    CLASSIFIER_INPUT_DEAD_CSV,
    MASTER_CSV,
    SCRAPE_PROCESSED_DEAD_CSV,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER_CSV)
    parser.add_argument("--processed", type=Path, default=SCRAPE_PROCESSED_DEAD_CSV)
    parser.add_argument("--output", type=Path, default=CLASSIFIER_INPUT_DEAD_CSV)
    args = parser.parse_args()
    build_classifier_input_2023(args.master, args.processed, args.output)


if __name__ == "__main__":
    main()
