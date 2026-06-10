#!/usr/bin/env python3
"""Stage D CLI: scrape output -> classifier_input_2023.csv (free).

Run after the extract finishes. The result drops straight into the existing
classifier (same columns as classifier_input.csv, retrievable companies only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.classifier_input import build_classifier_input_2023  # noqa: E402
from wayback_machine.paths import (  # noqa: E402
    CLASSIFIER_INPUT_2023_CSV,
    MASTER_CSV,
    SCRAPE_PROCESSED_CSV,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER_CSV)
    parser.add_argument("--processed", type=Path, default=SCRAPE_PROCESSED_CSV)
    parser.add_argument("--output", type=Path, default=CLASSIFIER_INPUT_2023_CSV)
    args = parser.parse_args()
    build_classifier_input_2023(args.master, args.processed, args.output)


if __name__ == "__main__":
    main()
