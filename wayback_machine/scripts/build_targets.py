#!/usr/bin/env python3
"""Stage B CLI: freeze coverage_full.csv -> data/scrape_targets.csv.

Free, deterministic, re-runnable. Run this once the coverage probe is done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.config import COHORT_FOUNDED_CUTOFF  # noqa: E402
from wayback_machine.paths import COVERAGE_FULL_CSV, SCRAPE_TARGETS_CSV  # noqa: E402
from wayback_machine.targets import build_targets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=COVERAGE_FULL_CSV)
    parser.add_argument("--output", type=Path, default=SCRAPE_TARGETS_CSV)
    parser.add_argument("--founded-cutoff", default=COHORT_FOUNDED_CUTOFF,
                        help="Keep only companies founded on/before this YYYY-MM (existed at GPT-4 launch).")
    args = parser.parse_args()
    build_targets(args.coverage, args.output, args.founded_cutoff)


if __name__ == "__main__":
    main()
