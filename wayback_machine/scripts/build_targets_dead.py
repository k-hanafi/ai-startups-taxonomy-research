#!/usr/bin/env python3
"""Stage B CLI (survivorship): death_coverage.csv -> data/scrape_targets_dead.csv.

Free, deterministic, re-runnable. Run once the death-anchored probe is done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.paths import DEATH_COVERAGE_CSV, SCRAPE_TARGETS_DEAD_CSV  # noqa: E402
from wayback_machine.targets_dead import build_targets_dead  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=DEATH_COVERAGE_CSV)
    parser.add_argument("--output", type=Path, default=SCRAPE_TARGETS_DEAD_CSV)
    args = parser.parse_args()
    build_targets_dead(args.coverage, args.output)


if __name__ == "__main__":
    main()
