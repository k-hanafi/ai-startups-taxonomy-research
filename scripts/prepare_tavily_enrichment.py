#!/usr/bin/env python3
"""Build the lean classifier_input.csv (44k rows) for GPT + Tavily."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.enrichment import (
    DEFAULT_CLASSIFIER_INPUT_CSV,
    DEFAULT_MASTER_CSV,
    DEFAULT_SUBSET_CSV,
    write_enrichment_outputs,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join the 44k subset to master resource fields; write classifier_input.csv.",
    )
    parser.add_argument("--subset", type=_path, default=DEFAULT_SUBSET_CSV)
    parser.add_argument("--master", type=_path, default=DEFAULT_MASTER_CSV)
    parser.add_argument("--output", type=_path, default=DEFAULT_CLASSIFIER_INPUT_CSV)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = write_enrichment_outputs(
        subset_csv=args.subset,
        master_csv=args.master,
        enriched_csv=args.output,
    )
    print(report.format_report())
    print(f"  Output CSV:               {args.output}")


if __name__ == "__main__":
    main()
