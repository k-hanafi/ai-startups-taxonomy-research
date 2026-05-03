#!/usr/bin/env python3
"""Prepare the 44k classifier enrichment CSV and Tavily crawl queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.enrichment import (
    DEFAULT_CRAWL_QUEUE_CSV,
    DEFAULT_ENRICHED_CSV,
    DEFAULT_MASTER_CSV,
    DEFAULT_SUBSET_CSV,
    write_enrichment_outputs,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join the 44k startup subset to selected master fields for Tavily enrichment.",
    )
    parser.add_argument("--subset", type=_path, default=DEFAULT_SUBSET_CSV)
    parser.add_argument("--master", type=_path, default=DEFAULT_MASTER_CSV)
    parser.add_argument("--enriched-output", type=_path, default=DEFAULT_ENRICHED_CSV)
    parser.add_argument("--queue-output", type=_path, default=DEFAULT_CRAWL_QUEUE_CSV)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = write_enrichment_outputs(
        subset_csv=args.subset,
        master_csv=args.master,
        enriched_csv=args.enriched_output,
        crawl_queue_csv=args.queue_output,
    )
    print(report.format_report())
    print(f"  Enriched CSV:             {args.enriched_output}")
    print(f"  Crawl queue CSV:          {args.queue_output}")


if __name__ == "__main__":
    main()
