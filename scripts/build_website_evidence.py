#!/usr/bin/env python3
"""Build classifier-ready CSV with compact Tavily website evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.enrichment import DEFAULT_CLASSIFIER_INPUT_CSV
from src.tavily_crawl import DEFAULT_RAW_RESULTS_JSONL
from src.website_evidence import (
    DEFAULT_CLASSIFIER_INPUT_CSV,
    DEFAULT_MAX_EVIDENCE_CHARS,
    DEFAULT_MAX_PAGE_CHARS,
    build_classifier_input_with_evidence,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Tavily JSONL into classifier_input.csv (website evidence columns).",
    )
    parser.add_argument(
        "--input",
        "--enriched",
        type=_path,
        dest="input",
        default=DEFAULT_CLASSIFIER_INPUT_CSV,
    )
    parser.add_argument("--raw-jsonl", type=_path, default=DEFAULT_RAW_RESULTS_JSONL)
    parser.add_argument(
        "--output",
        type=_path,
        default=None,
        help="Defaults to --input (in-place update).",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=_optional_positive_int,
        default=DEFAULT_MAX_EVIDENCE_CHARS,
        help="Maximum total website evidence chars. Defaults to no truncation; use 0 to disable.",
    )
    parser.add_argument(
        "--max-page-chars",
        type=_optional_positive_int,
        default=DEFAULT_MAX_PAGE_CHARS,
        help="Maximum chars per crawled page. Defaults to no truncation; use 0 to disable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = args.output if args.output is not None else args.input
    report = build_classifier_input_with_evidence(
        enriched_csv=args.input,
        raw_jsonl=args.raw_jsonl,
        output_csv=out,
        max_evidence_chars=args.max_evidence_chars,
        max_page_chars=args.max_page_chars,
    )
    print(report.format_report())


if __name__ == "__main__":
    main()
