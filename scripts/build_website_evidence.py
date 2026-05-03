#!/usr/bin/env python3
"""Build classifier-ready CSV with compact Tavily website evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.enrichment import DEFAULT_ENRICHED_CSV
from src.tavily_crawl import DEFAULT_RAW_RESULTS_JSONL
from src.website_evidence import (
    DEFAULT_CLASSIFIER_INPUT_CSV,
    DEFAULT_MAX_EVIDENCE_CHARS,
    DEFAULT_MAX_PAGE_CHARS,
    build_classifier_input_with_evidence,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join compact Tavily evidence onto the enriched classifier input CSV.",
    )
    parser.add_argument("--enriched", type=_path, default=DEFAULT_ENRICHED_CSV)
    parser.add_argument("--raw-jsonl", type=_path, default=DEFAULT_RAW_RESULTS_JSONL)
    parser.add_argument("--output", type=_path, default=DEFAULT_CLASSIFIER_INPUT_CSV)
    parser.add_argument("--max-evidence-chars", type=int, default=DEFAULT_MAX_EVIDENCE_CHARS)
    parser.add_argument("--max-page-chars", type=int, default=DEFAULT_MAX_PAGE_CHARS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_classifier_input_with_evidence(
        enriched_csv=args.enriched,
        raw_jsonl=args.raw_jsonl,
        output_csv=args.output,
        max_evidence_chars=args.max_evidence_chars,
        max_page_chars=args.max_page_chars,
    )
    print(report.format_report())


if __name__ == "__main__":
    main()
