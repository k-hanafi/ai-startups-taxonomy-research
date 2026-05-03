#!/usr/bin/env python3
"""Run the cost-controlled Tavily Crawl enrichment job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tavily_crawl import (
    DEFAULT_CRAWL_STATE_JSON,
    DEFAULT_RAW_RESULTS_JSONL,
    TavilyCrawlConfig,
    run_tavily_crawl,
)
from src.enrichment import DEFAULT_CLASSIFIER_INPUT_CSV


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Tavily Crawl over the prepared company homepage queue.",
    )
    parser.add_argument(
        "--input",
        "--queue",
        type=_path,
        dest="input",
        default=DEFAULT_CLASSIFIER_INPUT_CSV,
        help="classifier_input.csv (Tavily skips invalid URLs and website_alive=false).",
    )
    parser.add_argument("--output", type=_path, default=DEFAULT_RAW_RESULTS_JSONL)
    parser.add_argument("--state", type=_path, default=DEFAULT_CRAWL_STATE_JSON)
    parser.add_argument("--budget-credits", type=float, default=100_000.0)
    parser.add_argument("--max-companies", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-breadth", type=int, default=20)
    parser.add_argument("--chunks-per-source", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--allow-external", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TavilyCrawlConfig(
        limit=args.limit,
        max_depth=args.max_depth,
        max_breadth=args.max_breadth,
        chunks_per_source=args.chunks_per_source,
        timeout=args.timeout,
        allow_external=args.allow_external,
    )
    report = run_tavily_crawl(
        queue_csv=args.input,
        output_jsonl=args.output,
        state_json=args.state,
        config=config,
        budget_credits=args.budget_credits,
        max_companies=args.max_companies,
        sleep_seconds=args.sleep_seconds,
    )
    print(report.format_report())
    print(f"  Raw results JSONL:        {args.output}")
    print(f"  Crawl state JSON:         {args.state}")


if __name__ == "__main__":
    main()
