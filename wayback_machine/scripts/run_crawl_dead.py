#!/usr/bin/env python3
"""Stage C CLI (survivorship): the paid archive crawl over the dead target list.

Resumable and safe to interrupt (Ctrl-C drains at the next row boundary;
re-running skips finished companies and never re-bills them). For the full
overnight run, wrap in caffeinate OUTSIDE the Cursor sandbox:

    caffeinate -ims python3 wayback_machine/scripts/run_crawl_dead.py

Smoke-test first with --max-companies 20.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tavily_crawl import TavilyCrawlConfig  # noqa: E402
from wayback_machine.config import DEFAULT_BUDGET_CREDITS  # noqa: E402
from wayback_machine.crawl_dead import (  # noqa: E402
    DEFAULT_CRAWL_DEAD_CONCURRENCY,
    run_crawl_dead,
)
from wayback_machine.paths import SCRAPE_TARGETS_DEAD_CSV  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=SCRAPE_TARGETS_DEAD_CSV)
    parser.add_argument("--max-companies", type=int, default=None,
                        help="Cap rows this run (smoke test / incremental).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CRAWL_DEAD_CONCURRENCY)
    parser.add_argument("--budget-credits", type=float, default=DEFAULT_BUDGET_CREDITS)
    parser.add_argument("--crawl-rpm", type=float, default=None,
                        help="Override crawl RPM cap. 0 disables the limiter.")
    args = parser.parse_args()

    report = run_crawl_dead(
        targets_csv=args.targets,
        config=TavilyCrawlConfig(),
        budget_credits=args.budget_credits,
        max_companies=args.max_companies,
        max_concurrent_rows=args.concurrency,
        crawl_rpm=args.crawl_rpm,
    )
    print("\n" + report.format_report(), file=sys.stderr)


if __name__ == "__main__":
    main()
