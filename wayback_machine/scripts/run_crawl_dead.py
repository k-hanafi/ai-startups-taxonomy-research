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
from wayback_machine.config import (  # noqa: E402
    DEFAULT_BUDGET_CREDITS,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_MAX_OUTAGE_SECONDS,
)
from wayback_machine.crawl_dead import (  # noqa: E402
    DEFAULT_CRAWL_DEAD_CONCURRENCY,
    run_crawl_dead,
)
from wayback_machine.paths import (  # noqa: E402
    CRAWL_DEAD_JSONL,
    CRAWL_STATE_DEAD_JSON,
    SCRAPE_PROCESSED_DEAD_CSV,
    SCRAPE_TARGETS_DEAD_CSV,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=SCRAPE_TARGETS_DEAD_CSV)
    parser.add_argument("--max-companies", type=int, default=None,
                        help="Cap rows this run (smoke test / incremental).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CRAWL_DEAD_CONCURRENCY)
    parser.add_argument("--budget-credits", type=float, default=DEFAULT_BUDGET_CREDITS)
    parser.add_argument("--crawl-rpm", type=float, default=None,
                        help="Override crawl RPM cap. 0 disables the limiter.")
    parser.add_argument("--max-outage-seconds", type=float, default=DEFAULT_MAX_OUTAGE_SECONDS,
                        help="Cap the transient-failure retry window (lower it for fast smoke tests).")
    parser.add_argument("--heartbeat-every", type=int, default=DEFAULT_HEARTBEAT_EVERY,
                        help="Log progress every N companies (0 disables).")
    # Output overrides: point these at a throwaway dir to smoke-test without
    # touching the real run's resumable JSONL / processed CSV / state.
    parser.add_argument("--output-jsonl", type=Path, default=CRAWL_DEAD_JSONL)
    parser.add_argument("--processed", type=Path, default=SCRAPE_PROCESSED_DEAD_CSV)
    parser.add_argument("--state", type=Path, default=CRAWL_STATE_DEAD_JSON)
    args = parser.parse_args()

    report = run_crawl_dead(
        targets_csv=args.targets,
        output_jsonl=args.output_jsonl,
        state_json=args.state,
        config=TavilyCrawlConfig(),
        processed_csv=args.processed,
        budget_credits=args.budget_credits,
        max_companies=args.max_companies,
        max_concurrent_rows=args.concurrency,
        crawl_rpm=args.crawl_rpm,
        max_outage_seconds=args.max_outage_seconds,
        heartbeat_every=args.heartbeat_every,
    )
    print("\n" + report.format_report(), file=sys.stderr)


if __name__ == "__main__":
    main()
