"""CLI entry point: python -m evals <sample|run|score|report>."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Golden-set evaluation harness for the startup classifier.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("sample", help="Draw the stratified golden set (Stage 1)")
    subs.add_parser("run", help="Run one model config against the golden set (Stage 3)")
    subs.add_parser("score", help="Score run predictions against gold labels (Stage 6)")
    subs.add_parser("report", help="Build the benchmark dashboard (Stage 8)")

    args = parser.parse_args()

    if args.command == "sample":
        from evals.sampling import build_golden_set

        build_golden_set()
        return

    # Later stages land in subsequent PRs; fail loudly instead of silently.
    sys.exit(f"'{args.command}' is not implemented yet (see the eval-harness plan).")


if __name__ == "__main__":
    main()
