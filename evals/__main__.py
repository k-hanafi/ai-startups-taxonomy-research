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
    subs.add_parser(
        "export-labeling", help="Export prompt-view files for gold drafting (Stage 2)"
    )
    p_drafts = subs.add_parser(
        "apply-drafts", help="Merge a validated draft-label JSON into golden_set.csv"
    )
    p_drafts.add_argument("drafts_json", help="Path to a drafts JSON batch")
    subs.add_parser("review-page", help="Render the human-review HTML page (Stage 2)")
    subs.add_parser("run", help="Run one model config against the golden set (Stage 3)")
    subs.add_parser("score", help="Score run predictions against gold labels (Stage 6)")
    subs.add_parser("report", help="Build the benchmark dashboard (Stage 8)")

    args = parser.parse_args()

    if args.command == "sample":
        from evals.sampling import build_golden_set

        build_golden_set()
        return
    if args.command == "export-labeling":
        from evals.labeling import export_labeling_workspace

        export_labeling_workspace()
        return
    if args.command == "apply-drafts":
        from evals.labeling import apply_drafts

        apply_drafts(args.drafts_json)
        return
    if args.command == "review-page":
        from evals.labeling import render_review_page

        render_review_page()
        return

    # Later stages land in subsequent PRs; fail loudly instead of silently.
    sys.exit(f"'{args.command}' is not implemented yet (see the eval-harness plan).")


if __name__ == "__main__":
    main()
