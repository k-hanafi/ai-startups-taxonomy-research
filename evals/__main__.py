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
    p_run = subs.add_parser(
        "run", help="Run one model config against the golden set (Stage 3)"
    )
    p_run.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
    p_run.add_argument("--effort", default=None, help="Reasoning effort (default: screen effort)")
    p_run.add_argument("--repeat", type=int, default=1, help="Repeat index for the run_id")
    p_run.add_argument("--run-id", default=None, help="Override run_id to resume a partial run")
    p_run.add_argument("--limit", type=int, default=None, help="Cap rows (cheap smoke test)")
    p_run.add_argument("--dry-run", action="store_true", help="Print plan + cost, no API call")
    p_run2 = subs.add_parser(
        "run-two-pass",
        help="Run the two-pass classifier (binary gate + family-constrained subclass) (Stage 5)",
    )
    p_run2.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
    p_run2.add_argument("--effort-b", default=None, help="Pass B reasoning effort (default: high)")
    p_run2.add_argument("--repeat", type=int, default=1, help="Repeat index for the run_id")
    p_run2.add_argument("--run-id", default=None, help="Override run_id to resume a partial run")
    p_run2.add_argument("--limit", type=int, default=None, help="Cap rows (cheap smoke test)")
    p_run2.add_argument("--dry-run", action="store_true", help="Print plan + cost, no API call")
    p_score = subs.add_parser(
        "score", help="Score run predictions against gold labels (Stage 7)"
    )
    p_score.add_argument("run_id", help="Run directory name under evals/runs/")
    p_score.add_argument(
        "--baseline", default=None,
        help="Baseline run_id for paired-bootstrap deltas",
    )
    p_score.add_argument(
        "--confidence", default=None,
        help="Optional JSON file mapping org_uuid/custom_id -> binary "
             "confidence (enables calibration metrics)",
    )
    p_parity = subs.add_parser(
        "batch-parity",
        help="PAID: 10-row Batch-vs-sync parity smoke on Pass A (gate Q4, Stage 7)",
    )
    p_parity.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
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
    if args.command == "run":
        from evals import config as cfg
        from evals.runner import run

        run(
            model=args.model or cfg.EVAL_MODELS[0],
            effort=args.effort or cfg.SCREEN_REASONING_EFFORT,
            repeat=args.repeat,
            limit=args.limit,
            dry_run=args.dry_run,
            run_id=args.run_id,
        )
        return
    if args.command == "run-two-pass":
        from evals import config as cfg
        from evals.two_pass import run_two_pass

        run_two_pass(
            model=args.model or cfg.EVAL_MODELS[0],
            effort_b=args.effort_b or cfg.PASS_B_EFFORT,
            repeat=args.repeat,
            limit=args.limit,
            dry_run=args.dry_run,
            run_id=args.run_id,
        )
        return

    if args.command == "score":
        from evals.scoring import score_cli

        score_cli(args.run_id, args.baseline, args.confidence)
        return
    if args.command == "batch-parity":
        from evals import config as cfg
        from evals.batch_parity import run_parity

        report = run_parity(model=args.model or cfg.EVAL_MODELS[0])
        if report["verdict"] != "PASS":
            sys.exit(1)
        return

    # Later stages land in subsequent PRs; fail loudly instead of silently.
    sys.exit(f"'{args.command}' is not implemented yet (see the eval-harness plan).")


if __name__ == "__main__":
    main()
