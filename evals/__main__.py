"""CLI entry point: python -m evals <sample|run-two-pass|score|matrix|…>."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description=(
            "Golden-set evaluation harness for the startup classifier. "
            "Stage 8 uses two-pass only (run-two-pass / matrix)."
        ),
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
        "run",
        help=(
            "LEGACY single-pass runner (retired for Stage 8). Prefer "
            "run-two-pass. Kept only to rescore old banked runs."
        ),
    )
    p_run.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
    p_run.add_argument("--effort", default=None, help="Reasoning effort (default: screen effort)")
    p_run.add_argument("--repeat", type=int, default=1, help="Repeat index for the run_id")
    p_run.add_argument("--run-id", default=None, help="Override run_id to resume a partial run")
    p_run.add_argument("--limit", type=int, default=None, help="Cap rows (cheap smoke test)")
    p_run.add_argument("--dry-run", action="store_true", help="Print plan + cost, no API call")
    p_run2 = subs.add_parser(
        "run-two-pass",
        help=(
            "Run the two-pass classifier (Pass A binary gate + Pass B "
            "family-constrained subclass). Stage 8 paid path."
        ),
    )
    p_run2.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
    p_run2.add_argument(
        "--effort-b",
        default=None,
        help="Pass B reasoning effort (default: high). Stage 8 uses low/medium/high.",
    )
    p_run2.add_argument("--repeat", type=int, default=1, help="Repeat index for the run_id")
    p_run2.add_argument("--run-id", default=None, help="Override run_id to resume a partial run")
    p_run2.add_argument("--limit", type=int, default=None, help="Cap rows (cheap smoke test)")
    p_run2.add_argument("--dry-run", action="store_true", help="Print plan + cost, no API call")
    p_run2.add_argument(
        "--reuse-pass-a-from",
        default=None,
        metavar="RUN_ID",
        help=(
            "Reuse banked Pass A verdicts + raw logprobs from RUN_ID (same "
            "model). Required for Stage 8 effort arms after the first cell "
            "per model so Pass B deltas are not confounded by resampling "
            "the gate."
        ),
    )
    p_run2.add_argument(
        "--require-stage8-cell",
        action="store_true",
        help="Refuse models/efforts outside the locked Stage 8 matrix.",
    )
    p_matrix = subs.add_parser(
        "matrix",
        help=(
            "Enumerate the locked Stage 8 9-cell matrix "
            "(EVAL_MODELS × low/medium/high). Default is dry-run commands."
        ),
    )
    p_matrix.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print planned commands only (default).",
    )
    p_matrix.add_argument(
        "--pass-a-bank-template",
        default="<pass-a-bank-run-id-for-MODEL>",
        help="Placeholder shown for --reuse-pass-a-from on effort arms 2–3.",
    )
    p_score = subs.add_parser(
        "score", help="Score run predictions against gold labels (Stage 7)"
    )
    p_score.add_argument("run_id", help="Run directory name under evals/runs/")
    p_score.add_argument(
        "--baseline", default=None,
        help=(
            "Baseline run_id for paired-bootstrap deltas (same golden rows). "
            "Use for Stage 8 model/config comparisons; surfaces as "
            "vs_baseline in scored.json and the dashboard when present."
        ),
    )
    conf_src = p_score.add_mutually_exclusive_group()
    conf_src.add_argument(
        "--confidence", default=None,
        help="Optional JSON file mapping org_uuid/custom_id -> binary "
             "confidence (enables calibration metrics)",
    )
    conf_src.add_argument(
        "--confidence-from-raw", action="store_true",
        help="Derive binary confidence from the run's raw/ logprob responses "
             "(chosen-digit probability mass, pivot 6) and compute calibration",
    )
    p_score.add_argument(
        "--allow-partial",
        action="store_true",
        help="Score even when n_scored < expected rows (config n_rows or full "
             "golden set). Default refuses so a mid-flight resume cannot look "
             "like a finished screen.",
    )
    p_score.add_argument(
        "--allow-partial-confidence",
        action="store_true",
        help="Allow calibration when confidence covers fewer than all "
             "eligible rows (incomplete raw/ or one-sided binary pools). "
             "Default refuses incomplete confidence coverage.",
    )
    p_parity = subs.add_parser(
        "batch-parity",
        help="PAID: 10-row Batch-vs-sync parity smoke on Pass A (gate Q4, Stage 7)",
    )
    p_parity.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
    p_report = subs.add_parser(
        "report",
        help="Render production-cost extrapolation for a scored run (pivot 8)",
    )
    p_report.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run directory under evals/runs/ (default: most recently scored)",
    )
    p_dash = subs.add_parser(
        "dashboard",
        help="Build Stage 9 eval dashboard HTML (fixture or scored.json)",
    )
    p_dash.add_argument(
        "--fixture",
        nargs="?",
        const=True,
        default=None,
        help="Use synthetic mock fixture (optional path). Same as default when --runs/--scored are omitted.",
    )
    p_dash.add_argument(
        "--force-fixture",
        action="store_true",
        help="Explicitly use the mock fixture (same as default when --runs/--scored are omitted)",
    )
    p_dash.add_argument(
        "--scored",
        nargs="+",
        default=None,
        help="One or more scored.json paths (required for real runs; no auto-discovery)",
    )
    p_dash.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Run ids under evals/runs/ (required for real runs; no auto-discovery)",
    )
    p_dash.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: Presentation Materials/eval_dashboard.html)",
    )

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
        logging.warning(
            "Single-pass `run` is LEGACY. Stage 8 science uses "
            "`run-two-pass` (bank Pass A once per model, sweep Pass B effort)."
        )
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
        from evals.two_pass import run_two_pass, validate_stage8_cell

        model = args.model or cfg.EVAL_MODELS[0]
        effort_b = args.effort_b or cfg.PASS_B_EFFORT
        if args.require_stage8_cell:
            validate_stage8_cell(model, effort_b)
        run_two_pass(
            model=model,
            effort_b=effort_b,
            repeat=args.repeat,
            limit=args.limit,
            dry_run=args.dry_run,
            run_id=args.run_id,
            reuse_pass_a_from=args.reuse_pass_a_from,
        )
        return

    if args.command == "matrix":
        from evals import config as cfg
        from evals.two_pass import stage8_matrix_cells

        cells = stage8_matrix_cells()
        print(f"Stage 8 locked matrix: {len(cells)} cells")
        print(f"  models = {cfg.EVAL_MODELS}")
        print(f"  Pass B efforts = {cfg.STAGE8_PASS_B_EFFORTS}")
        print()
        print("Bank Pass A once per model (first effort arm), then reuse:")
        by_model: dict[str, list[str]] = {}
        for model, effort in cells:
            by_model.setdefault(model, []).append(effort)
        for model, efforts in by_model.items():
            bank_placeholder = args.pass_a_bank_template.replace("MODEL", model)
            first, *rest = efforts
            print(
                f"  # {model}: bank Pass A on {first}, reuse for "
                + ", ".join(rest)
            )
            print(
                f"  python -m evals run-two-pass --model {model} "
                f"--effort-b {first} --require-stage8-cell"
            )
            for effort in rest:
                print(
                    f"  python -m evals run-two-pass --model {model} "
                    f"--effort-b {effort} --require-stage8-cell "
                    f"--reuse-pass-a-from {bank_placeholder}"
                )
            print(
                f"  python -m evals score <run_id> --confidence-from-raw "
                f"[--baseline <other_run_id>]"
            )
            print()
        print(
            "Dry-run cost preflight (no API): add --dry-run to any "
            "run-two-pass line above."
        )
        return

    if args.command == "score":
        from evals.scoring import load_confidence_file, score_cli

        confidence = None
        if args.confidence:
            confidence = load_confidence_file(args.confidence)
        elif args.confidence_from_raw:
            from evals.logprob_extract import LogprobExtractionError, run_confidence
            from evals.paths import run_raw_dir

            try:
                confidence = run_confidence(run_raw_dir(args.run_id))
            except LogprobExtractionError as exc:
                sys.exit(f"--confidence-from-raw failed: {exc}")
        score_cli(
            args.run_id,
            args.baseline,
            confidence,
            allow_partial=args.allow_partial,
            allow_partial_confidence=args.allow_partial_confidence,
        )
        return
    if args.command == "batch-parity":
        from evals import config as cfg
        from evals.batch_parity import run_parity

        report = run_parity(model=args.model or cfg.EVAL_MODELS[0])
        # Nonzero exit on any non-PASS verdict, including batch_error runs
        # (batch timed out / no output file). The report is still written so
        # the paid sync results survive either way.
        if report["verdict"] != "PASS":
            sys.exit(1)
        return

    if args.command == "report":
        from evals.report import report_cli

        report_cli(args.run_id)
        return

    if args.command == "dashboard":
        import importlib.util
        from pathlib import Path

        from evals.paths import PROJECT_ROOT

        builder = (
            PROJECT_ROOT
            / "data visualization"
            / "02_Analysis_Code"
            / "build_eval_dashboard.py"
        )
        spec = importlib.util.spec_from_file_location("build_eval_dashboard", builder)
        if spec is None or spec.loader is None:
            sys.exit(f"Cannot load dashboard builder at {builder}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ns = argparse.Namespace(
            fixture=args.fixture,
            force_fixture=args.force_fixture,
            scored=args.scored,
            runs=args.runs,
            output=Path(args.output) if args.output else mod.OUTPUT_PATH,
        )
        metrics = mod.resolve_metrics(ns)
        ns.output.parent.mkdir(parents=True, exist_ok=True)
        ns.output.write_text(mod.build_html(metrics), encoding="utf-8")
        mode = "SYNTHETIC" if metrics.get("synthetic") else "scored"
        print(f"[{mode}] {metrics['n_configs']} configs → {ns.output}")
        return

    # Later stages land in subsequent PRs; fail loudly instead of silently.
    sys.exit(f"'{args.command}' is not implemented yet (see the eval-harness plan).")


if __name__ == "__main__":
    main()
