"""CLI entry point: python -m evals <sample|run-classification|score|matrix|…>."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description=(
            "Golden-set evaluation harness for the startup classifier. "
            "Paid screen path: cost-preview → run-evals → open-dashboard "
            "(or the lower-level run-classification / matrix / score)."
        ),
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p_cost = subs.add_parser(
        "cost-preview",
        help=(
            "Print estimated cost for every locked matrix config and the "
            "grand total (Pass A counted once per model). No API calls."
        ),
    )
    p_cost.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows for a cheaper smoke estimate (default: full golden set)",
    )
    p_run_evals = subs.add_parser(
        "run-evals",
        help=(
            "Run the full locked matrix from scratch every time: rebuild "
            "Pass A banks (3 models in parallel), re-run Batch parity, run "
            "9 Pass B cells in parallel, score each, build the dashboard. "
            "Live checklist in the terminal. Prior scored cells are not "
            "resumed (re-paying is intentional)."
        ),
    )
    p_run_evals.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the interactive cost confirm gate",
    )
    p_run_evals.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows for a cheap end-to-end smoke (applied to every cell)",
    )
    subs.add_parser(
        "open-dashboard",
        help=(
            "Open the eval instance archive index in your browser "
            "(eval_instances/index.html, newest run at the top)"
        ),
    )
    p_bank = subs.add_parser(
        "bank-pass-a",
        help=(
            "Bank Pass A only for one model (used by run-evals phase 1). "
            "Resumes incomplete banks by default; pass --rerun to rebuild."
        ),
    )
    p_bank.add_argument("--model", required=True, help="Model name to bank")
    p_bank.add_argument(
        "--limit", type=int, default=None, help="Cap rows (smoke test)",
    )
    p_bank.add_argument(
        "--dry-run", action="store_true", help="Print cost only, no API call",
    )
    p_bank.add_argument(
        "--rerun",
        action="store_true",
        help="Delete the stable per-model Pass A bank and rebuild it",
    )

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
            "LEGACY single-pass runner (retired for the locked matrix). Prefer "
            "run-classification. Kept only to rescore old banked runs."
        ),
    )
    p_run.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
    p_run.add_argument("--effort", default=None, help="Reasoning effort (default: screen effort)")
    p_run.add_argument("--repeat", type=int, default=1, help="Repeat index for the run_id")
    p_run.add_argument("--run-id", default=None, help="Override run_id to resume a partial run")
    p_run.add_argument("--limit", type=int, default=None, help="Cap rows (cheap smoke test)")
    p_run.add_argument("--dry-run", action="store_true", help="Print plan + cost, no API call")

    def _add_classification_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
        p.add_argument(
            "--effort-b",
            default=None,
            help=(
                "Pass B reasoning effort (default: high). Locked matrix uses "
                "low/medium/high."
            ),
        )
        p.add_argument("--repeat", type=int, default=1, help="Repeat index for the run_id")
        p.add_argument("--run-id", default=None, help="Override run_id to resume a partial run")
        p.add_argument("--limit", type=int, default=None, help="Cap rows (cheap smoke test)")
        p.add_argument("--dry-run", action="store_true", help="Print plan + cost, no API call")
        p.add_argument(
            "--pass-a-from",
            default=None,
            metavar="RUN_ID",
            help=(
                "Advanced override: pin Pass A to a specific historical run_id "
                "instead of the stable per-model bank under "
                "evals/runs/pass_a_banks/<model>/."
            ),
        )
        p.add_argument(
            "--rerun-pass-a",
            action="store_true",
            help=(
                "Invalidate the stable per-model Pass A bank and run Pass A "
                "again (escape hatch). Default is to reuse an existing bank."
            ),
        )
        p.add_argument(
            "--reuse-pass-a-from",
            default=None,
            metavar="RUN_ID",
            help=(
                "Deprecated alias for --pass-a-from. Pass A auto-reuses the "
                "per-model bank by default; do not use this for normal matrix runs."
            ),
        )
        p.add_argument(
            "--require-matrix-cell",
            action="store_true",
            help=(
                "Refuse models/efforts outside the locked EVAL_MODELS × "
                "MATRIX_PASS_B_EFFORTS matrix."
            ),
        )
        p.add_argument(
            "--require-stage8-cell",
            action="store_true",
            help="Deprecated alias for --require-matrix-cell.",
        )

    p_run_cls = subs.add_parser(
        "run-classification",
        help=(
            "Run Pass A (binary gate) + Pass B (family-constrained subclass). "
            "Paid matrix path."
        ),
    )
    _add_classification_run_args(p_run_cls)
    p_run_legacy = subs.add_parser(
        "run-two-pass",
        help="Deprecated alias for run-classification.",
    )
    _add_classification_run_args(p_run_legacy)
    p_matrix = subs.add_parser(
        "matrix",
        help=(
            "Enumerate the locked 9-cell matrix "
            "(EVAL_MODELS × MATRIX_PASS_B_EFFORTS). Prints planned commands only."
        ),
    )
    p_score = subs.add_parser(
        "score", help="Score run predictions against gold labels"
    )
    p_score.add_argument("run_id", help="Run directory name under evals/runs/")
    p_score.add_argument(
        "--baseline", default=None,
        help=(
            "Baseline run_id for paired-bootstrap deltas (same golden rows). "
            "Use for model/config comparisons; surfaces as vs_baseline in "
            "scored.json and the dashboard when present."
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
    p_score.add_argument(
        "--allow-missing-confidence",
        action="store_true",
        help=(
            "With --confidence-from-raw: if no row yields binary confidence "
            "(API returned a one-sided {0,1} pool), score accuracy axes "
            "without calibration instead of exiting. Used by run-evals so "
            "mini/luna sweeps are not blocked when top_logprobs omit the "
            "opposing digit."
        ),
    )
    p_parity = subs.add_parser(
        "batch-parity",
        help="PAID: 10-row Batch-vs-sync parity smoke on Pass A (gate Q4, Stage 7)",
    )
    p_parity.add_argument("--model", default=None, help="Model name (default: first EVAL_MODEL)")
    p_parity.add_argument(
        "--run-id",
        default=None,
        help="Override parity run directory under evals/runs/",
    )
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
    p_dash.add_argument(
        "--save-instance",
        action="store_true",
        help=(
            "Also archive this build as eval_instances/eval_instance_NN.html "
            "(automatic for real runs; use this to keep a mock build)"
        ),
    )

    args = parser.parse_args()

    if args.command == "cost-preview":
        from evals.cost_preview import print_matrix_preview
        from evals.runner import load_golden_rows

        rows = load_golden_rows()
        if args.limit is not None:
            if args.limit < 1:
                sys.exit(f"--limit must be a positive row cap, got {args.limit}")
            rows = rows[: args.limit]
        print_matrix_preview(rows)
        return
    if args.command == "run-evals":
        from evals.orchestrate import run_evals

        raise SystemExit(run_evals(yes=args.yes, limit=args.limit))
    if args.command == "open-dashboard":
        from evals.orchestrate import open_dashboard_index

        open_dashboard_index()
        return
    if args.command == "bank-pass-a":
        from evals.classification import bank_pass_a

        bank_pass_a(
            model=args.model,
            limit=args.limit,
            dry_run=args.dry_run,
            rerun=args.rerun,
        )
        return

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
            "Single-pass `run` is LEGACY. Paid science uses "
            "`run-classification` (bank Pass A once per model, sweep Pass B "
            "effort)."
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
    if args.command in ("run-classification", "run-two-pass"):
        if args.command == "run-two-pass":
            logging.warning(
                "`run-two-pass` is deprecated; use `run-classification`."
            )
        from evals import config as cfg
        from evals.classification import run_classification, validate_matrix_cell

        model = args.model or cfg.EVAL_MODELS[0]
        effort_b = args.effort_b or cfg.PASS_B_EFFORT
        require_matrix = args.require_matrix_cell or args.require_stage8_cell
        if args.require_stage8_cell and not args.require_matrix_cell:
            logging.warning(
                "`--require-stage8-cell` is deprecated; use "
                "`--require-matrix-cell`."
            )
        if require_matrix:
            validate_matrix_cell(model, effort_b)
        run_classification(
            model=model,
            effort_b=effort_b,
            repeat=args.repeat,
            limit=args.limit,
            dry_run=args.dry_run,
            run_id=args.run_id,
            pass_a_from=args.pass_a_from,
            rerun_pass_a=args.rerun_pass_a,
            reuse_pass_a_from=args.reuse_pass_a_from,
        )
        return

    if args.command == "matrix":
        from evals import config as cfg
        from evals.classification import matrix_cells

        cells = matrix_cells()
        print(f"Locked eval matrix: {len(cells)} cells")
        print(f"  models = {cfg.EVAL_MODELS}")
        print(f"  Pass B efforts = {cfg.MATRIX_PASS_B_EFFORTS}")
        print()
        print(
            "Pass A banks auto-create on the first effort arm per model, "
            "then reuse (no reuse flag needed):"
        )
        by_model: dict[str, list[str]] = {}
        for model, effort in cells:
            by_model.setdefault(model, []).append(effort)
        for model, efforts in by_model.items():
            first, *rest = efforts
            print(
                f"  # {model}: creates bank on {first}, auto-reuses for "
                + ", ".join(rest)
            )
            for effort in efforts:
                print(
                    f"  python -m evals run-classification --model {model} "
                    f"--effort-b {effort} --require-matrix-cell"
                )
            print(
                f"  python -m evals score <run_id> --confidence-from-raw "
                f"[--baseline <other_run_id>]"
            )
            print()
        print(
            "Escape hatch: --rerun-pass-a rebuilds the per-model bank. "
            "Advanced: --pass-a-from <run_id> pins a historical bank."
        )
        print(
            "Dry-run cost preflight (no API): add --dry-run to any "
            "run-classification line above."
        )
        return

    if args.command == "score":
        from evals.scoring import load_confidence_file, score_cli

        confidence = None
        robustness: dict | None = None
        if args.confidence:
            confidence = load_confidence_file(args.confidence)
        elif args.confidence_from_raw:
            from evals.batch_parity import load_parity_summary_for_model
            from evals.logprob_extract import (
                LogprobExtractionError,
                chosen_confidence,
                extract_confidence_rows,
                valid_mass_summary,
            )
            from evals.paths import run_config_path, run_raw_dir

            try:
                conf_rows = extract_confidence_rows(run_raw_dir(args.run_id))
                confidence = {
                    row["custom_id"]: chosen_confidence(row) for row in conf_rows
                }
                robustness = {"valid_mass": valid_mass_summary(conf_rows)}
            except LogprobExtractionError as exc:
                if args.allow_missing_confidence:
                    logging.warning(
                        "--confidence-from-raw unavailable (%s); "
                        "scoring accuracy without calibration "
                        "(--allow-missing-confidence)",
                        exc,
                    )
                    confidence = None
                else:
                    sys.exit(f"--confidence-from-raw failed: {exc}")
            model = None
            cfg_path = run_config_path(args.run_id)
            if cfg_path.exists():
                try:
                    model = json.loads(cfg_path.read_text(encoding="utf-8")).get(
                        "model"
                    )
                except (OSError, ValueError):
                    model = None
            if model:
                parity = load_parity_summary_for_model(str(model))
                if parity:
                    robustness = {**(robustness or {}), "batch_parity": parity}
        score_cli(
            args.run_id,
            args.baseline,
            confidence,
            allow_partial=args.allow_partial,
            allow_partial_confidence=args.allow_partial_confidence,
            robustness=robustness,
        )
        return
    if args.command == "batch-parity":
        from evals import config as cfg
        from evals.batch_parity import run_parity

        report = run_parity(
            model=args.model or cfg.EVAL_MODELS[0],
            run_id=args.run_id,
        )
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
        mod.write_dashboard(
            mod.resolve_metrics(ns), ns.output, save_instance=args.save_instance
        )
        return

    # Later stages land in subsequent PRs; fail loudly instead of silently.
    sys.exit(f"'{args.command}' is not implemented yet (see the eval-harness plan).")


if __name__ == "__main__":
    main()
