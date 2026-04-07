#!/usr/bin/env python3
"""CLI entry point for the v2 startup classification pipeline.

Each subcommand does exactly one thing and reads state.json to know where
to start. --dry-run on prepare and run prints the full cost plan without
touching the API.

Usage:
    python classify.py prepare [--dry-run]
    python classify.py submit  [--concurrency 3]
    python classify.py status
    python classify.py download
    python classify.py retry
    python classify.py merge   [--output path]
    python classify.py run     [--dry-run] [--concurrency 3]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.builder import build_batch_files, load_system_prompt
from src.config import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, ESTIMATED_TOKENS_PER_REQUEST
from src.downloader import collect_failed_custom_ids, download_completed
from src.formatter import build_custom_id, format_user_message
from src.logger import setup_logging
from src.merger import DEFAULT_OUTPUT_PATH, merge_batch_csvs, print_report
from src.monitor import print_status, submit_and_monitor
from src.state import BatchRecord, PipelineState
from src.tokens import estimate_cost

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CSV = _PROJECT_ROOT / "data" / "company_us_short_long_desc_.csv"


# -- Subcommand handlers -------------------------------------------------------


def _cmd_prepare(args: argparse.Namespace) -> None:
    """Build JSONL batch files from the dataset. With --dry-run, print cost only."""
    setup_logging()

    row_slice = _parse_rows(args.rows)
    df = pd.read_csv(DATA_CSV)
    if row_slice is not None:
        df = df.iloc[row_slice]

    user_messages = [format_user_message(row._asdict()) for row in df.itertuples(index=False)]
    system_prompt = load_system_prompt()

    estimate = estimate_cost(system_prompt, user_messages, args.model, args.batch_size)
    print(estimate.format_report())

    if args.dry_run:
        return

    files = build_batch_files(
        DATA_CSV, model=args.model, batch_size=args.batch_size, row_slice=row_slice,
    )

    state = PipelineState.load()
    state.run_id = ""
    state.model = args.model
    state.total_companies = len(df)

    for idx, fpath in enumerate(files, start=1):
        row_start = (idx - 1) * args.batch_size
        row_end = min(row_start + args.batch_size - 1, len(df) - 1)
        key = fpath.stem
        state.batches[key] = BatchRecord(
            batch_number=idx,
            file_path=str(fpath),
            row_range=f"{row_start}-{row_end}",
            estimated_tokens=args.batch_size * ESTIMATED_TOKENS_PER_REQUEST,
        )

    state.save()
    logger.info("Prepared %d batch files. Run 'classify.py submit' next.", len(files))


def _cmd_submit(args: argparse.Namespace) -> None:
    """Submit pending batches and monitor until all complete."""
    setup_logging()
    state = PipelineState.load()

    if not state.batches:
        logger.error("No batches prepared. Run 'classify.py prepare' first.")
        sys.exit(1)

    submit_and_monitor(
        state,
        concurrency=args.concurrency,
        model=args.model,
        batch_size=args.batch_size,
    )


def _cmd_status(args: argparse.Namespace) -> None:
    """Print a one-shot status table of all tracked batches."""
    setup_logging()
    state = PipelineState.load()
    print_status(state)


def _cmd_download(args: argparse.Namespace) -> None:
    """Download results for all completed batches."""
    setup_logging()
    state = PipelineState.load()
    download_completed(state)


def _cmd_retry(args: argparse.Namespace) -> None:
    """Collect failed custom_ids from error files and re-submit as a new batch."""
    setup_logging()
    state = PipelineState.load()
    failed_ids = collect_failed_custom_ids(state)

    if not failed_ids:
        logger.info("No failed requests to retry.")
        return

    logger.info("Found %d failed requests. Building retry batch...", len(failed_ids))

    df = pd.read_csv(DATA_CSV)
    id_set = {cid.removeprefix("startup-") for cid in failed_ids}
    retry_df = df[df["org_uuid"].isin(id_set)]

    if retry_df.empty:
        logger.warning("Could not match any failed custom_ids to the dataset.")
        return

    files = build_batch_files(
        DATA_CSV, model=args.model, batch_size=len(retry_df),
        row_slice=slice(0, 0),  # placeholder, we override below
    )

    # Build a single retry JSONL manually
    from src.builder import OUTPUT_DIR, _openai_strict_schema, build_request_body, load_system_prompt
    import json

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retry_path = OUTPUT_DIR / "retry_batch.jsonl"
    system_prompt = load_system_prompt()
    schema = _openai_strict_schema()

    with open(retry_path, "w", encoding="utf-8") as f:
        for row in retry_df.itertuples(index=False):
            row_dict = row._asdict()
            user_msg = format_user_message(row_dict)
            cid = build_custom_id(str(row_dict.get("org_uuid", "")))
            body = build_request_body(user_msg, cid, system_prompt, schema, args.model)
            f.write(json.dumps(body, ensure_ascii=False) + "\n")

    next_num = max((b.batch_number for b in state.batches.values()), default=0) + 1
    state.batches[retry_path.stem] = BatchRecord(
        batch_number=next_num,
        file_path=str(retry_path),
        row_range=f"retry-{len(retry_df)}",
        estimated_tokens=len(retry_df) * ESTIMATED_TOKENS_PER_REQUEST,
    )
    state.save()

    logger.info("Retry batch created with %d requests. Run 'classify.py submit'.", len(retry_df))


def _cmd_merge(args: argparse.Namespace) -> None:
    """Merge all batch outputs into one CSV and print the report."""
    setup_logging()
    state = PipelineState.load()
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_PATH
    merge_batch_csvs(state, output_path)
    print_report(state, output_path)


def _cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: prepare, submit, download, merge."""
    setup_logging()

    args_ns = argparse.Namespace(**vars(args))
    _cmd_prepare(args_ns)

    if args.dry_run:
        return

    _cmd_submit(args_ns)
    _cmd_download(args_ns)
    _cmd_merge(args_ns)


# -- Argument parsing -----------------------------------------------------------


def _parse_rows(rows_str: str | None) -> slice | None:
    """Parse a row range string like '0:50000' into a slice."""
    if not rows_str:
        return None
    parts = rows_str.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid row range: {rows_str}. Use start:end.")
    return slice(int(parts[0]), int(parts[1]))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --model and --batch-size to a subcommand parser."""
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, dest="batch_size",
        help=f"Requests per JSONL file (default: {DEFAULT_BATCH_SIZE})",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="classify.py",
        description="v2 AI-native startup classifier: two-axis taxonomy via OpenAI Batch API",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # prepare
    p = subs.add_parser(
        "prepare",
        help="Build JSONL batch files from the dataset and show cost estimate",
    )
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print cost breakdown only. No files written, no API calls.")
    p.add_argument("--rows", default=None,
                   help="Row range to process, e.g. '0:50000'")
    p.set_defaults(func=_cmd_prepare)

    # submit
    p = subs.add_parser(
        "submit",
        help="Submit pending batches and monitor until all complete",
    )
    _add_common_args(p)
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max batches in-flight simultaneously (default: 1)")
    p.set_defaults(func=_cmd_submit)

    # status
    p = subs.add_parser("status", help="Print status of all tracked batches")
    p.set_defaults(func=_cmd_status)

    # download
    p = subs.add_parser("download", help="Download results for completed batches")
    p.set_defaults(func=_cmd_download)

    # retry
    p = subs.add_parser(
        "retry",
        help="Re-submit failed requests from error files as a new batch",
    )
    _add_common_args(p)
    p.set_defaults(func=_cmd_retry)

    # merge
    p = subs.add_parser(
        "merge",
        help="Merge batch outputs into final CSV and print distribution report",
    )
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: outputs/classified_startups_v2.csv)")
    p.set_defaults(func=_cmd_merge)

    # run
    p = subs.add_parser(
        "run",
        help="Full pipeline: prepare -> submit -> download -> merge",
    )
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Run prepare in dry-run mode only")
    p.add_argument("--rows", default=None,
                   help="Row range to process, e.g. '0:50000'")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max batches in-flight simultaneously (default: 1)")
    p.add_argument("--output", default=None)
    p.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
