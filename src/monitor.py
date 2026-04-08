"""Async concurrent batch monitor with sliding-window queue pressure control.

Submitting all batches simultaneously can exhaust the 15B token queue before
any complete, deadlocking the pipeline. The monitor polls all in-flight
batches via asyncio.gather(). The concurrency parameter acts as a sliding
window: it only submits the next batch when a running one completes and
releases tokens from the queue.
"""

from __future__ import annotations

import asyncio
import logging
import time

from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    ESTIMATED_TOKENS_PER_REQUEST,
    MAX_BATCH_QUEUE_TOKENS,
)
from src.state import BatchRecord, PipelineState
from src.submitter import (
    BillingLimitError,
    create_batch,
    generate_run_id,
    get_client,
    upload_batch_file,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: int = 30


def _emit_billing_resume_help(state: PipelineState) -> None:
    """Print clear next steps after OpenAI billing hard limit blocks submission."""
    submitted = sum(1 for b in state.batches.values() if b.batch_id)
    pending = len(state.pending_batches())
    body = (
        "[bold]OpenAI billing hard limit reached[/] (monthly budget cap).\n\n"
        f"Batches already created on OpenAI: [cyan]{submitted}[/]\n"
        f"Batches not yet submitted: [yellow]{pending}[/]\n\n"
        "[bold]1.[/] Raise org (and project) limits:\n"
        "    [link=https://platform.openai.com/settings/organization/limits]"
        "https://platform.openai.com/settings/organization/limits[/]\n\n"
        "[bold]2.[/] Resume submission and wait for completion:\n"
        "    [green]python classify.py submit[/]\n\n"
        "[bold]3.[/] Fetch results and merge:\n"
        "    [green]python classify.py download[/]\n"
        "    [green]python classify.py merge[/]\n\n"
        "State is saved; you do not need to re-run [italic]prepare[/]."
    )
    Console().print()
    Console().print(Panel(body, title="Resume pipeline", border_style="yellow"))
    Console().print()


def _build_status_table(state: PipelineState) -> Table:
    """Build a rich table summarising all tracked batches."""
    table = Table(title="Batch Status", expand=True)
    table.add_column("Batch #", justify="right", style="cyan")
    table.add_column("Batch ID", style="dim")
    table.add_column("Status", style="bold")
    table.add_column("Rows", justify="right")
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")

    for key in sorted(state.batches, key=lambda k: state.batches[k].batch_number):
        b = state.batches[key]
        status_style = {
            "completed": "[green]completed[/]",
            "in_progress": "[yellow]in_progress[/]",
            "submitted": "[blue]submitted[/]",
            "failed": "[red]failed[/]",
            "expired": "[red]expired[/]",
        }.get(b.status, b.status)

        table.add_row(
            str(b.batch_number),
            b.batch_id[:16] + "..." if len(b.batch_id) > 16 else b.batch_id,
            status_style,
            b.row_range,
            str(b.completed_count),
            str(b.failed_count),
        )

    queued = state.estimated_queued_tokens()
    table.caption = (
        f"In-flight: {len(state.in_flight_batches())}  |  "
        f"Queued tokens: {queued:,} / {MAX_BATCH_QUEUE_TOKENS:,}"
    )
    return table


async def _poll_batch(client: OpenAI, record: BatchRecord) -> None:
    """Poll one batch and update its record in place."""
    try:
        batch = client.batches.retrieve(record.batch_id)
    except Exception:
        logger.warning("Failed to poll batch %s", record.batch_id, exc_info=True)
        return

    status_map = {
        "validating": "submitted",
        "in_progress": "in_progress",
        "finalizing": "in_progress",
        "completed": "completed",
        "failed": "failed",
        "expired": "expired",
        "cancelled": "cancelled",
    }
    record.status = status_map.get(batch.status, record.status)

    if batch.request_counts:
        record.completed_count = batch.request_counts.completed or 0
        record.failed_count = batch.request_counts.failed or 0
        record.request_count = batch.request_counts.total or 0

    if batch.output_file_id:
        record.output_file_id = batch.output_file_id
    if batch.error_file_id:
        record.error_file_id = batch.error_file_id


async def poll_all(state: PipelineState) -> None:
    """Poll every in-flight batch concurrently and update state."""
    client = get_client()
    in_flight = state.in_flight_batches()
    if not in_flight:
        return
    await asyncio.gather(*[_poll_batch(client, b) for b in in_flight])
    state.save()


def _can_submit_more(state: PipelineState, concurrency: int) -> bool:
    """Check whether the sliding window allows another submission."""
    in_flight = len(state.in_flight_batches())
    if in_flight >= concurrency:
        return False
    if state.estimated_queued_tokens() >= MAX_BATCH_QUEUE_TOKENS * 0.90:
        logger.info("Queue pressure at 90%%. Holding submission.")
        return False
    return True


def submit_and_monitor(
    state: PipelineState,
    *,
    concurrency: int = 1,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Submit pending batches with queue pressure control, then monitor.

    The concurrency parameter is a sliding window. It caps how many batches
    are in-flight simultaneously and only submits the next when a running
    one completes and releases tokens from the 15B queue limit.
    """
    client = get_client()
    run_id = state.run_id or generate_run_id(model)
    state.run_id = run_id
    state.model = model
    total_batches = len(state.batches)

    logger.info(
        "Starting monitor: %d pending, %d in-flight, concurrency=%d",
        len(state.pending_batches()),
        len(state.in_flight_batches()),
        concurrency,
    )

    with Live(_build_status_table(state), refresh_per_second=0.5) as live:
        while state.pending_batches() or state.in_flight_batches():
            while state.pending_batches() and _can_submit_more(state, concurrency):
                rec = state.pending_batches()[0]
                logger.info("Submitting batch %d ...", rec.batch_number)

                try:
                    file_id = upload_batch_file(client, rec.file_path)
                    batch_id = create_batch(
                        client,
                        file_id,
                        run_id=run_id,
                        batch_number=rec.batch_number,
                        total_batches=total_batches,
                        row_range=rec.row_range,
                        model=model,
                    )
                except BillingLimitError:
                    state.save()
                    logger.error(
                        "Stopped at batch %d: billing hard limit. "
                        "Earlier batches are unchanged in state.json.",
                        rec.batch_number,
                    )
                    _emit_billing_resume_help(state)
                    raise

                rec.file_id = file_id
                rec.batch_id = batch_id
                rec.status = "submitted"
                rec.estimated_tokens = batch_size * ESTIMATED_TOKENS_PER_REQUEST
                state.save()

            asyncio.run(poll_all(state))
            live.update(_build_status_table(state))

            if state.in_flight_batches():
                time.sleep(POLL_INTERVAL_SECONDS)

    logger.info(
        "All batches terminal: %d completed, %d failed/expired",
        len(state.completed_batches()),
        len(state.failed_batches()),
    )


def print_status(state: PipelineState) -> None:
    """Print a one-shot status table (for classify.py status)."""
    if not state.batches:
        logger.info("No batches tracked yet.")
        return

    client = get_client()
    asyncio.run(asyncio.gather(
        *[_poll_batch(client, b) for b in state.in_flight_batches()]
    ))
    state.save()

    from rich.console import Console
    Console().print(_build_status_table(state))
