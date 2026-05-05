"""Classification distribution and cost report.

Results are written directly to production_classifications.csv during
download, so no separate merge step is needed. This module provides the
rich terminal report: subclass distribution, RAD score breakdown, cohort
splits, token usage, cache hit rate, and actual cost.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.paths import DEFAULT_CLASSIFICATION_OUTPUT_CSV
from src.state import PipelineState
from src.tokens import BATCH_DISCOUNT, CACHE_DISCOUNT, MODEL_PRICING

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = DEFAULT_CLASSIFICATION_OUTPUT_CSV


def _build_distribution_table(rows: list[dict]) -> Table:
    """Subclass and RAD score frequency table."""
    subclass_counts = Counter(r.get("subclass", "") for r in rows)
    rad_counts = Counter(r.get("rad_score", "") for r in rows)
    cohort_counts = Counter(r.get("cohort", "") for r in rows)
    native_counts = Counter(r.get("ai_native", "") for r in rows)

    table = Table(title="Classification Distribution", expand=True)

    table.add_column("Subclass", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("", min_width=3)
    table.add_column("RAD Score", style="magenta")
    table.add_column("Count", justify="right")

    subclass_order = [
        "1A", "1B", "1C", "1D", "1E", "1F", "1G",
        "0A", "0B", "0C",
    ]
    rad_order = ["RAD-H", "RAD-M", "RAD-L", "RAD-NA"]

    max_rows = max(len(subclass_order), len(rad_order))
    for i in range(max_rows):
        sc = subclass_order[i] if i < len(subclass_order) else ""
        sc_count = str(subclass_counts.get(sc, 0)) if sc else ""
        rd = rad_order[i] if i < len(rad_order) else ""
        rd_count = str(rad_counts.get(rd, 0)) if rd else ""
        table.add_row(sc, sc_count, "", rd, rd_count)

    table.add_section()
    n_native = native_counts.get("1", 0)
    n_not = native_counts.get("0", 0)
    table.add_row(
        f"AI-native: {n_native}", f"Not: {n_not}", "",
        f"PRE-GENAI: {cohort_counts.get('PRE-GENAI', 0)}",
        f"GENAI-ERA: {cohort_counts.get('GENAI-ERA', 0)}",
    )

    return table


def _build_cost_table(state: PipelineState) -> Table:
    """Token usage and cost breakdown table."""
    pricing = MODEL_PRICING.get(state.model, MODEL_PRICING["gpt-5.4-nano"])

    prompt_toks = state.total_prompt_tokens
    completion_toks = state.total_completion_tokens
    cached_toks = state.total_cached_tokens
    uncached_toks = prompt_toks - cached_toks

    cache_rate = cached_toks / prompt_toks * 100 if prompt_toks > 0 else 0.0

    cost_input_uncached = uncached_toks / 1e6 * pricing["input"] * BATCH_DISCOUNT
    cost_input_cached = cached_toks / 1e6 * pricing["input"] * BATCH_DISCOUNT * CACHE_DISCOUNT
    cost_output = completion_toks / 1e6 * pricing["output"] * BATCH_DISCOUNT
    cost_total = cost_input_uncached + cost_input_cached + cost_output

    cost_without_cache = prompt_toks / 1e6 * pricing["input"] * BATCH_DISCOUNT + cost_output
    saved = cost_without_cache - cost_total

    n_completed = len(state.completed_batches())
    n_failed = len(state.failed_batches())

    table = Table(title="Pipeline Cost Report", expand=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Model", state.model)
    table.add_row("Batches completed", str(n_completed))
    table.add_row("Batches failed/expired", str(n_failed))
    table.add_row("", "")
    table.add_row("Prompt tokens", f"{prompt_toks:,}")
    table.add_row("Completion tokens", f"{completion_toks:,}")
    table.add_row("Cached tokens", f"{cached_toks:,}")
    table.add_row("Cache hit rate", f"{cache_rate:.1f}%")
    table.add_row("", "")
    table.add_row("Cost (input, uncached)", f"${cost_input_uncached:,.2f}")
    table.add_row("Cost (input, cached)", f"${cost_input_cached:,.2f}")
    table.add_row("Cost (output)", f"${cost_output:,.2f}")
    table.add_row("[bold]Total cost[/]", f"[bold]${cost_total:,.2f}[/]")
    table.add_row("Saved by caching", f"${saved:,.2f}")

    return table


def print_report(state: PipelineState, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Print the full classification distribution and cost report."""
    console = Console()

    rows: list[dict] = []
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    if rows:
        console.print()
        console.print(_build_distribution_table(rows))

    console.print()
    console.print(_build_cost_table(state))
    console.print()

    logger.info(
        "Report: %d companies classified, %d batches, cache hit %.1f%%",
        len(rows),
        len(state.completed_batches()),
        state.total_cached_tokens / max(state.total_prompt_tokens, 1) * 100,
    )
