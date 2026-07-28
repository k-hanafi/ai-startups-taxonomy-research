"""Offline cost estimates for the locked eval matrix.

No OpenAI import and no single_pass_classifier.config import: cost-preview must run without
an API key. The formula here is the single source of truth for both
``python -m evals cost-preview`` and classification ``--dry-run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from single_pass_classifier.formatter import format_user_message

from evals import config as cfg
from evals.paths import (
    BINARY_GATE_PROMPT,
    FAMILY_BLOCK_AI,
    FAMILY_BLOCK_NOT,
    SUBCLASS_RAD_PROMPT,
)

FAMILY_BLOCK_PLACEHOLDER = "{family_block}"


@dataclass(frozen=True)
class CostEstimate:
    """One priced line: Pass A bank, Pass B cell, or a combined dry-run cell."""

    label: str
    model: str
    kind: str  # "pass_a" | "pass_b" | "cell"
    effort_b: str | None
    n_rows: int
    est_input_tokens: int
    est_output_tokens: int
    est_input_cost: float
    est_output_cost: float
    includes_pass_a: bool

    @property
    def est_total_cost(self) -> float:
        return self.est_input_cost + self.est_output_cost


@dataclass(frozen=True)
class MatrixEstimate:
    """Full 9-cell screen: Pass A once per model + Pass B per cell."""

    pass_a: list[CostEstimate]
    cells: list[CostEstimate]
    n_rows: int

    @property
    def total_cost(self) -> float:
        return sum(e.est_total_cost for e in self.pass_a) + sum(
            e.est_total_cost for e in self.cells
        )


def _load_pass_a_prompt() -> str:
    return BINARY_GATE_PROMPT.read_text(encoding="utf-8").strip()


def _load_pass_b_prompt(family: int) -> str:
    template = SUBCLASS_RAD_PROMPT.read_text(encoding="utf-8").strip()
    block_path = FAMILY_BLOCK_AI if family == 1 else FAMILY_BLOCK_NOT
    block = block_path.read_text(encoding="utf-8").strip()
    if FAMILY_BLOCK_PLACEHOLDER not in template:
        raise AssertionError(
            f"{SUBCLASS_RAD_PROMPT.name} is missing the "
            f"{FAMILY_BLOCK_PLACEHOLDER} placeholder"
        )
    return template.replace(FAMILY_BLOCK_PLACEHOLDER, block)


def _pass_a_message(row: dict[str, Any]) -> str:
    trimmed = dict(row)
    trimmed["website_pages_used"] = ""
    return format_user_message(trimmed)


def _input_chars_pass_a(rows: list[dict[str, Any]], prompt_a: str | None = None) -> int:
    prompt = prompt_a if prompt_a is not None else _load_pass_a_prompt()
    return sum(len(prompt) + len(_pass_a_message(r)) for r in rows)


def _input_chars_pass_b(rows: list[dict[str, Any]]) -> int:
    # Pass B prompt size depends on the family; use the mean of both.
    prompt_b1 = _load_pass_b_prompt(1)
    prompt_b0 = _load_pass_b_prompt(0)
    b_prompt_mean = (len(prompt_b1) + len(prompt_b0)) / 2
    # +40 chars for PriorBinaryVerdict / Cohort conditioning lines.
    return sum(b_prompt_mean + len(format_user_message(r)) + 40 for r in rows)


def _price(model: str, est_input: float, est_out: float) -> tuple[float, float]:
    pricing = cfg.require_model_pricing(model)
    return (
        est_input / 1e6 * pricing["input"],
        est_out / 1e6 * pricing["output"],
    )


def estimate_pass_a(model: str, rows: list[dict[str, Any]]) -> CostEstimate:
    """Cost of banking Pass A once for *model* over *rows*."""
    n = len(rows)
    est_input = _input_chars_pass_a(rows) / 4
    est_out = n * cfg.PASS_A_OUTPUT_TOKEN_ESTIMATE
    in_cost, out_cost = _price(model, est_input, est_out)
    short = model.split("-")[-1] if "-" in model else model
    return CostEstimate(
        label=f"Pass A bank ({short})",
        model=model,
        kind="pass_a",
        effort_b=None,
        n_rows=n,
        est_input_tokens=int(est_input),
        est_output_tokens=int(est_out),
        est_input_cost=in_cost,
        est_output_cost=out_cost,
        includes_pass_a=True,
    )


def estimate_pass_b(
    model: str, effort_b: str, rows: list[dict[str, Any]]
) -> CostEstimate:
    """Cost of one Pass B cell (Pass A assumed already banked)."""
    n = len(rows)
    est_input = _input_chars_pass_b(rows) / 4
    est_out = n * cfg.PASS_B_OUTPUT_TOKEN_ESTIMATE.get(effort_b, 1_000)
    in_cost, out_cost = _price(model, est_input, est_out)
    short = model.split("-")[-1] if "-" in model else model
    return CostEstimate(
        label=f"{short} / {effort_b}",
        model=model,
        kind="pass_b",
        effort_b=effort_b,
        n_rows=n,
        est_input_tokens=int(est_input),
        est_output_tokens=int(est_out),
        est_input_cost=in_cost,
        est_output_cost=out_cost,
        includes_pass_a=False,
    )


def estimate_cell(
    model: str,
    effort_b: str,
    rows: list[dict[str, Any]],
    *,
    include_pass_a: bool,
) -> CostEstimate:
    """Single-cell dry-run estimate (Pass B, optionally + Pass A)."""
    b = estimate_pass_b(model, effort_b, rows)
    if not include_pass_a:
        return b
    a = estimate_pass_a(model, rows)
    short = model.split("-")[-1] if "-" in model else model
    return CostEstimate(
        label=f"{short} / {effort_b} (A+B)",
        model=model,
        kind="cell",
        effort_b=effort_b,
        n_rows=len(rows),
        est_input_tokens=a.est_input_tokens + b.est_input_tokens,
        est_output_tokens=a.est_output_tokens + b.est_output_tokens,
        est_input_cost=a.est_input_cost + b.est_input_cost,
        est_output_cost=a.est_output_cost + b.est_output_cost,
        includes_pass_a=True,
    )


def estimate_matrix(rows: list[dict[str, Any]]) -> MatrixEstimate:
    """Full locked matrix: Pass A once per model + Pass B for each of 9 cells."""
    pass_a = [estimate_pass_a(model, rows) for model in cfg.EVAL_MODELS]
    cells = [
        estimate_pass_b(model, effort, rows)
        for model in cfg.EVAL_MODELS
        for effort in cfg.MATRIX_PASS_B_EFFORTS
    ]
    return MatrixEstimate(pass_a=pass_a, cells=cells, n_rows=len(rows))


def format_matrix_table(estimate: MatrixEstimate) -> str:
    """Plain-text table for the terminal (no rich dependency here)."""
    lines: list[str] = []
    lines.append(
        f"Locked eval matrix cost preview ({estimate.n_rows} golden rows)"
    )
    lines.append(
        f"  models = {cfg.EVAL_MODELS}"
    )
    lines.append(
        f"  Pass B efforts = {cfg.MATRIX_PASS_B_EFFORTS}"
    )
    lines.append("")
    lines.append("Pass A (banked once per model):")
    for e in estimate.pass_a:
        lines.append(
            f"  {e.label:<28}  ~${e.est_total_cost:.4f}  "
            f"(in ~{e.est_input_tokens:,} + out ~{e.est_output_tokens:,})"
        )
    lines.append("")
    lines.append("Pass B cells (reuse Pass A bank):")
    for e in estimate.cells:
        lines.append(
            f"  {e.label:<28}  ~${e.est_total_cost:.4f}  "
            f"(in ~{e.est_input_tokens:,} + out ~{e.est_output_tokens:,})"
        )
    lines.append("")
    lines.append(f"TOTAL estimated spend: ~${estimate.total_cost:.4f}")
    lines.append(
        "Note: output/reasoning token estimates are a floor, not a cap. "
        "medium/high Pass B can dominate spend."
    )
    return "\n".join(lines)


def print_matrix_preview(rows: list[dict[str, Any]]) -> MatrixEstimate:
    """Estimate the locked matrix and print the table. Returns the estimate."""
    estimate = estimate_matrix(rows)
    print(format_matrix_table(estimate))
    return estimate
