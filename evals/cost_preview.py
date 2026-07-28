"""Offline cost preview for the production-owned classifier eval matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from two_pass_classifier import config as production_config
from two_pass_classifier.costing import count_request_input_tokens
from two_pass_classifier.request_builder import (
    RequestSettings,
    build_pass_a_request,
    build_pass_b_request,
)

from evals import config as cfg
from evals.cost_extrapolate import resolve_production_population


@dataclass(frozen=True)
class CostEstimate:
    """One priced Pass A bank, Pass B cell, or combined cell."""

    label: str
    model: str
    kind: str
    effort_b: str | None
    n_rows: int
    est_input_tokens: int
    input_tokens_min: int
    input_tokens_max: int
    est_output_tokens: int
    one_attempt_cap_tokens: int
    est_input_cost: float
    est_output_cost: float
    one_attempt_cap_cost: float
    includes_pass_a: bool

    @property
    def est_total_cost(self) -> float:
        return self.est_input_cost + self.est_output_cost

    @property
    def one_attempt_cap_total_cost(self) -> float:
        return self.est_input_cost + self.one_attempt_cap_cost


@dataclass(frozen=True)
class MatrixEstimate:
    """Three Pass A banks plus nine Pass B cells."""

    pass_a: list[CostEstimate]
    cells: list[CostEstimate]
    n_rows: int
    production_row_count: int
    production_population_label: str
    production_population_source: str

    @property
    def total_cost(self) -> float:
        return sum(item.est_total_cost for item in self.pass_a + self.cells)

    @property
    def one_attempt_cap_total_cost(self) -> float:
        return sum(
            item.one_attempt_cap_total_cost
            for item in self.pass_a + self.cells
        )


def _pricing(model: str) -> dict[str, float]:
    try:
        return production_config.require_model_pricing(model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _settings(model: str, effort_b: str) -> RequestSettings:
    return RequestSettings(model=model, pass_b_effort=effort_b)


def _costs(
    model: str,
    input_tokens: int,
    output_tokens: int,
    one_attempt_cap_tokens: int,
) -> tuple[float, float, float]:
    pricing = _pricing(model)
    return (
        input_tokens / 1_000_000 * pricing["input"],
        output_tokens / 1_000_000 * pricing["output"],
        one_attempt_cap_tokens / 1_000_000 * pricing["output"],
    )


def estimate_pass_a(model: str, rows: list[dict[str, Any]]) -> CostEstimate:
    """Price one production Pass A bank over the golden rows."""
    settings = _settings(model, production_config.DEFAULT_PASS_B_EFFORT)
    input_tokens = sum(
        count_request_input_tokens(build_pass_a_request(row, settings))
        for row in rows
    )
    output_tokens = (
        len(rows) * production_config.PASS_A_PROVISIONAL_OUTPUT_TOKENS
    )
    output_cap = len(rows) * settings.pass_a_max_output_tokens
    input_cost, output_cost, cap_cost = _costs(
        model,
        input_tokens,
        output_tokens,
        output_cap,
    )
    return CostEstimate(
        label=f"Pass A bank ({model.rsplit('-', 1)[-1]})",
        model=model,
        kind="pass_a",
        effort_b=None,
        n_rows=len(rows),
        est_input_tokens=input_tokens,
        input_tokens_min=input_tokens,
        input_tokens_max=input_tokens,
        est_output_tokens=output_tokens,
        one_attempt_cap_tokens=output_cap,
        est_input_cost=input_cost,
        est_output_cost=output_cost,
        one_attempt_cap_cost=cap_cost,
        includes_pass_a=True,
    )


def estimate_pass_b(
    model: str,
    effort_b: str,
    rows: list[dict[str, Any]],
) -> CostEstimate:
    """Price one Pass B cell using both production family routes."""
    settings = _settings(model, effort_b)
    family_totals = {
        family: sum(
            count_request_input_tokens(
                build_pass_b_request(row, family, settings)
            )
            for row in rows
        )
        for family in (0, 1)
    }
    input_min = min(family_totals.values())
    input_max = max(family_totals.values())
    input_tokens = round((input_min + input_max) / 2)
    output_tokens = (
        len(rows)
        * production_config.PASS_B_PREVIEW_OUTPUT_TOKENS[effort_b]
    )
    output_cap = len(rows) * int(settings.pass_b_max_output_tokens or 0)
    input_cost, output_cost, cap_cost = _costs(
        model,
        input_tokens,
        output_tokens,
        output_cap,
    )
    return CostEstimate(
        label=f"{model.rsplit('-', 1)[-1]} / {effort_b}",
        model=model,
        kind="pass_b",
        effort_b=effort_b,
        n_rows=len(rows),
        est_input_tokens=input_tokens,
        input_tokens_min=input_min,
        input_tokens_max=input_max,
        est_output_tokens=output_tokens,
        one_attempt_cap_tokens=output_cap,
        est_input_cost=input_cost,
        est_output_cost=output_cost,
        one_attempt_cap_cost=cap_cost,
        includes_pass_a=False,
    )


def estimate_cell(
    model: str,
    effort_b: str,
    rows: list[dict[str, Any]],
    *,
    include_pass_a: bool,
) -> CostEstimate:
    """Price Pass B and optionally include a not-yet-banked Pass A."""
    pass_b = estimate_pass_b(model, effort_b, rows)
    if not include_pass_a:
        return pass_b
    pass_a = estimate_pass_a(model, rows)
    return CostEstimate(
        label=f"{model.rsplit('-', 1)[-1]} / {effort_b} (A+B)",
        model=model,
        kind="cell",
        effort_b=effort_b,
        n_rows=len(rows),
        est_input_tokens=(
            pass_a.est_input_tokens + pass_b.est_input_tokens
        ),
        input_tokens_min=(
            pass_a.input_tokens_min + pass_b.input_tokens_min
        ),
        input_tokens_max=(
            pass_a.input_tokens_max + pass_b.input_tokens_max
        ),
        est_output_tokens=(
            pass_a.est_output_tokens + pass_b.est_output_tokens
        ),
        one_attempt_cap_tokens=(
            pass_a.one_attempt_cap_tokens
            + pass_b.one_attempt_cap_tokens
        ),
        est_input_cost=pass_a.est_input_cost + pass_b.est_input_cost,
        est_output_cost=pass_a.est_output_cost + pass_b.est_output_cost,
        one_attempt_cap_cost=(
            pass_a.one_attempt_cap_cost
            + pass_b.one_attempt_cap_cost
        ),
        includes_pass_a=True,
    )


def estimate_matrix(
    rows: list[dict[str, Any]],
    *,
    manifest_path: str | Path | None = None,
) -> MatrixEstimate:
    """Price the locked matrix with Pass A paid once per model."""
    population = resolve_production_population(manifest_path)
    pass_a = [estimate_pass_a(model, rows) for model in cfg.EVAL_MODELS]
    cells = [
        estimate_pass_b(model, effort, rows)
        for model in cfg.EVAL_MODELS
        for effort in cfg.MATRIX_PASS_B_EFFORTS
    ]
    return MatrixEstimate(
        pass_a=pass_a,
        cells=cells,
        n_rows=len(rows),
        production_row_count=population.row_count,
        production_population_label=population.label,
        production_population_source=population.source,
    )


def format_matrix_table(estimate: MatrixEstimate) -> str:
    """Render the paid cost gate without applying a Batch discount."""
    lines = [
        f"Locked eval matrix cost preview ({estimate.n_rows} golden rows)",
        f"  models = {list(cfg.EVAL_MODELS)}",
        f"  Pass B efforts = {list(cfg.MATRIX_PASS_B_EFFORTS)}",
        (
            "  production N reference = "
            f"{estimate.production_row_count:,} "
            f"({estimate.production_population_label}; "
            f"{estimate.production_population_source})"
        ),
        "",
        "Pass A (banked once per model):",
    ]
    for item in estimate.pass_a:
        lines.append(
            f"  {item.label:<28}  ~${item.est_total_cost:.4f}  "
            f"(in {item.est_input_tokens:,} + out ~{item.est_output_tokens:,}; "
            f"one-attempt cap ${item.one_attempt_cap_total_cost:.4f})"
        )
    lines.extend(("", "Pass B cells (reuse Pass A bank):"))
    for item in estimate.cells:
        lines.append(
            f"  {item.label:<28}  ~${item.est_total_cost:.4f}  "
            f"(in {item.input_tokens_min:,}-{item.input_tokens_max:,} + "
            f"out ~{item.est_output_tokens:,}; "
            f"one-attempt cap ${item.one_attempt_cap_total_cost:.4f})"
        )
    lines.extend(
        (
            "",
            f"TOTAL estimated spend: ~${estimate.total_cost:.4f}",
            (
                "TOTAL one-attempt cap projection: "
                f"${estimate.one_attempt_cap_total_cost:.4f}"
            ),
            (
                "The cap projection assumes one attempt per planned request; "
                "retries or resumed attempts can exceed it."
            ),
            "Normal Responses API prices are used. No Batch discount is applied.",
        )
    )
    return "\n".join(lines)


def print_matrix_preview(
    rows: list[dict[str, Any]],
    *,
    manifest_path: str | Path | None = None,
) -> MatrixEstimate:
    """Estimate and print the locked matrix."""
    estimate = estimate_matrix(rows, manifest_path=manifest_path)
    print(format_matrix_table(estimate))
    return estimate
