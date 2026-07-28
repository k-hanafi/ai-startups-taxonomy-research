"""Offline tests for the locked-matrix cost preview (no API key, no network)."""

from __future__ import annotations

import pytest

from evals import config as cfg
from evals.cost_preview import (
    estimate_cell,
    estimate_matrix,
    estimate_pass_a,
    estimate_pass_b,
    format_matrix_table,
)
from two_pass_classifier import config as production_config


def _fake_rows(n: int = 3) -> list[dict]:
    return [
        {
            "org_uuid": f"u{i}",
            "name": f"Co{i}",
            "short_description": "AI product",
            "website_evidence": "We build AI tools " * 20,
            "website_pages_used": "https://example.com",
            "founded_on": "2024-01-01",
            "category_list": "Artificial Intelligence",
            "category_groups_list": "Software",
            "employee_count": "11-50",
            "total_funding_usd": "1000000",
        }
        for i in range(n)
    ]


def test_pass_a_billed_once_per_model_not_three_times():
    rows = _fake_rows(5)
    matrix = estimate_matrix(rows)
    assert len(matrix.pass_a) == len(cfg.EVAL_MODELS)
    assert len(matrix.cells) == 9
    # Three cells per model must not re-count Pass A in the cell lines.
    assert all(not c.includes_pass_a for c in matrix.cells)
    assert all(c.kind == "pass_b" for c in matrix.cells)
    assert all(a.kind == "pass_a" for a in matrix.pass_a)


def test_total_equals_sum_of_parts():
    rows = _fake_rows(4)
    matrix = estimate_matrix(rows)
    parts = sum(e.est_total_cost for e in matrix.pass_a) + sum(
        e.est_total_cost for e in matrix.cells
    )
    assert matrix.total_cost == pytest.approx(parts)
    assert matrix.total_cost > 0


def test_nine_cells_cover_locked_matrix():
    matrix = estimate_matrix(_fake_rows(2))
    pairs = {(c.model, c.effort_b) for c in matrix.cells}
    expected = {
        (m, e) for m in cfg.EVAL_MODELS for e in cfg.MATRIX_PASS_B_EFFORTS
    }
    assert pairs == expected


def test_unknown_model_refuses_rather_than_zero():
    with pytest.raises((SystemExit, ValueError), match="unsupported model|unknown pricing"):
        estimate_pass_a("gpt-not-a-real-model", _fake_rows(1))
    with pytest.raises((SystemExit, ValueError), match="unsupported model|unknown pricing"):
        estimate_pass_b("gpt-not-a-real-model", "low", _fake_rows(1))


def test_higher_effort_costs_at_least_as_much():
    rows = _fake_rows(3)
    model = cfg.EVAL_MODELS[0]
    low = estimate_pass_b(model, "low", rows)
    high = estimate_pass_b(model, "high", rows)
    assert high.est_total_cost >= low.est_total_cost


def test_estimate_cell_include_pass_a_adds_cost():
    rows = _fake_rows(2)
    model = cfg.EVAL_MODELS[0]
    b_only = estimate_cell(model, "low", rows, include_pass_a=False)
    both = estimate_cell(model, "low", rows, include_pass_a=True)
    a = estimate_pass_a(model, rows)
    assert both.est_total_cost == pytest.approx(
        a.est_total_cost + b_only.est_total_cost
    )
    assert both.includes_pass_a
    assert not b_only.includes_pass_a


def test_format_matrix_table_mentions_total():
    matrix = estimate_matrix(_fake_rows(2))
    text = format_matrix_table(matrix)
    assert "TOTAL estimated spend" in text
    assert f"${matrix.total_cost:.4f}" in text
    assert "TOTAL one-attempt cap projection" in text
    assert "retries or resumed attempts can exceed it" in text
    assert "ceiling" not in text.lower()
    assert "No Batch discount is applied" in text
    for model in cfg.EVAL_MODELS:
        assert model.split("-")[-1] in text


def test_preview_uses_production_output_estimates_and_caps():
    rows = _fake_rows(2)
    pass_a = estimate_pass_a("gpt-5.4-nano", rows)
    pass_b = estimate_pass_b("gpt-5.4-nano", "medium", rows)

    assert production_config.PASS_A_PROVISIONAL_OUTPUT_TOKENS == 320
    assert pass_a.est_output_tokens == (
        len(rows) * production_config.PASS_A_PROVISIONAL_OUTPUT_TOKENS
    )
    assert pass_a.one_attempt_cap_tokens == (
        len(rows) * production_config.PASS_A_MAX_OUTPUT_TOKENS
    )
    assert pass_b.est_output_tokens == (
        len(rows)
        * production_config.PASS_B_PREVIEW_OUTPUT_TOKENS["medium"]
    )
    assert pass_b.one_attempt_cap_tokens == (
        len(rows) * production_config.PASS_B_MAX_OUTPUT_TOKENS["medium"]
    )
