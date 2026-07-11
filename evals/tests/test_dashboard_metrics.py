"""Offline tests for Stage 9 dashboard metrics (fixture + filter keys).

No OpenAI key required: dashboard_metrics does not import src.config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.dashboard_metrics import (
    DEFAULT_FIXTURE,
    MODEL_GROUP_ORDER,
    build_metrics,
    config_row_from_scored,
    load_fixture,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOCK = FIXTURES / "dashboard" / "dashboard_mock_runs.json"

# Fixture run_ids (unique filter keys). Labels stay model×effort.
FIXTURE_IDS = [
    "mock_gpt-5.4-nano_low_r1",
    "mock_gpt-5.4-nano_medium_r1",
    "mock_gpt-5.4-nano_high_r1",
    "mock_gpt-5.4-mini_low_r1",
    "mock_gpt-5.4-mini_medium_r1",
    "mock_gpt-5.4-mini_high_r1",
    "mock_gpt-5.6-luna_low_r1",
    "mock_gpt-5.6-luna_medium_r1",
    "mock_gpt-5.6-luna_high_r1",
]


def test_default_fixture_path_exists():
    assert DEFAULT_FIXTURE.exists()
    assert MOCK.exists()


def test_load_fixture_has_nine_stage8_configs():
    metrics = load_fixture(MOCK)
    assert metrics["synthetic"] is True
    assert metrics["n_configs"] == 9
    assert len(metrics["configs"]) == 9
    assert len(metrics["config_ids"]) == 9
    # Filter keys are unique per run (run_id), not collapsed model×effort.
    assert set(metrics["config_ids"]) == set(FIXTURE_IDS)
    assert len(set(metrics["config_ids"])) == 9
    assert metrics["model_group_order"] == list(MODEL_GROUP_ORDER)
    for g in MODEL_GROUP_ORDER:
        assert g in metrics["model_groups"]
        assert len(metrics["model_groups"][g]["ids"]) == 3


def test_fixture_rows_carry_chart_fields():
    metrics = load_fixture(MOCK)
    row = next(c for c in metrics["configs"] if c["id"] == "mock_gpt-5.4-mini_medium_r1")
    assert row["label"] == "mini / medium"
    assert row["model"] == "gpt-5.4-mini"
    assert row["effort_b"] == "medium"
    assert row["model_group"] == "mini"
    assert row["kind"] == "two_pass"
    assert 0.8 < row["subclass_acc"] < 0.9
    assert row["projected_usd"] == 412
    assert row["latency_p50"] == 4.6
    assert row["share_above_90"] == 0.61
    assert row["ece"] == 0.038
    assert row["n_scored"] == 100
    assert row["n_expected"] == 100
    assert row["is_partial"] is False


def test_config_row_prefers_scored_metadata_over_run_id_parse():
    """Explicit scored.json fields win when run_id would parse differently."""
    stub = {
        "run_id": "mock_gpt-5.4-nano_high_r1",  # would parse nano/high
        "model": "gpt-5.6-luna",
        "effort_b": "low",
        "kind": "two_pass",
        "n_scored": 100,
        "axes": {
            "subclass": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.75},
            "ai_native": {"accuracy": 0.9, "accuracy_ci95": [0.8, 1.0], "macro_f1": 0.9},
            "rad": {"accuracy": 0.85, "accuracy_ci95": [0.8, 0.9], "macro_f1": 0.8},
        },
    }
    row = config_row_from_scored(stub)
    assert row["model"] == "gpt-5.6-luna"
    assert row["model_group"] == "luna"
    assert row["effort_b"] == "low"
    assert row["label"] == "luna / low"


def test_config_row_from_minimal_scored_stub():
    stub = {
        "run_id": "2026-07-10_gpt-5.4-nano_high_r1",
        "model": "gpt-5.4-nano",
        "effort_b": "high",
        "kind": "two_pass",
        "n_scored": 10,
        "axes": {
            "subclass": {
                "accuracy": 0.7,
                "accuracy_ci95": [0.6, 0.8],
                "macro_f1": 0.65,
            },
            "ai_native": {"accuracy": 0.9, "accuracy_ci95": [0.8, 1.0], "macro_f1": 0.9},
            "rad": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.75},
        },
        "production_cost_estimate": {
            "available": True,
            "steps": {"4_scale": {"available": True, "estimated_production_usd": 120.5}},
        },
        "latency": {"latency_s": {"p50": 3.0, "p95": 7.0}},
        "calibration": None,
    }
    row = config_row_from_scored(stub)
    assert row["id"] == "2026-07-10_gpt-5.4-nano_high_r1"
    assert row["label"] == "nano / high"
    assert row["model_group"] == "nano"
    assert row["kind"] == "two_pass"
    assert row["subclass_acc"] == 0.7
    assert row["subclass_ci"] == pytest.approx(0.1)
    assert row["projected_usd"] == 120.5
    assert row["latency_p50"] == 3.0


def test_build_metrics_filter_keys_stable_order():
    raw = json.loads(MOCK.read_text(encoding="utf-8"))
    # Feed runs out of order; output should still be nano→mini→luna × low→med→high.
    shuffled = list(reversed(raw["scored_runs"]))
    metrics = build_metrics(shuffled, synthetic=True, source="test")
    assert metrics["config_ids"] == FIXTURE_IDS


def test_duplicate_model_effort_keeps_distinct_filter_ids():
    """Stage-2 finalist repeats share model×effort but must not collide in Set filters."""
    base = {
        "model": "gpt-5.4-mini",
        "effort_b": "medium",
        "n_scored": 100,
        "axes": {
            "subclass": {"accuracy": 0.85, "accuracy_ci95": [0.8, 0.9], "macro_f1": 0.8},
            "ai_native": {"accuracy": 0.95, "accuracy_ci95": [0.9, 1.0], "macro_f1": 0.95},
            "rad": {"accuracy": 0.88, "accuracy_ci95": [0.82, 0.94], "macro_f1": 0.86},
        },
        "screen": {"id": "mini-med", "label": "mini / medium"},
    }
    runs = [
        {**base, "run_id": "2026-07-10_gpt-5.4-mini_medium_r1"},
        {**base, "run_id": "2026-07-10_gpt-5.4-mini_medium_r2"},
        {**base, "run_id": "2026-07-10_gpt-5.4-mini_medium_r3"},
    ]
    metrics = build_metrics(runs, synthetic=True, source="finalist-repeats")
    assert metrics["n_configs"] == 3
    assert len(set(metrics["config_ids"])) == 3
    assert metrics["config_ids"] == [
        "2026-07-10_gpt-5.4-mini_medium_r1",
        "2026-07-10_gpt-5.4-mini_medium_r2",
        "2026-07-10_gpt-5.4-mini_medium_r3",
    ]
    # Colliding model×effort labels get · rN so chart pills/axes stay distinct.
    assert [c["label"] for c in metrics["configs"]] == [
        "mini / medium · r1",
        "mini / medium · r2",
        "mini / medium · r3",
    ]
    # Group pill still lists every repeat (not collapsed to one mini-med id).
    assert len(metrics["model_groups"]["mini"]["ids"]) == 3


def test_single_pass_kind_for_banked_none_effort():
    row = config_row_from_scored(
        {
            "run_id": "2026-07-06_gpt-5.4-nano_none_r1",
            "model": "gpt-5.4-nano",
            "effort": "none",
            "kind": "single_pass",
            "n_scored": 10,
            "axes": {
                "subclass": {"accuracy": 0.41, "accuracy_ci95": [0.3, 0.5], "macro_f1": 0.3},
                "ai_native": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.8},
                "rad": {"accuracy": 0.5, "accuracy_ci95": [0.4, 0.6], "macro_f1": 0.4},
            },
        }
    )
    assert row["kind"] == "single_pass"
    assert row["effort_b"] == "none"
    assert row["subclass_acc"] == 0.41


def test_build_html_pareto_y_axis_is_data_driven():
    """Generator must not ship the mock-only 60–95% Pareto band."""
    _, mod = _load_eval_dashboard_builder()
    html = mod.build_html(load_fixture(MOCK))
    assert "range: [0.6, 0.95]" not in html
    assert "yRange" in html
    assert "effortCaption" in html
    assert "Pass B ' + c.effort_b" not in html


def test_committed_html_keeps_pareto_axis_in_sync():
    """Checked-in eval_dashboard.html must match the generator's data-driven axis."""
    from evals.paths import PROJECT_ROOT

    html = (
        PROJECT_ROOT
        / "data visualization"
        / "01_Presentation_Materials"
        / "eval_dashboard.html"
    ).read_text(encoding="utf-8")
    assert "range: [0.6, 0.95]" not in html
    assert "yRange" in html
    assert "effortCaption" in html


def test_projected_usd_none_when_cost_unavailable():
    """Legacy / blocked cost estimates yield null; Pareto must omit those points."""
    stub = {
        "run_id": "2026-07-10_gpt-5.4-nano_low_r1",
        "model": "gpt-5.4-nano",
        "effort_b": "low",
        "n_scored": 10,
        "axes": {
            "subclass": {
                "accuracy": 0.7,
                "accuracy_ci95": [0.6, 0.8],
                "macro_f1": 0.65,
            },
            "ai_native": {"accuracy": 0.9, "accuracy_ci95": [0.8, 1.0], "macro_f1": 0.9},
            "rad": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.75},
        },
        "production_cost_estimate": {"available": False},
        "latency": {"latency_s": {"p50": 2.0, "p95": 5.0}},
        "calibration": None,
    }
    row = config_row_from_scored(stub)
    assert row["projected_usd"] is None


def _load_eval_dashboard_builder():
    import argparse
    import importlib.util

    from evals.paths import PROJECT_ROOT

    builder = (
        PROJECT_ROOT
        / "data visualization"
        / "02_Analysis_Code"
        / "build_eval_dashboard.py"
    )
    spec = importlib.util.spec_from_file_location("build_eval_dashboard", builder)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return argparse, mod


def test_config_row_unknown_effort_when_metadata_missing():
    """Missing effort must not silently become medium (mislabels the cell)."""
    stub = {
        "run_id": "2026-07-10_gpt-5.4-nano_r1",  # no effort token in id
        "model": "gpt-5.4-nano",
        "n_scored": 100,
        "n_expected": 100,
        "axes": {
            "subclass": {"accuracy": 0.7, "accuracy_ci95": [0.6, 0.8], "macro_f1": 0.65},
            "ai_native": {"accuracy": 0.9, "accuracy_ci95": [0.8, 1.0], "macro_f1": 0.9},
            "rad": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.75},
        },
    }
    row = config_row_from_scored(stub)
    assert row["effort_b"] == "unknown"
    assert row["label"] == "nano / unknown"
    assert row["is_partial"] is False


def test_config_row_marks_partial_when_n_scored_below_expected():
    """--allow-partial screens must carry n_expected so the UI can warn."""
    stub = {
        "run_id": "2026-07-10_gpt-5.4-mini_medium_r1",
        "model": "gpt-5.4-mini",
        "effort_b": "medium",
        "n_scored": 40,
        "n_expected": 100,
        "axes": {
            "subclass": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.75},
            "ai_native": {"accuracy": 0.9, "accuracy_ci95": [0.8, 1.0], "macro_f1": 0.9},
            "rad": {"accuracy": 0.85, "accuracy_ci95": [0.8, 0.9], "macro_f1": 0.8},
        },
    }
    row = config_row_from_scored(stub)
    assert row["n_scored"] == 40
    assert row["n_expected"] == 100
    assert row["is_partial"] is True


def test_unknown_effort_sorts_without_crashing():
    """Unknown effort ranks after known efforts; sort must not KeyError."""
    stubs = [
        {
            "run_id": "run_unknown",
            "model": "gpt-5.4-nano",
            "n_scored": 10,
            "n_expected": 10,
            "axes": {
                "subclass": {"accuracy": 0.5, "macro_f1": 0.4},
                "ai_native": {"accuracy": 0.5, "macro_f1": 0.4},
                "rad": {"accuracy": 0.5, "macro_f1": 0.4},
            },
        },
        {
            "run_id": "run_low",
            "model": "gpt-5.4-nano",
            "effort_b": "low",
            "n_scored": 10,
            "n_expected": 10,
            "axes": {
                "subclass": {"accuracy": 0.5, "macro_f1": 0.4},
                "ai_native": {"accuracy": 0.5, "macro_f1": 0.4},
                "rad": {"accuracy": 0.5, "macro_f1": 0.4},
            },
        },
    ]
    metrics = build_metrics(stubs, synthetic=True, source="test")
    efforts = [c["effort_b"] for c in metrics["configs"]]
    assert efforts == ["low", "unknown"]


def test_build_html_surfaces_partial_and_unknown_effort():
    """Generator JS must expose partial badge + unknown effort captions."""
    _, mod = _load_eval_dashboard_builder()
    html = mod.build_html(load_fixture(MOCK))
    assert "partial-badge" in html
    assert "isPartial" in html
    assert "sampleCaption" in html
    assert "effort unknown" in html
    assert "partial screen" in html


def test_resolve_metrics_defaults_to_fixture_not_discovered_runs():
    """Bare dashboard build must stay on the mock matrix (no auto-load of banked scored.json)."""
    argparse, mod = _load_eval_dashboard_builder()
    ns = argparse.Namespace(
        fixture=None,
        force_fixture=False,
        scored=None,
        runs=None,
    )
    metrics = mod.resolve_metrics(ns)
    assert metrics["synthetic"] is True
    assert metrics["n_configs"] == 9

    ns_force = argparse.Namespace(
        fixture=None,
        force_fixture=True,
        scored=None,
        runs=None,
    )
    assert mod.resolve_metrics(ns_force)["synthetic"] is True
