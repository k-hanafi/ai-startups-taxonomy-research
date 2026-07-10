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
MOCK = FIXTURES / "dashboard_mock_runs.json"


def test_default_fixture_path_exists():
    assert DEFAULT_FIXTURE.exists()
    assert MOCK.exists()


def test_load_fixture_has_nine_stage8_configs():
    metrics = load_fixture(MOCK)
    assert metrics["synthetic"] is True
    assert metrics["n_configs"] == 9
    assert len(metrics["configs"]) == 9
    assert len(metrics["config_ids"]) == 9
    # Filter keys the HTML toolbar needs.
    assert set(metrics["config_ids"]) == {
        "nano-low", "nano-med", "nano-high",
        "mini-low", "mini-med", "mini-high",
        "luna-low", "luna-med", "luna-high",
    }
    assert metrics["model_group_order"] == list(MODEL_GROUP_ORDER)
    for g in MODEL_GROUP_ORDER:
        assert g in metrics["model_groups"]
        assert len(metrics["model_groups"][g]["ids"]) == 3


def test_fixture_rows_carry_chart_fields():
    metrics = load_fixture(MOCK)
    row = next(c for c in metrics["configs"] if c["id"] == "mini-med")
    assert row["model"] == "gpt-5.4-mini"
    assert row["effort_b"] == "medium"
    assert row["model_group"] == "mini"
    assert 0.8 < row["subclass_acc"] < 0.9
    assert row["projected_usd"] == 412
    assert row["latency_p50"] == 4.6
    assert row["share_above_90"] == 0.61
    assert row["ece"] == 0.038


def test_config_row_from_minimal_scored_stub():
    stub = {
        "run_id": "2026-07-10_gpt-5.4-nano_high_r1",
        "model": "gpt-5.4-nano",
        "effort_b": "high",
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
    assert row["id"] == "nano-high"
    assert row["model_group"] == "nano"
    assert row["subclass_acc"] == 0.7
    assert row["subclass_ci"] == pytest.approx(0.1)
    assert row["projected_usd"] == 120.5
    assert row["latency_p50"] == 3.0


def test_build_metrics_filter_keys_stable_order():
    raw = json.loads(MOCK.read_text(encoding="utf-8"))
    # Feed runs out of order; output should still be nano→mini→luna × low→med→high.
    shuffled = list(reversed(raw["scored_runs"]))
    metrics = build_metrics(shuffled, synthetic=True, source="test")
    assert metrics["config_ids"] == [
        "nano-low", "nano-med", "nano-high",
        "mini-low", "mini-med", "mini-high",
        "luna-low", "luna-med", "luna-high",
    ]
