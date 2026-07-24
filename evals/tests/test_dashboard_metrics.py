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


def test_load_fixture_has_nine_matrix_configs():
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
    assert row["kind"] == "classification"
    assert 0.8 < row["subclass_acc"] < 0.9
    # Fixture ladders are generated with the real extrapolation formula, so
    # the projected $ is the exact step-4 output (displays as $412).
    assert row["projected_usd"] == pytest.approx(412.3517, abs=1e-3)
    assert row["latency_p50"] == 4.6
    assert row["share_above_90"] == 0.61
    assert row["ece"] == pytest.approx(0.046185)
    assert row["n_scored"] == 100
    assert row["n_expected"] == 100
    assert row["is_partial"] is False
    assert len(row["reliability_bins"]) == 10
    assert len(row["selective_curve"]) == 10


def test_fixture_pass_a_metrics_identical_within_model():
    """Banked Pass A design: ai_native / ECE / confidence do not vary with Pass B effort."""
    metrics = load_fixture(MOCK)
    by_model: dict[str, list] = {}
    for c in metrics["configs"]:
        by_model.setdefault(c["model"], []).append(c)
    for model, rows in by_model.items():
        assert len(rows) == 3
        ai = {r["ai_native_acc"] for r in rows}
        ece = {r["ece"] for r in rows}
        conf = {r["mean_confidence"] for r in rows}
        share = {r["share_above_90"] for r in rows}
        assert len(ai) == 1, model
        assert len(ece) == 1, model
        assert len(conf) == 1, model
        assert len(share) == 1, model
        # Pass B axes may still differ across efforts.
        subclass = {r["subclass_acc"] for r in rows}
        assert len(subclass) >= 2, model


def test_fixture_calibration_recomputes_from_bins():
    """Every derived confidence number must recompute from the same rows.

    The fixture generates bins, curve, mean confidence, share above 0.9 and
    ECE from one set of 100 synthetic per-row pairs per model, so each value
    must be re-derivable from the others.
    """
    metrics = load_fixture(MOCK)
    for c in metrics["configs"]:
        bins = c["reliability_bins"]
        curve = c["selective_curve"]
        assert bins and curve, c["id"]
        total = sum(b["count"] for b in bins)
        assert total == 100, c["id"]
        ece = sum(
            b["count"] / total * abs(b["accuracy"] - b["mean_confidence"])
            for b in bins
            if b["count"]
        )
        assert ece == pytest.approx(c["ece"], abs=1e-5), c["id"]
        mean_conf = (
            sum(b["count"] * b["mean_confidence"] for b in bins if b["count"])
            / total
        )
        assert mean_conf == pytest.approx(c["mean_confidence"], abs=1e-5), c["id"]
        share = sum(b["count"] for b in bins if b["range"][0] >= 0.9) / total
        assert share == pytest.approx(c["share_above_90"]), c["id"]
        # Selective curve: full coverage equals the binary-axis accuracy
        # (calibration measures the ai_native decision), and the 50% point
        # matches the screen's selective_acc_50.
        full = next(p for p in curve if p["coverage"] == 1.0)
        assert full["accuracy"] == pytest.approx(c["ai_native_acc"]), c["id"]
        half = next(p for p in curve if p["coverage"] == 0.5)
        assert half["accuracy"] == pytest.approx(c["selective_acc_50"]), c["id"]


def test_fixture_nano_seeds_documented_early_signal():
    """nano anchors the measured early signal: ECE near 0.077 and a fully
    correct top-confidence half (selective accuracy 1.0 at 50% coverage)."""
    metrics = load_fixture(MOCK)
    nano = next(
        c for c in metrics["configs"] if c["id"] == "mock_gpt-5.4-nano_low_r1"
    )
    assert abs(nano["ece"] - 0.077) < 0.005
    assert nano["selective_acc_50"] == 1.0


def test_fixture_robustness_checks_all_pass():
    metrics = load_fixture(MOCK)
    checks = metrics["robustness"]["checks"]
    assert [c["id"] for c in checks] == [
        "tokenization_pinned",
        "probability_mass",
        "batch_parity",
    ]
    for check in checks:
        assert check["status"] == "pass", check["id"]
        assert check["stats"], check["id"]
        assert check["meaning"], check["id"]
        assert len(check["per_model"]) == 3, check["id"]
        assert check["pending_note"] is None, check["id"]
    # Pass A is banked once per model: 3 models x 100 golden companies = 300
    # unique extractions, even though the mock carries 9 matrix cells.
    tok_stats = {s["label"]: s["value"] for s in checks[0]["stats"]}
    assert tok_stats["Unique companies with confidence"] == "300"
    assert tok_stats["Unique companies checked"] == "300"
    assert tok_stats["Models checked"] == "3"


def _axes_stub(run_id: str = "2026-08-01_gpt-5.4-nano_low_r1", **over):
    stub = {
        "run_id": run_id,
        "model": "gpt-5.4-nano",
        "effort_b": "low",
        "kind": "classification",
        "n_scored": 100,
        "n_expected": 100,
        "axes": {
            "subclass": {"accuracy": 0.7, "accuracy_ci95": [0.6, 0.8], "macro_f1": 0.65},
            "ai_native": {"accuracy": 0.9, "accuracy_ci95": [0.8, 1.0], "macro_f1": 0.9},
            "rad": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.75},
        },
    }
    stub.update(over)
    return stub


def test_robustness_pending_when_nothing_recorded():
    """Real runs without calibration or robustness blocks fabricate nothing."""
    metrics = build_metrics([_axes_stub()], synthetic=False, source="test")
    checks = metrics["robustness"]["checks"]
    assert {c["status"] for c in checks} == {"pending"}
    for check in checks:
        assert check["pending_note"], check["id"]
        assert check["stats"] == [], check["id"]
        assert check["per_model"] == [], check["id"]


def _calibration_block(n: int = 100, n_eligible: int = 100) -> dict:
    return {
        "axis": "ai_native",
        "n": n,
        "n_eligible": n_eligible,
        "reliability": {"bins": [], "ece": 0.05},
        "selective_prediction": [],
    }


def test_robustness_tokenization_from_calibration_coverage():
    """Full calibration coverage certifies extraction; a gap fails the check."""
    full = _axes_stub(calibration=_calibration_block())
    metrics = build_metrics([full], synthetic=False, source="test")
    tok = metrics["robustness"]["checks"][0]
    assert tok["id"] == "tokenization_pinned"
    assert tok["status"] == "pass"
    assert tok["per_model"] == [{
        "model": "gpt-5.4-nano",
        "status": "pass",
        "detail": "100 of 100 golden-set companies",
    }]

    partial = _axes_stub(calibration=_calibration_block(n=97))
    metrics = build_metrics([partial], synthetic=False, source="test")
    tok = metrics["robustness"]["checks"][0]
    assert tok["status"] == "fail"


def _banked_matrix_runs(gap_model: str | None = None) -> list[dict]:
    """9 matrix cells (3 models x 3 efforts) reusing one banked Pass A per
    model. gap_model, if set, gets n=97 in every copy of its banked block."""
    runs = []
    for model in ("gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.6-luna"):
        n = 97 if model == gap_model else 100
        for effort in ("low", "medium", "high"):
            runs.append(_axes_stub(
                run_id=f"2026-08-01_{model}_{effort}_r1",
                model=model,
                effort_b=effort,
                calibration=_calibration_block(n=n),
            ))
    return runs


def test_robustness_tokenization_dedupes_banked_pass_a():
    """9 cells sharing 3 banked Pass A blocks count 300 unique rows, not 900."""
    metrics = build_metrics(_banked_matrix_runs(), synthetic=False, source="test")
    tok = metrics["robustness"]["checks"][0]
    assert tok["status"] == "pass"
    stats = {s["label"]: s["value"] for s in tok["stats"]}
    assert stats["Unique companies with confidence"] == "300"
    assert stats["Unique companies checked"] == "300"
    assert stats["Models checked"] == "3"
    assert len(tok["per_model"]) == 3
    assert all(
        r["detail"] == "100 of 100 golden-set companies"
        for r in tok["per_model"]
    )


def test_robustness_tokenization_dedupe_does_not_hide_gap():
    """A model whose banked Pass A misses rows still fails after deduping."""
    metrics = build_metrics(
        _banked_matrix_runs(gap_model="gpt-5.4-mini"),
        synthetic=False,
        source="test",
    )
    tok = metrics["robustness"]["checks"][0]
    assert tok["status"] == "fail"
    by_model = {r["model"]: r for r in tok["per_model"]}
    assert by_model["gpt-5.4-mini"]["status"] == "fail"
    assert by_model["gpt-5.4-mini"]["detail"] == "97 of 100 golden-set companies"
    assert by_model["gpt-5.4-nano"]["status"] == "pass"
    stats = {s["label"]: s["value"] for s in tok["stats"]}
    assert stats["Unique companies with confidence"] == "297"
    assert stats["Unique companies checked"] == "300"


def test_robustness_valid_mass_and_parity_fail_paths():
    stub = _axes_stub(robustness={
        "valid_mass": {
            "n": 100,
            "min": 0.91,
            "p50": 0.998,
            "mean": 0.995,
            "threshold": 0.98,
            "n_below_threshold": 3,
        },
        "batch_parity": {
            "verdict": "FAIL",
            "n_rows": 10,
            "n_checks": 190,
            "n_failed": 2,
        },
    })
    metrics = build_metrics([stub], synthetic=False, source="test")
    by_id = {c["id"]: c for c in metrics["robustness"]["checks"]}
    assert by_id["probability_mass"]["status"] == "fail"
    assert by_id["batch_parity"]["status"] == "fail"
    assert by_id["tokenization_pinned"]["status"] == "pending"


def test_fixture_cost_breakdowns_recompute_to_displayed_total():
    """Every fixture row's popover arithmetic must recompute by hand."""
    metrics = load_fixture(MOCK)
    for c in metrics["configs"]:
        b = c["cost_breakdown"]
        assert b is not None, c["id"]
        assert b["available"] is True, c["id"]
        p = b["pricing_per_mtok"]
        assert p is not None, c["id"]

        # Step 1: sync list on measured tokens.
        sync = (
            b["total_input_tokens"] / 1e6 * p["input"]
            + b["total_output_tokens"] / 1e6 * p["output"]
        )
        assert sync == pytest.approx(b["golden_sync_usd"]), c["id"]

        # Step 2: cache discount on the cached input portion.
        uncached = b["total_input_tokens"] - b["total_cached_tokens"]
        after_cache = (
            uncached / 1e6 * p["input"]
            + b["total_cached_tokens"] / 1e6 * p["input"] * b["cache_discount"]
            + b["total_output_tokens"] / 1e6 * p["output"]
        )
        assert after_cache == pytest.approx(b["golden_after_cache_usd"]), c["id"]
        assert b["cache_hit_rate"] == pytest.approx(
            b["total_cached_tokens"] / b["total_input_tokens"]
        ), c["id"]

        # Step 3: batch discount, step 4: scale to production N.
        after_batch = after_cache * b["batch_discount"]
        assert after_batch == pytest.approx(b["golden_after_batch_usd"]), c["id"]
        projected = after_batch * (b["n_prod"] / b["n_golden"])
        assert projected == pytest.approx(b["estimated_production_usd"]), c["id"]
        assert projected == pytest.approx(c["projected_usd"]), c["id"]

        # Per-pass split sums to the run totals.
        pp = b["per_pass"]
        assert pp is not None, c["id"]
        for kind, total_key in (
            ("input", "total_input_tokens"),
            ("cached", "total_cached_tokens"),
            ("output", "total_output_tokens"),
        ):
            assert (
                pp["pass_a"][kind] + pp["pass_b"][kind] == b[total_key]
            ), (c["id"], kind)


def test_fixture_pass_a_cost_tokens_identical_within_model():
    """Bank-once Pass A: its token totals do not vary with Pass B effort."""
    metrics = load_fixture(MOCK)
    by_model: dict[str, list] = {}
    for c in metrics["configs"]:
        by_model.setdefault(c["model"], []).append(c)
    for model, rows in by_model.items():
        pass_a = {json.dumps(r["cost_breakdown"]["per_pass"]["pass_a"], sort_keys=True) for r in rows}
        assert len(pass_a) == 1, model


def test_cost_breakdown_none_without_recorded_cost_data():
    """No cost / production_cost_estimate blocks → no popover payload."""
    stub = {
        "run_id": "2026-07-10_gpt-5.4-nano_low_r1",
        "model": "gpt-5.4-nano",
        "effort_b": "low",
        "n_scored": 10,
        "axes": {
            "subclass": {"accuracy": 0.7, "accuracy_ci95": [0.6, 0.8], "macro_f1": 0.65},
            "ai_native": {"accuracy": 0.9, "accuracy_ci95": [0.8, 1.0], "macro_f1": 0.9},
            "rad": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.75},
        },
    }
    assert config_row_from_scored(stub)["cost_breakdown"] is None


def test_cost_breakdown_marks_unavailable_ladder_without_fabricating():
    """Legacy runs (no cached_tokens) keep step 1 and null the blocked steps."""
    stub = {
        "run_id": "2026-07-06_gpt-5.4-nano_none_r1",
        "model": "gpt-5.4-nano",
        "effort": "none",
        "kind": "single_pass",
        "n_scored": 100,
        "axes": {
            "subclass": {"accuracy": 0.41, "accuracy_ci95": [0.3, 0.5], "macro_f1": 0.3},
            "ai_native": {"accuracy": 0.8, "accuracy_ci95": [0.7, 0.9], "macro_f1": 0.8},
            "rad": {"accuracy": 0.5, "accuracy_ci95": [0.4, 0.6], "macro_f1": 0.4},
        },
        "cost": {
            "model": "gpt-5.4-nano",
            "n_rows": 100,
            "total_input_tokens": 1_000_000,
            "total_output_tokens": 100_000,
            "total_cached_tokens": None,
            "cache_field_present": False,
            "total_usd": 0.325,
            "mean_usd_per_row": 0.00325,
            "pricing_per_mtok": {"input": 0.20, "output": 1.25},
        },
        "production_cost_estimate": {
            "available": False,
            "reason": "cached_tokens_unavailable",
            "assumptions": {
                "n_prod": 41_076,
                "n_prod_label": "alive_plus_dead",
                "n_golden": 100,
                "model": "gpt-5.4-nano",
                "batch_discount": 0.5,
                "cache_discount": 0.5,
                "cache_source": "unavailable_legacy_run_missing_cached_tokens",
            },
            "steps": {
                "1_golden_sync": {
                    "n_rows": 100,
                    "total_input_tokens": 1_000_000,
                    "total_output_tokens": 100_000,
                    "total_usd": 0.325,
                },
                "2_cache": {
                    "available": False,
                    "reason": "predictions records lack cached_tokens; re-run to measure.",
                    "total_cached_tokens": None,
                    "cache_hit_rate": None,
                    "total_usd_after_cache": None,
                },
                "3_batch": {"available": False, "total_usd_after_batch": None},
                "4_scale": {"available": False, "estimated_production_usd": None},
            },
            "golden_sync_usd": 0.325,
        },
    }
    row = config_row_from_scored(stub)
    b = row["cost_breakdown"]
    assert b is not None
    assert b["available"] is False
    assert b["reason"] == "cached_tokens_unavailable"
    assert b["golden_sync_usd"] == pytest.approx(0.325)
    # Blocked steps stay None (rendered as "not recorded", never invented).
    assert b["total_cached_tokens"] is None
    assert b["cache_hit_rate"] is None
    assert b["golden_after_cache_usd"] is None
    assert b["golden_after_batch_usd"] is None
    assert b["estimated_production_usd"] is None
    assert b["per_pass"] is None
    assert "cached_tokens" in b["cache_step_reason"]
    assert row["projected_usd"] is None


def test_config_row_prefers_scored_metadata_over_run_id_parse():
    """Explicit scored.json fields win when run_id would parse differently."""
    stub = {
        "run_id": "mock_gpt-5.4-nano_high_r1",  # would parse nano/high
        "model": "gpt-5.6-luna",
        "effort_b": "low",
        "kind": "classification",
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
        "kind": "classification",
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
    assert row["kind"] == "classification"
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
    # 3 individual repeats + 1 mean±range aggregate.
    assert metrics["n_configs"] == 4
    assert len(set(metrics["config_ids"])) == 4
    assert metrics["config_ids"][:3] == [
        "2026-07-10_gpt-5.4-mini_medium_r1",
        "2026-07-10_gpt-5.4-mini_medium_r2",
        "2026-07-10_gpt-5.4-mini_medium_r3",
    ]
    assert metrics["config_ids"][3] == "agg_gpt-5.4-mini_medium"
    # Colliding model×effort labels get · rN so chart pills/axes stay distinct.
    assert [c["label"] for c in metrics["configs"] if not c.get("is_aggregate")] == [
        "mini / medium · r1",
        "mini / medium · r2",
        "mini / medium · r3",
    ]
    agg = next(c for c in metrics["configs"] if c.get("is_aggregate"))
    assert agg["n_repeats"] == 3
    assert agg["subclass_acc"] == pytest.approx(0.85)
    assert agg["subclass_acc_range"] == [0.85, 0.85]
    # Mean-of-repeats rows have no single measured ladder → no popover.
    assert agg["cost_breakdown"] is None
    # Group pill lists repeats + aggregate.
    assert len(metrics["model_groups"]["mini"]["ids"]) == 4


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


def test_build_html_includes_cost_breakdown_popover():
    """Generator must ship the cost-info icon + popover machinery."""
    _, mod = _load_eval_dashboard_builder()
    html = mod.build_html(load_fixture(MOCK))
    assert "cost-info" in html
    assert "cost-popover" in html
    assert "costBreakdownHtml" in html
    assert "data-cost-info" in html
    assert "not recorded" in html
    # Breakdown payloads ride along in the embedded metrics JSON.
    assert '"cost_breakdown"' in html
    assert '"golden_after_batch_usd"' in html


def test_committed_html_includes_cost_breakdown_popover():
    """Checked-in eval_dashboard.html must carry the popover machinery."""
    from evals.paths import PROJECT_ROOT

    html = (
        PROJECT_ROOT
        / "data visualization"
        / "01_Presentation_Materials"
        / "eval_dashboard.html"
    ).read_text(encoding="utf-8")
    assert "cost-info" in html
    assert "cost-popover" in html
    assert '"cost_breakdown"' in html


def test_build_html_is_three_tab_suite():
    """Generator must ship the Classifier Eval Suite: exactly three tabs,
    product voice, no design-inspiration references, no retired views."""
    _, mod = _load_eval_dashboard_builder()
    page = mod.build_html(load_fixture(MOCK))
    assert "Classifier Eval Suite" in page
    assert page.count('data-tab="') == 3
    assert 'data-tab="robustness"' in page
    assert 'data-tab="benchmarks"' in page
    assert 'data-tab="confidence"' in page
    # Retired views are gone by decision.
    assert "Confusion" not in page
    assert "chart-summary" not in page
    assert "baseline-table" not in page
    # No design-inspiration or meta references leak into the product.
    assert "langsmith" not in page.lower()
    # Robustness panel renders server-side with badges.
    assert 'id="check-tokenization_pinned"' in page
    assert 'id="check-probability_mass"' in page
    assert 'id="check-batch_parity"' in page
    assert '<span class="badge pass">pass</span>' in page
    # Confidence tab charts.
    assert "chart-reliability" in page
    assert "chart-ece" in page
    assert "chart-selective" in page
    # Synthetic notice is present but flat (no shouting banner class).
    assert "synthetic-notice" in page
    assert "internally consistent placeholders" in page


def test_build_html_no_langsmith_in_builder_source():
    """The builder itself must not reference the retired design inspiration."""
    from evals.paths import PROJECT_ROOT

    src = (
        PROJECT_ROOT
        / "data visualization"
        / "02_Analysis_Code"
        / "build_eval_dashboard.py"
    ).read_text(encoding="utf-8")
    assert "langsmith" not in src.lower()


def test_committed_html_is_the_suite():
    """Checked-in eval_dashboard.html must be the regenerated three-tab suite."""
    from evals.paths import PROJECT_ROOT

    page = (
        PROJECT_ROOT
        / "data visualization"
        / "01_Presentation_Materials"
        / "eval_dashboard.html"
    ).read_text(encoding="utf-8")
    assert "Classifier Eval Suite" in page
    assert page.count('data-tab="') == 3
    assert "langsmith" not in page.lower()
    assert 'id="check-batch_parity"' in page


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
