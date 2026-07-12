"""Offline tests for the Stage 7 scorer: metric math on known-answer synthetic
fixtures, bootstrap determinism, cost computation, the calibration-absent
path, and an end-to-end score_run over a tmp_path mini golden set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import scoring


# --- accuracy / macro-F1 / confusion (known answers) ------------------------

GOLD = ["a", "a", "b", "c"]
PRED = ["a", "b", "b", "b"]


def test_accuracy_known_answer():
    assert scoring.accuracy(GOLD, PRED) == 0.5
    assert scoring.accuracy(["x"], ["x"]) == 1.0


def test_accuracy_empty_rejected():
    with pytest.raises(ValueError):
        scoring.accuracy([], [])


def test_macro_f1_known_answer():
    # a: P=1, R=1/2 -> F1=2/3. b: P=1/3, R=1 -> F1=1/2. c: never predicted -> 0.
    expected = (2 / 3 + 1 / 2 + 0) / 3
    assert scoring.macro_f1(GOLD, PRED) == pytest.approx(expected)


def test_macro_f1_perfect_and_worst():
    assert scoring.macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert scoring.macro_f1(["a", "a"], ["b", "b"]) == 0.0


def test_confusion_matrix_counts():
    cm = scoring.confusion_matrix(GOLD, PRED)
    assert cm["labels"] == ["a", "b", "c"]
    assert cm["matrix"]["a"] == {"a": 1, "b": 1}
    assert cm["matrix"]["b"] == {"b": 1}
    assert cm["matrix"]["c"] == {"b": 1}


# --- bootstrap ---------------------------------------------------------------

def test_bootstrap_ci_deterministic_with_fixed_seed():
    correct = [True] * 70 + [False] * 30
    a = scoring.bootstrap_accuracy_ci(correct, resamples=500, seed=7)
    b = scoring.bootstrap_accuracy_ci(correct, resamples=500, seed=7)
    assert a == b
    # The point estimate must sit inside its own CI.
    assert a[0] <= 0.7 <= a[1]


def test_bootstrap_ci_degenerate_sample():
    assert scoring.bootstrap_accuracy_ci([True] * 10, resamples=100) == [1.0, 1.0]


def test_paired_bootstrap_detects_a_real_gap():
    run = [True] * 90 + [False] * 10
    base = [True] * 50 + [False] * 50
    delta = scoring.paired_bootstrap_delta(run, base, resamples=2000, seed=3)
    assert delta["delta_accuracy"] == pytest.approx(0.4)
    assert delta["significant"] is True
    assert delta["ci95"][0] > 0


def test_paired_bootstrap_identical_runs_not_significant():
    same = [True, False] * 50
    delta = scoring.paired_bootstrap_delta(same, same, resamples=500, seed=3)
    assert delta["delta_accuracy"] == 0.0
    assert delta["ci95"] == [0.0, 0.0]
    assert delta["significant"] is False


def test_paired_bootstrap_requires_aligned_samples():
    with pytest.raises(ValueError):
        scoring.paired_bootstrap_delta([True], [True, False])


# --- cost + token usage --------------------------------------------------------

def test_cost_from_actual_usage_single_pass():
    records = [
        {"input_tokens": 1_000_000, "output_tokens": 0, "reasoning_tokens": 0},
        {"input_tokens": 0, "output_tokens": 1_000_000, "reasoning_tokens": 400},
    ]
    cost = scoring.cost_and_tokens(records, "gpt-5.4-nano")
    # 1M input @ $0.20 + 1M output @ $1.25.
    assert cost["total_usd"] == pytest.approx(0.20 + 1.25)
    assert cost["mean_usd_per_row"] == pytest.approx((0.20 + 1.25) / 2)
    assert cost["output_tokens"]["max"] == 1_000_000
    assert cost["reasoning_tokens"]["max"] == 400


def test_cost_sums_two_pass_usage_fields():
    records = [{
        "a_input_tokens": 100, "a_output_tokens": 10, "a_reasoning_tokens": 0,
        "b_input_tokens": 200, "b_output_tokens": 40, "b_reasoning_tokens": 30,
    }]
    cost = scoring.cost_and_tokens(records, "gpt-5.4-nano")
    assert cost["total_input_tokens"] == 300
    assert cost["total_output_tokens"] == 50
    assert cost["reasoning_tokens"]["max"] == 30


def test_cost_unknown_model_reports_tokens_without_dollars():
    cost = scoring.cost_and_tokens(
        [{"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}],
        "some-future-model",
    )
    assert cost["total_usd"] is None
    assert cost["mean_usd_per_row"] is None
    assert cost["total_input_tokens"] == 10


# --- latency -------------------------------------------------------------------

def test_latency_summary_known_answer():
    records = [{"latency_s": 1.0}, {"latency_s": 2.0}, {"latency_s": 3.0},
               {"latency_s": 10.0}]
    summary = scoring.latency_summary(records)
    stats = summary["latency_s"]
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["p50"] == pytest.approx(2.5)
    assert stats["max"] == 10.0
    # Single-pass records carry no per-pass fields.
    assert "a_latency_s" not in summary and "b_latency_s" not in summary


def test_latency_summary_two_pass_per_pass_breakdown():
    records = [
        {"latency_s": 10.0, "a_latency_s": 1.0, "b_latency_s": 9.0},
        {"latency_s": 20.0, "a_latency_s": 2.0, "b_latency_s": 18.0},
    ]
    summary = scoring.latency_summary(records)
    assert summary["latency_s"]["mean"] == pytest.approx(15.0)
    assert summary["a_latency_s"]["mean"] == pytest.approx(1.5)
    assert summary["b_latency_s"]["mean"] == pytest.approx(13.5)


def test_latency_summary_none_for_legacy_records():
    # Banked pre-latency runs must keep scoring: summary is None, not a crash.
    legacy = [{"input_tokens": 100, "output_tokens": 10}]
    assert scoring.latency_summary(legacy) is None
    assert scoring.latency_summary([{"latency_s": None}]) is None


def test_latency_summary_skips_null_rows():
    # A resumed run can mix legacy rows (no field) with new ones.
    records = [{"latency_s": 4.0}, {}, {"latency_s": None}]
    stats = scoring.latency_summary(records)["latency_s"]
    assert stats["n"] == 1 and stats["mean"] == 4.0


# --- calibration ---------------------------------------------------------------

def test_resolve_confidence_absent_everywhere_is_empty():
    records = [{"org_uuid": "u1", "custom_id": "startup-u1"}]
    assert scoring.resolve_confidence(records) == {}
    assert scoring.resolve_confidence(records, external={}) == {}


def test_resolve_confidence_record_field_and_external_mapping():
    records = [
        {"org_uuid": "u1", "custom_id": "startup-u1", "binary_confidence": 0.9},
        {"org_uuid": "u2", "custom_id": "startup-u2"},
        {"org_uuid": "u3", "custom_id": "startup-u3"},
    ]
    # External values win; custom_id keys are accepted too.
    conf = scoring.resolve_confidence(
        records, external={"u1": 0.5, "startup-u2": 0.7}
    )
    assert conf == {"u1": 0.5, "u2": 0.7}
    # Without the external mapping, the record field is used.
    assert scoring.resolve_confidence(records) == {"u1": 0.9}


def test_calibration_report_none_when_no_confidence():
    assert scoring.calibration_report({}, {}, {}) is None


def test_reliability_bins_known_answer():
    # All rows in the top bin at 0.95 confidence, 9/10 correct: ECE = 0.05.
    conf = [0.95] * 10
    correct = [True] * 9 + [False]
    rel = scoring.reliability_bins(conf, correct, n_bins=10)
    top = rel["bins"][-1]
    assert top["count"] == 10
    assert top["accuracy"] == pytest.approx(0.9)
    assert rel["ece"] == pytest.approx(0.05)


def test_reliability_top_bin_includes_confidence_one():
    rel = scoring.reliability_bins([1.0], [True], n_bins=10)
    assert rel["bins"][-1]["count"] == 1


def test_selective_prediction_curve_slopes_up_toward_confident_head():
    # High-confidence rows correct, low-confidence rows wrong.
    conf = [0.9] * 5 + [0.1] * 5
    correct = [True] * 5 + [False] * 5
    curve = scoring.selective_prediction_curve(conf, correct, [0.5, 1.0])
    assert curve[0] == {"coverage": 0.5, "n": 5, "accuracy": 1.0}
    assert curve[1] == {"coverage": 1.0, "n": 10, "accuracy": 0.5}


# --- end-to-end score_run over a tmp mini golden set ---------------------------

GOLD_HEADER = "org_uuid,draft_ai_native,draft_subclass,draft_rad\n"


@pytest.fixture
def mini_run(tmp_path, monkeypatch):
    """3-row gold CSV + a run where u1/u2 are right and u3 is wrong."""
    gold_csv = tmp_path / "golden_set.csv"
    gold_csv.write_text(
        GOLD_HEADER
        + "u1,1,1E,RAD-M\n"
        + "u2,0,0A,RAD-NA\n"
        + "u3,0,0B,RAD-NA\n",
        encoding="utf-8",
    )
    run_id = "test-run"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    predictions = [
        {"custom_id": "startup-u1", "org_uuid": "u1", "model": "gpt-5.4-nano",
         "status": "completed", "ai_native": 1, "subclass": "1E",
         "rad_score": "RAD-M", "input_tokens": 100, "output_tokens": 10,
         "reasoning_tokens": 0},
        {"custom_id": "startup-u2", "org_uuid": "u2", "model": "gpt-5.4-nano",
         "status": "completed", "ai_native": 0, "subclass": "0A",
         "rad_score": "RAD-NA", "input_tokens": 100, "output_tokens": 10,
         "reasoning_tokens": 0},
        {"custom_id": "startup-u3", "org_uuid": "u3", "model": "gpt-5.4-nano",
         "status": "completed", "ai_native": 1, "subclass": "1A",
         "rad_score": "RAD-H", "input_tokens": 100, "output_tokens": 10,
         "reasoning_tokens": 0},
        # A failed row must not be scored.
        {"custom_id": "startup-u4", "org_uuid": "u4", "status": "failed"},
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8"
    )
    monkeypatch.setattr(scoring, "GOLDEN_SET_CSV", gold_csv)
    monkeypatch.setattr(scoring, "run_predictions_path",
                        lambda rid: tmp_path / "runs" / rid / "predictions.jsonl")
    monkeypatch.setattr(scoring, "run_config_path",
                        lambda rid: tmp_path / "runs" / rid / "config.json")
    monkeypatch.setattr(scoring, "run_scored_path",
                        lambda rid: tmp_path / "runs" / rid / "scored.json")
    return run_id, run_dir


def test_score_run_end_to_end(mini_run):
    run_id, run_dir = mini_run
    report = scoring.score_run(run_id)

    assert report["n_scored"] == 3
    assert report["axes"]["ai_native"]["accuracy"] == pytest.approx(2 / 3)
    assert report["axes"]["subclass"]["accuracy"] == pytest.approx(2 / 3)
    assert report["axes"]["rad"]["accuracy"] == pytest.approx(2 / 3)
    # No confidence input anywhere: calibration must be skipped, not crash.
    assert report["calibration"] is None
    assert report["cost"]["n_rows"] == 3
    assert report["cost"]["total_input_tokens"] == 300
    # Legacy records carry no latency fields: summary degrades to None.
    assert report["latency"] is None
    # Without config.json, model still comes from prediction records.
    assert report["model"] == "gpt-5.4-nano"
    assert report["kind"] == "single_pass"

    scored = json.loads((run_dir / "scored.json").read_text(encoding="utf-8"))
    assert scored["run_id"] == run_id
    assert scored["model"] == "gpt-5.4-nano"
    assert scored["kind"] == "single_pass"


def test_score_run_writes_two_pass_metadata_from_config(mini_run):
    run_id, run_dir = mini_run
    (run_dir / "config.json").write_text(
        json.dumps({
            "kind": "two_pass",
            "model": "gpt-5.6-luna",
            "effort_b": "medium",
            "n_rows": 3,
        }),
        encoding="utf-8",
    )
    report = scoring.score_run(run_id, write=False)
    assert report["model"] == "gpt-5.6-luna"
    assert report["effort_b"] == "medium"
    assert report["kind"] == "two_pass"
    assert "effort" not in report


def test_score_run_refuses_partial_by_default(mini_run):
    run_id, run_dir = mini_run
    # Keep only one completed prediction against a 3-row gold set.
    records = [
        json.loads(l)
        for l in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kept = next(r for r in records if r.get("org_uuid") == "u1")
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(kept) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="scored 1/3"):
        scoring.score_run(run_id, write=False)


def test_score_run_allow_partial_scores_incomplete(mini_run):
    run_id, run_dir = mini_run
    records = [
        json.loads(l)
        for l in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kept = next(r for r in records if r.get("org_uuid") == "u1")
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(kept) + "\n", encoding="utf-8"
    )
    report = scoring.score_run(run_id, write=False, allow_partial=True)
    assert report["n_scored"] == 1
    assert report["n_expected"] == 3


def test_score_run_honors_config_n_rows_as_expected(mini_run):
    """A --limit smoke writes n_rows=2; scoring 2/2 must pass without --allow-partial."""
    run_id, run_dir = mini_run
    (run_dir / "config.json").write_text(
        json.dumps({"kind": "two_pass", "model": "gpt-5.4-nano",
                    "effort_b": "low", "n_rows": 2}),
        encoding="utf-8",
    )
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    # Keep two completed gold rows (u1, u2).
    kept = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("status") == "completed" and rec.get("org_uuid") in ("u1", "u2"):
            kept.append(line)
    (run_dir / "predictions.jsonl").write_text(
        "".join(l + "\n" for l in kept), encoding="utf-8"
    )
    report = scoring.score_run(run_id, write=False)
    assert report["n_scored"] == 2
    assert report["n_expected"] == 2
    assert report["effort_b"] == "low"
    assert report["kind"] == "two_pass"


def test_score_run_surfaces_latency_when_recorded(mini_run):
    run_id, run_dir = mini_run
    path = run_dir / "predictions.jsonl"
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    for i, rec in enumerate(records):
        if rec["status"] == "completed":
            rec["latency_s"] = float(i + 1)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    report = scoring.score_run(run_id, write=False)
    assert report["latency"]["latency_s"]["n"] == 3
    assert report["latency"]["latency_s"]["mean"] == pytest.approx(2.0)


def test_score_run_with_confidence_computes_calibration(mini_run):
    run_id, _ = mini_run
    report = scoring.score_run(
        run_id,
        confidence={"u1": 0.95, "u2": 0.9, "u3": 0.2},
        write=False,
    )
    cal = report["calibration"]
    assert cal is not None and cal["n"] == 3
    # u3 (wrong) has the lowest confidence, so the confident half is perfect.
    half = next(p for p in cal["selective_prediction"] if p["coverage"] == 0.5)
    assert half["accuracy"] == 1.0


def test_score_run_paired_baseline_delta(mini_run, tmp_path):
    run_id, _ = mini_run
    base_id = "base-run"
    base_dir = tmp_path / "runs" / base_id
    base_dir.mkdir(parents=True)
    # Baseline gets every row wrong on every axis.
    rows = [
        {"custom_id": f"startup-{u}", "org_uuid": u, "status": "completed",
         "ai_native": 9, "subclass": "XX", "rad_score": "XX"}
        for u in ("u1", "u2", "u3")
    ]
    (base_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    report = scoring.score_run(run_id, baseline_run_id=base_id, write=False)
    vs = report["vs_baseline"]
    assert vs["n_paired"] == 3
    assert vs["deltas"]["ai_native"]["delta_accuracy"] == pytest.approx(2 / 3)


def test_calibration_refuses_partial_confidence_by_default(mini_run):
    run_id, _ = mini_run
    with pytest.raises(SystemExit, match="Confidence covers only 2 of 3"):
        scoring.score_run(
            run_id, confidence={"u1": 0.95, "u3": 0.2}, write=False
        )


def test_calibration_allow_partial_confidence(mini_run, caplog):
    run_id, _ = mini_run
    import logging
    with caplog.at_level(logging.WARNING, logger="evals.scoring"):
        report = scoring.score_run(
            run_id,
            confidence={"u1": 0.95, "u3": 0.2},
            write=False,
            allow_partial_confidence=True,
        )
    cal = report["calibration"]
    assert cal["n"] == 2
    assert cal["n_eligible"] == 3
    assert any("allow-partial-confidence" in r.message for r in caplog.records)


def test_pass_b_isolating_metrics_in_scored_report(mini_run):
    run_id, run_dir = mini_run
    # Mark boundary_disagreement on completed rows.
    path = run_dir / "predictions.jsonl"
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    for rec in records:
        if rec.get("status") == "completed":
            rec["boundary_disagreement"] = rec.get("org_uuid") == "u2"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    report = scoring.score_run(run_id, write=False)
    pbm = report["pass_b_metrics"]
    assert "subclass_family_conditional" in pbm
    assert "rad_ai_native_only" in pbm
    assert pbm["boundary_disagreement"]["n"] == 3
    assert pbm["boundary_disagreement"]["rate"] == pytest.approx(1 / 3)
    assert "definitions" in pbm


def test_calibration_full_coverage_matches_eligible(mini_run):
    run_id, _ = mini_run
    report = scoring.score_run(
        run_id, confidence={"u1": 0.95, "u2": 0.9, "u3": 0.2}, write=False
    )
    assert report["calibration"]["n"] == report["calibration"]["n_eligible"] == 3


def test_score_run_calibration_from_raw_logprob_fixtures(tmp_path, monkeypatch):
    """Full wire-up on real (anonymized) tokenization: fixtures as raw/,
    run_confidence as the external mapping, calibration populated, and the
    minority-sampling row landing below 0.5 confidence (pivot 6)."""
    from evals.logprob_extract import run_confidence

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    run_id = "raw-run"
    run_dir = tmp_path / "runs" / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)

    gold_lines, predictions, minority_cid = [], [], None
    for i, src in enumerate(sorted(fixtures_dir.glob("*.json"))):
        fixture = json.loads(src.read_text(encoding="utf-8"))
        uuid = f"u{i}"
        cid = f"startup-{uuid}"
        (raw_dir / f"{cid}.json").write_text(src.read_text(encoding="utf-8"))
        sampled = fixture["expected"]["ai_native"]
        if src.stem == "chose_minority":
            minority_cid = cid
        gold_lines.append(f"{uuid},{sampled},1A,RAD-H\n")
        predictions.append(
            {"custom_id": cid, "org_uuid": uuid, "status": "completed",
             "ai_native": sampled, "subclass": "1A", "rad_score": "RAD-H",
             "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}
        )
    assert minority_cid is not None

    gold_csv = tmp_path / "golden_set.csv"
    gold_csv.write_text(GOLD_HEADER + "".join(gold_lines), encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8"
    )
    monkeypatch.setattr(scoring, "GOLDEN_SET_CSV", gold_csv)
    monkeypatch.setattr(scoring, "run_predictions_path",
                        lambda rid: tmp_path / "runs" / rid / "predictions.jsonl")
    monkeypatch.setattr(scoring, "run_config_path",
                        lambda rid: tmp_path / "runs" / rid / "config.json")

    confidence = run_confidence(raw_dir)
    assert confidence[minority_cid] < 0.5

    report = scoring.score_run(run_id, confidence=confidence, write=False)
    cal = report["calibration"]
    assert cal is not None and cal["n"] == len(predictions)
    assert sum(b["count"] for b in cal["reliability"]["bins"]) == len(predictions)
    assert 0.0 <= cal["reliability"]["ece"] <= 1.0
    assert cal["selective_prediction"][-1]["coverage"] == 1.0


def test_gold_with_empty_labels_refused(tmp_path, monkeypatch):
    gold_csv = tmp_path / "golden_set.csv"
    gold_csv.write_text(GOLD_HEADER + "u1,1,,RAD-M\n", encoding="utf-8")
    monkeypatch.setattr(scoring, "GOLDEN_SET_CSV", gold_csv)
    with pytest.raises(SystemExit, match="empty draft"):
        scoring.load_gold()


def test_missing_prediction_axis_counts_as_wrong(mini_run, monkeypatch):
    run_id, run_dir = mini_run
    # u1's record loses its subclass (e.g. Pass B never produced one).
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines]
    records[0]["subclass"] = None
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    report = scoring.score_run(run_id, write=False)
    assert report["axes"]["subclass"]["accuracy"] == pytest.approx(1 / 3)
    # The miss is visible in the confusion matrix, not silently dropped.
    assert scoring.MISSING in report["axes"]["subclass"]["confusion"]["matrix"]["1E"]
