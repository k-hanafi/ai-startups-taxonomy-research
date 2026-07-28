"""Offline tests for production cost extrapolation (pivot 8)."""

from __future__ import annotations

import csv
import json

import pytest

from evals import config as cfg
from evals import cost_extrapolate as ce
from evals import report as report_mod
from evals import scoring
from two_pass_classifier.input_contract import SOURCE_COLUMNS
from two_pass_classifier.manifest import build_manifest, write_manifest


@pytest.fixture(autouse=True)
def _empty_manifest_directory(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    monkeypatch.setattr(ce, "MANIFESTS_DIR", manifests)


def test_ladder_known_answer_classification_with_cache():
    # 1M input (500k cached) + 1M output on nano: sync in=$0.20, out=$1.25.
    # After cache: uncached 0.5M @ 0.20 = $0.10, cached 0.5M @ 0.10 = $0.05
    # → $0.15 + $1.25 = $1.40. Production runs the sync Responses API, so
    # No Batch discount: scale × fallback N (n_golden=1) gives $1.40 * N.
    records = [{
        "a_input_tokens": 400_000, "a_output_tokens": 100_000,
        "a_cached_tokens": 200_000, "a_reasoning_tokens": 0,
        "b_input_tokens": 600_000, "b_output_tokens": 900_000,
        "b_cached_tokens": 300_000, "b_reasoning_tokens": 50,
    }]
    est = ce.production_cost_from_records(records, "gpt-5.4-nano")
    assert est["available"] is True
    assert est["assumptions"]["architecture"] == "classification"
    assert est["assumptions"]["n_prod"] == cfg.OFFLINE_PRODUCTION_ROW_FALLBACK
    assert est["assumptions"]["n_prod_label"] == "offline_fallback_37746"
    assert est["assumptions"]["production_population_source"] == (
        "offline_fallback"
    )
    assert est["assumptions"]["cache_source"] == "measured_from_run"

    s1 = est["steps"]["1_golden_sync"]
    assert s1["total_input_tokens"] == 1_000_000
    assert s1["total_output_tokens"] == 1_000_000
    assert s1["total_usd"] == pytest.approx(0.20 + 1.25)

    s2 = est["steps"]["2_cache"]
    assert s2["available"] is True
    assert s2["total_cached_tokens"] == 500_000
    assert s2["cache_hit_rate"] == pytest.approx(0.5)
    assert s2["total_usd_after_cache"] == pytest.approx(1.40)

    assert "3_batch" not in est["steps"]
    s3 = est["steps"]["3_scale"]
    assert s3["n_prod"] == cfg.OFFLINE_PRODUCTION_ROW_FALLBACK
    assert s3["estimated_production_usd"] == pytest.approx(
        1.40 * cfg.OFFLINE_PRODUCTION_ROW_FALLBACK
    )


def test_corrupt_manifests_fall_back_like_empty_directory(tmp_path, monkeypatch):
    manifests = tmp_path / "corrupt_manifests"
    manifests.mkdir()
    (manifests / "manifest_bad.jsonl").write_text(
        "{not valid jsonl\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ce, "MANIFESTS_DIR", manifests)

    population = ce.resolve_production_population()

    assert population.row_count == cfg.OFFLINE_PRODUCTION_ROW_FALLBACK
    assert population.source == "offline_fallback"
    assert population.label == "offline_fallback_37746"
    assert population.manifest_path is None


def test_population_comes_from_manifest_when_present(tmp_path, monkeypatch):
    live = tmp_path / "live.csv"
    dead = tmp_path / "dead.csv"
    live_raw = tmp_path / "raw_results.jsonl"
    dead_scrape = tmp_path / "scrape_processed_dead.csv"
    base = {column: "" for column in SOURCE_COLUMNS}
    rows = [
        {
            **base,
            "org_uuid": f"u{index}",
            "name": f"Company {index}",
            "founded_date": "2024-01",
            "website_evidence": "Direct company evidence.",
        }
        for index in range(3)
    ]
    with live.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with dead.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
    with live_raw.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "org_uuid": row["org_uuid"],
                        "ok": True,
                        "requested_at": "2026-05-04T17:12:06.086815+00:00",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with dead_scrape.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "org_uuid",
                "name",
                "homepage_url",
                "snapshot_ts",
                "website_pages_used",
                "website_evidence",
            ],
        )
        writer.writeheader()
    manifest = build_manifest(
        live,
        dead,
        live_raw_results=live_raw,
        dead_scrape_processed=dead_scrape,
    )
    manifests = tmp_path / "production_manifests"
    write_manifest(manifest, manifests)
    monkeypatch.setattr(ce, "MANIFESTS_DIR", manifests)

    estimate = ce.production_cost_from_records(
        [{"input_tokens": 100, "output_tokens": 10, "cached_tokens": 0}],
        "gpt-5.4-nano",
    )

    assert estimate["assumptions"]["n_prod"] == 3
    assert estimate["assumptions"]["production_population_source"] == "manifest"
    assert estimate["assumptions"]["manifest_path"]


def test_legacy_run_without_cached_field_marks_cache_unavailable():
    records = [
        {"input_tokens": 1_000_000, "output_tokens": 0, "reasoning_tokens": 0},
    ]
    est = ce.production_cost_from_records(records, "gpt-5.4-nano")
    assert est["available"] is False
    assert est["reason"] == "cached_tokens_unavailable"
    assert est["steps"]["1_golden_sync"]["total_usd"] == pytest.approx(0.20)
    assert est["steps"]["2_cache"]["available"] is False
    assert "invent" in est["steps"]["2_cache"]["reason"].lower() or \
        "re-run" in est["steps"]["2_cache"]["reason"].lower()
    assert est["steps"]["3_scale"]["available"] is False
    assert est["assumptions"]["architecture"] == "single-pass"


def test_zero_cached_with_field_present_is_real_zero_not_unavailable():
    records = [{"input_tokens": 1000, "output_tokens": 0, "cached_tokens": 0}]
    est = ce.production_cost_from_records(records, "gpt-5.4-nano")
    assert est["available"] is True
    assert est["steps"]["2_cache"]["cache_hit_rate"] == 0.0
    assert est["steps"]["2_cache"]["total_usd_after_cache"] == pytest.approx(
        est["steps"]["1_golden_sync"]["total_usd"]
    )


def test_mixed_cached_field_coverage_is_unavailable():
    # Partial resume: one new row with cached_tokens, one legacy without.
    records = [
        {"input_tokens": 1000, "output_tokens": 0, "cached_tokens": 500},
        {"input_tokens": 1000, "output_tokens": 0},
    ]
    est = ce.production_cost_from_records(records, "gpt-5.4-nano")
    assert est["available"] is False
    assert est["reason"] == "cached_tokens_partial_coverage"
    assert "mixed" in est["steps"]["2_cache"]["reason"].lower()
    # Must not silently treat the legacy row as a 0% cache hit.
    assert est["steps"]["2_cache"].get("cache_hit_rate") is None


def test_token_totals_parity_scoring_vs_cost_extrapolate():
    """Displayed cost and production projection must sum the same tokens."""
    from evals.usage import token_totals

    classification = {
        "a_input_tokens": 100,
        "b_input_tokens": 200,
        "a_output_tokens": 10,
        "b_output_tokens": 40,
        "a_reasoning_tokens": 0,
        "b_reasoning_tokens": 5,
        "a_cached_tokens": 20,
        "b_cached_tokens": 30,
    }
    single = {
        "input_tokens": 300,
        "output_tokens": 50,
        "reasoning_tokens": 5,
        "cached_tokens": 50,
    }
    assert token_totals(classification) == token_totals(single)
    assert scoring._record_tokens(classification) == token_totals(classification)

    est = ce.production_cost_from_records([classification], "gpt-5.4-nano")
    cost = scoring.cost_and_tokens([classification], "gpt-5.4-nano")
    assert est["steps"]["1_golden_sync"]["total_input_tokens"] == cost["total_input_tokens"]
    assert est["steps"]["1_golden_sync"]["total_output_tokens"] == cost["total_output_tokens"]
    assert est["steps"]["1_golden_sync"]["total_usd"] == pytest.approx(cost["total_usd"])
    assert est["steps"]["2_cache"]["total_cached_tokens"] == cost["total_cached_tokens"]


def test_format_cost_ladder_mentions_assumptions():
    records = [{"input_tokens": 100, "output_tokens": 10, "cached_tokens": 20}]
    text = ce.format_cost_ladder(ce.production_cost_from_records(records, "gpt-5.4-nano"))
    assert "PRODUCTION COST EXTRAPOLATION" in text
    assert "Cache source" in text
    fallback = cfg.OFFLINE_PRODUCTION_ROW_FALLBACK
    assert str(fallback) in text or f"{fallback:,}" in text


def test_score_run_includes_production_cost_estimate(mini_run_factory):
    run_id, _ = mini_run_factory(with_cache=True)
    report = scoring.score_run(run_id, write=False)
    est = report["production_cost_estimate"]
    assert est["available"] is True
    assert report["cost"]["cache_field_present"] is True
    assert report["cost"]["total_cached_tokens"] == 30


def test_score_run_legacy_omits_fake_cache(mini_run_factory):
    run_id, _ = mini_run_factory(with_cache=False)
    report = scoring.score_run(run_id, write=False)
    assert report["production_cost_estimate"]["available"] is False
    assert report["cost"]["cache_field_present"] is False
    assert report["cost"]["total_cached_tokens"] is None


def test_report_writes_html(tmp_path, monkeypatch, mini_run_factory):
    run_id, run_dir = mini_run_factory(with_cache=True)
    scoring.score_run(run_id, write=True)
    monkeypatch.setattr(report_mod, "run_scored_path",
                        lambda rid: tmp_path / "runs" / rid / "scored.json")
    monkeypatch.setattr(report_mod, "cost_report_path",
                        lambda rid: tmp_path / "runs" / rid / "cost_report.html")
    monkeypatch.setattr(report_mod, "run_dir",
                        lambda rid: tmp_path / "runs" / rid)
    out = report_mod.write_cost_report(run_id)
    html = out.read_text(encoding="utf-8")
    assert "Production cost extrapolation" in html
    assert "After cache" in html
    assert str(cfg.OFFLINE_PRODUCTION_ROW_FALLBACK) in html


GOLD_HEADER = "org_uuid,draft_ai_native,draft_subclass,draft_rad\n"


@pytest.fixture
def mini_run_factory(tmp_path, monkeypatch):
    def _make(*, with_cache: bool):
        gold_csv = tmp_path / "golden_set.csv"
        gold_csv.write_text(
            GOLD_HEADER
            + "u1,1,1E,RAD-M\n"
            + "u2,0,0A,RAD-NA\n"
            + "u3,0,0B,RAD-NA\n",
            encoding="utf-8",
        )
        run_id = "cost-run" if with_cache else "legacy-run"
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        predictions = []
        for u, ai, sub, rad in (
            ("u1", 1, "1E", "RAD-M"),
            ("u2", 0, "0A", "RAD-NA"),
            ("u3", 1, "1A", "RAD-H"),
        ):
            rec = {
                "custom_id": f"startup-{u}", "org_uuid": u, "model": "gpt-5.4-nano",
                "status": "completed", "ai_native": ai, "subclass": sub,
                "rad_score": rad, "input_tokens": 100, "output_tokens": 10,
                "reasoning_tokens": 0,
            }
            if with_cache:
                rec["cached_tokens"] = 10
            predictions.append(rec)
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

    return _make
