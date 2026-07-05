"""Sampler tests on synthetic frames (no production data, no API key)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evals.sampling as sampling
from evals.sampling import GOLDEN_SET_COLUMNS, build_golden_set, sample_golden_set

SUBCLASSES = ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "0A", "0B", "0C"]


def _synthetic_pool(rows_per_subclass: int = 40, seed: int = 7):
    """Build (predictions, classifier_input) frames mimicking production shape."""
    rng = np.random.default_rng(seed)
    n = rows_per_subclass * len(SUBCLASSES)
    ids = [f"uuid-{i:05d}" for i in range(n)]

    predictions = pd.DataFrame(
        {
            "CompanyID": ids,
            "CompanyName": [f"Co {i}" for i in range(n)],
            "ai_native": [1 if s.startswith("1") else 0 for s in SUBCLASSES] * rows_per_subclass,
            "subclass": SUBCLASSES * rows_per_subclass,
            "rad_score": ["RAD-M" if s.startswith("1") else "RAD-NA" for s in SUBCLASSES]
            * rows_per_subclass,
        }
    )
    classifier_input = pd.DataFrame(
        {
            "org_uuid": ids,
            "name": [f"Co {i}" for i in range(n)],
            "website_evidence": ["x" * int(rng.integers(100, 30_000)) for _ in range(n)],
        }
    )
    return predictions, classifier_input


def test_quotas_met_exactly():
    predictions, classifier_input = _synthetic_pool()
    quotas = {s: 5 for s in SUBCLASSES}
    golden = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=1)
    assert len(golden) == 50
    counts = golden["predicted_subclass"].value_counts()
    assert all(counts[s] == 5 for s in SUBCLASSES)


def test_deterministic_across_calls():
    predictions, classifier_input = _synthetic_pool()
    quotas = {s: 4 for s in SUBCLASSES}
    a = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=42)
    b = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_seed_changes_draw():
    predictions, classifier_input = _synthetic_pool()
    quotas = {s: 4 for s in SUBCLASSES}
    a = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=1)
    b = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=2)
    assert set(a["org_uuid"]) != set(b["org_uuid"])


def test_input_row_order_does_not_change_draw():
    predictions, classifier_input = _synthetic_pool()
    quotas = {s: 4 for s in SUBCLASSES}
    a = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=9)
    shuffled = predictions.sample(frac=1, random_state=0).reset_index(drop=True)
    b = sample_golden_set(shuffled, classifier_input, quotas=quotas, seed=9)
    pd.testing.assert_frame_equal(a, b)


def test_no_evidence_text_in_output():
    predictions, classifier_input = _synthetic_pool()
    quotas = {s: 3 for s in SUBCLASSES}
    golden = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=1)
    assert list(golden.columns) == GOLDEN_SET_COLUMNS
    assert "website_evidence" not in golden.columns
    assert "short_description" not in golden.columns


def test_empty_evidence_rows_excluded():
    predictions, classifier_input = _synthetic_pool()
    classifier_input.loc[:99, "website_evidence"] = ""
    quotas = {s: 3 for s in SUBCLASSES}
    golden = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=1)
    excluded = set(classifier_input.loc[:99, "org_uuid"])
    assert not (set(golden["org_uuid"]) & excluded)


def test_sentinel_evidence_rows_excluded():
    predictions = pd.DataFrame(
        {
            "CompanyID": ["valid", "spaces", "none", "nan"],
            "CompanyName": ["Valid Co", "Spaces Co", "None Co", "Nan Co"],
            "ai_native": [1, 1, 1, 1],
            "subclass": ["1A", "1A", "1A", "1A"],
            "rad_score": ["RAD-M", "RAD-M", "RAD-M", "RAD-M"],
        }
    )
    classifier_input = pd.DataFrame(
        {
            "org_uuid": ["valid", "spaces", "none", "nan"],
            "name": ["Valid Co", "Spaces Co", "None Co", "Nan Co"],
            "website_evidence": ["  actual evidence  ", "   ", "none", "nan"],
        }
    )

    golden = sample_golden_set(predictions, classifier_input, quotas={"1A": 1}, seed=1)

    assert golden["org_uuid"].tolist() == ["valid"]
    assert golden["evidence_chars"].tolist() == [15]


def test_undersized_stratum_raises():
    predictions, classifier_input = _synthetic_pool(rows_per_subclass=2)
    quotas = {s: 5 for s in SUBCLASSES}
    with pytest.raises(ValueError, match="quota"):
        sample_golden_set(predictions, classifier_input, quotas=quotas, seed=1)


def test_duplicate_predictions_do_not_satisfy_quota():
    predictions = pd.DataFrame(
        {
            "CompanyID": ["dup", "dup", "other"],
            "CompanyName": ["Dup Co", "Dup Co", "Other Co"],
            "ai_native": [1, 1, 1],
            "subclass": ["1A", "1A", "1A"],
            "rad_score": ["RAD-M", "RAD-M", "RAD-M"],
        }
    )
    classifier_input = pd.DataFrame(
        {
            "org_uuid": ["dup", "other"],
            "name": ["Dup Co", "Other Co"],
            "website_evidence": ["x" * 10, "x" * 100],
        }
    )
    with pytest.raises(ValueError, match="2 unique evidence-bearing companies"):
        sample_golden_set(predictions, classifier_input, quotas={"1A": 3}, seed=1)


def test_duplicate_predictions_are_collapsed():
    predictions, classifier_input = _synthetic_pool(rows_per_subclass=10)
    duplicate = predictions[predictions["subclass"] == "1A"].iloc[[0]]
    predictions = pd.concat([predictions, duplicate, duplicate], ignore_index=True)

    golden = sample_golden_set(predictions, classifier_input, quotas={"1A": 5}, seed=1)

    assert len(golden) == 5
    assert golden["org_uuid"].is_unique


def test_conflicting_duplicate_predictions_raise():
    predictions, classifier_input = _synthetic_pool(rows_per_subclass=10)
    duplicate = predictions[predictions["subclass"] == "1A"].iloc[[0]].copy()
    duplicate["subclass"] = "1B"
    predictions = pd.concat([predictions, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="disagree on sampling fields"):
        sample_golden_set(predictions, classifier_input, quotas={"1A": 5}, seed=1)


def test_terciles_spread_within_stratum():
    predictions, classifier_input = _synthetic_pool(rows_per_subclass=60)
    quotas = {"1E": 9}
    golden = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=3)
    spread = golden["evidence_tercile"].value_counts()
    assert len(golden) == 9
    assert set(spread.index) == {"short", "medium", "long"}
    assert spread.min() >= 2


def test_build_golden_set_refuses_to_overwrite_human_labels(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden_set.csv"
    pd.DataFrame(
        [
            {
                "org_uuid": "already-labeled",
                "gold_verdict": "approved",
            }
        ]
    ).to_csv(golden_path, index=False)
    monkeypatch.setattr(sampling, "GOLDEN_SET_CSV", golden_path)

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        build_golden_set()
