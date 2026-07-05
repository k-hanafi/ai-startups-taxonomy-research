"""Sampler tests on synthetic frames (no production data, no API key)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evals.sampling import GOLDEN_SET_COLUMNS, sample_golden_set

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


def test_undersized_stratum_raises():
    predictions, classifier_input = _synthetic_pool(rows_per_subclass=2)
    quotas = {s: 5 for s in SUBCLASSES}
    with pytest.raises(ValueError, match="quota"):
        sample_golden_set(predictions, classifier_input, quotas=quotas, seed=1)


def test_terciles_spread_within_stratum():
    predictions, classifier_input = _synthetic_pool(rows_per_subclass=60)
    quotas = {"1E": 9}
    golden = sample_golden_set(predictions, classifier_input, quotas=quotas, seed=3)
    spread = golden["evidence_tercile"].value_counts()
    assert len(golden) == 9
    assert set(spread.index) == {"short", "medium", "long"}
    assert spread.min() >= 2
