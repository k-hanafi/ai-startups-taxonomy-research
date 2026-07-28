"""Offline tests for shared eval run mechanics."""

from __future__ import annotations

import json

import pytest

from evals import runner
from evals.jsonl_io import MalformedJSONLError


def test_completed_custom_ids_reads_legacy_and_completed_rows(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"custom_id": "startup-a"}) + "\n"
        + json.dumps(
            {"custom_id": "startup-b", "status": "completed"}
        )
        + "\n"
        + json.dumps({"custom_id": "startup-c", "status": "failed"})
        + "\n",
        encoding="utf-8",
    )

    assert runner._completed_custom_ids(path) == {
        "startup-a",
        "startup-b",
    }


def test_completed_custom_ids_missing_file(tmp_path):
    assert runner._completed_custom_ids(tmp_path / "missing.jsonl") == set()


def test_completed_custom_ids_tolerates_truncated_final_line(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"custom_id": "startup-a"}) + "\n"
        + '{"custom_id": "startup-b", "status": "comple',
        encoding="utf-8",
    )

    assert runner._completed_custom_ids(path) == {"startup-a"}


def test_completed_custom_ids_rejects_interior_corruption(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"custom_id": "startup-a"}) + "\n"
        + "{broken\n"
        + json.dumps({"custom_id": "startup-c"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MalformedJSONLError, match="line 2"):
        runner._completed_custom_ids(path)


def test_load_golden_rows_preserves_committed_order(tmp_path, monkeypatch):
    golden = tmp_path / "golden.csv"
    classifier_input = tmp_path / "classifier_input.csv"
    golden.write_text("org_uuid\nu2\nu1\n", encoding="utf-8")
    classifier_input.write_text(
        "org_uuid,name,website_evidence\n"
        "u1,First,evidence one\n"
        "u2,Second,evidence two\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "GOLDEN_SET_CSV", golden)
    monkeypatch.setattr(runner, "CLASSIFIER_INPUT_CSV", classifier_input)

    rows = runner.load_golden_rows()

    assert [row["org_uuid"] for row in rows] == ["u2", "u1"]


def test_runner_contains_no_classifier_builder():
    assert not hasattr(runner, "build_request_kwargs")
    assert not hasattr(runner, "run")
