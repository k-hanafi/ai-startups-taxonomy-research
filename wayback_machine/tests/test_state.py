"""Tests for JSONL-derived resume state and processed CSV backfill."""

from __future__ import annotations

import json
from pathlib import Path

from wayback_machine.extract import backfill_processed_csv
from wayback_machine.state import (
    ExtractState,
    reconcile_extract_state,
    tally_outcomes_from_jsonl,
)

_SUCCESS_RESPONSE = {
    "results": [{
        "url": "https://acme.ai/",
        "raw_content": (
            "# Acme AI\n"
            "We build an AI platform for automation and workflow integration.\n"
            "Our product uses machine learning models to serve customers."
        ),
    }]
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_tally_outcomes_from_jsonl(tmp_path: Path) -> None:
    jsonl = tmp_path / "snapshots.jsonl"
    _write_jsonl(jsonl, [
        {"org_uuid": "a", "ok": True, "status": "success", "retryable": False},
        {"org_uuid": "b", "ok": True, "status": "empty_results", "retryable": False},
        {"org_uuid": "c", "ok": False, "status": "http_429", "retryable": True},
        {"org_uuid": "d", "ok": False, "status": "http_400", "retryable": False},
    ])
    tally = tally_outcomes_from_jsonl(jsonl)
    assert tally.successful == 1
    assert tally.empty == 1
    assert tally.failed == 2
    assert tally.completed_ids == {"a", "b", "d"}
    assert "c" not in tally.completed_ids


def test_reconcile_extract_state_overwrites_counters(tmp_path: Path) -> None:
    jsonl = tmp_path / "snapshots.jsonl"
    _write_jsonl(jsonl, [
        {"org_uuid": "a", "ok": True, "status": "success", "retryable": False},
        {"org_uuid": "b", "ok": True, "status": "success", "retryable": False},
    ])
    state = ExtractState(successful=0, empty=99, failed=99)
    reconcile_extract_state(state, jsonl)
    assert state.successful == 2
    assert state.empty == 0
    assert state.failed == 0


def test_backfill_processed_csv_from_jsonl(tmp_path: Path) -> None:
    jsonl = tmp_path / "snapshots.jsonl"
    processed = tmp_path / "scrape_processed.csv"
    _write_jsonl(jsonl, [{
        "org_uuid": "uuid-1",
        "name": "Acme",
        "homepage_url": "https://acme.ai/",
        "snapshot_ts": "20230314120000",
        "ok": True,
        "status": "success",
        "retryable": False,
        "response": _SUCCESS_RESPONSE,
    }])

    added = backfill_processed_csv(jsonl, processed)
    assert added == 1
    text = processed.read_text(encoding="utf-8")
    assert "uuid-1" in text
    assert "website_evidence" in text

    assert backfill_processed_csv(jsonl, processed) == 0


def test_backfill_after_simulated_crash_gap(tmp_path: Path) -> None:
    """JSONL has a success row but processed CSV is empty — startup backfill restores it."""
    jsonl = tmp_path / "snapshots.jsonl"
    processed = tmp_path / "scrape_processed.csv"
    _write_jsonl(jsonl, [{
        "org_uuid": "uuid-2",
        "name": "Beta",
        "homepage_url": "https://beta.ai/",
        "snapshot_ts": "20230314120000",
        "ok": True,
        "status": "success",
        "retryable": False,
        "response": _SUCCESS_RESPONSE,
    }])
    assert backfill_processed_csv(jsonl, processed) == 1
    assert "uuid-2" in processed.read_text(encoding="utf-8")
