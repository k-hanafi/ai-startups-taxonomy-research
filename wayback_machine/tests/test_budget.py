"""Tests for pessimistic in-flight budget accounting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wayback_machine.config import estimate_credits
from wayback_machine.extract import _RowOutcome, run_extract


def test_budget_blocks_when_in_flight_would_exceed() -> None:
    """Four in-flight rows at 4 successes should block the next pop at budget=1."""
    budget_credits = 1.0
    state_successful = 4
    in_flight_rows = 4
    assert (
        estimate_credits(state_successful + in_flight_rows, extract_depth="basic")
        >= budget_credits
    )


def test_resumed_run_budget_uses_this_run_delta(tmp_path: Path, monkeypatch) -> None:
    """Prior successful rows should not consume a resumed run's fresh budget."""
    targets = tmp_path / "targets.csv"
    snapshots = tmp_path / "snapshots.jsonl"

    with targets.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["org_uuid", "name", "homepage_url", "closest_ts", "snapshot_url"],
        )
        writer.writeheader()
        writer.writerow({
            "org_uuid": "pending-1",
            "name": "Pending",
            "homepage_url": "https://pending.example",
            "closest_ts": "20230314120000",
            "snapshot_url": (
                "http://web.archive.org/web/20230314120000id_/https://pending.example"
            ),
        })

    with snapshots.open("w", encoding="utf-8") as f:
        for i in range(600):
            f.write(json.dumps({
                "org_uuid": f"done-{i}",
                "ok": True,
                "status": "success",
                "retryable": False,
            }) + "\n")

    monkeypatch.setattr("wayback_machine.extract._api_key", lambda: "dummy")

    def fake_run_row(**kwargs) -> _RowOutcome:
        target = kwargs["target"]
        record = {
            "org_uuid": target["org_uuid"],
            "name": target["name"],
            "homepage_url": target["homepage_url"],
            "snapshot_ts": target["closest_ts"],
            "ok": True,
            "status": "success",
            "retryable": False,
            "response": {"results": []},
        }
        return _RowOutcome(record, "success", True, False, "1", "real evidence", False)

    monkeypatch.setattr("wayback_machine.extract._run_row_with_outage", fake_run_row)

    report = run_extract(
        targets_csv=targets,
        output_jsonl=snapshots,
        state_json=tmp_path / "state.json",
        processed_csv=tmp_path / "processed.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
        manifest_csv=tmp_path / "manifest.csv",
        budget_credits=100.0,
        max_concurrent_rows=1,
        extract_rpm=0,
    )

    assert report.attempted == 1
    assert report.succeeded == 1
    assert report.budget_reached is False
    assert report.est_credits == pytest.approx(0.2)
