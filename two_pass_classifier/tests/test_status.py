"""Unit tests for offline run status metrics."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from two_pass_classifier import status as status_module
from two_pass_classifier import workflow
from two_pass_classifier.journal import JOURNAL_VERSION
from two_pass_classifier.manifest import write_manifest
from two_pass_classifier.request_builder import (
    RequestSettings,
    request_fingerprint,
    request_identity,
)
from two_pass_classifier.workflow import build_run_metadata, load_run_context

from .test_cli import _invoke, _manifest_artifact


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    runs = tmp_path / "runs"
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(workflow, "RUNS_DIR", runs)
    monkeypatch.setattr(workflow, "MANIFESTS_DIR", manifests)
    return {"runs": runs, "manifests": manifests}


def test_active_elapsed_skips_idle_gaps_between_events() -> None:
    start = datetime(2026, 7, 28, 20, 0, 0, tzinfo=UTC)
    context = SimpleNamespace(
        header={"created_at": start.isoformat()},
        events=(
            {"finished_at": (start + timedelta(seconds=30)).isoformat()},
            {"finished_at": (start + timedelta(seconds=60)).isoformat()},
            # Multi-minute downtime must not count as productive work.
            {"finished_at": (start + timedelta(seconds=1860)).isoformat()},
            {"finished_at": (start + timedelta(seconds=1890)).isoformat()},
        ),
    )

    active = status_module._active_elapsed_seconds(context)

    assert active == 90.0


def test_status_throughput_and_eta_ignore_resume_downtime(
    tmp_path: Path,
    registry: dict[str, Path],
) -> None:
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=4,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "eta-active-time"
    run_path = registry["runs"] / run_id
    run_manifest = write_manifest(manifest, run_path / "inputs")
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=run_manifest,
        manifest=manifest,
        settings=settings,
        parent_manifest_path=artifact,
        parent_manifest=manifest,
    )
    created_at = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
    rows = list(manifest.rows)
    events: list[dict[str, Any]] = [
        {
            "event_type": "run_started",
            "event_id": "header-event",
            "journal_version": JOURNAL_VERSION,
            "created_at": created_at.isoformat(),
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_rows_sha256": manifest.rows_sha256,
            "manifest_row_count": manifest.row_count,
            "request_fingerprint": request_fingerprint(settings),
            "request_identity": request_identity(settings),
            "run_config": metadata,
        }
    ]
    # Two companies finish in the first minute, then a long kill/resume gap,
    # then nothing else. Wall clock is ~1h; active processing is ~60s.
    finish_times = (
        created_at + timedelta(seconds=30),
        created_at + timedelta(seconds=60),
    )
    for row, finished_at in zip(rows[:2], finish_times, strict=True):
        events.append(
            {
                "event_type": "pass_a_completed",
                "event_id": f"a-{row.company_id}",
                "company_id": row.company_id,
                "company_name": row.company_name,
                "input_hash": row.input_hash,
                "finished_at": finished_at.isoformat(),
                "normalized": {
                    "ai_native": 1,
                    "ai_native_reasoning": "AI is the core mechanism.",
                    "sources_used": ["website_evidence"],
                    "ai_native_critique": "Implementation details are limited.",
                },
                "ai_native_confidence": 0.8,
            }
        )
        events.append(
            {
                "event_type": "company_completed",
                "event_id": f"b-{row.company_id}",
                "company_id": row.company_id,
                "company_name": row.company_name,
                "input_hash": row.input_hash,
                "finished_at": finished_at.isoformat(),
                "normalized": {
                    "subclass": "1E",
                    "rad_score": "RAD-M",
                    "subclass_confidence": 4,
                    "rad_confidence": 3,
                    "subclass_reasoning": "Deep vertical AI.",
                    "rad_reasoning": "Some proprietary depth.",
                    "sources_used": ["website_evidence"],
                    "subclass_critique": "A thick integrator is possible.",
                    "rad_critique": "Model ownership is unclear.",
                },
            }
        )
    # Idle marker after the downtime so wall elapsed includes the pause.
    events.append(
        {
            "event_type": "pass_a_completed",
            "event_id": f"a-{rows[2].company_id}",
            "company_id": rows[2].company_id,
            "company_name": rows[2].company_name,
            "input_hash": rows[2].input_hash,
            "finished_at": (created_at + timedelta(hours=1)).isoformat(),
            "normalized": {
                "ai_native": 0,
                "ai_native_reasoning": "No AI product mechanism.",
                "sources_used": ["website_evidence"],
                "ai_native_critique": "Evidence is thin.",
            },
            "ai_native_confidence": 0.7,
        }
    )
    (run_path / "events.jsonl").write_text(
        "".join(
            json.dumps(event, separators=(",", ":")) + "\n" for event in events
        ),
        encoding="utf-8",
    )

    code, raw = _invoke(["status", run_id, "--json"])
    payload = json.loads(raw)
    context = load_run_context(run_id)
    active = status_module._active_elapsed_seconds(context)

    assert code == 0
    assert payload["complete"] == 2
    assert payload["elapsed_seconds"] == 3600.0
    assert active == 60.0
    # 2 completes in 60 active seconds => 120 companies/hour.
    assert payload["throughput_companies_per_hour"] == 120.0
    # 2 remaining runnable (one untouched + one pass_a_only) at 120/hr => 1 minute.
    assert payload["remaining_runnable"] == 2
    assert payload["eta_seconds"] == 60.0
