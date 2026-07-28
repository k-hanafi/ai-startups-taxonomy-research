from __future__ import annotations

import asyncio
import csv
import json
import os

import pytest

from two_pass_classifier.journal import (
    JOURNAL_VERSION,
    AsyncJSONLWriter,
    JournalCorruptionError,
    JournalState,
    RunArtifactPaths,
    RunLock,
    RunLockedError,
    load_journal_state,
    rebuild_derived_artifacts,
)
from two_pass_classifier.manifest import Manifest, ManifestRow


def _manifest() -> Manifest:
    rows = tuple(
        ManifestRow(
            company_id=f"company-{index}",
            company_name=f"Company {index}",
            cohort="GENAI-ERA",
            company_alive="yes",
            website_snapshot_date="2026-05-04",
            evidence_source="live",
            source_row_number=index + 1,
            input_hash=f"hash-{index}",
            inputs={
                "org_uuid": f"company-{index}",
                "name": f"Company {index}",
                "short_description": "AI product",
                "Long description": "Long description",
                "category_list": "AI",
                "category_groups_list": "Software",
                "founded_date": "2024-01",
                "employee_count": "1-10",
                "total_funding_usd": "1",
                "website_pages_used": "https://example.test",
                "website_evidence": "Evidence",
            },
        )
        for index in (1, 2)
    )
    return Manifest(
        rows=rows,
        sources=(),
        rows_sha256="rows-sha",
        manifest_sha256="manifest-sha",
    )


def _header() -> dict:
    return {
        "event_type": "run_started",
        "journal_version": JOURNAL_VERSION,
        "manifest_sha256": "manifest-sha",
        "manifest_row_count": 2,
        "request_fingerprint": "fingerprint",
    }


def _pass_a(row: ManifestRow) -> dict:
    return {
        "event_type": "pass_a_completed",
        "event_id": f"a-{row.company_id}",
        "company_id": row.company_id,
        "input_hash": row.input_hash,
        "normalized": {
            "ai_native": 1,
            "ai_native_reasoning": "AI is the core mechanism.",
            "sources_used": ["website_evidence"],
            "ai_native_critique": "Implementation details are limited.",
        },
        "ai_native_confidence": 0.8,
    }


def _complete(row: ManifestRow) -> dict:
    return {
        "event_type": "company_completed",
        "event_id": f"b-{row.company_id}",
        "company_id": row.company_id,
        "input_hash": row.input_hash,
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


def test_read_only_replay_tolerates_truncated_tail_without_mutation(tmp_path):
    path = tmp_path / "events.jsonl"
    first = json.dumps(_header(), separators=(",", ":")).encode()
    original = first + b"\n{\"event_type\":\"pass_a"
    path.write_bytes(original)

    state = load_journal_state(path)

    assert state.header == _header()
    assert path.read_bytes() == original


def test_truncated_final_line_is_healed_in_repair_mode(tmp_path):
    path = tmp_path / "events.jsonl"
    first = json.dumps(_header(), separators=(",", ":")).encode()
    path.write_bytes(first + b"\n{\"event_type\":\"pass_a")

    state = load_journal_state(path, replay_mode="repair")

    assert state.header == _header()
    assert path.read_bytes() == first + b"\n"


def test_interior_corruption_fails_loudly(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(_header())
        + "\n"
        + '{"event_type":broken}\n'
        + json.dumps(
            {
                "event_type": "request_error",
                "company_id": "company-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JournalCorruptionError, match="line 2"):
        load_journal_state(path)


@pytest.mark.asyncio
async def test_writer_acknowledges_only_after_fsync_seam(tmp_path):
    trace: list[str] = []

    def fsync(fd: int) -> None:
        os.fsync(fd)
        trace.append("fsync")

    writer = AsyncJSONLWriter(
        tmp_path / "events.jsonl",
        queue_size=4,
        group_max_events=4,
        group_max_wait_seconds=0.01,
        fsync=fsync,
    )
    await writer.start()

    async def submit(name: str) -> None:
        await writer.submit({"event_type": name})
        trace.append(f"ack-{name}")

    await asyncio.gather(submit("one"), submit("two"))
    await writer.close()

    assert trace[0] == "fsync"
    assert trace.index("fsync") < trace.index("ack-one")
    assert trace.index("fsync") < trace.index("ack-two")
    assert trace.count("fsync") == 1


def test_run_lock_excludes_a_second_process_handle(tmp_path):
    lock_path = tmp_path / "run.lock"
    first = RunLock(lock_path).acquire()
    try:
        with pytest.raises(RunLockedError):
            RunLock(lock_path).acquire()
    finally:
        first.release()

    with RunLock(lock_path):
        pass


def test_exports_only_complete_rows_and_gates_final_file(tmp_path):
    manifest = _manifest()
    row1, row2 = manifest.rows
    paths = RunArtifactPaths.from_run_dir(tmp_path / "run")
    state = JournalState(header=_header())
    state.pass_a[row1.company_id] = _pass_a(row1)
    state.pass_a[row2.company_id] = _pass_a(row2)
    state.completed[row2.company_id] = _complete(row2)

    summary = rebuild_derived_artifacts(
        manifest,
        state,
        paths,
        stopped=False,
    )

    assert summary["completed_count"] == 1
    assert not paths.final_csv.exists()
    with paths.in_progress_csv.open(encoding="utf-8", newline="") as handle:
        partial = list(csv.DictReader(handle))
    assert [row["company_id"] for row in partial] == ["company-2"]

    state.completed[row1.company_id] = _complete(row1)
    summary = rebuild_derived_artifacts(
        manifest,
        state,
        paths,
        stopped=False,
    )

    assert summary["all_complete"] is True
    with paths.final_csv.open(encoding="utf-8", newline="") as handle:
        final = list(csv.DictReader(handle))
    assert [row["company_id"] for row in final] == [
        "company-1",
        "company-2",
    ]
