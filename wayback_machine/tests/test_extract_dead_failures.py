"""Failure root-cause instrumentation for the dead-cohort extract.

Covers the pure classification helpers (synthetic error payloads in, derived
``failure_reason`` + ``retryable`` out), the end-to-end ``_process_single_row``
diagnostics with the extract call stubbed, the resume tally in ``_scan_jsonl``,
and the offline ``summarize_crawl_failures`` report.
"""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

from wayback_machine.config import ExtractConfig
from wayback_machine.extract_dead import (
    NETWORK_ERROR,
    NO_ARCHIVE_CONTENT,
    RATE_LIMITED,
    TRANSIENT_ERROR,
    UNKNOWN_FAILURE,
    _attempt_from_error,
    _classify_failure_reason,
    _process_single_row,
    _RowOutcome,
    _row_failure_reason,
    _run_row_with_outage,
    _scan_jsonl,
    run_extract_dead,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "msg", {}, io.BytesIO(body))


# ---------------------------------------------------------------------------
# Pure classification helpers
# ---------------------------------------------------------------------------


def test_attempt_from_error_flags_429_as_rate_limited() -> None:
    attempt = _attempt_from_error("extract", {"type": "HTTPError", "status": 429, "body": ""})
    assert attempt == {"phase": "extract", "error_type": "HTTPError",
                       "http_status": 429, "rate_limited": True}


def test_attempt_from_error_flags_rate_limit_text_without_429() -> None:
    attempt = _attempt_from_error(
        "extract", {"type": "HTTPError", "status": 403, "body": "Monthly quota exceeded"},
    )
    assert attempt["rate_limited"] is True


def test_attempt_from_error_network_exception_has_no_status() -> None:
    attempt = _attempt_from_error("extract", {"type": "TimeoutError", "message": "timed out"})
    assert attempt == {"phase": "extract", "error_type": "TimeoutError"}


@pytest.mark.parametrize(
    "attempts, expected",
    [
        # Every call returned a clean 200 but no usable content → genuine gap.
        ([{"phase": "extract", "http_status": 200}, {"phase": "extract", "http_status": 200}],
         (NO_ARCHIVE_CONTENT, False)),
        # A 429 anywhere wins, even alongside clean 200s.
        ([{"phase": "extract", "http_status": 200},
          {"phase": "extract", "error_type": "HTTPError", "http_status": 429, "rate_limited": True}],
         (RATE_LIMITED, True)),
        # Network exception → retryable network_error.
        ([{"phase": "extract", "error_type": "URLError"}], (NETWORK_ERROR, True)),
        # Transient HTTP status → retryable transient_error.
        ([{"phase": "extract", "error_type": "HTTPError", "http_status": 503}],
         (TRANSIENT_ERROR, True)),
        # Errored but unclassifiable (e.g. auth) → terminal unknown.
        ([{"phase": "extract", "error_type": "HTTPError", "http_status": 403}],
         (UNKNOWN_FAILURE, False)),
        # No attempts at all → treated as a content gap.
        ([], (NO_ARCHIVE_CONTENT, False)),
    ],
)
def test_classify_failure_reason(attempts, expected) -> None:
    assert _classify_failure_reason(attempts) == expected


def test_rate_limited_beats_transient_priority() -> None:
    attempts = [
        {"phase": "extract", "error_type": "HTTPError", "http_status": 503},
        {"phase": "extract", "error_type": "HTTPError", "http_status": 429, "rate_limited": True},
    ]
    assert _classify_failure_reason(attempts) == (RATE_LIMITED, True)


def test_row_failure_reason_legacy_empty() -> None:
    assert _row_failure_reason({"status": "empty_results"}) == "legacy_empty"
    assert _row_failure_reason({"status": "empty_results",
                                "failure_reason": "no_archive_content"}) == "no_archive_content"


# ---------------------------------------------------------------------------
# _process_single_row end-to-end (extract call stubbed)
# ---------------------------------------------------------------------------

_TARGET = {
    "org_uuid": "u1",
    "name": "Co",
    "homepage_url": "https://co.example",
    "snapshot_url": "http://web.archive.org/web/20230314120000if_/https://co.example",
    "closest_ts": "20230314120000",
    "select_paths": "",
}


def _run_row(monkeypatch, *, extract, cfg=None, stop_check=None):
    """Run _process_single_row with the extract call stubbed."""
    monkeypatch.setattr("wayback_machine.extract_dead.call_tavily_extract", lambda *a, **k: extract())
    return _process_single_row(
        target=dict(_TARGET),
        cfg=cfg or ExtractConfig(),
        api_key="k",
        rate_limiter=None,
        stop_check=stop_check,
        stop_sleep=None,
    )


def test_process_row_no_archive_content_is_terminal_empty(monkeypatch) -> None:
    outcome = _run_row(monkeypatch, extract=lambda: {"results": []})
    assert outcome.status == "empty_results"
    assert outcome.ok is True
    assert outcome.retryable is False
    assert outcome.record["failure_reason"] == NO_ARCHIVE_CONTENT
    assert outcome.record["attempts"]  # both snapshot candidates recorded


def test_process_row_extract_rate_limit_is_retryable(monkeypatch) -> None:
    def extract():
        raise _http_error(429, b'{"detail":"rate limit"}')

    outcome = _run_row(monkeypatch, extract=extract)
    assert outcome.status == RATE_LIMITED
    assert outcome.ok is False
    assert outcome.retryable is True
    assert outcome.transient_failure is True
    assert outcome.record["failure_reason"] == RATE_LIMITED


def test_process_row_extract_network_error_is_retryable(monkeypatch) -> None:
    def extract():
        raise TimeoutError("timed out")

    outcome = _run_row(monkeypatch, extract=extract)
    assert outcome.status == NETWORK_ERROR
    assert outcome.retryable is True
    assert outcome.record["failure_reason"] == NETWORK_ERROR


def test_process_row_usable_results_skip_failure_diagnostics(monkeypatch) -> None:
    # An extract with usable raw_content takes the evidence branch, never the
    # no-evidence branch, so no failure_reason is attached. (The vendored cleaner
    # needs real Wayback chrome to emit evidence, so this lands on empty_results
    # for synthetic content — the point is only that a clean 200 is NOT a fetch
    # failure and the row is terminal, not retryable.)
    usable = {"results": [{"url": "https://co.example",
                           "raw_content": "Co builds widgets for teams everywhere."}]}
    outcome = _run_row(monkeypatch, extract=lambda: usable)
    assert outcome.ok is True
    assert outcome.retryable is False
    assert outcome.status in {"success", "empty_results"}


def test_process_row_stop_signal_finishes_current_row(monkeypatch) -> None:
    # Once a company starts, finish its snapshot attempts and write one JSONL row.
    # That row-boundary stop behavior avoids partial paid work with no resume
    # record, which would double-bill the first snapshot on restart.
    calls = 0

    def extract():
        nonlocal calls
        calls += 1
        return {"results": []}

    outcome = _run_row(monkeypatch, extract=extract, stop_check=lambda: True)
    assert calls == 2
    assert outcome.status == "empty_results"
    assert outcome.record["usage_credits"] == pytest.approx(0.4)


def test_no_archive_content_credits_track_extract_depth(monkeypatch) -> None:
    # Two 200-but-empty snapshots billed at the depth's per-extraction rate:
    # basic = 0.2 each, advanced = 0.4 each (2 credits per 5 extractions).
    basic = _run_row(monkeypatch, extract=lambda: {"results": []},
                     cfg=ExtractConfig(extract_depth="basic"))
    advanced = _run_row(monkeypatch, extract=lambda: {"results": []},
                        cfg=ExtractConfig(extract_depth="advanced"))
    assert basic.record["usage_credits"] == pytest.approx(0.4)
    assert advanced.record["usage_credits"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Resume tally: retryable empties stay pending; terminal empties complete
# ---------------------------------------------------------------------------


def test_scan_jsonl_splits_retryable_and_terminal_empties(tmp_path: Path) -> None:
    rows = [
        {"org_uuid": "done", "ok": True, "status": "success", "retryable": False,
         "usage_credits": 1.0},
        {"org_uuid": "gap", "ok": True, "status": "empty_results", "retryable": False,
         "failure_reason": NO_ARCHIVE_CONTENT},
        {"org_uuid": "throttled", "ok": False, "status": RATE_LIMITED, "retryable": True,
         "failure_reason": RATE_LIMITED},
        {"org_uuid": "legacy", "ok": True, "status": "empty_results", "retryable": False},
    ]
    jsonl = tmp_path / "crawl_dead.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    scan = _scan_jsonl(jsonl)
    assert scan.successful == 1
    assert scan.empty == 1  # no_archive_content only; legacy crawl empties re-enter pending
    assert scan.failed == 1  # rate_limited
    # Terminal rows complete; throttled + legacy crawl-empty rows stay pending.
    assert scan.completed_ids == {"done", "gap"}
    assert "throttled" not in scan.completed_ids
    assert "legacy" not in scan.completed_ids
    assert scan.failure_reasons[NO_ARCHIVE_CONTENT] == 1
    assert scan.failure_reasons[RATE_LIMITED] == 1
    assert scan.failure_reasons["legacy_empty"] == 1


def test_run_extract_dead_reserves_budget_for_concurrent_rows(tmp_path: Path, monkeypatch) -> None:
    targets = tmp_path / "targets.csv"
    targets.write_text(
        "org_uuid,name,homepage_url,snapshot_url,closest_ts,select_paths\n"
        + "\n".join(
            f"u{i},Co {i},https://co{i}.example,"
            f"http://web.archive.org/web/20230314120000if_/https://co{i}.example,"
            "20230314120000,"
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("wayback_machine.extract_dead._api_key", lambda: "k")
    monkeypatch.setattr("wayback_machine.extract_dead._preflight_checks", lambda **_: None)

    def fake_run_row(**kwargs):
        org_uuid = kwargs["target"]["org_uuid"]
        return _RowOutcome(
            record={
                "org_uuid": org_uuid,
                "name": kwargs["target"]["name"],
                "homepage_url": kwargs["target"]["homepage_url"],
                "snapshot_ts": kwargs["target"]["closest_ts"],
                "ok": True,
                "status": "success",
                "retryable": False,
            },
            status="success",
            ok=True,
            retryable=False,
            pages_used=kwargs["target"]["homepage_url"],
            evidence="homepage evidence",
            credits_added=0.2,
            transient_failure=False,
        )

    monkeypatch.setattr("wayback_machine.extract_dead._run_row_with_outage", fake_run_row)

    report = run_extract_dead(
        targets_csv=targets,
        output_jsonl=tmp_path / "crawl_dead.jsonl",
        state_json=tmp_path / "crawl_state_dead.json",
        processed_csv=tmp_path / "scrape_processed_dead.csv",
        heartbeat_log=tmp_path / "crawl_dead.log",
        manifest_csv=tmp_path / "run_manifest_dead.csv",
        budget_credits=0.4,
        max_concurrent_rows=12,
        heartbeat_every=0,
    )

    assert report.attempted == 1
    assert report.budget_reached is True


def test_outage_retry_preserves_paid_credits_from_earlier_attempts(monkeypatch) -> None:
    outcomes = iter([
        _RowOutcome(
            record={
                "org_uuid": "u1",
                "ok": False,
                "status": RATE_LIMITED,
                "retryable": True,
                "usage_credits": 0.4,
                "failure_reason": RATE_LIMITED,
                "attempts": [{"phase": "extract", "http_status": 200},
                             {"phase": "extract", "error_type": "HTTPError",
                              "http_status": 429, "rate_limited": True}],
            },
            status=RATE_LIMITED,
            ok=False,
            retryable=True,
            pages_used="",
            evidence="",
            credits_added=0.4,
            transient_failure=True,
        ),
        _RowOutcome(
            record={"org_uuid": "u1", "ok": True, "status": "success",
                    "retryable": False, "usage_credits": 0.2},
            status="success",
            ok=True,
            retryable=False,
            pages_used="https://co.example",
            evidence="homepage evidence",
            credits_added=0.2,
            transient_failure=False,
        ),
    ])
    monkeypatch.setattr("wayback_machine.extract_dead._process_single_row",
                        lambda **_: next(outcomes))

    class Stop:
        stop_requested = False

        @staticmethod
        def sleep(_seconds):
            return True

    outcome = _run_row_with_outage(
        target=dict(_TARGET),
        cfg=ExtractConfig(),
        api_key="k",
        rate_limiter=None,
        stop=Stop(),
        max_outage_seconds=60,
        outage_backoff_min_seconds=0,
        outage_backoff_max_seconds=0,
    )

    assert outcome.status == "success"
    assert outcome.credits_added == pytest.approx(0.6)
    assert outcome.record["usage_credits"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Offline summary tool
# ---------------------------------------------------------------------------


def _load_summary_module():
    path = PROJECT_ROOT / "wayback_machine" / "scripts" / "summarize_crawl_failures.py"
    spec = importlib.util.spec_from_file_location("summarize_crawl_failures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summarize_crawl_failures_counts(tmp_path: Path) -> None:
    mod = _load_summary_module()
    rows = [
        {"org_uuid": "a", "status": "success"},
        {"org_uuid": "b", "status": "success_extract_fallback"},  # legacy crawl-era win
        {"org_uuid": "c", "status": "empty_results", "failure_reason": NO_ARCHIVE_CONTENT},
        {"org_uuid": "d", "status": RATE_LIMITED, "failure_reason": RATE_LIMITED,
         "retryable": True},
        {"org_uuid": "e", "status": "empty_results"},  # legacy
        {"org_uuid": "f", "status": "thin_evidence"},
    ]
    jsonl = tmp_path / "crawl_dead.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary = mod.summarize(jsonl)
    assert summary["total"] == 6
    assert summary["successes"] == {"success": 1, "success_extract_fallback": 1}
    assert summary["thin_evidence"] == 1
    assert summary["failures"] == {NO_ARCHIVE_CONTENT: 1, RATE_LIMITED: 1, "legacy_empty": 1}
    assert summary["retryable_failures"] == 1
    report = mod.format_report(summary)
    assert "FAILURE BREAKDOWN" in report
    assert RATE_LIMITED in report


def test_summarize_missing_file_is_empty(tmp_path: Path) -> None:
    mod = _load_summary_module()
    summary = mod.summarize(tmp_path / "nope.jsonl")
    assert summary["total"] == 0
