"""Reliability tests for the Tavily crawl runner.

These cover the layers added on top of the per-call retries:
    * atomic state save
    * JSONL self-heal on resume
    * outage retry loop
    * SIGINT graceful exit
    * budget pre-reservation
    * pre-flight eligibility check
    * heartbeat and run manifest

None of these tests hit the live Tavily API.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from src.tavily_crawl import (
    MANIFEST_FIELDS,
    TavilyCrawlConfig,
    TavilyCrawlState,
    _CrawlSlidingWindowLimiter,
    _GracefulStopController,
    _heal_jsonl_tail,
    run_tavily_crawl,
)


def _master_row(org_uuid: str, name: str, url: str, website_alive: str = "true") -> dict:
    return {
        "org_uuid": org_uuid,
        "name": name,
        "homepage_url": url,
        "short_description": "",
        "Long description": "",
        "category_list": "",
        "category_groups_list": "",
        "founded_date": "",
        "employee_count": "",
        "total_funding_usd": "",
        "website_alive": website_alive,
    }


def _two_row_queue(tmp_path: Path) -> Path:
    queue = pd.DataFrame([
        _master_row("org-1", "First", "https://first.test"),
        _master_row("org-2", "Second", "https://second.test"),
    ])
    queue_path = tmp_path / "queue.csv"
    queue.to_csv(queue_path, index=False)
    return queue_path


def _success_response(url: str, credits: float = 1.0) -> dict:
    return {
        "results": [{"url": url, "raw_content": f"Content for {url}."}],
        "usage": {"credits": credits},
    }


def test_state_save_is_atomic_when_replace_fails(tmp_path, monkeypatch):
    """A crash between temp-write and rename leaves the original state file intact."""
    state_path = tmp_path / "state.json"
    original = TavilyCrawlState(total_credits=42.0, completed=7, last_org_uuid="org-prev")
    original.save(state_path)
    original_bytes = state_path.read_bytes()

    def boom(_src, _dst):
        raise OSError("simulated mid-rename crash")

    monkeypatch.setattr("src.tavily_crawl.os.replace", boom)

    next_state = TavilyCrawlState(total_credits=999.0, completed=99, last_org_uuid="org-next")
    with pytest.raises(OSError, match="simulated mid-rename crash"):
        next_state.save(state_path)

    assert state_path.read_bytes() == original_bytes
    assert not state_path.with_suffix(state_path.suffix + ".tmp").exists() or \
        state_path.with_suffix(state_path.suffix + ".tmp").read_bytes() != original_bytes


def test_heal_jsonl_tail_truncates_unterminated_partial(tmp_path):
    path = tmp_path / "raw.jsonl"
    good = json.dumps({"org_uuid": "org-1", "ok": True}) + "\n"
    partial = '{"org_uuid":"org-2","ok":'  # truncated mid-write, no newline
    path.write_text(good + partial, encoding="utf-8")

    truncated = _heal_jsonl_tail(path)

    assert truncated == len(partial)
    assert path.read_text(encoding="utf-8") == good
    parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert parsed == [{"org_uuid": "org-1", "ok": True}]


def test_heal_jsonl_tail_appends_missing_newline(tmp_path):
    path = tmp_path / "raw.jsonl"
    good = json.dumps({"org_uuid": "org-1"}) + "\n"
    valid_no_newline = json.dumps({"org_uuid": "org-2"})
    path.write_text(good + valid_no_newline, encoding="utf-8")

    truncated = _heal_jsonl_tail(path)

    assert truncated == 0
    parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert parsed == [{"org_uuid": "org-1"}, {"org_uuid": "org-2"}]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_runner_self_heals_jsonl_on_startup(tmp_path, monkeypatch):
    queue_path = _two_row_queue(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"

    pre_existing_good = json.dumps({"org_uuid": "org-pre", "ok": True, "retryable": False}) + "\n"
    output_path.write_text(pre_existing_good + '{"org_uuid":"org-bad",', encoding="utf-8")

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: _success_response(url),
    )

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["org_uuid"] == "org-pre"
    assert {p["org_uuid"] for p in parsed[1:]} == {"org-1", "org-2"}
    assert report.completed == 2


def test_outage_loop_retries_row_through_transient_errors(tmp_path, monkeypatch):
    """Persistent transient errors trigger a long-cap outage retry, then succeed."""
    queue = pd.DataFrame([
        {
            "org_uuid": "org-outage",
            "name": "OutageCo",
            "homepage_url": "https://outage.test",
            "short_description": "",
            "website_alive": "true",
        }
    ])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)

    calls = {"n": 0}

    def flaky_call(url, config, api_key, **_kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("simulated outage")
        return _success_response(url, credits=1.0)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", flaky_call)

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        config=TavilyCrawlConfig(retry_backoff_seconds=0, max_retries=0),
        outage_backoff_min_seconds=0,
        outage_backoff_max_seconds=0,
        max_outage_seconds=600,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert calls["n"] == 3
    assert report.completed == 1
    assert report.failed == 0
    assert records[0]["crawl_status"] == "success"
    assert records[0]["usage_credits"] == 1.0


def test_outage_loop_gives_up_after_max_outage_seconds(tmp_path, monkeypatch):
    """If transient errors persist past the wall-time cap, write a retryable failure and move on."""
    queue = pd.DataFrame([
        {
            "org_uuid": "org-cap",
            "name": "CapCo",
            "homepage_url": "https://cap.test",
            "short_description": "",
            "website_alive": "true",
        }
    ])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: (_ for _ in ()).throw(TimeoutError("never recovers")),
    )

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        config=TavilyCrawlConfig(retry_backoff_seconds=0, max_retries=0),
        outage_backoff_min_seconds=0,
        outage_backoff_max_seconds=0,
        max_outage_seconds=0,  # immediately give up after first transient failure
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert report.failed == 1
    assert records[0]["crawl_status"] == "transient_error"
    assert records[0]["retryable"] is True
    assert report.errors_by_status.get("transient_error") == 1


def test_runner_exits_at_row_boundary_on_sigint(tmp_path, monkeypatch):
    """A SIGINT mid-run drains state cleanly at the next row boundary."""
    queue_path = _two_row_queue(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")

    raised_signal = {"done": False}

    def fake_call(url, config, api_key, **_kwargs):
        if not raised_signal["done"]:
            raised_signal["done"] = True
            os.kill(os.getpid(), 2)  # SIGINT delivered to the controller
        return _success_response(url)

    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", fake_call)

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    # Row 1 completes (signal was caught, not raised). Loop bails before row 2.
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert {r["org_uuid"] for r in records} == {"org-1"}
    assert report.completed == 1
    assert report.exit_reason == "user_interrupt"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_org_uuid"] == "org-1"


def test_preflight_aborts_when_no_eligible_rows(tmp_path, monkeypatch):
    """An empty website_alive column must fail loudly before any API call is made."""
    queue = pd.DataFrame([
        {
            "org_uuid": "org-1",
            "name": "First",
            "homepage_url": "https://first.test",
            "short_description": "",
            "website_alive": "",
        }
    ])
    queue_path = tmp_path / "queue.csv"
    queue.to_csv(queue_path, index=False)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"

    calls = {"n": 0}

    def fake_call(*_args, **_kwargs):
        calls["n"] += 1
        return {"results": [], "usage": {"credits": 0}}

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", fake_call)

    with pytest.raises(RuntimeError, match="master_csv.csv"):
        run_tavily_crawl(
            queue_path,
            output_path,
            state_path,
            processed_csv=tmp_path / "processed.csv",
            classifier_input_csv=tmp_path / "classifier_input.csv",
            manifest_csv=tmp_path / "manifest.csv",
            heartbeat_log=tmp_path / "heartbeat.log",
        )

    assert calls["n"] == 0
    assert not output_path.exists()


def test_budget_pre_reservation_blocks_next_row(tmp_path, monkeypatch):
    """When the rolling per-row credits would push past budget, stop before the next call."""
    queue_path = _two_row_queue(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"

    calls = {"n": 0}

    def fake_call(url, config, api_key, **_kwargs):
        calls["n"] += 1
        return _success_response(url, credits=1.0)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", fake_call)

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        budget_credits=1.5,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    assert calls["n"] == 1
    assert report.attempted == 1
    assert report.completed == 1
    assert report.budget_reached is True
    assert report.exit_reason == "budget_reached"


def test_run_manifest_appends_one_row_per_run(tmp_path, monkeypatch):
    queue_path = _two_row_queue(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    manifest_path = tmp_path / "manifest.csv"

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: _success_response(url),
    )

    run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        max_companies=1,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=manifest_path,
        heartbeat_log=tmp_path / "heartbeat.log",
    )
    run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=manifest_path,
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    rows = list(manifest_path.read_text(encoding="utf-8").splitlines())
    assert rows[0].split(",") == MANIFEST_FIELDS
    assert len(rows) == 3  # header + two runs
    second_run = dict(zip(MANIFEST_FIELDS, rows[1].split(",")))
    third_run = dict(zip(MANIFEST_FIELDS, rows[2].split(",")))
    assert second_run["exit_reason"] == "max_companies"
    assert third_run["exit_reason"] == "completed"


def test_heartbeat_writes_progress_lines(tmp_path, monkeypatch):
    queue_path = _two_row_queue(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    heartbeat_path = tmp_path / "heartbeat.log"

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: _success_response(url),
    )

    run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        heartbeat_every=1,
        heartbeat_log=heartbeat_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
    )

    text = heartbeat_path.read_text(encoding="utf-8")
    assert "processed=1" in text
    assert "processed=2" in text
    assert "rows/min" in text


def test_crawl_sliding_window_rate_limiter_throttles_fourth_call():
    """Fourth call in a 3-per-window limiter should wait until the window slides."""
    lim = _CrawlSlidingWindowLimiter(3.0, window_seconds=0.15)
    t0 = time.monotonic()
    for _ in range(4):
        lim.acquire()
    assert time.monotonic() - t0 >= 0.08


def test_concurrent_run_finishes_all_rows(tmp_path, monkeypatch):
    rows = [
        {
            "org_uuid": f"org-{i}",
            "name": f"C{i}",
            "homepage_url": f"https://c{i}.test",
            "short_description": "",
            "website_alive": "true",
        }
        for i in range(4)
    ]
    queue_path = tmp_path / "queue.csv"
    pd.DataFrame(rows).to_csv(queue_path, index=False)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: _success_response(url),
    )

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        max_concurrent_rows=3,
        crawl_rpm=0,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )
    parsed = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == 4
    assert report.completed == 4
    assert report.attempted == 4


def test_graceful_stop_controller_interruptible_sleep():
    """The controller's sleep wakes immediately when the stop flag flips."""
    controller = _GracefulStopController()
    controller.stop_requested = True
    assert controller.sleep(60) is False


def test_resume_picks_up_after_kill(tmp_path, monkeypatch):
    """After a SIGINT mid-run, a second invocation skips the completed row and crawls the rest."""
    queue_path = _two_row_queue(tmp_path)
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")

    raised = {"done": False}

    def first_call(url, config, api_key, **_kwargs):
        if not raised["done"]:
            raised["done"] = True
            os.kill(os.getpid(), 2)
        return _success_response(url)

    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", first_call)
    first = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )
    assert first.exit_reason == "user_interrupt"

    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: _success_response(url),
    )
    second = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert {r["org_uuid"] for r in records} == {"org-1", "org-2"}
    assert second.skipped_existing == 1
    assert second.completed == 1
