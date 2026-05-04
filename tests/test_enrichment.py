import json

import pandas as pd
import pytest

from src.master_csv import (
    CLASSIFIER_INPUT_COLUMNS,
    MASTER_CSV_COLUMNS,
    is_valid_homepage_url,
    tavily_eligible_mask,
)
from src.tavily_crawl import (
    PROCESSED_OUTPUT_FIELDS,
    TavilyCrawlConfig,
    extract_usage_credits,
    run_tavily_crawl,
)
from src.website_evidence import (
    MIN_USEFUL_EVIDENCE_CHARS,
    compact_tavily_response,
)


# ---------------------------------------------------------------------------
# is_valid_homepage_url
# ---------------------------------------------------------------------------

def test_is_valid_homepage_url():
    assert is_valid_homepage_url("https://example.com")
    assert is_valid_homepage_url("http://example.com/path")
    assert not is_valid_homepage_url("")
    assert not is_valid_homepage_url("example.com")
    assert not is_valid_homepage_url("mailto:hello@example.com")


# ---------------------------------------------------------------------------
# tavily_eligible_mask
# ---------------------------------------------------------------------------

def test_tavily_eligible_mask_requires_website_alive_true():
    df = pd.DataFrame([
        {"org_uuid": "1", "homepage_url": "https://live.test", "website_alive": "true"},
        {"org_uuid": "2", "homepage_url": "https://dead.test", "website_alive": "false"},
        {"org_uuid": "3", "homepage_url": "https://blank.test", "website_alive": ""},
        {"org_uuid": "4", "homepage_url": "not-a-url", "website_alive": "true"},
    ])
    mask = tavily_eligible_mask(df)
    assert list(mask) == [True, False, False, False]


# ---------------------------------------------------------------------------
# TavilyCrawlConfig
# ---------------------------------------------------------------------------

def test_tavily_config_uses_cost_control_defaults():
    payload = TavilyCrawlConfig().request_payload("https://example.com")

    assert payload["limit"] == 5
    assert payload["max_breadth"] == 20
    assert payload["chunks_per_source"] == 3
    assert payload["extract_depth"] == "basic"
    assert payload["include_usage"] is True
    assert payload["allow_external"] is False
    assert "instructions" in payload
    assert len(payload["instructions"]) <= 400


def test_tavily_fallback_payload_omits_instruction_only_fields():
    payload = TavilyCrawlConfig(instructions="", chunks_per_source=1).request_payload("https://example.com")

    assert "instructions" not in payload
    assert "chunks_per_source" not in payload


def test_extract_usage_credits_accepts_common_shapes():
    assert extract_usage_credits({"usage": {"total_credits": 2}}) == 2.0
    assert extract_usage_credits({"usage": {"credits": "3.5"}}) == 3.5
    assert extract_usage_credits({"usage": {"map": 1, "extract": 2}}) == 3.0


# ---------------------------------------------------------------------------
# compact_tavily_response
# ---------------------------------------------------------------------------

def test_compact_tavily_response_builds_source_linked_evidence():
    padding = "\n".join([f"Supplemental classifier signal {i:03d}." for i in range(120)])
    pages_used, evidence = compact_tavily_response({
        "results": [
            {"url": "https://acme.test/product", "raw_content": f"Product uses LLM agents.\n{padding}"},
            {"url": "https://acme.test/about", "raw_content": f"About our proprietary AI.\n{padding}"},
            {"url": "https://acme.test/", "raw_content": f"Homepage overview.\n{padding}"},
        ]
    })

    assert pages_used == "https://acme.test/product | https://acme.test/about | https://acme.test/"
    assert "[Page 1: product]" in evidence
    assert "[Page 2: about]" in evidence
    assert "[Page 3: homepage]" in evidence
    assert "Product uses LLM agents." in evidence


def test_compact_tavily_response_does_not_truncate_by_default():
    long_content = "x" * 10_000
    _, evidence = compact_tavily_response({
        "results": [
            {"url": "https://acme.test/product", "raw_content": long_content},
        ]
    })

    assert long_content in evidence
    assert "[truncated]" not in evidence


def test_compact_tavily_response_removes_common_boilerplate():
    padding = "\n".join([f"Supplemental classifier signal {i:03d}." for i in range(120)])
    page_lines = [
        "top of page",
        "![Hero](https://acme.test/hero.png)",
        "Book a Demo",
        "{{templateValue}}",
        "sales@acme.test",
        "Terms & Conditions",
        "Social Media [...] Product analytics for enterprise workflows.",
        "AI workflow platform for insurance teams.",
        "AI workflow platform for insurance teams.",
        "2026 Acme Inc. All rights reserved",
    ]
    pages_used, evidence = compact_tavily_response({
        "results": [
            {
                "url": "https://acme.test/product",
                "raw_content": "\n".join(page_lines) + "\n" + padding,
            }
        ]
    })

    assert pages_used == "https://acme.test/product"
    assert "AI workflow platform for insurance teams." in evidence
    assert evidence.count("AI workflow platform for insurance teams.") == 1
    assert "hero.png" not in evidence
    assert "Book a Demo" not in evidence
    assert "top of page" not in evidence
    assert "{{templateValue}}" not in evidence
    assert "sales@acme.test" not in evidence
    assert "Terms & Conditions" not in evidence
    assert "Product analytics for enterprise workflows." in evidence
    assert "Social Media [...]" not in evidence
    assert "All rights reserved" not in evidence


def test_compact_tavily_response_drops_thin_evidence_below_min_chars():
    # Fewer than 100 chars of real content after cleaning should still be dropped
    thin_lines = [f"line {i}" for i in range(3)]  # ~18 chars total
    pages_used, evidence = compact_tavily_response({
        "results": [
            {"url": "https://acme.test/", "raw_content": "\n".join(thin_lines)},
        ]
    })

    assert pages_used == ""
    assert evidence == ""


def test_compact_tavily_response_keeps_evidence_at_or_above_min_chars():
    thick_lines = [f"signal line {i:02d}" for i in range(40)]
    pages_used, evidence = compact_tavily_response({
        "results": [
            {"url": "https://acme.test/", "raw_content": "\n".join(thick_lines)},
        ]
    })

    assert pages_used == "https://acme.test/"
    assert len(evidence) >= MIN_USEFUL_EVIDENCE_CHARS


# ---------------------------------------------------------------------------
# run_tavily_crawl — inline processed CSV writing
# ---------------------------------------------------------------------------

def _master_csv_row(org_uuid: str, name: str, url: str, website_alive: str = "true") -> dict:
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


def _success_response(url: str, credits: float = 1.0) -> dict:
    padding = "\n".join([f"Supplemental classifier signal {i:03d}." for i in range(120)])
    return {
        "results": [{"url": url, "raw_content": f"Homepage content for {url}.\n{padding}"}],
        "usage": {"credits": credits},
    }


def test_run_tavily_crawl_appends_to_processed_csv(tmp_path, monkeypatch):
    """On each successful crawl, one row should be appended to tavily_processed_output.csv."""
    queue = pd.DataFrame([
        _master_csv_row("org-1", "First", "https://first.test"),
        _master_csv_row("org-2", "Second", "https://second.test"),
    ])
    queue_path = tmp_path / "master_csv.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    processed_path = tmp_path / "processed.csv"

    queue.to_csv(queue_path, index=False)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: _success_response(url),
    )

    run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=processed_path,
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    assert processed_path.exists()
    proc = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
    assert list(proc.columns) == PROCESSED_OUTPUT_FIELDS
    assert len(proc) == 2
    assert set(proc["org_uuid"].tolist()) == {"org-1", "org-2"}
    assert proc.loc[proc["org_uuid"] == "org-1", "website_evidence"].iloc[0] != ""


def test_run_tavily_crawl_writes_classifier_input_on_completion(tmp_path, monkeypatch):
    """On clean completion, classifier_input.csv should join master + evidence."""
    queue = pd.DataFrame([
        _master_csv_row("org-live", "Live", "https://live.test", website_alive="true"),
        _master_csv_row("org-dead", "Dead", "https://dead.test", website_alive="false"),
    ])
    queue_path = tmp_path / "master_csv.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    processed_path = tmp_path / "processed.csv"
    classifier_input_path = tmp_path / "classifier_input.csv"

    queue.to_csv(queue_path, index=False)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: _success_response(url),
    )

    run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=processed_path,
        classifier_input_csv=classifier_input_path,
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    assert classifier_input_path.exists()
    ci = pd.read_csv(classifier_input_path, dtype=str, keep_default_na=False)
    assert list(ci.columns) == list(CLASSIFIER_INPUT_COLUMNS)
    assert len(ci) == 2

    live_row = ci[ci["org_uuid"] == "org-live"].iloc[0]
    dead_row = ci[ci["org_uuid"] == "org-dead"].iloc[0]

    assert live_row["website_evidence"] != ""
    assert dead_row["website_evidence"] == ""
    assert dead_row["website_pages_used"] == ""


def test_run_tavily_crawl_classifier_input_not_written_on_interrupt(tmp_path, monkeypatch):
    """classifier_input.csv must NOT be written on user_interrupt so partial joins don't silently occur."""
    import os

    queue = pd.DataFrame([
        _master_csv_row("org-1", "First", "https://first.test"),
        _master_csv_row("org-2", "Second", "https://second.test"),
    ])
    queue_path = tmp_path / "master_csv.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    classifier_input_path = tmp_path / "classifier_input.csv"
    queue.to_csv(queue_path, index=False)

    raised = {"done": False}

    def fake_call(url, config, api_key, **_kwargs):
        if not raised["done"]:
            raised["done"] = True
            os.kill(os.getpid(), 2)
        return _success_response(url)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", fake_call)

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=classifier_input_path,
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )

    assert report.exit_reason == "user_interrupt"
    assert not classifier_input_path.exists()


# ---------------------------------------------------------------------------
# run_tavily_crawl — existing fallback / retry behaviour
# ---------------------------------------------------------------------------

def test_run_tavily_crawl_falls_back_on_empty_results(tmp_path, monkeypatch):
    queue = pd.DataFrame([_master_csv_row("org-1", "Acme", "https://acme.test")])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)
    calls = []

    def fake_call(url, config, api_key, **_kwargs):
        calls.append(config.instructions)
        if len(calls) == 1:
            return {"results": [], "usage": {"credits": 0}}
        return {
            "results": [
                {"url": "https://acme.test/", "raw_content": "Recovered homepage content."}
            ],
            "usage": {"credits": 1},
        }

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", fake_call)

    report = run_tavily_crawl(
        queue_path, output_path, state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert report.completed == 1
    assert report.empty_results == 0
    assert calls[0]
    assert calls[1] == ""
    assert records[0]["crawl_status"] == "success_fallback"
    assert records[0]["usage_credits"] == 1


def test_run_tavily_crawl_records_terminal_empty_results(tmp_path, monkeypatch):
    queue = pd.DataFrame([_master_csv_row("org-empty", "EmptyCo", "https://empty.test")])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key, **_kwargs: {"results": [], "usage": {"credits": 0}},
    )

    report = run_tavily_crawl(
        queue_path, output_path, state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )
    report_again = run_tavily_crawl(
        queue_path, output_path, state_path,
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert report.empty_results == 1
    assert report_again.skipped_existing == 1
    assert len(records) == 1
    assert records[0]["crawl_status"] == "empty_results"
    assert records[0]["retryable"] is False


def test_run_tavily_crawl_retries_transient_errors(tmp_path, monkeypatch):
    queue = pd.DataFrame([_master_csv_row("org-retry", "RetryCo", "https://retry.test")])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)
    calls = 0

    def flaky_call(url, config, api_key, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out")
        return {
            "results": [
                {"url": "https://retry.test/", "raw_content": "Recovered after retry."}
            ],
            "usage": {"credits": 1},
        }

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", flaky_call)

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        config=TavilyCrawlConfig(retry_backoff_seconds=0),
        processed_csv=tmp_path / "processed.csv",
        classifier_input_csv=tmp_path / "classifier_input.csv",
        manifest_csv=tmp_path / "manifest.csv",
        heartbeat_log=tmp_path / "heartbeat.log",
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert calls == 2
    assert report.completed == 1
    assert records[0]["crawl_status"] == "success"


