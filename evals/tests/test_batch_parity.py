"""Offline tests for the Stage 7 Batch parity smoke (gate Q4).

The paid submission path is exercised live outside the sandbox; here the
request construction and every parity assertion run against fixture payloads
shaped like real Responses API bodies (sync and batch sides are identical
shapes, which is exactly what the checks assert)."""

from __future__ import annotations

import copy
import json

import pytest

from evals import batch_parity, config as cfg
from evals.two_pass import pass_a_kwargs


@pytest.fixture
def sample_row() -> dict[str, str]:
    return {
        "org_uuid": "abc123-uuid",
        "name": "Acme AI",
        "short_description": "An AI thing.",
        "Long description": "A longer description.",
        "category_list": "Artificial Intelligence (AI)",
        "category_groups_list": "Software",
        "founded_date": "2023-05",
        "employee_count": "1-10",
        "total_funding_usd": "1000000",
        "website_pages_used": "https://acme.ai/",
        "website_evidence": "We build AI agents.",
    }


def _logprob_entry(token: str, n_top: int) -> dict:
    return {
        "token": token,
        "bytes": [ord(c) for c in token],
        "logprob": -0.01,
        "top_logprobs": [
            {"token": f"alt{i}", "bytes": [65], "logprob": -0.5 - i}
            for i in range(n_top)
        ],
    }


def _response_body(verdict: int = 1, n_top: int = cfg.TOP_LOGPROBS,
                   status: str = "completed") -> dict:
    """A payload dict shaped like a real Pass A Responses API body."""
    text = json.dumps({"ai_native": verdict})
    return {
        "status": status,
        "model": "gpt-5.4-nano",
        "temperature": 1.0,
        "top_logprobs": cfg.TOP_LOGPROBS,
        "reasoning": {"effort": "none"},
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "output": [
            {"type": "reasoning", "content": []},
            {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "logprobs": [_logprob_entry(tok, n_top)
                                 for tok in ("{\"", "ai", "_native", "\":",
                                             str(verdict), "}")],
                }],
            },
        ],
    }


def _request_body() -> dict:
    return {
        "model": "gpt-5.4-nano",
        "top_logprobs": cfg.TOP_LOGPROBS,
        "reasoning": {"effort": "none"},
        "include": list(cfg.LOGPROB_INCLUDE),
    }


# --- request construction -------------------------------------------------------

def test_batch_line_wraps_the_exact_sync_body(sample_row):
    body = pass_a_kwargs(sample_row, "PROMPT", "gpt-5.4-nano")
    line = batch_parity.batch_input_line("startup-abc123-uuid", body)
    assert line["url"] == "/v1/responses"
    assert line["method"] == "POST"
    # Byte-identical params: the batch body IS the sync kwargs dict.
    assert line["body"] is body
    assert json.dumps(line["body"], sort_keys=True) == json.dumps(body, sort_keys=True)


# --- payload inspection ----------------------------------------------------------

def test_extract_logprob_entries_finds_message_logprobs():
    entries = batch_parity.extract_logprob_entries(_response_body())
    assert len(entries) == 6
    assert entries[0]["token"] == "{\""


def test_extract_logprob_entries_empty_when_absent():
    body = _response_body()
    del body["output"][1]["content"][0]["logprobs"]
    assert batch_parity.extract_logprob_entries(body) == []
    assert batch_parity.extract_logprob_entries({}) == []


def test_binary_verdict_parsed_from_output_text():
    assert batch_parity._binary_verdict(_response_body(verdict=0)) == 0
    assert batch_parity._binary_verdict(_response_body(verdict=1)) == 1
    broken = _response_body()
    broken["output"][1]["content"][0]["text"] = "not json"
    assert batch_parity._binary_verdict(broken) is None


# --- parity checks ---------------------------------------------------------------

def _all_ok(checks: list[dict]) -> bool:
    return all(c["ok"] for c in checks)


def test_parity_checks_pass_on_matching_payloads():
    checks = batch_parity.parity_checks(
        _request_body(), _response_body(), _response_body()
    )
    assert _all_ok(checks)


def test_parity_fails_when_batch_drops_logprobs():
    batch_body = _response_body()
    del batch_body["output"][1]["content"][0]["logprobs"]
    checks = batch_parity.parity_checks(
        _request_body(), _response_body(), batch_body
    )
    failed = {c["name"] for c in checks if not c["ok"]}
    assert "batch_logprobs_present" in failed
    # The sync side is untouched and must still pass its shape checks.
    assert "sync_logprobs_present" not in failed


def test_parity_fails_when_batch_truncates_top_logprobs():
    batch_body = _response_body(n_top=5)
    checks = batch_parity.parity_checks(
        _request_body(), _response_body(), batch_body
    )
    failed = {c["name"] for c in checks if not c["ok"]}
    assert "batch_top_logprobs_honored" in failed


def test_parity_fails_when_batch_ignores_reasoning_effort():
    batch_body = _response_body()
    batch_body["reasoning"] = {"effort": "medium"}
    batch_body["usage"]["output_tokens_details"]["reasoning_tokens"] = 640
    checks = batch_parity.parity_checks(
        _request_body(), _response_body(), batch_body
    )
    failed = {c["name"] for c in checks if not c["ok"]}
    assert "batch_echoes_reasoning_effort" in failed
    assert "batch_reasoning_tokens_zero" in failed


def test_parity_fails_on_temperature_divergence():
    batch_body = _response_body()
    batch_body["temperature"] = 0.7
    checks = batch_parity.parity_checks(
        _request_body(), _response_body(), batch_body
    )
    assert "temperature_parity" in {c["name"] for c in checks if not c["ok"]}


def test_parity_fails_on_incomplete_status():
    batch_body = _response_body(status="incomplete")
    checks = batch_parity.parity_checks(
        _request_body(), _response_body(), batch_body
    )
    assert "batch_completed" in {c["name"] for c in checks if not c["ok"]}


# --- report aggregation -----------------------------------------------------------

def test_report_pass_verdict_when_all_rows_ok():
    requests = {"startup-1": _request_body(), "startup-2": _request_body()}
    sync = {cid: _response_body() for cid in requests}
    batch = copy.deepcopy(sync)
    report = batch_parity.build_parity_report(requests, sync, batch, "gpt-5.4-nano")
    assert report["verdict"] == "PASS"
    assert report["gate"] == "Q4"
    assert report["n_rows"] == 2
    assert all(row["ok"] for row in report["rows"].values())


def test_report_fail_verdict_on_missing_batch_row():
    requests = {"startup-1": _request_body()}
    sync = {"startup-1": _response_body()}
    report = batch_parity.build_parity_report(requests, sync, {}, "gpt-5.4-nano")
    assert report["verdict"] == "FAIL"
    assert report["rows"]["startup-1"]["ok"] is False


def test_report_fail_verdict_on_one_bad_row():
    requests = {"startup-1": _request_body(), "startup-2": _request_body()}
    sync = {cid: _response_body() for cid in requests}
    batch = copy.deepcopy(sync)
    batch["startup-2"]["status"] = "failed"
    report = batch_parity.build_parity_report(requests, sync, batch, "gpt-5.4-nano")
    assert report["verdict"] == "FAIL"
    assert report["rows"]["startup-1"]["ok"] is True
    assert report["rows"]["startup-2"]["ok"] is False
