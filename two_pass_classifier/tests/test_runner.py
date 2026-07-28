from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest

from two_pass_classifier.journal import (
    ResumeMismatchError,
    load_journal_state,
)
from two_pass_classifier.manifest import Manifest, ManifestRow
from two_pass_classifier.request_builder import RequestSettings
from two_pass_classifier.runner import (
    ProductionRunner,
    RunnerSettings,
    _idempotency_key,
)


class _FakeResponse:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        self.status = raw.get("status")
        self.id = raw.get("id")
        self._request_id = f"request-{self.id}"
        self.output_text = _output_text(raw)

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        return self._raw


class _FakeResponses:
    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[_FakeResponse]],
    ) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return await self._handler(kwargs)


class _FakeClient:
    max_retries = 0

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[_FakeResponse]],
    ) -> None:
        self.responses = _FakeResponses(handler)


class _RawResult:
    def __init__(
        self,
        response: _FakeResponse,
        headers: dict[str, str],
    ) -> None:
        self._response = response
        self.headers = headers

    def parse(self) -> _FakeResponse:
        return self._response


class _RawResponses:
    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[_FakeResponse]],
    ) -> None:
        self.with_raw_response = self
        self._handler = handler

    async def create(self, **kwargs: Any) -> _RawResult:
        response = await self._handler(kwargs)
        return _RawResult(
            response,
            {
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-limit-tokens": "10000000",
                "x-ratelimit-remaining-requests": "999",
                "x-ratelimit-remaining-tokens": "9990000",
                "x-request-id": f"header-{response.id}",
            },
        )


class _RawClient:
    max_retries = 0

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[_FakeResponse]],
    ) -> None:
        self.responses = _RawResponses(handler)


class _FakeHTTPError(Exception):
    def __init__(self, status: int, message: str, headers: dict[str, str]) -> None:
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers)


def _manifest(count: int = 1) -> Manifest:
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
                "short_description": "AI workflow",
                "Long description": "A detailed AI workflow.",
                "category_list": "Artificial Intelligence",
                "category_groups_list": "Software",
                "founded_date": "2024-01",
                "employee_count": "1-10",
                "total_funding_usd": "100",
                "website_pages_used": "https://company.test",
                "website_evidence": "We build and operate an AI workflow.",
            },
        )
        for index in range(1, count + 1)
    )
    return Manifest(
        rows=rows,
        sources=(),
        rows_sha256="rows-sha",
        manifest_sha256="manifest-sha",
    )


def _runner_settings(
    *,
    requests: RequestSettings | None = None,
    attempts: int = 2,
) -> RunnerSettings:
    return RunnerSettings(
        requests=requests or RequestSettings(model="gpt-5.4-nano"),
        initial_concurrency=4,
        max_concurrency=4,
        company_queue_size=4,
        writer_queue_size=8,
        writer_group_max_events=8,
        writer_group_max_wait_seconds=0.0,
        max_request_attempts=attempts,
        retry_base_delay_seconds=0.0,
        retry_max_delay_seconds=0.0,
        retry_jitter_fraction=0.0,
    )


def _pass_a_response(
    verdict: int = 1,
    *,
    confidence_available: bool = True,
    response_id: str = "a-1",
) -> _FakeResponse:
    payload = {
        "ai_native": verdict,
        "ai_native_reasoning": "AI is the core product mechanism.",
        "sources_used": ["website_evidence"],
        "ai_native_critique": "The evidence may omit implementation details.",
    }
    text = json.dumps(payload, separators=(",", ":"))
    decision_offset = text.index(f'"ai_native":{verdict}') + len('"ai_native":')
    entries: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        entry = {
            "token": character,
            "bytes": list(character.encode()),
            "logprob": 0.0,
            "top_logprobs": [],
        }
        if index == decision_offset:
            if confidence_available:
                chosen_mass = 0.8
                opponent_mass = 0.2
                entry["logprob"] = math.log(chosen_mass)
                entry["top_logprobs"] = [
                    {
                        "token": str(1 - verdict),
                        "bytes": list(str(1 - verdict).encode()),
                        "logprob": math.log(opponent_mass),
                    }
                ]
            else:
                entry["logprob"] = math.log(0.5)
                entry["top_logprobs"] = [
                    {
                        "token": " ",
                        "bytes": [32],
                        "logprob": math.log(0.1),
                    }
                ]
        entries.append(entry)
    return _FakeResponse(
        {
            "id": response_id,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "logprobs": entries,
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_tokens_details": {"cached_tokens": 20},
            },
        }
    )


def _pass_b_response(
    verdict: int = 1,
    *,
    response_id: str = "b-1",
) -> _FakeResponse:
    if verdict == 1:
        payload = {
            "subclass": "1E",
            "rad_score": "RAD-M",
            "subclass_confidence": 4,
            "rad_confidence": 3,
            "subclass_reasoning": "The product is deep vertical AI.",
            "rad_reasoning": "Proprietary data offsets provider dependency.",
            "sources_used": ["website_evidence", "resource_context"],
            "subclass_critique": "A thick integrator is plausible.",
            "rad_critique": "Model ownership is not explicit.",
        }
    else:
        payload = {
            "subclass": "0A",
            "subclass_confidence": 4,
            "subclass_reasoning": "The core is conventional software.",
            "rad_reasoning": "RAD is not applicable to the non-AI family.",
            "sources_used": ["website_evidence"],
            "subclass_critique": "A shipped AI feature could move it to 0B.",
        }
    text = json.dumps(payload, separators=(",", ":"))
    return _FakeResponse(
        {
            "id": response_id,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {
                "input_tokens": 120,
                "output_tokens": 80,
                "input_tokens_details": {"cached_tokens": 40},
            },
        }
    )


def _output_text(raw: dict[str, Any]) -> str:
    for item in raw.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                return str(content.get("text") or "")
    return ""


@pytest.mark.asyncio
async def test_pass_b_starts_only_after_pass_a_fsync_ack(tmp_path):
    run_dir = tmp_path / "run"
    journal = run_dir / "events.jsonl"
    pass_a_durable = False

    def fsync(fd: int) -> None:
        nonlocal pass_a_durable
        os.fsync(fd)
        if journal.exists() and "pass_a_completed" in journal.read_text(
            encoding="utf-8"
        ):
            pass_a_durable = True

    async def handler(kwargs: dict[str, Any]) -> _FakeResponse:
        if "top_logprobs" in kwargs:
            return _pass_a_response()
        assert pass_a_durable is True
        return _pass_b_response()

    client = _FakeClient(handler)
    runner = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=client,
        settings=_runner_settings(),
        install_signal_handlers=False,
        writer_fsync=fsync,
    )
    result = await runner.run()

    assert result.all_complete is True
    assert ["top_logprobs" in call for call in client.responses.calls] == [
        True,
        False,
    ]
    state = load_journal_state(runner.paths.journal, manifest=_manifest())
    pass_a = state.pass_a["company-1"]
    complete = state.completed["company-1"]
    assert pass_a["raw_response"]["id"] == "a-1"
    assert complete["raw_response"]["id"] == "b-1"
    assert pass_a["attempt"] == complete["attempt"] == 1
    assert pass_a["client_request_id"]
    assert pass_a["openai_request_id"] == "request-a-1"
    assert pass_a["usage"]["input_tokens_details"]["cached_tokens"] == 20
    assert runner.paths.final_csv.exists()


@pytest.mark.asyncio
async def test_resume_pass_a_only_then_skip_completed_company(tmp_path):
    run_dir = tmp_path / "resume"
    holder: dict[str, ProductionRunner] = {}

    async def first_handler(kwargs: dict[str, Any]) -> _FakeResponse:
        assert "top_logprobs" in kwargs
        holder["runner"].request_shutdown()
        return _pass_a_response()

    first_client = _FakeClient(first_handler)
    first = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=first_client,
        settings=_runner_settings(),
        install_signal_handlers=False,
    )
    holder["runner"] = first
    first_result = await first.run()

    assert first_result.stopped is True
    assert first_result.pass_a_checkpoint_count == 1
    assert first_result.completed_count == 0
    assert not first.paths.final_csv.exists()
    with first.paths.in_progress_csv.open(
        encoding="utf-8", newline=""
    ) as handle:
        assert list(csv.DictReader(handle)) == []

    async def second_handler(kwargs: dict[str, Any]) -> _FakeResponse:
        assert "top_logprobs" not in kwargs
        return _pass_b_response()

    second_client = _FakeClient(second_handler)
    second = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=second_client,
        settings=_runner_settings(),
        install_signal_handlers=False,
    )
    second_result = await second.run()
    assert second_result.all_complete is True
    assert len(second_client.responses.calls) == 1

    async def forbidden_handler(kwargs: dict[str, Any]) -> _FakeResponse:
        raise AssertionError("completed companies must not be resubmitted")

    third_client = _FakeClient(forbidden_handler)
    third = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=third_client,
        settings=_runner_settings(),
        install_signal_handlers=False,
    )
    third_result = await third.run()
    assert third_result.all_complete is True
    assert third_client.responses.calls == []


@pytest.mark.asyncio
async def test_same_runner_clears_shutdown_flag_before_next_run(tmp_path):
    """In-process resume must not inherit a sticky shutdown flag."""
    run_dir = tmp_path / "reuse-shutdown"
    holder: dict[str, ProductionRunner] = {}

    async def shutdown_handler(kwargs: dict[str, Any]) -> _FakeResponse:
        assert "top_logprobs" in kwargs
        holder["runner"].request_shutdown()
        return _pass_a_response(response_id="a-1")

    client = _FakeClient(shutdown_handler)
    runner = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=client,
        settings=_runner_settings(),
        install_signal_handlers=False,
    )
    holder["runner"] = runner
    first = await runner.run()

    assert first.stopped is True
    assert first.completed_count == 0
    assert first.pass_a_checkpoint_count == 1
    assert runner.shutdown_event.is_set()

    async def resume_handler(kwargs: dict[str, Any]) -> _FakeResponse:
        assert "top_logprobs" not in kwargs
        return _pass_b_response(response_id="b-1")

    client.responses._handler = resume_handler
    second = await runner.run()

    assert second.stopped is False
    assert second.all_complete is True
    assert second.completed_count == 1
    assert len(client.responses.calls) == 2
    assert not runner.shutdown_event.is_set()


@pytest.mark.asyncio
async def test_unavailable_confidence_still_completes_with_blank_csv(tmp_path):
    async def handler(kwargs: dict[str, Any]) -> _FakeResponse:
        if "top_logprobs" in kwargs:
            return _pass_a_response(confidence_available=False)
        return _pass_b_response()

    runner = ProductionRunner(
        manifest=_manifest(),
        run_dir=tmp_path / "unavailable",
        client=_FakeClient(handler),
        settings=_runner_settings(),
        install_signal_handlers=False,
    )
    result = await runner.run()

    assert result.all_complete is True
    state = load_journal_state(runner.paths.journal, manifest=_manifest())
    pass_a = state.pass_a["company-1"]
    assert pass_a["ai_native_confidence"] is None
    assert pass_a["confidence_extraction"]["status"] == "unavailable"
    with runner.paths.final_csv.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["ai_native_confidence"] == ""


@pytest.mark.asyncio
async def test_rate_limit_headers_are_recorded_and_update_live_limits(tmp_path):
    async def handler(kwargs: dict[str, Any]) -> _FakeResponse:
        if "top_logprobs" in kwargs:
            return _pass_a_response()
        return _pass_b_response()

    runner = ProductionRunner(
        manifest=_manifest(),
        run_dir=tmp_path / "headers",
        client=_RawClient(handler),
        settings=_runner_settings(),
        install_signal_handlers=False,
    )
    await runner.run()

    state = load_journal_state(runner.paths.journal, manifest=_manifest())
    pass_a = state.pass_a["company-1"]
    assert pass_a["openai_request_id"] == "header-a-1"
    assert (
        pass_a["rate_limit_headers"]["x-ratelimit-limit-requests"]
        == "1000"
    )
    assert runner.rate_controller.target_limits("gpt-5.4-nano") == (
        800,
        8_000_000,
    )


@pytest.mark.asyncio
async def test_429_is_journaled_retried_and_reduces_global_concurrency(tmp_path):
    attempt = 0

    async def handler(kwargs: dict[str, Any]) -> _FakeResponse:
        nonlocal attempt
        if "top_logprobs" in kwargs:
            attempt += 1
            if attempt == 1:
                raise _FakeHTTPError(
                    429,
                    "too many requests",
                    {"retry-after": "0", "x-request-id": "rate-request"},
                )
            return _pass_a_response()
        return _pass_b_response()

    runner = ProductionRunner(
        manifest=_manifest(),
        run_dir=tmp_path / "retry",
        client=_FakeClient(handler),
        settings=_runner_settings(attempts=2),
        install_signal_handlers=False,
    )
    result = await runner.run()

    assert result.all_complete is True
    assert runner.concurrency_controller.limit == 2
    events = [
        json.loads(line)
        for line in runner.paths.journal.read_text(encoding="utf-8").splitlines()
    ]
    error = next(event for event in events if event["event_type"] == "request_error")
    assert error["category"] == "rate_limit"
    assert error["attempt"] == 1
    assert error["will_retry"] is True
    assert error["openai_request_id"] == "rate-request"
    pass_a = next(
        event for event in events if event["event_type"] == "pass_a_completed"
    )
    assert pass_a["attempt"] == 2


def test_idempotency_key_is_stable_across_attempts_for_one_stage():
    first = _idempotency_key(
        company_id="company-1",
        stage="pass_b",
        input_hash="hash-1",
    )
    second = _idempotency_key(
        company_id="company-1",
        stage="pass_b",
        input_hash="hash-1",
    )
    other_stage = _idempotency_key(
        company_id="company-1",
        stage="pass_a",
        input_hash="hash-1",
    )
    assert first == second == "company-1:pass_b:hash-1"
    assert other_stage != first


@pytest.mark.asyncio
async def test_pass_b_sends_stable_idempotency_key(tmp_path):
    seen_pass_a_keys: list[str] = []

    async def handler(kwargs: dict[str, Any]) -> _FakeResponse:
        headers = kwargs.get("extra_headers") or {}
        if "top_logprobs" in kwargs:
            seen_pass_a_keys.append(headers["Idempotency-Key"])
            assert headers["Idempotency-Key"] == _idempotency_key(
                company_id="company-1",
                stage="pass_a",
                input_hash="hash-1",
            )
            if len(seen_pass_a_keys) == 1:
                raise _FakeHTTPError(
                    429,
                    "too many requests",
                    {"retry-after": "0", "x-request-id": "rate-request"},
                )
            return _pass_a_response()
        assert headers["Idempotency-Key"] == _idempotency_key(
            company_id="company-1",
            stage="pass_b",
            input_hash="hash-1",
        )
        assert headers["X-Client-Request-Id"] == headers["Idempotency-Key"]
        return _pass_b_response()

    runner = ProductionRunner(
        manifest=_manifest(),
        run_dir=tmp_path / "idempotency",
        client=_FakeClient(handler),
        settings=_runner_settings(attempts=2),
        install_signal_handlers=False,
    )
    result = await runner.run()
    assert result.all_complete is True
    assert seen_pass_a_keys == [
        "company-1:pass_a:hash-1",
        "company-1:pass_a:hash-1",
    ]


@pytest.mark.asyncio
async def test_empty_run_metadata_cannot_resume_recorded_run_config(tmp_path):
    run_dir = tmp_path / "run-config"
    holder: dict[str, ProductionRunner] = {}

    async def stop_after_a(kwargs: dict[str, Any]) -> _FakeResponse:
        holder["runner"].request_shutdown()
        return _pass_a_response()

    first = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=_FakeClient(stop_after_a),
        settings=_runner_settings(),
        run_metadata={"cohort_label": "pilot"},
        install_signal_handlers=False,
    )
    holder["runner"] = first
    await first.run()

    second = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=_FakeClient(stop_after_a),
        settings=_runner_settings(),
        run_metadata={},
        install_signal_handlers=False,
    )
    with pytest.raises(ResumeMismatchError, match="run configuration"):
        await second.run()


@pytest.mark.asyncio
async def test_changed_output_cap_refuses_unsafe_resume(tmp_path):
    run_dir = tmp_path / "fingerprint"
    holder: dict[str, ProductionRunner] = {}

    async def stop_after_a(kwargs: dict[str, Any]) -> _FakeResponse:
        holder["runner"].request_shutdown()
        return _pass_a_response()

    baseline = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=_FakeClient(stop_after_a),
        settings=_runner_settings(),
        install_signal_handlers=False,
    )
    holder["runner"] = baseline
    await baseline.run()

    changed_requests = RequestSettings(
        model="gpt-5.4-nano",
        pass_b_effort="low",
        pass_b_max_output_tokens=4_097,
    )
    changed = ProductionRunner(
        manifest=_manifest(),
        run_dir=run_dir,
        client=_FakeClient(stop_after_a),
        settings=_runner_settings(requests=changed_requests),
        install_signal_handlers=False,
    )
    with pytest.raises(ResumeMismatchError, match="fingerprint"):
        await changed.run()
