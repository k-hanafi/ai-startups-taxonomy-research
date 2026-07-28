"""High-throughput resumable runner over the normal Responses API."""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import signal
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from openai import AsyncOpenAI
from pydantic import ValidationError

from . import config
from .confidence import (
    BinaryConfidenceUnavailable,
    LogprobExtractionError,
    extract_binary_confidence,
)
from .journal import (
    JOURNAL_VERSION,
    AsyncJSONLWriter,
    JournalState,
    ResumeMismatchError,
    RunArtifactPaths,
    RunLock,
    load_journal_state,
    rebuild_derived_artifacts,
)
from .manifest import Manifest, ManifestRow
from .rate_control import (
    AdmissionStopped,
    AdaptiveConcurrencyController,
    CacheRouteWarmer,
    DualRateAdmissionController,
)
from .request_builder import (
    RequestSettings,
    build_pass_a_request,
    build_pass_b_request,
    cache_route_for_pass,
    estimate_input_tokens,
    request_fingerprint,
    request_identity,
)
from .schema import (
    PassAResult,
    PassBAINativeResult,
    PassBNotAINativeResult,
)


class ResponseContractError(ValueError):
    """A successful provider response violates the locked output contract."""


class StageFailed(RuntimeError):
    """A company stage exhausted retries or reached a permanent failure."""

    def __init__(
        self,
        message: str,
        *,
        disposition: "ErrorDisposition",
    ) -> None:
        super().__init__(message)
        self.disposition = disposition


class ShutdownRequested(RuntimeError):
    """The run stopped before another physical request could start."""


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    requests: RequestSettings = field(default_factory=RequestSettings)
    initial_concurrency: int = config.INITIAL_CONCURRENCY
    max_concurrency: int = config.MAX_CONCURRENCY
    company_queue_size: int = config.COMPANY_QUEUE_SIZE
    writer_queue_size: int = config.WRITER_QUEUE_SIZE
    writer_group_max_events: int = config.WRITER_GROUP_MAX_EVENTS
    writer_group_max_wait_seconds: float = (
        config.WRITER_GROUP_MAX_WAIT_SECONDS
    )
    max_request_attempts: int = config.MAX_REQUEST_ATTEMPTS
    retry_base_delay_seconds: float = config.RETRY_BASE_DELAY_SECONDS
    retry_max_delay_seconds: float = config.RETRY_MAX_DELAY_SECONDS
    retry_jitter_fraction: float = config.RETRY_JITTER_FRACTION

    def __post_init__(self) -> None:
        if not 1 <= self.initial_concurrency <= self.max_concurrency:
            raise ValueError(
                "concurrency must satisfy 1 <= initial_concurrency "
                "<= max_concurrency"
            )
        if self.max_concurrency > config.MAX_CONCURRENCY:
            raise ValueError(
                f"max_concurrency exceeds local safety ceiling "
                f"{config.MAX_CONCURRENCY}"
            )
        if self.company_queue_size < 1 or self.writer_queue_size < 1:
            raise ValueError("queue sizes must be positive")
        if self.max_request_attempts < 1:
            raise ValueError("max_request_attempts must be positive")
        if not 0 <= self.retry_jitter_fraction <= 1:
            raise ValueError("retry_jitter_fraction must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    manifest_row_count: int
    pass_a_checkpoint_count: int
    completed_count: int
    incomplete_count: int
    all_complete: bool
    stopped: bool


@dataclass(frozen=True, slots=True)
class ParsedPassA:
    normalized: dict[str, Any]
    ai_native_confidence: float | None
    confidence_extraction: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PhysicalCall:
    stage: str
    attempt: int
    client_request_id: str
    openai_request_id: str | None
    provider_response_id: str | None
    model: str
    usage: dict[str, Any] | None
    latency_seconds: float
    started_at: str
    finished_at: str
    rate_limit_headers: dict[str, str]
    concurrency_limit: int
    active_concurrency: int
    rate_utilization: dict[str, float]
    rate_limit_targets: dict[str, int]
    raw_response: dict[str, Any]
    parsed: Any


@dataclass(frozen=True, slots=True)
class ErrorDisposition:
    category: str
    retriable: bool
    ambiguous_provider_billing: bool
    status_code: int | None


def create_async_openai_client(
    *,
    api_key: str | None = None,
) -> AsyncOpenAI:
    """Construct the one production SDK client with SDK retries disabled."""
    return AsyncOpenAI(api_key=api_key, max_retries=0)


class ProductionRunner:
    """Run coupled Pass A and Pass B company tasks with JSONL-only resume."""

    def __init__(
        self,
        *,
        manifest: Manifest,
        run_dir: str | Path,
        client: Any | None = None,
        settings: RunnerSettings | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        rate_controller: DualRateAdmissionController | None = None,
        concurrency_controller: AdaptiveConcurrencyController | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        install_signal_handlers: bool = True,
        writer_fsync: Callable[[int], None] | None = None,
    ) -> None:
        self.manifest = manifest
        self.paths = RunArtifactPaths.from_run_dir(run_dir)
        self.settings = settings or RunnerSettings()
        self.run_metadata = _json_round_trip(dict(run_metadata or {}))
        self.client = client or create_async_openai_client()
        self._owns_client = client is None
        client_retries = getattr(self.client, "max_retries", None)
        if client_retries is not None and client_retries != 0:
            raise ValueError(
                "AsyncOpenAI SDK retries must be disabled with max_retries=0"
            )
        self.rate_controller = rate_controller or DualRateAdmissionController()
        self.concurrency_controller = (
            concurrency_controller
            or AdaptiveConcurrencyController(
                initial=self.settings.initial_concurrency,
                ceiling=self.settings.max_concurrency,
            )
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self._rng = rng or random.Random()
        self._install_signal_handlers = install_signal_handlers
        self._writer_fsync = writer_fsync
        self.shutdown_event = asyncio.Event()
        self._writer: AsyncJSONLWriter | None = None
        self._warmer = CacheRouteWarmer(
            (
                cache_route_for_pass("pass_a"),
                cache_route_for_pass("pass_b", 0),
                cache_route_for_pass("pass_b", 1),
            )
        )

    def request_shutdown(self) -> None:
        self.shutdown_event.set()

    async def run(self) -> RunResult:
        """Run or resume one manifest under an exclusive process lock."""
        # A prior SIGINT/request_shutdown/fatal error must not sticky-block
        # the next in-process run on this same runner instance.
        self.shutdown_event.clear()
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = request_fingerprint(self.settings.requests)

        with RunLock(self.paths.lock):
            state = load_journal_state(
                self.paths.journal,
                manifest=self.manifest,
                expected_fingerprint=fingerprint,
                replay_mode="repair",
            )
            if state.header is not None:
                recorded_metadata = state.header.get("run_config") or {}
                if recorded_metadata != self.run_metadata:
                    raise ResumeMismatchError(
                        "immutable run configuration changed; resume with the "
                        "configuration recorded in events.jsonl"
                    )
            writer_kwargs: dict[str, Any] = {}
            if self._writer_fsync is not None:
                writer_kwargs["fsync"] = self._writer_fsync
            self._writer = AsyncJSONLWriter(
                self.paths.journal,
                queue_size=self.settings.writer_queue_size,
                group_max_events=self.settings.writer_group_max_events,
                group_max_wait_seconds=(
                    self.settings.writer_group_max_wait_seconds
                ),
                **writer_kwargs,
            )
            await self._writer.start()
            if state.header is None:
                header = self._run_started_event(fingerprint)
                await self._writer.submit(header)
                state.header = header
                state.event_count += 1

            rebuild_derived_artifacts(
                self.manifest,
                state,
                self.paths,
                stopped=self.shutdown_event.is_set(),
            )
            restore_signals = self._register_signal_handlers()
            run_error: BaseException | None = None
            try:
                await self._run_company_queue(state)
            except BaseException as exc:
                run_error = exc
                self.shutdown_event.set()
            finally:
                restore_signals()
                await self._writer.close()
                self._writer = None
                final_state = load_journal_state(
                    self.paths.journal,
                    manifest=self.manifest,
                    expected_fingerprint=fingerprint,
                    replay_mode="repair",
                )
                summary = rebuild_derived_artifacts(
                    self.manifest,
                    final_state,
                    self.paths,
                    stopped=self.shutdown_event.is_set(),
                )
                if self._owns_client:
                    await _close_client(self.client)

            if run_error is not None:
                raise run_error
            return RunResult(
                run_dir=self.paths.run_dir,
                manifest_row_count=self.manifest.row_count,
                pass_a_checkpoint_count=summary["pass_a_checkpoint_count"],
                completed_count=summary["completed_count"],
                incomplete_count=summary["incomplete_count"],
                all_complete=summary["all_complete"],
                stopped=summary["stopped"],
            )

    async def _run_company_queue(self, state: JournalState) -> None:
        remaining = [
            row
            for row in self.manifest.rows
            if row.company_id not in state.completed
            and (
                row.company_id not in state.latest_errors
                or bool(state.latest_errors[row.company_id].get("retriable"))
            )
        ]
        if not remaining or self.shutdown_event.is_set():
            return

        worker_count = min(self.settings.max_concurrency, len(remaining))
        queue: asyncio.Queue[ManifestRow | None] = asyncio.Queue(
            maxsize=self.settings.company_queue_size
        )
        fatal_errors: list[BaseException] = []

        async def worker() -> None:
            while True:
                row = await queue.get()
                try:
                    if row is None:
                        return
                    if self.shutdown_event.is_set():
                        continue
                    try:
                        await self._process_company(row, state)
                    except StageFailed as exc:
                        if exc.disposition.category in {
                            "authentication",
                            "permission",
                            "quota_or_billing",
                        }:
                            fatal_errors.append(exc)
                            self.shutdown_event.set()
                        continue
                    except ShutdownRequested:
                        continue
                    except Exception as exc:
                        fatal_errors.append(exc)
                        self.shutdown_event.set()
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker(), name=f"company-worker-{index}")
            for index in range(worker_count)
        ]
        for row in remaining:
            if self.shutdown_event.is_set():
                break
            await queue.put(row)
        for _ in workers:
            await queue.put(None)
        await queue.join()
        await asyncio.gather(*workers)
        if fatal_errors:
            raise fatal_errors[0]

    async def _process_company(
        self,
        row: ManifestRow,
        state: JournalState,
    ) -> None:
        pass_a_event = state.pass_a.get(row.company_id)
        if pass_a_event is None:
            request_a = build_pass_a_request(row, self.settings.requests)
            call_a = await self._warmer.run(
                cache_route_for_pass("pass_a"),
                lambda: self._execute_request(
                    row=row,
                    stage="pass_a",
                    request=request_a,
                    parser=_parse_pass_a,
                ),
            )
            parsed_a = call_a.parsed
            if not isinstance(parsed_a, ParsedPassA):
                raise TypeError("Pass A parser returned an invalid result")
            pass_a_event = {
                "event_type": "pass_a_completed",
                "event_id": uuid.uuid4().hex,
                "company_id": row.company_id,
                "company_name": row.company_name,
                "input_hash": row.input_hash,
                **self._physical_metadata(call_a),
                "raw_response": call_a.raw_response,
                "normalized": parsed_a.normalized,
                "ai_native_confidence": parsed_a.ai_native_confidence,
                "confidence_extraction": parsed_a.confidence_extraction,
            }
            writer_ack_started = self._monotonic()
            await self._require_writer().submit(pass_a_event)
            writer_ack_seconds = self._monotonic() - writer_ack_started
            state.pass_a[row.company_id] = pass_a_event
            await self._record_success(call_a, writer_ack_seconds)

        if self.shutdown_event.is_set():
            raise ShutdownRequested("shutdown began before Pass B")

        verdict = int(pass_a_event["normalized"]["ai_native"])
        request_b = build_pass_b_request(
            row,
            verdict,
            self.settings.requests,
        )
        call_b = await self._warmer.run(
            cache_route_for_pass("pass_b", verdict),
            lambda: self._execute_request(
                row=row,
                stage="pass_b",
                request=request_b,
                parser=lambda response, raw: _parse_pass_b(
                    response,
                    raw,
                    verdict,
                ),
            ),
        )
        complete_event = {
            "event_type": "company_completed",
            "event_id": uuid.uuid4().hex,
            "company_id": row.company_id,
            "company_name": row.company_name,
            "input_hash": row.input_hash,
            "pass_a_event_id": pass_a_event["event_id"],
            **self._physical_metadata(call_b),
            "raw_response": call_b.raw_response,
            "normalized": call_b.parsed,
        }
        writer_ack_started = self._monotonic()
        await self._require_writer().submit(complete_event)
        writer_ack_seconds = self._monotonic() - writer_ack_started
        state.completed[row.company_id] = complete_event
        await self._record_success(call_b, writer_ack_seconds)

    async def _execute_request(
        self,
        *,
        row: ManifestRow,
        stage: str,
        request: dict[str, Any],
        parser: Callable[[Any, dict[str, Any]], Any],
    ) -> PhysicalCall:
        estimated_input = estimate_input_tokens(request)
        output_allowance = int(request["max_output_tokens"])

        # One key per company/stage/input so resume after any attempt still
        # hits the provider idempotency cache (attempt number is local only).
        client_request_id = _idempotency_key(
            company_id=row.company_id,
            stage=stage,
            input_hash=row.input_hash,
        )
        for attempt in range(1, self.settings.max_request_attempts + 1):
            if self.shutdown_event.is_set():
                raise ShutdownRequested(
                    f"shutdown began before {stage} attempt {attempt}"
                )
            kwargs = dict(request)
            extra_headers = dict(kwargs.get("extra_headers") or {})
            extra_headers["X-Client-Request-Id"] = client_request_id
            extra_headers["Idempotency-Key"] = client_request_id
            kwargs["extra_headers"] = extra_headers

            try:
                reservation = await self.rate_controller.acquire(
                    self.settings.requests.model,
                    estimated_input_tokens=estimated_input,
                    output_allowance=output_allowance,
                    stop_event=self.shutdown_event,
                )
            except AdmissionStopped as exc:
                raise ShutdownRequested(str(exc)) from exc

            started_at = _utc_now()
            started_clock = self._monotonic()
            response: Any | None = None
            raw_response: dict[str, Any] | None = None
            headers: dict[str, str] = {}
            try:
                try:
                    async with self.concurrency_controller.slot(
                        self.shutdown_event
                    ):
                        if self.shutdown_event.is_set():
                            raise ShutdownRequested(
                                f"shutdown began before {stage} dispatch"
                            )
                        response, headers = await _call_responses_api(
                            self.client,
                            kwargs,
                        )
                    latency = self._monotonic() - started_clock
                    raw_response = _response_to_dict(response)
                    usage = _usage_dict(raw_response)
                    await self.rate_controller.reconcile(
                        reservation,
                        actual_input_tokens=_usage_token(usage, "input_tokens"),
                        actual_output_tokens=_usage_token(
                            usage, "output_tokens"
                        ),
                    )
                    await self.rate_controller.observe_headers(
                        self.settings.requests.model,
                        headers,
                    )
                    parsed = parser(response, raw_response)
                    operational = await self._operational_snapshot()
                except AdmissionStopped as exc:
                    raise ShutdownRequested(str(exc)) from exc
                except ShutdownRequested:
                    raise
                except Exception as exc:
                    latency = self._monotonic() - started_clock
                    if not headers:
                        headers = _exception_headers(exc)
                    if headers:
                        await self.rate_controller.observe_headers(
                            self.settings.requests.model,
                            headers,
                        )
                    disposition = categorize_error(exc)
                    retry_after = _retry_after_seconds(headers)
                    if disposition.category == "rate_limit":
                        await self.rate_controller.pause_model(
                            self.settings.requests.model,
                            retry_after
                            if retry_after is not None
                            else config.DEFAULT_429_RETRY_AFTER_SECONDS,
                        )
                    await self.concurrency_controller.record_error(
                        disposition.category
                    )
                    operational = await self._operational_snapshot()
                    will_retry = (
                        disposition.retriable
                        and attempt < self.settings.max_request_attempts
                        and not self.shutdown_event.is_set()
                    )
                    error_event = {
                        "event_type": "request_error",
                        "event_id": uuid.uuid4().hex,
                        "company_id": row.company_id,
                        "company_name": row.company_name,
                        "input_hash": row.input_hash,
                        "stage": stage,
                        "attempt": attempt,
                        "client_request_id": client_request_id,
                        "openai_request_id": _openai_request_id(
                            response, headers, exc
                        ),
                        "provider_response_id": _provider_response_id(
                            response, raw_response
                        ),
                        "model": self.settings.requests.model,
                        "usage": (
                            _usage_dict(raw_response)
                            if raw_response is not None
                            else None
                        ),
                        "latency_seconds": round(latency, 6),
                        "started_at": started_at,
                        "finished_at": _utc_now(),
                        "rate_limit_headers": _rate_limit_headers(headers),
                        **operational,
                        "category": disposition.category,
                        "status_code": disposition.status_code,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "retriable": disposition.retriable,
                        "will_retry": will_retry,
                        "ambiguous_provider_billing": (
                            disposition.ambiguous_provider_billing
                        ),
                        "retry_after_seconds": retry_after,
                    }
                    if raw_response is not None:
                        error_event["raw_response"] = raw_response
                    await self._require_writer().submit(error_event)

                    if not will_retry:
                        if self.shutdown_event.is_set():
                            raise ShutdownRequested(
                                f"shutdown stopped {stage} retries"
                            ) from exc
                        raise StageFailed(
                            f"{stage} failed for {row.company_id}: "
                            f"{disposition.category}: {exc}",
                            disposition=disposition,
                        ) from exc
                    delay = self._retry_delay(attempt)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    await self._sleep_before_retry(delay)
                    continue

                return PhysicalCall(
                    stage=stage,
                    attempt=attempt,
                    client_request_id=client_request_id,
                    openai_request_id=_openai_request_id(
                        response, headers, None
                    ),
                    provider_response_id=_provider_response_id(
                        response, raw_response
                    ),
                    model=self.settings.requests.model,
                    usage=_usage_dict(raw_response),
                    latency_seconds=latency,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    rate_limit_headers=_rate_limit_headers(headers),
                    concurrency_limit=operational["concurrency_limit"],
                    active_concurrency=operational["active_concurrency"],
                    rate_utilization=operational["rate_utilization"],
                    rate_limit_targets=operational["rate_limit_targets"],
                    raw_response=raw_response,
                    parsed=parsed,
                )
            finally:
                await self.rate_controller.release(reservation)

        raise AssertionError("request attempt loop exited unexpectedly")

    async def _record_success(
        self,
        call: PhysicalCall,
        writer_ack_seconds: float,
    ) -> None:
        utilization = await self.rate_controller.utilization(call.model)
        await self.concurrency_controller.record_success(
            latency_seconds=call.latency_seconds,
            writer_ack_seconds=writer_ack_seconds,
            rate_utilization=utilization,
        )

    def _physical_metadata(self, call: PhysicalCall) -> dict[str, Any]:
        return {
            "stage": call.stage,
            "attempt": call.attempt,
            "client_request_id": call.client_request_id,
            "openai_request_id": call.openai_request_id,
            "provider_response_id": call.provider_response_id,
            "model": call.model,
            "usage": call.usage,
            "latency_seconds": round(call.latency_seconds, 6),
            "started_at": call.started_at,
            "finished_at": call.finished_at,
            "rate_limit_headers": call.rate_limit_headers,
            "concurrency_limit": call.concurrency_limit,
            "active_concurrency": call.active_concurrency,
            "rate_utilization": call.rate_utilization,
            "rate_limit_targets": call.rate_limit_targets,
        }

    async def _operational_snapshot(self) -> dict[str, Any]:
        request_target, token_target = self.rate_controller.target_limits(
            self.settings.requests.model
        )
        return {
            "concurrency_limit": self.concurrency_controller.limit,
            "active_concurrency": self.concurrency_controller.active,
            "rate_utilization": await self.rate_controller.utilization_breakdown(
                self.settings.requests.model
            ),
            "rate_limit_targets": {
                "rpm": request_target,
                "tpm": token_target,
            },
        }

    def _retry_delay(self, attempt: int) -> float:
        base = min(
            self.settings.retry_max_delay_seconds,
            self.settings.retry_base_delay_seconds * (2 ** (attempt - 1)),
        )
        spread = base * self.settings.retry_jitter_fraction
        return max(0.0, base + self._rng.uniform(-spread, spread))

    async def _sleep_before_retry(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        sleep_task = asyncio.create_task(self._sleep(delay))
        stop_task = asyncio.create_task(self.shutdown_event.wait())
        done, pending = await asyncio.wait(
            {sleep_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if stop_task in done and self.shutdown_event.is_set():
            raise ShutdownRequested("shutdown began during retry backoff")

    def _run_started_event(self, fingerprint: str) -> dict[str, Any]:
        return {
            "event_type": "run_started",
            "event_id": uuid.uuid4().hex,
            "journal_version": JOURNAL_VERSION,
            "created_at": _utc_now(),
            "manifest_sha256": self.manifest.manifest_sha256,
            "manifest_rows_sha256": self.manifest.rows_sha256,
            "manifest_row_count": self.manifest.row_count,
            "request_fingerprint": fingerprint,
            "request_identity": request_identity(self.settings.requests),
            "run_config": self.run_metadata,
        }

    def _register_signal_handlers(self) -> Callable[[], None]:
        if not self._install_signal_handlers:
            return lambda: None
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        previous: dict[signal.Signals, Any] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.getsignal(sig)
                loop.add_signal_handler(sig, self.request_shutdown)
                installed.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                continue

        def restore() -> None:
            for sig in installed:
                loop.remove_signal_handler(sig)
                signal.signal(sig, previous[sig])

        return restore

    def _require_writer(self) -> AsyncJSONLWriter:
        if self._writer is None:
            raise RuntimeError("runner journal writer is not active")
        return self._writer


def _idempotency_key(
    *,
    company_id: str,
    stage: str,
    input_hash: str,
) -> str:
    """Return a stable key for one company stage under a fixed input hash.

    Resume after a crash between provider success and journal fsync reuses the
    same key so OpenAI can return the cached response instead of re-billing.
    Attempt number is omitted on purpose: resume cannot recover which attempt
    was in flight when the process died before the journal write.
    """
    return f"{company_id}:{stage}:{input_hash}"


def categorize_error(exc: BaseException) -> ErrorDisposition:
    """Classify one physical failure for the runner-owned retry policy."""
    if isinstance(
        exc,
        (ResponseContractError, ValidationError, LogprobExtractionError),
    ):
        return ErrorDisposition("schema", False, False, None)

    status = _status_code(exc)
    name = type(exc).__name__.lower()
    text = _error_text(exc).lower()
    quota = any(
        marker in text
        for marker in (
            "insufficient_quota",
            "billing",
            "quota exceeded",
            "current quota",
            "credit balance",
            "insufficient credit",
        )
    )
    if quota:
        return ErrorDisposition("quota_or_billing", False, False, status)
    if status == 401 or "authentication" in name:
        return ErrorDisposition("authentication", False, False, status)
    if status == 403 or "permission" in name:
        return ErrorDisposition("permission", False, False, status)
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return ErrorDisposition("rate_limit", True, False, status)
    if (
        status in (408, 504)
        or isinstance(exc, (asyncio.TimeoutError, TimeoutError))
        or "timeout" in name
    ):
        return ErrorDisposition("timeout", True, True, status)
    if (
        isinstance(exc, ConnectionError)
        or "connection" in name
        or "connecterror" in name
    ):
        return ErrorDisposition("connection", True, True, status)
    if status == 409:
        return ErrorDisposition("conflict", True, False, status)
    if status is not None and 500 <= status <= 599:
        category = (
            "service_unavailable" if status == 503 else "server_error"
        )
        return ErrorDisposition(category, True, True, status)
    if status in (400, 404, 405, 413, 415, 422):
        return ErrorDisposition("bad_request", False, False, status)
    return ErrorDisposition("unknown", False, False, status)


async def _call_responses_api(
    client: Any,
    kwargs: dict[str, Any],
) -> tuple[Any, dict[str, str]]:
    responses = client.responses
    raw_api = getattr(responses, "with_raw_response", None)
    if raw_api is not None:
        raw_result = await raw_api.create(**kwargs)
        headers = _headers_dict(getattr(raw_result, "headers", None))
        parsed = raw_result.parse()
        if inspect.isawaitable(parsed):
            parsed = await parsed
        return parsed, headers
    response = await responses.create(**kwargs)
    headers = _headers_dict(getattr(response, "headers", None))
    if not headers:
        headers = _headers_dict(getattr(response, "_headers", None))
    return response, headers


def _parse_pass_a(response: Any, raw: dict[str, Any]) -> ParsedPassA:
    _require_completed(response, raw)
    text = _response_output_text(response, raw)
    try:
        result = PassAResult.model_validate_json(text)
    except (ValidationError, ValueError) as exc:
        raise ResponseContractError(f"invalid Pass A structured output: {exc}") from exc

    try:
        confidence = extract_binary_confidence(raw)
    except BinaryConfidenceUnavailable as exc:
        return ParsedPassA(
            normalized=result.model_dump(mode="json"),
            ai_native_confidence=None,
            confidence_extraction={
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
    except LogprobExtractionError as exc:
        raise ResponseContractError(
            f"invalid Pass A logprob payload: {exc}"
        ) from exc
    if confidence.ai_native != result.ai_native:
        raise ResponseContractError(
            "Pass A structured verdict disagrees with its sampled decision token"
        )
    return ParsedPassA(
        normalized=result.model_dump(mode="json"),
        ai_native_confidence=confidence.sampled_probability,
        confidence_extraction={
            "status": "available",
            **asdict(confidence),
        },
    )


def _parse_pass_b(
    response: Any,
    raw: dict[str, Any],
    verdict: int,
) -> dict[str, Any]:
    _require_completed(response, raw)
    text = _response_output_text(response, raw)
    model_cls = PassBAINativeResult if verdict == 1 else PassBNotAINativeResult
    try:
        result = model_cls.model_validate_json(text)
    except (ValidationError, ValueError) as exc:
        raise ResponseContractError(f"invalid Pass B structured output: {exc}") from exc
    return result.model_dump(mode="json")


def _require_completed(response: Any, raw: Mapping[str, Any]) -> None:
    status = getattr(response, "status", None) or raw.get("status")
    if status is not None and status != "completed":
        raise ResponseContractError(
            f"provider response status is {status!r}, not 'completed'"
        )


def _response_output_text(
    response: Any,
    raw: Mapping[str, Any],
) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    raw_text = raw.get("output_text")
    if isinstance(raw_text, str) and raw_text:
        return raw_text
    for item in raw.get("output") or []:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if (
                isinstance(content, Mapping)
                and content.get("type") == "output_text"
                and isinstance(content.get("text"), str)
            ):
                return str(content["text"])
    raise ResponseContractError("provider response has no output text")


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return _json_round_trip(response)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return _json_round_trip(dumped)
    raise ResponseContractError(
        f"response type {type(response).__name__} cannot be serialized"
    )


def _json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _usage_dict(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    usage = raw.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else None


def _usage_token(usage: Mapping[str, Any] | None, key: str) -> int | None:
    if usage is None or usage.get(key) is None:
        return None
    try:
        return int(usage[key])
    except (TypeError, ValueError):
        return None


def _headers_dict(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    try:
        return {str(key).lower(): str(value) for key, value in headers.items()}
    except AttributeError:
        return {}


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.startswith("x-ratelimit-")
        or key in {"retry-after", "x-request-id"}
    }


def _exception_headers(exc: BaseException) -> dict[str, str]:
    response = getattr(exc, "response", None)
    return _headers_dict(getattr(response, "headers", None))


def _status_code(exc: BaseException) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _error_text(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    return f"{exc} {body if body is not None else ''}"


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _openai_request_id(
    response: Any | None,
    headers: Mapping[str, str],
    exc: BaseException | None,
) -> str | None:
    for value in (
        headers.get("x-request-id"),
        getattr(response, "_request_id", None),
        getattr(exc, "request_id", None) if exc is not None else None,
    ):
        if value:
            return str(value)
    return None


def _provider_response_id(
    response: Any | None,
    raw: Mapping[str, Any] | None,
) -> str | None:
    value = getattr(response, "id", None)
    if value is None and raw is not None:
        value = raw.get("id")
    return str(value) if value else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result
