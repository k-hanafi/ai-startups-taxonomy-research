"""Durable JSONL state, run locking, and derived export artifacts."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .exporter import assemble_professor_row, write_professor_csv
from .manifest import Manifest
from .schema import (
    PassAResult,
    PassBAINativeResult,
    PassBNotAINativeResult,
)

JOURNAL_VERSION = 1
JournalReplayMode = Literal["read_only", "repair"]


class JournalCorruptionError(ValueError):
    """The authoritative journal contains an invalid interior record."""


class ResumeMismatchError(ValueError):
    """A run cannot resume under a different manifest or request identity."""


class RunLockedError(RuntimeError):
    """Another process currently owns the run directory."""


@dataclass(frozen=True, slots=True)
class RunArtifactPaths:
    run_dir: Path
    journal: Path
    lock: Path
    in_progress_csv: Path
    final_csv: Path
    failure_summary: Path
    run_summary: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "RunArtifactPaths":
        root = Path(run_dir)
        return cls(
            run_dir=root,
            journal=root / "events.jsonl",
            lock=root / "run.lock",
            in_progress_csv=root / "classifications_in_progress.csv",
            final_csv=root / "classifications.csv",
            failure_summary=root / "failure_summary.json",
            run_summary=root / "summary.json",
        )


@dataclass(slots=True)
class JournalState:
    header: dict[str, Any] | None = None
    pass_a: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    retry_requests: list[dict[str, Any]] = field(default_factory=list)
    event_count: int = 0


class RunLock:
    """Hold a non-blocking process lock on one stable run lock file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RunLockedError(
                f"run is already locked by another process: {self.path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid()}, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *_: Any) -> None:
        self.release()


@dataclass(slots=True)
class _PendingWrite:
    event: Mapping[str, Any]
    acknowledgement: asyncio.Future[None]


_STOP_WRITER = object()


class AsyncJSONLWriter:
    """Serialize events through one group-committing async writer task."""

    def __init__(
        self,
        path: str | Path,
        *,
        queue_size: int,
        group_max_events: int,
        group_max_wait_seconds: float,
        fsync: Callable[[int], None] = os.fsync,
    ) -> None:
        if queue_size < 1 or group_max_events < 1:
            raise ValueError("writer queue and group sizes must be positive")
        if group_max_wait_seconds < 0:
            raise ValueError("writer group wait cannot be negative")
        self.path = Path(path)
        self._queue: asyncio.Queue[_PendingWrite | object] = asyncio.Queue(
            maxsize=queue_size
        )
        self._group_max_events = group_max_events
        self._group_max_wait_seconds = group_max_wait_seconds
        self._fsync = fsync
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._run(), name="jsonl-writer")

    async def submit(self, event: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("journal writer is closed")
        if self._task is None:
            raise RuntimeError("journal writer has not been started")
        if self._task.done():
            await self._task
            raise RuntimeError("journal writer stopped unexpectedly")
        acknowledgement = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _PendingWrite(event=dict(event), acknowledgement=acknowledgement)
        )
        done, _ = await asyncio.wait(
            {acknowledgement, self._task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if acknowledgement in done:
            await acknowledgement
            return
        if acknowledgement.done():
            acknowledgement.exception()
        await self._task
        raise RuntimeError("journal writer stopped before acknowledgement")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is None:
            return
        if not self._task.done():
            await self._queue.put(_STOP_WRITER)
        await self._task

    async def _run(self) -> None:
        try:
            with self.path.open("ab") as raw_handle:
                while True:
                    first = await self._queue.get()
                    if first is _STOP_WRITER:
                        self._queue.task_done()
                        return

                    batch = [first]
                    stop_after_batch = False
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + self._group_max_wait_seconds
                    while len(batch) < self._group_max_events:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            break
                        try:
                            item = await asyncio.wait_for(
                                self._queue.get(),
                                timeout=remaining,
                            )
                        except asyncio.TimeoutError:
                            break
                        if item is _STOP_WRITER:
                            self._queue.task_done()
                            stop_after_batch = True
                            break
                        batch.append(item)

                    pending = [
                        item for item in batch if isinstance(item, _PendingWrite)
                    ]
                    try:
                        payload = b"".join(
                            (
                                json.dumps(
                                    item.event,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            ).encode("utf-8")
                            for item in pending
                        )
                        raw_handle.write(payload)
                        raw_handle.flush()
                        self._fsync(raw_handle.fileno())
                    except BaseException as exc:
                        for item in pending:
                            if not item.acknowledgement.done():
                                item.acknowledgement.set_exception(exc)
                        raise
                    else:
                        for item in pending:
                            item.acknowledgement.set_result(None)
                    finally:
                        for _ in pending:
                            self._queue.task_done()

                    if stop_after_batch:
                        return
        except BaseException as exc:
            while True:
                try:
                    queued = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if (
                    isinstance(queued, _PendingWrite)
                    and not queued.acknowledgement.done()
                ):
                    queued.acknowledgement.set_exception(exc)
                self._queue.task_done()
            raise


def load_journal_state(
    path: str | Path,
    *,
    manifest: Manifest | None = None,
    expected_fingerprint: str | None = None,
    replay_mode: JournalReplayMode = "read_only",
) -> JournalState:
    """Load state from one journal snapshot.

    Read-only replay ignores an unterminated final fragment without changing
    the authoritative file. Repair mode may heal that fragment and must be
    used only while the caller holds the exclusive run lock.
    """
    artifact = Path(path)
    events = _read_jsonl(
        artifact,
        replay_mode=replay_mode,
    )
    return replay_journal_events(
        events,
        manifest=manifest,
        expected_fingerprint=expected_fingerprint,
    )


def replay_journal_events(
    events: Sequence[dict[str, Any]],
    *,
    manifest: Manifest | None = None,
    expected_fingerprint: str | None = None,
) -> JournalState:
    """Validate and reduce an already captured journal snapshot."""
    state = JournalState(event_count=len(events))
    if not events:
        return state

    header = events[0]
    if header.get("event_type") != "run_started":
        raise JournalCorruptionError("first journal event must be run_started")
    if header.get("journal_version") != JOURNAL_VERSION:
        raise ResumeMismatchError(
            f"unsupported journal version {header.get('journal_version')!r}"
        )
    state.header = header

    if expected_fingerprint is not None:
        actual = header.get("request_fingerprint")
        if actual != expected_fingerprint:
            raise ResumeMismatchError(
                "request fingerprint changed; start a new run directory "
                f"(journal={actual!r}, requested={expected_fingerprint!r})"
            )
    if manifest is not None:
        actual_manifest = header.get("manifest_sha256")
        if actual_manifest != manifest.manifest_sha256:
            raise ResumeMismatchError(
                "manifest changed; start a new run directory "
                f"(journal={actual_manifest!r}, "
                f"requested={manifest.manifest_sha256!r})"
            )
        if header.get("manifest_row_count") != manifest.row_count:
            raise ResumeMismatchError("manifest row count changed")

    known_rows = (
        {row.company_id: row for row in manifest.rows}
        if manifest is not None
        else None
    )
    for line_number, event in enumerate(events[1:], start=2):
        event_type = event.get("event_type")
        if event_type == "run_started":
            raise JournalCorruptionError(
                f"duplicate run_started event at JSONL line {line_number}"
            )
        if event_type == "request_error":
            company_id = _required_text(event, "company_id", line_number)
            if known_rows is not None and company_id not in known_rows:
                raise JournalCorruptionError(
                    f"unknown company_id {company_id!r} at JSONL line {line_number}"
                )
            if company_id in state.completed:
                raise JournalCorruptionError(
                    f"error recorded after completion for {company_id!r} "
                    f"at JSONL line {line_number}"
                )
            state.latest_errors[company_id] = event
            continue
        if event_type == "retry_requested":
            company_id = _validate_company_event(
                event, line_number, known_rows
            )
            prior = state.latest_errors.get(company_id)
            if prior is None:
                raise JournalCorruptionError(
                    f"retry request has no active failure for {company_id!r} "
                    f"at JSONL line {line_number}"
                )
            if not prior.get("retriable"):
                raise JournalCorruptionError(
                    f"retry request targets terminal failure for {company_id!r} "
                    f"at JSONL line {line_number}"
                )
            if event.get("failure_event_id") != prior.get("event_id"):
                raise JournalCorruptionError(
                    f"retry request targets a stale failure for {company_id!r} "
                    f"at JSONL line {line_number}"
                )
            if event.get("stage") != prior.get("stage"):
                raise JournalCorruptionError(
                    f"retry request stage does not match failure for "
                    f"{company_id!r} at JSONL line {line_number}"
                )
            state.retry_requests.append(event)
            state.latest_errors.pop(company_id, None)
            continue
        if event_type == "pass_a_completed":
            company_id = _validate_company_event(
                event, line_number, known_rows
            )
            if company_id in state.pass_a:
                raise JournalCorruptionError(
                    f"duplicate Pass A checkpoint for {company_id!r}"
                )
            normalized = event.get("normalized")
            PassAResult.model_validate(normalized)
            state.pass_a[company_id] = event
            latest_error = state.latest_errors.get(company_id)
            if latest_error is not None and latest_error.get("stage") == "pass_a":
                state.latest_errors.pop(company_id, None)
            continue
        if event_type == "company_completed":
            company_id = _validate_company_event(
                event, line_number, known_rows
            )
            if company_id not in state.pass_a:
                raise JournalCorruptionError(
                    f"company {company_id!r} completed before Pass A checkpoint"
                )
            if company_id in state.completed:
                raise JournalCorruptionError(
                    f"duplicate completed company event for {company_id!r}"
                )
            verdict = int(state.pass_a[company_id]["normalized"]["ai_native"])
            normalized = event.get("normalized")
            model_cls = (
                PassBAINativeResult if verdict == 1 else PassBNotAINativeResult
            )
            model_cls.model_validate(normalized)
            state.completed[company_id] = event
            state.latest_errors.pop(company_id, None)
            continue
        raise JournalCorruptionError(
            f"unknown event_type {event_type!r} at JSONL line {line_number}"
        )
    return state


def read_journal_events(
    path: str | Path,
    *,
    replay_mode: JournalReplayMode = "read_only",
) -> list[dict[str, Any]]:
    """Read one journal snapshot without mutating it by default."""
    return _read_jsonl(
        Path(path),
        replay_mode=replay_mode,
    )


def append_retry_events(
    run_dir: str | Path,
    *,
    manifest: Manifest,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    """Append durable retry requests for active retriable failures only."""
    if stage not in (None, "pass_a", "pass_b"):
        raise ValueError("retry stage must be pass_a, pass_b, or omitted")

    paths = RunArtifactPaths.from_run_dir(run_dir)
    with RunLock(paths.lock):
        state = load_journal_state(
            paths.journal,
            manifest=manifest,
            replay_mode="repair",
        )
        if state.header is None:
            raise JournalCorruptionError(
                f"run has no run_started event: {paths.journal}"
            )

        events: list[dict[str, Any]] = []
        for row in manifest.rows:
            failure = state.latest_errors.get(row.company_id)
            if failure is None or not failure.get("retriable"):
                continue
            if stage is not None and failure.get("stage") != stage:
                continue
            failure_event_id = failure.get("event_id")
            if not isinstance(failure_event_id, str) or not failure_event_id:
                raise JournalCorruptionError(
                    f"active failure for {row.company_id!r} has no event_id"
                )
            events.append(
                {
                    "event_type": "retry_requested",
                    "event_id": uuid.uuid4().hex,
                    "requested_at": datetime.now(UTC).isoformat(),
                    "company_id": row.company_id,
                    "company_name": row.company_name,
                    "input_hash": row.input_hash,
                    "stage": failure.get("stage"),
                    "category": failure.get("category"),
                    "failure_event_id": failure_event_id,
                }
            )
        _append_events_fsync(paths.journal, events)
        return events


def rebuild_derived_artifacts(
    manifest: Manifest,
    state: JournalState,
    paths: RunArtifactPaths,
    *,
    stopped: bool,
) -> dict[str, Any]:
    """Rebuild every CSV and summary from complete journal state only."""
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    professor_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in manifest.rows:
        pass_a = state.pass_a.get(row.company_id)
        complete = state.completed.get(row.company_id)
        if pass_a is not None and complete is not None:
            professor_rows.append(
                assemble_professor_row(
                    row,
                    pass_a["normalized"],
                    complete["normalized"],
                    pass_a.get("ai_native_confidence"),
                )
            )
            continue

        latest_error = state.latest_errors.get(row.company_id)
        failure: dict[str, Any] = {
            "company_id": row.company_id,
            "company_name": row.company_name,
            "stage": "pass_b" if pass_a is not None else "pass_a",
            "state": "pass_a_only" if pass_a is not None else "not_started",
        }
        if latest_error is not None:
            failure["latest_error"] = {
                key: latest_error.get(key)
                for key in (
                    "stage",
                    "attempt",
                    "category",
                    "message",
                    "retriable",
                    "will_retry",
                    "ambiguous_provider_billing",
                    "finished_at",
                )
            }
        elif stopped:
            failure["reason"] = "shutdown"
        else:
            failure["reason"] = "not_completed"
        failures.append(failure)

    _write_csv_atomic(professor_rows, paths.in_progress_csv)
    complete_count = len(professor_rows)
    all_complete = complete_count == manifest.row_count
    if all_complete:
        _write_csv_atomic(professor_rows, paths.final_csv)
    else:
        paths.final_csv.unlink(missing_ok=True)

    failure_payload = {
        "manifest_row_count": manifest.row_count,
        "completed_count": complete_count,
        "incomplete_count": len(failures),
        "stopped": stopped,
        "failures": failures,
    }
    summary_payload = {
        "manifest_sha256": manifest.manifest_sha256,
        "request_fingerprint": (
            state.header.get("request_fingerprint") if state.header else None
        ),
        "manifest_row_count": manifest.row_count,
        "pass_a_checkpoint_count": len(state.pass_a),
        "completed_count": complete_count,
        "incomplete_count": len(failures),
        "all_complete": all_complete,
        "stopped": stopped,
    }
    _write_json_atomic(failure_payload, paths.failure_summary)
    _write_json_atomic(summary_payload, paths.run_summary)
    return summary_payload


def _read_jsonl(
    path: Path,
    *,
    replay_mode: JournalReplayMode,
) -> list[dict[str, Any]]:
    if replay_mode not in ("read_only", "repair"):
        raise ValueError(f"unknown journal replay mode {replay_mode!r}")
    if not path.exists():
        return []
    data = path.read_bytes()
    if not data:
        return []

    has_final_newline = data.endswith(b"\n")
    parts = data.split(b"\n")
    if has_final_newline:
        parts.pop()
    events: list[dict[str, Any]] = []
    offset = 0
    for index, raw in enumerate(parts):
        is_unterminated_final = index == len(parts) - 1 and not has_final_newline
        if replay_mode == "read_only" and is_unterminated_final:
            return events
        if not raw.strip():
            offset += len(raw) + 1
            continue
        try:
            decoded = raw.decode("utf-8")
            event = json.loads(decoded)
            if not isinstance(event, dict):
                raise TypeError("record is not a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            if replay_mode == "repair" and is_unterminated_final:
                _truncate_and_sync(path, offset)
                return events
            raise JournalCorruptionError(
                f"malformed JSONL line {index + 1} in {path}: {exc}"
            ) from exc
        events.append(event)
        offset += len(raw) + 1

    if not has_final_newline:
        if replay_mode == "repair":
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return events


def _truncate_and_sync(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(offset)
        handle.flush()
        os.fsync(handle.fileno())


def _append_events_fsync(
    path: Path,
    events: list[Mapping[str, Any]],
) -> None:
    if not events:
        return
    payload = b"".join(
        (
            json.dumps(
                dict(event),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for event in events
    )
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _required_text(
    event: Mapping[str, Any],
    key: str,
    line_number: int,
) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value:
        raise JournalCorruptionError(
            f"event at JSONL line {line_number} has invalid {key}"
        )
    return value


def _validate_company_event(
    event: Mapping[str, Any],
    line_number: int,
    known_rows: dict[str, Any] | None,
) -> str:
    company_id = _required_text(event, "company_id", line_number)
    if known_rows is None:
        return company_id
    row = known_rows.get(company_id)
    if row is None:
        raise JournalCorruptionError(
            f"unknown company_id {company_id!r} at JSONL line {line_number}"
        )
    if event.get("input_hash") != row.input_hash:
        raise ResumeMismatchError(
            f"input hash changed for company {company_id!r}"
        )
    return company_id


def _write_csv_atomic(
    rows: list[Mapping[str, Any]],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        write_professor_csv(rows, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
