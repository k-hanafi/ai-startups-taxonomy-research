"""Resumable, budget-capped Tavily ``/extract`` runner over archive snapshots.

This is the historical analogue of ``src/tavily_crawl.py``. The job is simpler:
we already know one exact snapshot URL per company, so there is no crawl, no
fallback config, and no usage field to parse. What we keep are the reliability
layers that make a multi-hour paid run safe to close the laptop on:

* atomic resume state (``ExtractState``) + append-only fsynced JSONL
* startup healing of a crash-truncated JSONL tail
* per-call retries (429 ``Retry-After`` / transient backoff) wrapped in a longer
  outage-retry loop for multi-minute Archive/Tavily outages
* a shared sliding-window rate limiter across worker threads
* SIGINT/SIGTERM graceful stop at the next row boundary
* a call-count budget cap (basic extract bills 1 credit / 5 successes)
* heartbeats + a per-run manifest row for observability
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import (
    DEFAULT_BUDGET_CREDITS,
    DEFAULT_EXTRACT_RPM_HEADROOM,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_MAX_CONCURRENT_ROWS,
    DEFAULT_MAX_OUTAGE_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    TAVILY_EXTRACT_ENDPOINT,
    TAVILY_EXTRACT_RPM_DOCUMENTED,
    ExtractConfig,
    estimate_credits,
)
from .evidence import compact_tavily_response
from .paths import (
    EXTRACT_LOG,
    EXTRACT_STATE_JSON,
    PROJECT_ROOT,
    RUN_MANIFEST_CSV,
    SCRAPE_PROCESSED_CSV,
    SCRAPE_TARGETS_CSV,
    SNAPSHOTS_JSONL,
)
from .state import (
    ExtractState,
    completed_ids_from_jsonl,
    heal_jsonl_tail,
    processed_ids_from_csv,
    reconcile_extract_state,
)

PROCESSED_FIELDS = [
    "org_uuid", "name", "homepage_url", "snapshot_ts",
    "website_pages_used", "website_evidence",
]

MANIFEST_FIELDS = [
    "started_at", "ended_at", "attempted", "succeeded", "empty", "failed",
    "est_credits", "budget_reached", "output_jsonl_size_mb", "last_org_uuid",
    "exit_reason",
]


# ---------------------------------------------------------------------------
# API key + HTTP
# ---------------------------------------------------------------------------


def _api_key() -> str:
    load_dotenv(PROJECT_ROOT / "keys" / "tavily.env")
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TAVILY_API_KEY must be set before running the extract scrape")
    return key


class ExtractCallError(Exception):
    """Carries a normalized Tavily request error after retry handling."""

    def __init__(self, error: dict[str, Any]):
        super().__init__(str(error))
        self.error = error


class ExtractInterrupted(Exception):
    """Raised when the user stops the run during a rate-limit wait."""


class _SlidingWindowLimiter:
    """Thread-safe sliding window limiting POSTs to the extract endpoint."""

    def __init__(self, max_calls_per_minute: float, *, window_seconds: float = 60.0) -> None:
        self._max = max(1.0, float(max_calls_per_minute))
        self._window = float(window_seconds)
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, stop_check: Callable[[], bool] | None = None) -> None:
        while True:
            if stop_check is not None and stop_check():
                raise ExtractInterrupted("stop requested during rate wait")
            sleep_for: float | None = None
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < int(self._max):
                    self._timestamps.append(now)
                    return
                oldest = self._timestamps[0]
                sleep_for = max(0.001, self._window - (now - oldest) + 0.001)
            end = time.monotonic() + sleep_for
            while time.monotonic() < end:
                if stop_check is not None and stop_check():
                    raise ExtractInterrupted("stop requested during rate wait")
                time.sleep(min(0.2, end - time.monotonic()))


def call_tavily_extract(
    url: str,
    config: ExtractConfig,
    api_key: str,
    *,
    rate_limiter: _SlidingWindowLimiter | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Call Tavily ``/extract`` for one URL with stdlib HTTP (no SDK dep)."""
    if rate_limiter is not None:
        rate_limiter.acquire(stop_check)
    data = json.dumps(config.request_payload(url)).encode("utf-8")
    request = urllib.request.Request(
        TAVILY_EXTRACT_ENDPOINT,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=config.timeout + 10) as response:
        return json.loads(response.read().decode("utf-8"))


def _has_usable_results(response: dict[str, Any]) -> bool:
    results = response.get("results")
    if not isinstance(results, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("raw_content", "")).strip()
        for item in results
    )


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ExtractCallError):
        return exc.error
    if isinstance(exc, urllib.error.HTTPError):
        body = exc.read().decode("utf-8", errors="replace")
        return {"type": "HTTPError", "status": exc.code, "body": body}
    return {"type": type(exc).__name__, "message": str(exc)}


def _error_status(error: dict[str, Any]) -> tuple[str, bool]:
    """Return (`status_label`, `retryable`) for a captured error payload."""
    status = error.get("status")
    error_type = str(error.get("type", ""))
    if status in {401, 403}:
        return "auth_error", False
    if status in {400, 422}:
        return "request_error", False
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return "transient_error", True
    if error_type in {"TimeoutError", "URLError", "ConnectionError"}:
        return "transient_error", True
    return "error", False


def _sleep_retry_after_or_backoff(
    exc: urllib.error.HTTPError,
    *,
    attempt: int,
    config: ExtractConfig,
    stop_check: Callable[[], bool] | None,
    stop_sleep: Callable[[float], bool] | None,
) -> None:
    raw = exc.headers.get("Retry-After") if exc.headers else None
    if raw is not None:
        try:
            wait_s = float(raw)
        except (TypeError, ValueError):
            wait_s = 60.0
    else:
        wait_s = config.retry_backoff_seconds * (2 ** attempt)
    wait_s = max(0.0, wait_s)
    if stop_sleep is not None:
        stop_sleep(wait_s)
    else:
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            if stop_check is not None and stop_check():
                return
            time.sleep(min(0.2, end - time.monotonic()))


def _call_with_retries(
    url: str,
    config: ExtractConfig,
    api_key: str,
    *,
    rate_limiter: _SlidingWindowLimiter | None = None,
    stop_check: Callable[[], bool] | None = None,
    stop_sleep: Callable[[float], bool] | None = None,
) -> dict[str, Any]:
    last_error: dict[str, Any] | None = None
    for attempt in range(max(1, config.max_retries + 1)):
        try:
            return call_tavily_extract(
                url, config, api_key, rate_limiter=rate_limiter, stop_check=stop_check,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            error: dict[str, Any] = {"type": "HTTPError", "status": exc.code, "body": body}
            _, retryable = _error_status(error)
            last_error = error
            if not retryable or attempt >= config.max_retries:
                break
            _sleep_retry_after_or_backoff(
                exc, attempt=attempt, config=config, stop_check=stop_check, stop_sleep=stop_sleep,
            )
            if stop_check is not None and stop_check():
                break
        except Exception as exc:
            error = _error_payload(exc)
            _, retryable = _error_status(error)
            last_error = error
            if not retryable or attempt >= config.max_retries:
                break
            backoff = config.retry_backoff_seconds * (2 ** attempt)
            if stop_sleep is not None:
                stop_sleep(backoff)
            else:
                time.sleep(backoff)
    raise ExtractCallError(last_error or {"type": "UnknownError", "message": "unknown"})


class _GracefulStopController:
    """Catch SIGINT/SIGTERM, set a flag, and wake interruptible sleeps."""

    def __init__(self) -> None:
        self.stop_requested = False
        self._previous_handlers: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True

    def __enter__(self) -> "_GracefulStopController":
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(ValueError, OSError):
                self._previous_handlers[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *_exc: Any) -> None:
        for sig, prev in self._previous_handlers.items():
            with suppress(ValueError, OSError):
                signal.signal(sig, prev)

    def sleep(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, waking on stop. True if full time elapsed."""
        if seconds <= 0:
            return not self.stop_requested
        end = time.monotonic() + seconds
        while not self.stop_requested:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.5, remaining))
        return False


@dataclass
class _RowOutcome:
    record: dict[str, Any]
    status: str
    ok: bool
    retryable: bool
    pages_used: str
    evidence: str
    transient_failure: bool


def _evidence_from_response(response: dict[str, Any], homepage_url: str) -> tuple[str, str]:
    """Clean an extract response, rewriting the archive URL back to the homepage.

    Tavily returns the ``web.archive.org/.../id_/...`` URL it fetched. We swap it
    for the original homepage so the ``URL:`` line and page label in the evidence
    match the live crawl's format exactly — keeping the comparison clean.
    """
    results = response.get("results")
    if not isinstance(results, list):
        return "", ""
    normalized = {
        "results": [
            {"url": homepage_url, "raw_content": r.get("raw_content", "")}
            for r in results
            if isinstance(r, dict)
        ]
    }
    return compact_tavily_response(normalized)


def _process_single_row(
    *,
    target: dict[str, str],
    cfg: ExtractConfig,
    api_key: str,
    rate_limiter: _SlidingWindowLimiter | None,
    stop_check: Callable[[], bool] | None,
    stop_sleep: Callable[[float], bool] | None,
) -> _RowOutcome:
    org_uuid = str(target.get("org_uuid", "")).strip()
    homepage_url = str(target.get("homepage_url", "")).strip()
    snapshot_url = str(target.get("snapshot_url", "")).strip()
    snapshot_ts = str(target.get("closest_ts", "")).strip()
    record: dict[str, Any] = {
        "org_uuid": org_uuid,
        "name": target.get("name", ""),
        "homepage_url": homepage_url,
        "snapshot_url": snapshot_url,
        "snapshot_ts": snapshot_ts,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = _call_with_retries(
            snapshot_url, cfg, api_key,
            rate_limiter=rate_limiter, stop_check=stop_check, stop_sleep=stop_sleep,
        )
    except Exception as exc:
        error = _error_payload(exc)
        status, retryable = _error_status(error)
        record.update({"ok": False, "status": status, "retryable": retryable, "error": error})
        return _RowOutcome(record, status, False, retryable, "", "", retryable)

    if not _has_usable_results(response):
        # Tavily reached the Archive but got nothing usable (often a real gap, or
        # the Archive throttling Tavily's fetch). Terminal+non-retryable so resume
        # skips it; delete the JSONL line to force a retry later if desired.
        record.update({"ok": True, "status": "empty_results", "retryable": False,
                       "response": response})
        return _RowOutcome(record, "empty_results", True, False, "", "", False)

    pages_used, evidence = _evidence_from_response(response, homepage_url)
    if not evidence:
        record.update({"ok": True, "status": "thin_evidence", "retryable": False,
                       "response": response})
        return _RowOutcome(record, "thin_evidence", True, False, "", "", False)

    record.update({"ok": True, "status": "success", "retryable": False, "response": response})
    return _RowOutcome(record, "success", True, False, pages_used, evidence, False)


def _run_row_with_outage(
    *,
    target: dict[str, str],
    cfg: ExtractConfig,
    api_key: str,
    rate_limiter: _SlidingWindowLimiter | None,
    stop: _GracefulStopController,
    max_outage_seconds: float,
    outage_backoff_min_seconds: float,
    outage_backoff_max_seconds: float,
) -> _RowOutcome:
    outage_attempt = 0
    outage_started = time.monotonic()
    while True:
        outcome = _process_single_row(
            target=target, cfg=cfg, api_key=api_key, rate_limiter=rate_limiter,
            stop_check=lambda: stop.stop_requested, stop_sleep=stop.sleep,
        )
        if not outcome.transient_failure:
            return outcome
        if time.monotonic() - outage_started >= max_outage_seconds or stop.stop_requested:
            return outcome
        sleep_secs = min(outage_backoff_min_seconds * (2 ** outage_attempt),
                         outage_backoff_max_seconds)
        if not stop.sleep(sleep_secs):
            return outcome
        outage_attempt += 1


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN guard
        return "n/a"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    return f"{hours}h {rem // 60:02d}m"


def _emit_heartbeat(
    *, log_path: Path, processed: int, total: int, succeeded: int, empty: int,
    failed: int, est_credits: float, elapsed_seconds: float, last_org_uuid: str,
) -> None:
    rate = (processed / elapsed_seconds * 60.0) if elapsed_seconds > 0 else 0.0
    remaining = max(total - processed, 0)
    eta = (remaining / rate * 60.0) if rate > 0 else float("inf")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line1 = (f"[{ts}] processed={processed:,}/{total:,}  "
             f"ok={succeeded:,} empty={empty:,} fail={failed:,}")
    line2 = (f"   est_credits={est_credits:,.1f}  rate={rate:,.1f}/min  "
             f"ETA={_format_eta(eta)}  last={last_org_uuid or 'n/a'}")
    print(line1, file=sys.stderr, flush=True)
    print(line2, file=sys.stderr, flush=True)
    with suppress(OSError):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line1 + "\n" + line2 + "\n")


def processed_row_from_snapshot_record(record: dict[str, Any]) -> dict[str, str] | None:
    """Build a scrape_processed.csv row from one successful JSONL snapshot record."""
    if record.get("ok") is not True or record.get("status") != "success":
        return None
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    homepage_url = str(record.get("homepage_url", ""))
    pages_used, evidence = _evidence_from_response(response, homepage_url)
    if not evidence.strip():
        return None
    return {
        "org_uuid": str(record.get("org_uuid", "")).strip(),
        "name": str(record.get("name", "")),
        "homepage_url": homepage_url,
        "snapshot_ts": str(record.get("snapshot_ts", "")),
        "website_pages_used": pages_used,
        "website_evidence": evidence,
    }


def backfill_processed_csv(jsonl_path: Path, processed_csv: Path) -> int:
    """Append success rows from JSONL that are missing from the processed CSV."""
    existing = processed_ids_from_csv(processed_csv)
    to_add: list[dict[str, str]] = []
    if not jsonl_path.exists():
        return 0

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = processed_row_from_snapshot_record(record)
            if row is None:
                continue
            org_uuid = row["org_uuid"]
            if not org_uuid or org_uuid in existing:
                continue
            to_add.append(row)
            existing.add(org_uuid)

    for row in to_add:
        _append_processed_row(processed_csv, row, fsync=True)
    return len(to_add)


def _append_processed_row(
    processed_csv: Path, row: dict[str, Any], *, fsync: bool = False,
) -> None:
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not processed_csv.exists() or processed_csv.stat().st_size == 0
    with processed_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROCESSED_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PROCESSED_FIELDS})
        if fsync:
            fh.flush()
            os.fsync(fh.fileno())


def _append_run_manifest(manifest_csv: Path, row: dict[str, Any]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_csv.exists() or manifest_csv.stat().st_size == 0
    with manifest_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def _preflight_checks(*, pending_count: int, output_jsonl: Path, min_free_disk_gb: float) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    probe = output_jsonl.parent / ".extract_writable_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Output dir {output_jsonl.parent} not writable: {exc}") from exc
    finally:
        with suppress(OSError):
            probe.unlink()
    if min_free_disk_gb > 0:
        free_gb = shutil.disk_usage(output_jsonl.parent).free / (1024 ** 3)
        if free_gb < min_free_disk_gb:
            raise RuntimeError(
                f"Free disk {free_gb:.2f} GB < required {min_free_disk_gb:.2f} GB"
            )
    if pending_count <= 0:
        raise RuntimeError("No pending targets. Did build_targets.py run, and is everything done?")


@dataclass(frozen=True)
class ExtractRunReport:
    attempted: int
    succeeded: int
    empty: int
    failed: int
    skipped_existing: int
    est_credits: float
    budget_reached: bool
    errors_by_status: dict[str, int] = field(default_factory=dict)
    exit_reason: str = "completed"

    def format_report(self) -> str:
        lines = [
            "WAYBACK EXTRACT RUN REPORT",
            f"  Attempted this run:   {self.attempted:,}",
            f"  Succeeded:            {self.succeeded:,}",
            f"  Empty/thin:           {self.empty:,}",
            f"  Failed:               {self.failed:,}",
            f"  Skipped existing:     {self.skipped_existing:,}",
            f"  Est. credits (run):   {self.est_credits:,.1f}",
            f"  Budget reached:       {self.budget_reached}",
            f"  Exit reason:          {self.exit_reason}",
        ]
        if self.errors_by_status:
            lines.append("  By status:")
            for status, n in sorted(self.errors_by_status.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"    {status:<22} {n:,}")
        return "\n".join(lines)


def run_extract(
    targets_csv: str | Path = SCRAPE_TARGETS_CSV,
    output_jsonl: str | Path = SNAPSHOTS_JSONL,
    state_json: str | Path = EXTRACT_STATE_JSON,
    config: ExtractConfig | None = None,
    *,
    processed_csv: str | Path = SCRAPE_PROCESSED_CSV,
    budget_credits: float = DEFAULT_BUDGET_CREDITS,
    max_companies: int | None = None,
    heartbeat_every: int = DEFAULT_HEARTBEAT_EVERY,
    heartbeat_log: str | Path = EXTRACT_LOG,
    manifest_csv: str | Path = RUN_MANIFEST_CSV,
    max_concurrent_rows: int = DEFAULT_MAX_CONCURRENT_ROWS,
    extract_rpm: float | None = None,
    extract_rpm_headroom: float = DEFAULT_EXTRACT_RPM_HEADROOM,
    max_outage_seconds: float = DEFAULT_MAX_OUTAGE_SECONDS,
    outage_backoff_min_seconds: float = DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    outage_backoff_max_seconds: float = DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    min_free_disk_gb: float = 0.0,
) -> ExtractRunReport:
    """Run a resumable, budget-capped extract over the frozen targets list."""
    cfg = config or ExtractConfig()
    state_path = Path(state_json)
    out_path = Path(output_jsonl)
    processed_path = Path(processed_csv)
    heartbeat_path = Path(heartbeat_log)
    manifest_path = Path(manifest_csv)

    with Path(targets_csv).open(encoding="utf-8", newline="") as f:
        all_targets = list(csv.DictReader(f))

    healed = heal_jsonl_tail(out_path)
    if healed:
        print(f"[extract] healed {healed} trailing bytes from {out_path}", file=sys.stderr, flush=True)

    state = ExtractState.load(state_path)
    reconcile_extract_state(state, out_path)
    state.save(state_path)
    backfilled = backfill_processed_csv(out_path, processed_path)
    if backfilled:
        print(
            f"[extract] backfilled {backfilled} row(s) into {processed_path}",
            file=sys.stderr,
            flush=True,
        )
    completed_ids = completed_ids_from_jsonl(out_path)

    pending: list[dict[str, str]] = []
    skipped_existing = 0
    for t in all_targets:
        if str(t.get("org_uuid", "")).strip() in completed_ids:
            skipped_existing += 1
            continue
        if not str(t.get("snapshot_url", "")).strip():
            continue
        pending.append(t)
    if max_companies is not None:
        pending = pending[: int(max_companies)]

    _preflight_checks(pending_count=len(pending), output_jsonl=out_path,
                      min_free_disk_gb=min_free_disk_gb)

    if extract_rpm is not None and extract_rpm <= 0:
        effective_rpm: float | None = None
    elif extract_rpm is not None:
        effective_rpm = float(extract_rpm)
    else:
        effective_rpm = TAVILY_EXTRACT_RPM_DOCUMENTED * float(extract_rpm_headroom)
    rate_limiter = _SlidingWindowLimiter(effective_rpm) if effective_rpm else None

    api_key = _api_key()
    workers = max(1, int(max_concurrent_rows))
    heartbeat_total = skipped_existing + len(pending)

    attempted = 0
    succeeded_run = 0
    empty_run = 0
    failed_run = 0
    credits_start = estimate_credits(state.successful, extract_depth=cfg.extract_depth)
    budget_reached = False
    exit_reason = "completed"
    errors: Counter[str] = Counter()
    started = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()

    print(f"[extract] targets={len(all_targets):,} pending={len(pending):,} "
          f"skipped={skipped_existing:,} workers={workers} "
          f"rpm_cap={effective_rpm or 'off'}", file=sys.stderr, flush=True)

    rows_written = [0]

    def persist(out: Any, outcome: _RowOutcome) -> None:
        nonlocal succeeded_run, empty_run, failed_run
        org_uuid = str(outcome.record.get("org_uuid", "")).strip()
        state.last_org_uuid = org_uuid
        if outcome.ok and outcome.status == "success":
            state.successful += 1
            succeeded_run += 1
            completed_ids.add(org_uuid)
        elif outcome.ok:  # empty_results / thin_evidence — terminal, no evidence
            state.empty += 1
            empty_run += 1
            errors[outcome.status] += 1
            completed_ids.add(org_uuid)
        else:
            state.failed += 1
            failed_run += 1
            errors[outcome.status] += 1
            if not outcome.retryable:
                completed_ids.add(org_uuid)

        if outcome.status == "success":
            _append_processed_row(processed_path, {
                "org_uuid": org_uuid,
                "name": outcome.record.get("name", ""),
                "homepage_url": outcome.record.get("homepage_url", ""),
                "snapshot_ts": outcome.record.get("snapshot_ts", ""),
                "website_pages_used": outcome.pages_used,
                "website_evidence": outcome.evidence,
            }, fsync=True)

        out.write(json.dumps(outcome.record, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
        state.save(state_path)

        rows_written[0] += 1
        if heartbeat_every > 0 and rows_written[0] % heartbeat_every == 0:
            _emit_heartbeat(
                log_path=heartbeat_path,
                processed=rows_written[0] + skipped_existing,
                total=heartbeat_total, succeeded=succeeded_run, empty=empty_run,
                failed=failed_run,
                est_credits=estimate_credits(
                    state.successful, extract_depth=cfg.extract_depth,
                ) - credits_start,
                elapsed_seconds=time.monotonic() - started, last_org_uuid=state.last_org_uuid,
            )

    with _GracefulStopController() as stop, out_path.open("a", encoding="utf-8") as out:
        deque_lock = threading.Lock()
        write_lock = threading.Lock()
        rows_deque: deque[dict[str, str]] = deque(pending)
        worker_errors: list[BaseException] = []
        in_flight_rows = 0

        def _estimated_run_credits() -> float:
            return estimate_credits(
                state.successful + in_flight_rows,
                extract_depth=cfg.extract_depth,
            ) - credits_start

        def run_worker() -> None:
            nonlocal attempted, budget_reached, exit_reason, in_flight_rows
            try:
                while True:
                    if stop.stop_requested:
                        return
                    target: dict[str, str] | None = None
                    with deque_lock:
                        if not rows_deque:
                            return
                        if _estimated_run_credits() >= budget_credits:
                            budget_reached = True
                            exit_reason = "budget_reached"
                            return
                        target = rows_deque.popleft()
                        attempted += 1
                        in_flight_rows += 1
                    try:
                        outcome = _run_row_with_outage(
                            target=target, cfg=cfg, api_key=api_key, rate_limiter=rate_limiter,
                            stop=stop, max_outage_seconds=max_outage_seconds,
                            outage_backoff_min_seconds=outage_backoff_min_seconds,
                            outage_backoff_max_seconds=outage_backoff_max_seconds,
                        )
                    except ExtractInterrupted:
                        with deque_lock:
                            rows_deque.appendleft(target)
                            attempted -= 1
                            in_flight_rows -= 1
                        return
                    except Exception as exc:
                        with deque_lock:
                            in_flight_rows -= 1
                        worker_errors.append(exc)
                        return
                    with write_lock:
                        persist(out, outcome)
                    with deque_lock:
                        in_flight_rows -= 1
                    if stop.stop_requested:
                        exit_reason = "user_interrupt"
                        return
            except ExtractInterrupted:
                return
            except Exception as exc:
                worker_errors.append(exc)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in [pool.submit(run_worker) for _ in range(workers)]:
                fut.result()
        if worker_errors:
            raise worker_errors[0]

    state.save(state_path)
    if max_companies is not None and exit_reason == "completed" and attempted == max_companies:
        exit_reason = "max_companies"

    jsonl_mb = 0.0
    with suppress(OSError):
        jsonl_mb = out_path.stat().st_size / (1024 * 1024)

    report = ExtractRunReport(
        attempted=attempted, succeeded=succeeded_run, empty=empty_run, failed=failed_run,
        skipped_existing=skipped_existing,
        est_credits=estimate_credits(state.successful, extract_depth=cfg.extract_depth)
        - credits_start,
        budget_reached=budget_reached, errors_by_status=dict(errors), exit_reason=exit_reason,
    )
    with suppress(OSError):
        _append_run_manifest(manifest_path, {
            "started_at": started_iso, "ended_at": datetime.now(timezone.utc).isoformat(),
            "attempted": report.attempted, "succeeded": report.succeeded,
            "empty": report.empty, "failed": report.failed,
            "est_credits": f"{report.est_credits:.2f}", "budget_reached": report.budget_reached,
            "output_jsonl_size_mb": f"{jsonl_mb:.2f}", "last_org_uuid": state.last_org_uuid,
            "exit_reason": report.exit_reason,
        })
    return report
