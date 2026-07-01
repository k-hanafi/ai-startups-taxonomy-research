"""Resumable, budget-capped Tavily ``/extract`` runner over dead-cohort snapshots.

Survivorship Stage C. For every company we already know one pre-death Wayback
snapshot, so a single-page ``/extract`` is all we need for the classifier. The
live cohort was scraped with a multi-page ``/crawl``, but ``/crawl`` fights the
Internet Archive's per-IP playback limits (bursts of link-following fetches get
throttled mid-crawl and come back empty), so on archived sites we drop to one
homepage extract and note the single-page scope as a methodology limitation.

Per company we fetch the archived homepage, trying the ``if_`` iframe snapshot the
target list already carries first, then the ``id_`` raw-bytes snapshot as a second
chance. We strip any residual Wayback chrome, rewrite the archived URL back to the
origin homepage, and clean it with the SAME vendored evidence cleaner the live +
2023 cohorts used — so only the evidence itself differs across cohorts (the whole
point of the fair-comparison design).

Reuses the extract engine's reliability harness (``wayback_machine.extract``):
graceful stop, heartbeat, per-run manifest, atomic resume state, JSONL tail-healing,
sliding-window rate limiter, and a call-count budget cap. Keeps the failure-reason
instrumentation (``rate_limited`` vs ``no_archive_content`` vs transient/network) so
a resumed run re-attempts only the recoverable infrastructure failures.

Artifacts intentionally keep their crawl-era names (``crawl_dead.jsonl``,
``scrape_processed_dead.csv``, ``crawl_state_dead.json``) so the companies already
scraped resume cleanly and the downstream classifier input is unchanged.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_BUDGET_CREDITS,
    DEFAULT_EXTRACT_RPM_HEADROOM,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_MAX_OUTAGE_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    TAVILY_EXTRACT_RPM_DOCUMENTED,
    ExtractConfig,
    estimate_credits,
)
from .evidence import compact_tavily_response
from .extract import (
    ExtractInterrupted,
    _api_key,
    _append_processed_row,
    _append_run_manifest,
    _emit_heartbeat,
    _error_payload,
    _GracefulStopController,
    _has_usable_results,
    _preflight_checks,
    _SlidingWindowLimiter,
    call_tavily_extract,
)
from .paths import (
    CRAWL_DEAD_JSONL,
    CRAWL_DEAD_LOG,
    CRAWL_STATE_DEAD_JSON,
    RUN_MANIFEST_DEAD_CSV,
    SCRAPE_PROCESSED_DEAD_CSV,
    SCRAPE_TARGETS_DEAD_CSV,
)
from .state import ExtractState, heal_jsonl_tail, processed_ids_from_csv
from .tavily_archive_lab import _strip_wayback_chrome, archive_url

# One /extract = one archive page-fetch (a second only when the first snapshot is
# empty), so this is far less bursty than a 5-page /crawl and the Archive's
# playback limit (~480 safe req/min per IP) is never the binding constraint.
# Tavily's extract endpoint (100 RPM) is the real cap, so we can run many more
# workers than the crawl did (4) and still stay well under both ceilings.
DEFAULT_EXTRACT_DEAD_CONCURRENCY = 12

# New extract rows only ever write "success"; the two legacy labels are kept so a
# resume over the pre-migration crawl JSONL still tallies those wins correctly.
_SUCCESS_STATUSES = frozenset({"success", "success_fallback", "success_extract_fallback"})

# Controlled vocabulary for *why* a company yielded no usable evidence. Derived
# purely from the HTTP status / error type of each extract attempt, so an operator
# can tell a recoverable infrastructure problem (rate limit / outage / network)
# apart from a permanent property of the company (the Archive simply has nothing).
RATE_LIMITED = "rate_limited"
NO_ARCHIVE_CONTENT = "no_archive_content"
TRANSIENT_ERROR = "transient_error"
NETWORK_ERROR = "network_error"
UNKNOWN_FAILURE = "unknown"
# Pre-instrumentation rows recorded a bare ``empty_results`` with no attempts; the
# summary tool shows them under this bucket so old and new runs never blur.
LEGACY_EMPTY = "legacy_empty"

# Failure reasons that are an infrastructure problem, not a content gap: a resumed
# run should re-attempt these, so rows carrying them are written ``retryable=True``.
_RETRYABLE_FAILURE_REASONS = frozenset({RATE_LIMITED, TRANSIENT_ERROR, NETWORK_ERROR})

# Substrings that flag a rate-limit/quota error even when the HTTP status is not a
# clean 429 (some Tavily/Archive throttles surface as a 200/4xx body message).
_RATE_LIMIT_HINTS = ("rate limit", "ratelimit", "too many requests", "quota")
# HTTP statuses treated as transient/retryable.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 425, 500, 502, 503, 504})
# Exception type names that mean the request never reached a usable response.
_NETWORK_ERROR_TYPES = frozenset({"TimeoutError", "URLError", "ConnectionError", "timeout"})

csv.field_size_limit(1_000_000_000)


def _attempt_ok(phase: str) -> dict[str, Any]:
    """Compact record of one extract call that returned HTTP 200."""
    return {"phase": phase, "http_status": 200}


def _attempt_from_error(phase: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSONL-safe record of one FAILED extract call.

    ``payload`` is whatever ``_error_payload`` produced (HTTP status + body, or an
    exception type + message). We keep only the small, auditable fields and decide
    the rate-limit flag here (from status 429 or a hint in the body/message) so the
    summary never has to re-parse a large error body.
    """
    text = f"{payload.get('body', '')} {payload.get('message', '')}".lower()
    rate_limited = payload.get("status") == 429 or any(h in text for h in _RATE_LIMIT_HINTS)
    attempt: dict[str, Any] = {
        "phase": phase,
        "error_type": str(payload.get("type", "") or "error"),
    }
    if payload.get("status") is not None:
        attempt["http_status"] = payload["status"]
    if rate_limited:
        attempt["rate_limited"] = True
    return attempt


def _classify_failure_reason(attempts: list[dict[str, Any]]) -> tuple[str, bool]:
    """Derive ``(failure_reason, retryable)`` from the recorded attempts.

    Priority is most-actionable first: an explicit rate limit anywhere wins, then
    network/transient errors. ``no_archive_content`` is reserved for the case where
    every call succeeded (HTTP 200) yet produced nothing usable — a genuine gap.
    Anything errored but unclassifiable is ``unknown`` (terminal).
    """
    errored = [a for a in attempts if a.get("error_type")]
    if any(a.get("rate_limited") or a.get("http_status") == 429 for a in errored):
        return RATE_LIMITED, True
    if any(a.get("error_type") in _NETWORK_ERROR_TYPES for a in errored):
        return NETWORK_ERROR, True
    if any(a.get("http_status") in _TRANSIENT_HTTP_STATUSES for a in errored):
        return TRANSIENT_ERROR, True
    if errored:
        return UNKNOWN_FAILURE, False
    return NO_ARCHIVE_CONTENT, False


@dataclass
class _ScanResult:
    completed_ids: set[str]
    successful: int
    empty: int
    failed: int
    credits: float
    failure_reasons: Counter[str]


def _row_failure_reason(obj: dict[str, Any]) -> str:
    """Best-effort failure bucket for a non-success JSONL row (handles legacy).

    New rows carry an explicit ``failure_reason``. Pre-instrumentation rows only
    have ``status=="empty_results"`` — surface those as ``legacy_empty`` so the two
    eras are never silently merged. Any other status falls back to itself.
    """
    reason = str(obj.get("failure_reason", "")).strip()
    if reason:
        return reason
    status = str(obj.get("status", "")).strip()
    if status == "empty_results":
        return LEGACY_EMPTY
    return status or UNKNOWN_FAILURE


def _scan_jsonl(path: str | Path) -> _ScanResult:
    """One pass over the JSONL to rebuild resume + budget state.

    ``completed_ids`` = anything terminal (ok, or a non-retryable error), so a
    resumed run re-extracts only transient failures (rate-limited / outage /
    network empties, which are written ``ok=False, retryable=True``). Credits are
    summed from the recorded per-row usage so the budget cap stays cumulative
    across resumes, and ``failure_reasons`` tallies WHY rows yielded no evidence.
    """
    result = _ScanResult(set(), 0, 0, 0, 0.0, Counter())
    p = Path(path)
    if not p.exists():
        return result
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            org_uuid = str(obj.get("org_uuid", "")).strip()
            ok = obj.get("ok")
            status = str(obj.get("status", ""))
            legacy_empty = status == "empty_results" and not str(
                obj.get("failure_reason", "")
            ).strip()
            if org_uuid and not legacy_empty and (ok is True or obj.get("retryable") is False):
                result.completed_ids.add(org_uuid)
            if ok is True:
                if status in _SUCCESS_STATUSES:
                    result.successful += 1
                elif not legacy_empty:
                    result.empty += 1
                    result.failure_reasons[_row_failure_reason(obj)] += 1
                else:
                    result.failure_reasons[_row_failure_reason(obj)] += 1
            elif ok is False:
                result.failed += 1
                result.failure_reasons[_row_failure_reason(obj)] += 1
            with suppress(TypeError, ValueError):
                result.credits += float(obj.get("usage_credits", 0.0) or 0.0)
    return result


@dataclass
class _RowOutcome:
    record: dict[str, Any]
    status: str
    ok: bool
    retryable: bool
    pages_used: str
    evidence: str
    credits_added: float
    transient_failure: bool


def _evidence_from_extract_response(
    response: dict[str, Any],
    homepage_url: str,
) -> tuple[str, str]:
    """Clean a single-page extract response to live-cohort evidence format."""
    results = response.get("results")
    if not isinstance(results, list):
        return "", ""
    normalized = {
        "results": [
            {
                "url": homepage_url,
                "raw_content": _strip_wayback_chrome(str(r.get("raw_content", ""))),
            }
            for r in results
            if isinstance(r, dict)
        ]
    }
    return compact_tavily_response(normalized)


def _extract_candidates(target: dict[str, str]) -> tuple[str, ...]:
    """Snapshot URLs to try for one company, in order (deduped, non-empty).

    The target list already carries an ``if_`` iframe snapshot (toolbar stripped,
    the format validated during the crawl-era testing) — try that first, then the
    ``id_`` raw-bytes snapshot of the same capture as a second chance.
    """
    snapshot_url = str(target.get("snapshot_url", "")).strip()
    homepage_url = str(target.get("homepage_url", "")).strip()
    snapshot_ts = str(target.get("closest_ts", "")).strip()
    id_url = (
        archive_url(homepage_url, snapshot_ts, "id_")
        if homepage_url and snapshot_ts
        else ""
    )
    ordered: list[str] = []
    for url in (snapshot_url, id_url):
        if url and url not in ordered:
            ordered.append(url)
    return tuple(ordered)


def _process_single_row(
    *,
    target: dict[str, str],
    cfg: ExtractConfig,
    api_key: str,
    rate_limiter: _SlidingWindowLimiter | None,
    stop_check: Callable[[], bool] | None,
    stop_sleep: Callable[[float], bool] | None,
) -> _RowOutcome:
    """Extract one company's archived homepage (if_ snapshot, then id_)."""
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
    credits = 0.0
    attempts: list[dict[str, Any]] = []
    # Per successful extraction: 1 credit / 5 (basic) or 2 credits / 5 (advanced).
    credit_per_extract = estimate_credits(1, extract_depth=cfg.extract_depth)
    for url in _extract_candidates(target):
        try:
            response = call_tavily_extract(
                url, cfg, api_key, rate_limiter=rate_limiter, stop_check=None,
            )
        except ExtractInterrupted:
            raise  # stop hit during a rate-limit wait: requeue, don't record a failure
        except Exception as exc:  # noqa: BLE001 — normalize any Tavily/network error
            attempts.append(_attempt_from_error("extract", _error_payload(exc)))
            continue
        credits += credit_per_extract
        attempts.append(_attempt_ok("extract"))
        if not _has_usable_results(response):
            continue
        pages_used, evidence = _evidence_from_extract_response(response, homepage_url)
        if evidence.strip():
            record.update({
                "ok": True, "status": "success", "retryable": False,
                "usage_credits": credits,
                "website_pages_used": pages_used, "website_evidence": evidence,
            })
            return _RowOutcome(record, "success", True, False, pages_used, evidence, credits, False)

    # No usable evidence from any snapshot. Diagnose WHY from the attempts: a rate
    # limit / outage / network error is recoverable (retryable; recorded ok=False so
    # a resume retries it), while a clean-200-but-empty is a genuine content gap
    # (terminal). ``transient_failure`` also drives the in-run outage-retry loop.
    failure_reason, retryable = _classify_failure_reason(attempts)
    if retryable:
        record.update({"ok": False, "status": failure_reason, "retryable": True,
                       "usage_credits": credits, "failure_reason": failure_reason,
                       "attempts": attempts})
        return _RowOutcome(record, failure_reason, False, True, "", "", credits, True)
    record.update({"ok": True, "status": "empty_results", "retryable": False,
                   "usage_credits": credits, "failure_reason": failure_reason,
                   "attempts": attempts})
    return _RowOutcome(record, "empty_results", True, False, "", "", credits, False)


def _processed_row_from_dead_record(record: dict[str, Any]) -> dict[str, str] | None:
    """Build a scrape_processed_dead.csv row from one successful JSONL record."""
    if record.get("ok") is not True or record.get("status") not in _SUCCESS_STATUSES:
        return None
    evidence = str(record.get("website_evidence", ""))
    if not evidence.strip():
        return None
    return {
        "org_uuid": str(record.get("org_uuid", "")).strip(),
        "name": str(record.get("name", "")),
        "homepage_url": str(record.get("homepage_url", "")),
        "snapshot_ts": str(record.get("snapshot_ts", "")),
        "website_pages_used": str(record.get("website_pages_used", "")),
        "website_evidence": evidence,
    }


def backfill_processed_dead_csv(jsonl_path: Path, processed_csv: Path) -> int:
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
            row = _processed_row_from_dead_record(record)
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
    """Retry a row through multi-minute Tavily/Archive outages, honoring stop."""
    outage_attempt = 0
    outage_started = time.monotonic()
    credits_across_attempts = 0.0
    attempts_across_attempts: list[dict[str, Any]] = []

    def _attach_retry_accounting(outcome: _RowOutcome) -> _RowOutcome:
        """Carry paid work from earlier outage attempts into the final record."""
        if credits_across_attempts:
            outcome.credits_added = credits_across_attempts
            outcome.record["usage_credits"] = credits_across_attempts
        if attempts_across_attempts and "attempts" in outcome.record:
            outcome.record["attempts"] = attempts_across_attempts
        return outcome

    while True:
        outcome = _process_single_row(
            target=target, cfg=cfg, api_key=api_key, rate_limiter=rate_limiter,
            stop_check=lambda: stop.stop_requested, stop_sleep=stop.sleep,
        )
        credits_across_attempts += outcome.credits_added
        attempts = outcome.record.get("attempts")
        if isinstance(attempts, list):
            attempts_across_attempts.extend(
                attempt for attempt in attempts if isinstance(attempt, dict)
            )
        if not outcome.transient_failure:
            return _attach_retry_accounting(outcome)
        if credits_across_attempts > 0:
            # Once Tavily has billed this company, persist the retryable row before
            # any more retries. That keeps the append-only log and the global
            # budget counter current, even during a long Archive/Tavily outage.
            return _attach_retry_accounting(outcome)
        if time.monotonic() - outage_started >= max_outage_seconds or stop.stop_requested:
            return _attach_retry_accounting(outcome)
        sleep_secs = min(outage_backoff_min_seconds * (2 ** outage_attempt),
                         outage_backoff_max_seconds)
        if not stop.sleep(sleep_secs):
            return _attach_retry_accounting(outcome)
        outage_attempt += 1


@dataclass(frozen=True)
class ExtractDeadRunReport:
    attempted: int
    succeeded: int
    empty: int
    failed: int
    skipped_existing: int
    credits_used_this_run: float
    total_credits: float
    budget_reached: bool
    errors_by_status: dict[str, int]
    exit_reason: str

    def format_report(self) -> str:
        lines = [
            "WAYBACK DEAD-COHORT EXTRACT REPORT",
            f"  Attempted this run:   {self.attempted:,}",
            f"  Succeeded:            {self.succeeded:,}",
            f"  Empty/thin:           {self.empty:,}",
            f"  Failed:               {self.failed:,}",
            f"  Skipped existing:     {self.skipped_existing:,}",
            f"  Credits (run):        {self.credits_used_this_run:,.2f}",
            f"  Credits (cumulative): {self.total_credits:,.2f}",
            f"  Budget reached:       {self.budget_reached}",
            f"  Exit reason:          {self.exit_reason}",
        ]
        if self.errors_by_status:
            lines.append("  By status:")
            for status, n in sorted(self.errors_by_status.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"    {status:<22} {n:,}")
        return "\n".join(lines)


def run_extract_dead(
    targets_csv: str | Path = SCRAPE_TARGETS_DEAD_CSV,
    output_jsonl: str | Path = CRAWL_DEAD_JSONL,
    state_json: str | Path = CRAWL_STATE_DEAD_JSON,
    config: ExtractConfig | None = None,
    *,
    processed_csv: str | Path = SCRAPE_PROCESSED_DEAD_CSV,
    budget_credits: float = DEFAULT_BUDGET_CREDITS,
    max_companies: int | None = None,
    heartbeat_every: int = DEFAULT_HEARTBEAT_EVERY,
    heartbeat_log: str | Path = CRAWL_DEAD_LOG,
    manifest_csv: str | Path = RUN_MANIFEST_DEAD_CSV,
    max_concurrent_rows: int = DEFAULT_EXTRACT_DEAD_CONCURRENCY,
    extract_rpm: float | None = None,
    extract_rpm_headroom: float = DEFAULT_EXTRACT_RPM_HEADROOM,
    max_outage_seconds: float = DEFAULT_MAX_OUTAGE_SECONDS,
    outage_backoff_min_seconds: float = DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    outage_backoff_max_seconds: float = DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    min_free_disk_gb: float = 0.0,
) -> ExtractDeadRunReport:
    """Run a resumable, budget-capped archive extract over the dead target list."""
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
        print(f"[extract_dead] healed {healed} trailing bytes from {out_path}",
              file=sys.stderr, flush=True)

    backfilled = backfill_processed_dead_csv(out_path, processed_path)
    if backfilled:
        print(f"[extract_dead] backfilled {backfilled} rows into {processed_path}",
              file=sys.stderr, flush=True)

    scan = _scan_jsonl(out_path)
    state = ExtractState.load(state_path)
    state.successful, state.empty, state.failed = scan.successful, scan.empty, scan.failed
    state.save(state_path)
    completed_ids = scan.completed_ids

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
    credits_start = scan.credits
    credits_this_run = 0.0
    budget_reached = False
    exit_reason = "completed"
    errors: Counter[str] = Counter()
    started = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()

    print(f"[extract_dead] targets={len(all_targets):,} pending={len(pending):,} "
          f"skipped={skipped_existing:,} workers={workers} "
          f"extract_rpm_cap={effective_rpm or 'off'}", file=sys.stderr, flush=True)
    if scan.failure_reasons:
        breakdown = "  ".join(
            f"{reason}={n:,}"
            for reason, n in sorted(scan.failure_reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        print(f"[extract_dead] prior no-evidence rows by reason: {breakdown}",
              file=sys.stderr, flush=True)

    rows_written = [0]

    def persist(out: Any, outcome: _RowOutcome) -> None:
        nonlocal succeeded_run, empty_run, failed_run, credits_this_run
        org_uuid = str(outcome.record.get("org_uuid", "")).strip()
        state.last_org_uuid = org_uuid
        credits_this_run += outcome.credits_added
        pages_used = ""
        evidence = ""
        if outcome.ok and outcome.status in _SUCCESS_STATUSES:
            state.successful += 1
            succeeded_run += 1
            completed_ids.add(org_uuid)
            pages_used = outcome.pages_used
            evidence = outcome.evidence
        elif outcome.ok:
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

        out.write(json.dumps(outcome.record, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
        state.save(state_path)

        if pages_used or evidence:
            _append_processed_row(processed_path, {
                "org_uuid": org_uuid,
                "name": outcome.record.get("name", ""),
                "homepage_url": outcome.record.get("homepage_url", ""),
                "snapshot_ts": outcome.record.get("snapshot_ts", ""),
                "website_pages_used": pages_used,
                "website_evidence": evidence,
            }, fsync=True)

        rows_written[0] += 1
        if heartbeat_every > 0 and rows_written[0] % heartbeat_every == 0:
            _emit_heartbeat(
                log_path=heartbeat_path,
                processed=rows_written[0] + skipped_existing,
                total=heartbeat_total, succeeded=succeeded_run, empty=empty_run,
                failed=failed_run, est_credits=credits_this_run,
                elapsed_seconds=time.monotonic() - started, last_org_uuid=state.last_org_uuid,
            )

    with _GracefulStopController() as stop, out_path.open("a", encoding="utf-8") as out:
        deque_lock = threading.Lock()
        write_lock = threading.Lock()
        rows_deque: deque[dict[str, str]] = deque(pending)
        worker_errors: list[BaseException] = []
        in_flight_rows = 0

        # Reserve the worst-case paid work for each company: the if_ extract plus
        # the id_ second chance. This makes the budget gate safe even with many
        # workers starting rows before earlier rows finish and report actual spend.
        max_credits_per_row = 2 * estimate_credits(1, extract_depth=cfg.extract_depth)

        def _would_exceed_budget_after_starting_one() -> bool:
            reserved_in_flight = (in_flight_rows + 1) * max_credits_per_row
            return credits_start + credits_this_run + reserved_in_flight > budget_credits

        def run_worker() -> None:
            nonlocal attempted, budget_reached, exit_reason, in_flight_rows
            try:
                while True:
                    if stop.stop_requested:
                        return
                    with deque_lock:
                        if not rows_deque:
                            return
                        if _would_exceed_budget_after_starting_one():
                            budget_reached = True
                            exit_reason = "budget_reached"
                            return
                        target = rows_deque.popleft()
                        attempted += 1
                        in_flight_rows += 1
                    try:
                        outcome = _run_row_with_outage(
                            target=target, cfg=cfg, api_key=api_key,
                            rate_limiter=rate_limiter, stop=stop,
                            max_outage_seconds=max_outage_seconds,
                            outage_backoff_min_seconds=outage_backoff_min_seconds,
                            outage_backoff_max_seconds=outage_backoff_max_seconds,
                        )
                    except ExtractInterrupted:
                        with deque_lock:
                            rows_deque.appendleft(target)
                            attempted -= 1
                            in_flight_rows -= 1
                        return
                    except Exception as exc:  # noqa: BLE001 — surface to main thread
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
            except Exception as exc:  # noqa: BLE001
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

    report = ExtractDeadRunReport(
        attempted=attempted, succeeded=succeeded_run, empty=empty_run, failed=failed_run,
        skipped_existing=skipped_existing, credits_used_this_run=credits_this_run,
        total_credits=credits_start + credits_this_run, budget_reached=budget_reached,
        errors_by_status=dict(errors), exit_reason=exit_reason,
    )
    with suppress(OSError):
        _append_run_manifest(manifest_path, {
            "started_at": started_iso, "ended_at": datetime.now(timezone.utc).isoformat(),
            "attempted": report.attempted, "succeeded": report.succeeded,
            "empty": report.empty, "failed": report.failed,
            "est_credits": f"{report.credits_used_this_run:.2f}",
            "budget_reached": report.budget_reached,
            "output_jsonl_size_mb": f"{jsonl_mb:.2f}", "last_org_uuid": state.last_org_uuid,
            "exit_reason": report.exit_reason,
        })
    return report
