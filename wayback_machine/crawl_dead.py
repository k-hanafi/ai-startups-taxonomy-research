"""Resumable, budget-capped Tavily ``/crawl`` runner over dead-cohort snapshots.

The GO methodology: crawl each company's pre-death Wayback ``if_`` snapshot with
the EXACT live ``TavilyCrawlConfig`` (5 pages, depth 2, same instructions), scoped
to the company's own archived pages, then rewrite each archived page URL back to
its origin and clean it with the same evidence cleaner the live cohort used. The
output is byte-format-identical to the live + 2023 evidence, so only the evidence
itself differs across cohorts — the whole point of the fair-comparison design.

This is the crawl analogue of ``extract.py``. It reuses, rather than reimplements:

* the live crawl call + retry + fallback + usage parsing (``src.tavily_crawl``),
* the per-company archive scope + origin-rewrite cleaner (``tavily_archive_lab``),
* the reliability harness — graceful stop, heartbeat, manifest, processed-CSV
  append, preflight, JSONL tail-healing, atomic resume state (``extract`` + ``state``).

Resume keys off ``completed_ids`` from the append-only JSONL; compacted evidence is
stored in both the JSONL row and the processed CSV. Startup backfill heals any
JSONL rows that landed before a crash interrupted the CSV append, so paid work is
never lost and finished companies are never double-billed.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tavily_crawl import (
    DEFAULT_CRAWL_RPM_HEADROOM,
    TAVILY_CRAWL_RPM_DOCUMENTED,
    TavilyCrawlConfig,
    TavilyCrawlInterrupted,
    _api_key,
    _call_tavily_crawl_with_retries,
    _CrawlSlidingWindowLimiter,
    _error_payload,
    _error_status,
    _fallback_config,
    _has_usable_results,
    extract_usage_credits,
)

from .config import (
    DEFAULT_BUDGET_CREDITS,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_MAX_OUTAGE_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
)
from .extract import (
    _GracefulStopController,
    _append_processed_row,
    _append_run_manifest,
    _emit_heartbeat,
    _preflight_checks,
)
from .state import processed_ids_from_csv
from .paths import (
    CRAWL_DEAD_JSONL,
    CRAWL_DEAD_LOG,
    CRAWL_STATE_DEAD_JSON,
    RUN_MANIFEST_DEAD_CSV,
    SCRAPE_PROCESSED_DEAD_CSV,
    SCRAPE_TARGETS_DEAD_CSV,
)
from .state import ExtractState, heal_jsonl_tail
from .tavily_archive_lab import _ScopedCrawlConfig, clean_evidence

# Pin every crawl to the archive host so a scope miss can never escape to the
# (dead) live domain. Per-company path scope is layered on top when available.
_ARCHIVE_DOMAIN = r"^web\.archive\.org$"

# Many dead snapshots are big multi-page crawls, so default to a small worker
# pool: the binding constraint is the Internet Archive throttling Tavily's
# fetches, not our own crawl-RPM cap.
DEFAULT_CRAWL_DEAD_CONCURRENCY = 4

csv.field_size_limit(1_000_000_000)


def _scoped_config(base: TavilyCrawlConfig, select_paths: str) -> TavilyCrawlConfig:
    """Wrap the base crawl config with this company's archive scope.

    Forwards EVERY field of ``base`` (limit/depth/instructions/exclude_paths/…),
    not just ``extract_depth``, so a caller-supplied config is honored in full
    and the request payload stays byte-identical to the live cohort's.
    """
    paths = (select_paths,) if select_paths.strip() else ()
    return _ScopedCrawlConfig(
        **asdict(base),
        select_paths=paths,
        select_domains=(_ARCHIVE_DOMAIN,),
    )


@dataclass
class _ScanResult:
    completed_ids: set[str]
    successful: int
    empty: int
    failed: int
    credits: float


def _scan_jsonl(path: str | Path) -> _ScanResult:
    """One pass over the crawl JSONL to rebuild resume + budget state.

    ``completed_ids`` = anything terminal (ok, or a non-retryable error), so a
    resumed run re-crawls only transient failures. Credits are summed from the
    recorded per-row usage so the budget cap stays cumulative across resumes.
    """
    result = _ScanResult(set(), 0, 0, 0, 0.0)
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
            if org_uuid and (ok is True or obj.get("retryable") is False):
                result.completed_ids.add(org_uuid)
            if ok is True:
                if status in ("success", "success_fallback"):
                    result.successful += 1
                else:
                    result.empty += 1
            elif ok is False:
                result.failed += 1
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


def _process_single_row(
    *,
    target: dict[str, str],
    base_cfg: TavilyCrawlConfig,
    api_key: str,
    rate_limiter: _CrawlSlidingWindowLimiter | None,
    stop_check: Callable[[], bool] | None,
    stop_sleep: Callable[[float], bool] | None,
) -> _RowOutcome:
    """Crawl one company's snapshot (primary, then instructionless fallback)."""
    org_uuid = str(target.get("org_uuid", "")).strip()
    homepage_url = str(target.get("homepage_url", "")).strip()
    snapshot_url = str(target.get("snapshot_url", "")).strip()
    snapshot_ts = str(target.get("closest_ts", "")).strip()
    cfg = _scoped_config(base_cfg, str(target.get("select_paths", "")))
    record: dict[str, Any] = {
        "org_uuid": org_uuid,
        "name": target.get("name", ""),
        "homepage_url": homepage_url,
        "snapshot_url": snapshot_url,
        "snapshot_ts": snapshot_ts,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    credits = 0.0
    try:
        response = _call_tavily_crawl_with_retries(
            snapshot_url, cfg, api_key,
            rate_limiter=rate_limiter, stop_check=stop_check, stop_sleep=stop_sleep,
        )
        credits += extract_usage_credits(response)
        status = "success"
        fallback_used = False
        if not _has_usable_results(response):
            # Same empty-result fallback the live runner uses: drop the LLM
            # instructions and try once more before giving up on the company.
            fallback_cfg = _fallback_config(cfg)
            response = _call_tavily_crawl_with_retries(
                snapshot_url, fallback_cfg, api_key,
                rate_limiter=rate_limiter, stop_check=stop_check, stop_sleep=stop_sleep,
            )
            credits += extract_usage_credits(response)
            fallback_used = True
            status = "success_fallback"
    except Exception as exc:  # noqa: BLE001 — normalize any Tavily/network error
        error = _error_payload(exc)
        status, retryable = _error_status(error)
        record.update({"ok": False, "status": status, "retryable": retryable,
                       "usage_credits": credits, "error": error})
        return _RowOutcome(record, status, False, retryable, "", "", credits, retryable)

    if not _has_usable_results(response):
        record.update({"ok": True, "status": "empty_results", "retryable": False,
                       "usage_credits": credits})
        return _RowOutcome(record, "empty_results", True, False, "", "", credits, False)

    pages_used, evidence = clean_evidence(response)
    if not evidence.strip():
        record.update({"ok": True, "status": "thin_evidence", "retryable": False,
                       "usage_credits": credits})
        return _RowOutcome(record, "thin_evidence", True, False, "", "", credits, False)

    record.update({"ok": True, "status": status, "retryable": False,
                   "usage_credits": credits, "fallback_used": fallback_used,
                   "website_pages_used": pages_used, "website_evidence": evidence})
    return _RowOutcome(record, status, True, False, pages_used, evidence, credits, False)


def _processed_row_from_dead_record(record: dict[str, Any]) -> dict[str, str] | None:
    """Build a scrape_processed_dead.csv row from one successful JSONL record."""
    if record.get("ok") is not True or record.get("status") not in ("success", "success_fallback"):
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
    base_cfg: TavilyCrawlConfig,
    api_key: str,
    rate_limiter: _CrawlSlidingWindowLimiter | None,
    stop: _GracefulStopController,
    max_outage_seconds: float,
    outage_backoff_min_seconds: float,
    outage_backoff_max_seconds: float,
) -> _RowOutcome:
    """Retry a row through multi-minute Tavily/Archive outages, honoring stop."""
    outage_attempt = 0
    outage_started = time.monotonic()
    while True:
        outcome = _process_single_row(
            target=target, base_cfg=base_cfg, api_key=api_key, rate_limiter=rate_limiter,
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


@dataclass(frozen=True)
class CrawlDeadRunReport:
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
            "WAYBACK DEAD-COHORT CRAWL REPORT",
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


def run_crawl_dead(
    targets_csv: str | Path = SCRAPE_TARGETS_DEAD_CSV,
    output_jsonl: str | Path = CRAWL_DEAD_JSONL,
    state_json: str | Path = CRAWL_STATE_DEAD_JSON,
    config: TavilyCrawlConfig | None = None,
    *,
    processed_csv: str | Path = SCRAPE_PROCESSED_DEAD_CSV,
    budget_credits: float = DEFAULT_BUDGET_CREDITS,
    max_companies: int | None = None,
    heartbeat_every: int = DEFAULT_HEARTBEAT_EVERY,
    heartbeat_log: str | Path = CRAWL_DEAD_LOG,
    manifest_csv: str | Path = RUN_MANIFEST_DEAD_CSV,
    max_concurrent_rows: int = DEFAULT_CRAWL_DEAD_CONCURRENCY,
    crawl_rpm: float | None = None,
    crawl_rpm_headroom: float = DEFAULT_CRAWL_RPM_HEADROOM,
    max_outage_seconds: float = DEFAULT_MAX_OUTAGE_SECONDS,
    outage_backoff_min_seconds: float = DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    outage_backoff_max_seconds: float = DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    min_free_disk_gb: float = 0.0,
) -> CrawlDeadRunReport:
    """Run a resumable, budget-capped archive crawl over the dead target list."""
    base_cfg = config or TavilyCrawlConfig()
    state_path = Path(state_json)
    out_path = Path(output_jsonl)
    processed_path = Path(processed_csv)
    heartbeat_path = Path(heartbeat_log)
    manifest_path = Path(manifest_csv)

    with Path(targets_csv).open(encoding="utf-8", newline="") as f:
        all_targets = list(csv.DictReader(f))

    healed = heal_jsonl_tail(out_path)
    if healed:
        print(f"[crawl_dead] healed {healed} trailing bytes from {out_path}",
              file=sys.stderr, flush=True)

    backfilled = backfill_processed_dead_csv(out_path, processed_path)
    if backfilled:
        print(f"[crawl_dead] backfilled {backfilled} rows into {processed_path}",
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

    if crawl_rpm is not None and crawl_rpm <= 0:
        effective_rpm: float | None = None
    elif crawl_rpm is not None:
        effective_rpm = float(crawl_rpm)
    else:
        effective_rpm = TAVILY_CRAWL_RPM_DOCUMENTED * float(crawl_rpm_headroom)
    rate_limiter = _CrawlSlidingWindowLimiter(effective_rpm) if effective_rpm else None

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

    print(f"[crawl_dead] targets={len(all_targets):,} pending={len(pending):,} "
          f"skipped={skipped_existing:,} workers={workers} "
          f"crawl_rpm_cap={effective_rpm or 'off'}", file=sys.stderr, flush=True)

    rows_written = [0]

    def persist(out: Any, outcome: _RowOutcome) -> None:
        nonlocal succeeded_run, empty_run, failed_run, credits_this_run
        org_uuid = str(outcome.record.get("org_uuid", "")).strip()
        state.last_org_uuid = org_uuid
        credits_this_run += outcome.credits_added
        pages_used = ""
        evidence = ""
        if outcome.ok and outcome.status in ("success", "success_fallback"):
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

        def _per_row_estimate() -> float:
            return credits_this_run / attempted if attempted > 0 else 0.0

        def run_worker() -> None:
            nonlocal attempted, budget_reached, exit_reason
            try:
                while True:
                    if stop.stop_requested:
                        return
                    with deque_lock:
                        if not rows_deque:
                            return
                        if credits_start + credits_this_run + _per_row_estimate() > budget_credits:
                            budget_reached = True
                            exit_reason = "budget_reached"
                            return
                        target = rows_deque.popleft()
                        attempted += 1
                    try:
                        outcome = _run_row_with_outage(
                            target=target, base_cfg=base_cfg, api_key=api_key,
                            rate_limiter=rate_limiter, stop=stop,
                            max_outage_seconds=max_outage_seconds,
                            outage_backoff_min_seconds=outage_backoff_min_seconds,
                            outage_backoff_max_seconds=outage_backoff_max_seconds,
                        )
                    except TavilyCrawlInterrupted:
                        with deque_lock:
                            rows_deque.appendleft(target)
                            attempted -= 1
                        return
                    except Exception as exc:  # noqa: BLE001 — surface to main thread
                        worker_errors.append(exc)
                        return
                    with write_lock:
                        persist(out, outcome)
                    if stop.stop_requested:
                        exit_reason = "user_interrupt"
                        return
            except TavilyCrawlInterrupted:
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

    report = CrawlDeadRunReport(
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
