"""Cost-controlled Tavily Crawl runner for company homepage enrichment."""

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
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.master_csv import (
    CLASSIFIER_INPUT_COLUMNS,
    DEFAULT_MASTER_CSV,
    is_valid_homepage_url,
    tavily_eligible_mask,
)
from src.paths import LOGS_DIR, PROJECT_ROOT, TAVILY_PROCESSED_DIR, TAVILY_RAW_DIR
from src.website_evidence import compact_tavily_response

TAVILY_CRAWL_ENDPOINT = "https://api.tavily.com/crawl"
DEFAULT_RAW_RESULTS_JSONL = TAVILY_RAW_DIR / "raw_results.jsonl"
DEFAULT_TAVILY_PROCESSED_CSV = TAVILY_PROCESSED_DIR / "tavily_processed_output.csv"
DEFAULT_CLASSIFIER_INPUT_CSV = TAVILY_PROCESSED_DIR / "classifier_input.csv"

DEFAULT_CRAWL_STATE_JSON = TAVILY_RAW_DIR / "crawl_state.json"

PROCESSED_OUTPUT_FIELDS = ["org_uuid", "name", "homepage_url", "website_pages_used", "website_evidence"]
DEFAULT_HEARTBEAT_LOG = LOGS_DIR / "tavily_crawl.log"
DEFAULT_RUN_MANIFEST_CSV = TAVILY_RAW_DIR / "run_manifest.csv"
PREFLIGHT_DRY_CALL_URL = "https://example.com"

# Outage retry caps. Per-call retries handle 30s blips; this layer handles 30+ minute outages.
DEFAULT_MAX_OUTAGE_SECONDS = 1800.0
DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS = 60.0
DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS = 600.0

DEFAULT_HEARTBEAT_EVERY = 100

# Crawl endpoint RPM is documented separately from the default API RPM. Both dev and prod
# keys share the same crawl cap. See https://docs.tavily.com/documentation/rate-limits
TAVILY_CRAWL_RPM_DOCUMENTED = 100
DEFAULT_CRAWL_RPM_HEADROOM = 0.8
DEFAULT_MAX_CONCURRENT_ROWS = 12

DEFAULT_INSTRUCTIONS = (
    "Select up to 5 pages that best explain what this company does, sells, who "
    "it serves, and how the offering works. Prefer homepage, product/platform/"
    "solutions/services/use-case, about, pricing, docs, technical/research "
    "pages with concrete evidence."
)

DEFAULT_EXCLUDE_PATHS = [
    r"/privacy.*",
    r"/terms.*",
    r"/legal.*",
    r"/cookies?.*",
    r"/login.*",
    r"/signin.*",
    r"/signup.*",
    r"/register.*",
    r"/checkout.*",
    r"/cart.*",
    r"/account.*",
    r"/support.*",
    r"/help.*",
    r"/careers?.*",
    r"/jobs?.*",
    r"/hiring.*",
    r"/blog/page/.*",
    r"/news/page/.*",
    r"/press/page/.*",
    r"/wp-admin.*",
]


MANIFEST_FIELDS = [
    "started_at",
    "ended_at",
    "attempted",
    "completed",
    "failed",
    "empty_results",
    "credits_used",
    "budget_reached",
    "output_jsonl_size_mb",
    "last_org_uuid",
    "exit_reason",
]


@dataclass(frozen=True)
class TavilyCrawlConfig:
    """Tavily Crawl request defaults tuned for signal and budget control."""

    limit: int = 5
    max_depth: int = 2
    max_breadth: int = 20
    extract_depth: str = "basic"
    chunks_per_source: int = 3
    timeout: float = 60.0
    format: str = "markdown"
    allow_external: bool = False
    include_images: bool = False
    include_favicon: bool = False
    include_usage: bool = True
    instructions: str = DEFAULT_INSTRUCTIONS
    exclude_paths: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATHS))
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0

    def request_payload(self, url: str) -> dict[str, Any]:
        """Return the JSON body for one Tavily Crawl request."""
        payload = {
            "url": url,
            "max_depth": self.max_depth,
            "max_breadth": self.max_breadth,
            "limit": self.limit,
            "exclude_paths": self.exclude_paths,
            "allow_external": self.allow_external,
            "include_images": self.include_images,
            "extract_depth": self.extract_depth,
            "format": self.format,
            "include_favicon": self.include_favicon,
            "timeout": self.timeout,
            "include_usage": self.include_usage,
        }
        if self.instructions:
            payload["instructions"] = self.instructions
            payload["chunks_per_source"] = self.chunks_per_source
        return payload


@dataclass
class TavilyCrawlState:
    """Persistent resume and budget state for the crawl run."""

    total_credits: float = 0.0
    completed: int = 0
    failed: int = 0
    skipped_invalid_url: int = 0
    last_org_uuid: str = ""
    updated_at: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "TavilyCrawlState":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        """Persist state via temp file + atomic rename so a crash mid-write cannot corrupt it."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self), indent=2).encode("utf-8")
        tmp = p.with_suffix(p.suffix + ".tmp")
        # Crash mid-write would otherwise leave a half-truncated state file. os.replace is POSIX-atomic.
        with tmp.open("wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)


@dataclass(frozen=True)
class CrawlRunReport:
    """Summary of a Tavily crawl run."""

    attempted: int
    completed: int
    failed: int
    empty_results: int
    skipped_existing: int
    skipped_invalid_url: int
    credits_used_this_run: float
    total_credits: float
    budget_reached: bool
    errors_by_status: dict[str, int] = field(default_factory=dict)
    exit_reason: str = "completed"

    def format_report(self) -> str:
        lines = [
            "TAVILY CRAWL RUN REPORT",
            f"  Attempted this run:       {self.attempted:,}",
            f"  Completed this run:       {self.completed:,}",
            f"  Failed this run:          {self.failed:,}",
            f"  Empty crawl results:      {self.empty_results:,}",
            f"  Skipped existing:         {self.skipped_existing:,}",
            f"  Skipped invalid URL:      {self.skipped_invalid_url:,}",
            f"  Credits this run:         {self.credits_used_this_run:,.2f}",
            f"  Total tracked credits:    {self.total_credits:,.2f}",
            f"  Budget reached:           {self.budget_reached}",
            f"  Exit reason:              {self.exit_reason}",
        ]
        if self.errors_by_status:
            lines.append("  Errors by status:")
            for status, count in sorted(self.errors_by_status.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"    {status:<28} {count:,}")
        return "\n".join(lines)


def _api_key() -> str:
    load_dotenv(PROJECT_ROOT / "keys" / "tavily.env")
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TAVILY_API_KEY must be set before running Tavily crawl")
    return key


def _heal_jsonl_tail(path: str | Path) -> int:
    """Truncate any unterminated partial line at the tail of a JSONL file.

    Tavily-crawl writes one terminal record per row and fsyncs. A power loss between
    ``write`` and ``fsync`` can still leave a half-flushed final line that breaks
    strict JSONL readers downstream. On startup we walk back to the last newline,
    parse the trailing bytes, and either append a missing newline (if valid) or
    truncate the partial line (if not).

    Returns the number of bytes removed.
    """
    p = Path(path)
    if not p.exists():
        return 0
    size = p.stat().st_size
    if size == 0:
        return 0

    chunk_size = 4096
    last_newline = -1
    position = size
    with p.open("rb") as fh:
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            fh.seek(position)
            idx = fh.read(read_size).rfind(b"\n")
            if idx >= 0:
                last_newline = position + idx
                break
        tail_start = last_newline + 1
        fh.seek(tail_start)
        tail = fh.read()

    if not tail:
        return 0

    try:
        json.loads(tail.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        with p.open("rb+") as fh:
            fh.truncate(tail_start)
        return len(tail)

    # Tail parses cleanly; just missing the trailing newline. Append one so future
    # appenders never produce two records concatenated on the same line.
    with p.open("ab") as fh:
        fh.write(b"\n")
    return 0


def _completed_ids_from_jsonl(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()

    completed: set[str] = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            org_uuid = str(obj.get("org_uuid", "")).strip()
            if org_uuid and (obj.get("ok") is True or obj.get("retryable") is False):
                completed.add(org_uuid)
    return completed


def extract_usage_credits(response: dict[str, Any]) -> float:
    """Extract Tavily usage credits from flexible response shapes."""
    usage = response.get("usage")
    if isinstance(usage, dict):
        for key in ("total_credits", "credits", "api_credits", "cost"):
            value = usage.get(key)
            if isinstance(value, int | float):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    pass
        numeric_values = [v for v in usage.values() if isinstance(v, int | float)]
        if numeric_values:
            return float(sum(numeric_values))
    return 0.0


def call_tavily_crawl(
    url: str,
    config: TavilyCrawlConfig,
    api_key: str,
    *,
    rate_limiter: _CrawlSlidingWindowLimiter | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Call Tavily Crawl with stdlib HTTP so no SDK dependency is required."""
    if rate_limiter is not None:
        rate_limiter.acquire(stop_check)
    data = json.dumps(config.request_payload(url)).encode("utf-8")
    request = urllib.request.Request(
        TAVILY_CRAWL_ENDPOINT,
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



class TavilyCrawlCallError(Exception):
    """Carries a normalized Tavily request error after retry handling."""

    def __init__(self, error: dict[str, Any]):
        super().__init__(str(error))
        self.error = error


class TavilyCrawlInterrupted(Exception):
    """Raised when the user stops the run during a rate-limit wait."""

    pass


class _CrawlSlidingWindowLimiter:
    """Thread-safe sliding window limiting POSTs to the Tavily Crawl endpoint.

    Tavily documents crawl RPM separately from the general API. The window counts
    each ``acquire`` as one crawl HTTP request (primary, fallback, or retry).
    """

    def __init__(
        self,
        max_calls_per_minute: float,
        *,
        window_seconds: float = 60.0,
    ) -> None:
        self._max = max(1.0, float(max_calls_per_minute))
        self._window = float(window_seconds)
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, stop_check: Callable[[], bool] | None = None) -> None:
        """Block until a crawl call is allowed under the sliding window.

        Raises:
            TavilyCrawlInterrupted: If ``stop_check`` returns True while waiting.
        """
        while True:
            if stop_check is not None and stop_check():
                raise TavilyCrawlInterrupted("stop requested during crawl rate wait")
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
            if sleep_for is None:
                continue
            end = time.monotonic() + sleep_for
            while time.monotonic() < end:
                if stop_check is not None and stop_check():
                    raise TavilyCrawlInterrupted("stop requested during crawl rate wait")
                time.sleep(min(0.2, end - time.monotonic()))


def _has_usable_results(response: dict[str, Any]) -> bool:
    results = response.get("results")
    if not isinstance(results, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("raw_content", "")).strip()
        for item in results
    )


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, TavilyCrawlCallError):
        return exc.error
    if isinstance(exc, urllib.error.HTTPError):
        body = exc.read().decode("utf-8", errors="replace")
        return {"type": "HTTPError", "status": exc.code, "body": body}
    return {"type": type(exc).__name__, "message": str(exc)}


def _error_status(error: dict[str, Any]) -> tuple[str, bool]:
    """Return (`crawl_status`, `retryable`) for a captured error payload."""
    body = str(error.get("body", ""))
    status = error.get("status")
    error_type = str(error.get("type", ""))

    if status == 400 and "Invalid Start URL" in body:
        return "invalid_start_url", False
    if status == 422:
        return "request_validation_error", False
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return "transient_error", True
    if error_type in {"TimeoutError", "URLError", "ConnectionError"}:
        return "transient_error", True
    return "error", False


def _fallback_config(config: TavilyCrawlConfig) -> TavilyCrawlConfig:
    """Less selective fallback for zero-page responses."""
    return replace(config, instructions="", chunks_per_source=1)


def _sleep_retry_after_or_backoff(
    exc: urllib.error.HTTPError,
    *,
    attempt: int,
    config: TavilyCrawlConfig,
    stop_check: Callable[[], bool] | None,
    stop_sleep: Callable[[float], bool] | None,
) -> None:
    """Sleep for 429 ``Retry-After`` or exponential backoff, honoring stop signals."""
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


def _call_tavily_crawl_with_retries(
    url: str,
    config: TavilyCrawlConfig,
    api_key: str,
    *,
    rate_limiter: _CrawlSlidingWindowLimiter | None = None,
    stop_check: Callable[[], bool] | None = None,
    stop_sleep: Callable[[float], bool] | None = None,
) -> dict[str, Any]:
    last_error: dict[str, Any] | None = None
    for attempt in range(max(1, config.max_retries + 1)):
        try:
            return call_tavily_crawl(
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
    raise TavilyCrawlCallError(last_error or {"type": "UnknownError", "message": "unknown"})


class _GracefulStopController:
    """Catch SIGINT/SIGTERM, set a flag, and wake interruptible sleeps.

    The runner installs this for the duration of a crawl. Handlers only flip a
    flag, so the next row-boundary check drains state cleanly with one final
    ``state.save`` before exiting.
    """

    def __init__(self) -> None:
        self.stop_requested = False
        self._previous_handlers: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True

    def __enter__(self) -> "_GracefulStopController":
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(ValueError, OSError):
                # signal.signal raises ValueError if called outside the main thread.
                self._previous_handlers[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *_exc: Any) -> None:
        for sig, prev in self._previous_handlers.items():
            with suppress(ValueError, OSError):
                signal.signal(sig, prev)

    def sleep(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, waking immediately on SIGINT/SIGTERM.

        Returns True if the full duration elapsed, False if interrupted.
        """
        if seconds <= 0:
            return not self.stop_requested
        end = time.monotonic() + seconds
        while not self.stop_requested:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.5, remaining))
        return False


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN guard
        return "n/a"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours}h {minutes:02d}m"


def _emit_heartbeat(
    *,
    log_path: Path,
    processed: int,
    total: int,
    completed: int,
    fallbacks: int,
    empty: int,
    failed: int,
    credits: float,
    elapsed_seconds: float,
    last_org_uuid: str,
) -> None:
    """Write one heartbeat line to stderr and append to the heartbeat log."""
    rate_per_min = (processed / elapsed_seconds * 60.0) if elapsed_seconds > 0 else 0.0
    remaining = max(total - processed, 0)
    eta_seconds = (remaining / rate_per_min * 60.0) if rate_per_min > 0 else float("inf")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line1 = (
        f"[{timestamp}] processed={processed:,} / {total:,}  "
        f"ok={completed:,} fb={fallbacks:,} empty={empty:,} fail={failed:,}"
    )
    line2 = (
        f"   credits={credits:,.2f}  rate={rate_per_min:,.1f} rows/min  "
        f"ETA={_format_eta(eta_seconds)}  last={last_org_uuid or 'n/a'}"
    )
    print(line1, file=sys.stderr, flush=True)
    print(line2, file=sys.stderr, flush=True)
    with suppress(OSError):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line1 + "\n")
            fh.write(line2 + "\n")


def _preflight_checks(
    *,
    queue_eligible_count: int,
    output_jsonl: Path,
    state_json: Path,
    min_free_disk_gb: float,
) -> None:
    """Fail fast before a single API call if the environment is not safe to run.

    Order is deliberate: cheap local checks first, then anything that touches the API.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    state_json.parent.mkdir(parents=True, exist_ok=True)

    probe_path = output_jsonl.parent / ".tavily_crawl_writable_probe"
    try:
        probe_path.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Output directory {output_jsonl.parent} is not writable: {exc}. "
            f"Fix permissions before running."
        ) from exc
    finally:
        with suppress(OSError):
            probe_path.unlink()

    if min_free_disk_gb > 0:
        free_bytes = shutil.disk_usage(output_jsonl.parent).free
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < min_free_disk_gb:
            raise RuntimeError(
                f"Free disk at {output_jsonl.parent} is {free_gb:.2f} GB but "
                f"{min_free_disk_gb:.2f} GB is required. Free space and retry."
            )

    if queue_eligible_count <= 0:
        raise RuntimeError(
            "No Tavily-eligible rows found. Ensure `website_alive=true` is set on rows "
            "with valid homepage URLs in `data/master_csv.csv` before running the crawler."
        )


def _preflight_dry_call(
    api_key: str,
    config: TavilyCrawlConfig,
    *,
    rate_limiter: _CrawlSlidingWindowLimiter | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> None:
    """Spend ~1 credit to confirm the API key, network, and request shape all work."""
    dry_config = replace(
        config,
        instructions="",
        chunks_per_source=1,
        limit=1,
        max_depth=1,
        max_breadth=1,
        max_retries=0,
    )
    call_tavily_crawl(
        PREFLIGHT_DRY_CALL_URL,
        dry_config,
        api_key,
        rate_limiter=rate_limiter,
        stop_check=stop_check,
    )


def _append_run_manifest(manifest_csv: Path, row: dict[str, Any]) -> None:
    """Append one row to the per-run manifest CSV, creating the header if needed."""
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_csv.exists() or manifest_csv.stat().st_size == 0
    with manifest_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def _append_processed_row(processed_csv: Path, row: dict[str, Any]) -> None:
    """Append one post-processed result row to tavily_processed_output.csv."""
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not processed_csv.exists() or processed_csv.stat().st_size == 0
    with processed_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROCESSED_OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PROCESSED_OUTPUT_FIELDS})


def write_classifier_input(
    master_csv: Path,
    processed_csv: Path,
    output_csv: Path,
) -> None:
    """Join master_csv + tavily_processed_output into classifier_input.csv.

    Called once at the end of a completed crawl run. Dead/uncrawled rows receive
    empty strings for the two evidence columns.
    """
    import pandas as pd

    master = pd.read_csv(master_csv, dtype=str, keep_default_na=False)

    if processed_csv.exists() and processed_csv.stat().st_size > 0:
        processed = pd.read_csv(processed_csv, dtype=str, keep_default_na=False)
        ev_cols = [c for c in ("org_uuid", "website_pages_used", "website_evidence") if c in processed.columns]
        if len(ev_cols) == 3:
            # Keep only the last entry per org (handles duplicate rows from resume)
            processed = processed[ev_cols].drop_duplicates(subset=["org_uuid"], keep="last")
            output = master.merge(processed, on="org_uuid", how="left")
        else:
            output = master.copy()
    else:
        output = master.copy()

    for col in ("website_pages_used", "website_evidence"):
        if col not in output.columns:
            output[col] = ""
        output[col] = output[col].fillna("").astype(str)

    for col in CLASSIFIER_INPUT_COLUMNS:
        if col not in output.columns:
            output[col] = ""
    output = output[list(CLASSIFIER_INPUT_COLUMNS)].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    print(
        f"[tavily_crawl] wrote classifier_input.csv: {len(output):,} rows → {output_csv}",
        file=sys.stderr,
        flush=True,
    )


@dataclass
class _RowOutcome:
    """Result of a single row processing attempt, before persistence."""

    record: dict[str, Any]
    crawl_status: str
    ok: bool
    retryable: bool
    credits_added: float
    transient_failure: bool


def _process_single_row(
    *,
    row: dict[str, Any],
    org_uuid: str,
    homepage_url: str,
    cfg: TavilyCrawlConfig,
    api_key: str,
    rate_limiter: _CrawlSlidingWindowLimiter | None = None,
    stop_check: Callable[[], bool] | None = None,
    stop_sleep: Callable[[float], bool] | None = None,
) -> _RowOutcome:
    """Run primary-and-maybe-fallback for one row (thread-safe: no shared state)."""
    record: dict[str, Any] = {
        "org_uuid": org_uuid,
        "name": row.get("name", ""),
        "homepage_url": homepage_url,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    credits_added = 0.0
    try:
        response = _call_tavily_crawl_with_retries(
            homepage_url,
            cfg,
            api_key,
            rate_limiter=rate_limiter,
            stop_check=stop_check,
            stop_sleep=stop_sleep,
        )
        primary_credits = extract_usage_credits(response)
        credits_added += primary_credits

        if _has_usable_results(response):
            record.update({
                "ok": True,
                "crawl_status": "success",
                "retryable": False,
                "usage_credits": primary_credits,
                "response": response,
            })
            return _RowOutcome(
                record=record,
                crawl_status="success",
                ok=True,
                retryable=False,
                credits_added=credits_added,
                transient_failure=False,
            )

        fallback_cfg = _fallback_config(cfg)
        fallback_response = _call_tavily_crawl_with_retries(
            homepage_url,
            fallback_cfg,
            api_key,
            rate_limiter=rate_limiter,
            stop_check=stop_check,
            stop_sleep=stop_sleep,
        )
        fallback_credits = extract_usage_credits(fallback_response)
        credits_added += fallback_credits
        total_credits_for_row = primary_credits + fallback_credits

        if _has_usable_results(fallback_response):
            record.update({
                "ok": True,
                "crawl_status": "success_fallback",
                "retryable": False,
                "usage_credits": total_credits_for_row,
                "response": fallback_response,
            })
            return _RowOutcome(
                record=record,
                crawl_status="success_fallback",
                ok=True,
                retryable=False,
                credits_added=credits_added,
                transient_failure=False,
            )

        record.update({
            "ok": True,
            "crawl_status": "empty_results",
            "retryable": False,
            "usage_credits": total_credits_for_row,
            "response": fallback_response,
        })
        return _RowOutcome(
            record=record,
            crawl_status="empty_results",
            ok=True,
            retryable=False,
            credits_added=credits_added,
            transient_failure=False,
        )
    except Exception as exc:
        error = _error_payload(exc)
        crawl_status, retryable = _error_status(error)
        record.update({
            "ok": False,
            "crawl_status": crawl_status,
            "retryable": retryable,
            "usage_credits": 0.0,
            "error": error,
        })
        return _RowOutcome(
            record=record,
            crawl_status=crawl_status,
            ok=False,
            retryable=retryable,
            credits_added=credits_added,
            transient_failure=retryable,
        )


def _run_row_with_outage(
    *,
    row: dict[str, Any],
    org_uuid: str,
    homepage_url: str,
    cfg: TavilyCrawlConfig,
    api_key: str,
    rate_limiter: _CrawlSlidingWindowLimiter | None,
    stop: _GracefulStopController,
    max_outage_seconds: float,
    outage_backoff_min_seconds: float,
    outage_backoff_max_seconds: float,
) -> _RowOutcome:
    """Retry a row through long Tavily outages while honoring crawl RPM limits."""
    outage_attempt = 0
    outage_started = time.monotonic()
    while True:
        outcome = _process_single_row(
            row=row,
            org_uuid=org_uuid,
            homepage_url=homepage_url,
            cfg=cfg,
            api_key=api_key,
            rate_limiter=rate_limiter,
            stop_check=lambda: stop.stop_requested,
            stop_sleep=stop.sleep,
        )
        if not outcome.transient_failure:
            return outcome
        elapsed = time.monotonic() - outage_started
        if elapsed >= max_outage_seconds or stop.stop_requested:
            return outcome
        sleep_secs = min(
            outage_backoff_min_seconds * (2 ** outage_attempt),
            outage_backoff_max_seconds,
        )
        if not stop.sleep(sleep_secs):
            return outcome
        outage_attempt += 1


def run_tavily_crawl(
    queue_csv: str | Path = DEFAULT_MASTER_CSV,
    output_jsonl: str | Path = DEFAULT_RAW_RESULTS_JSONL,
    state_json: str | Path = DEFAULT_CRAWL_STATE_JSON,
    config: TavilyCrawlConfig | None = None,
    budget_credits: float = 100_000.0,
    max_companies: int | None = None,
    *,
    processed_csv: str | Path = DEFAULT_TAVILY_PROCESSED_CSV,
    classifier_input_csv: str | Path = DEFAULT_CLASSIFIER_INPUT_CSV,
    heartbeat_every: int = DEFAULT_HEARTBEAT_EVERY,
    heartbeat_log: str | Path = DEFAULT_HEARTBEAT_LOG,
    manifest_csv: str | Path = DEFAULT_RUN_MANIFEST_CSV,
    max_outage_seconds: float = DEFAULT_MAX_OUTAGE_SECONDS,
    outage_backoff_min_seconds: float = DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    outage_backoff_max_seconds: float = DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    min_free_disk_gb: float = 0.0,
    preflight_dry_call: bool = False,
    max_concurrent_rows: int = 1,
    crawl_rpm: float | None = None,
    crawl_rpm_headroom: float = DEFAULT_CRAWL_RPM_HEADROOM,
) -> CrawlRunReport:
    """Run a resumable, budget-capped Tavily crawl over the queue CSV.

    Reliability layers added on top of the per-call retries already in
    ``_call_tavily_crawl_with_retries``:

    * ``state_json`` is written via temp-file + ``os.replace`` for atomic durability.
    * The append-only JSONL is fsynced after every record so paid Tavily calls
      cannot outrun what's recorded on disk.
    * A trailing partial JSONL line from a prior crash is healed at startup.
    * SIGINT/SIGTERM flip a flag and the loop exits cleanly at the next row boundary.
    * A row that fails with a transient error is retried after a long sleep up to
      ``max_outage_seconds``, surviving multi-minute Tavily outages without losing
      the row.
    * The next row is refused if it would push spend past ``budget_credits``,
      based on a rolling credits-per-row average.
    * A heartbeat line is emitted every ``heartbeat_every`` rows.
    * One row is appended to ``manifest_csv`` per run for post-hoc observability.

    Concurrency (``max_concurrent_rows`` > 1) uses a shared sliding-window limiter on
    crawl POSTs. Tavily documents the crawl endpoint at 100 RPM for both development
    and production keys (separate from the higher default API RPM on production keys).
    Default ``crawl_rpm_headroom`` targets 80 RPM so bursts stay under the crawl cap.
    Set ``crawl_rpm`` explicitly to override the computed cap.

    With multiple workers, JSONL lines may appear in completion order, not CSV order.
    Resume still keys off ``org_uuid`` and terminal ``ok`` / ``retryable`` flags.
    """
    cfg = config or TavilyCrawlConfig()
    state_path = Path(state_json)
    out_path = Path(output_jsonl)
    processed_path = Path(processed_csv)
    classifier_input_path = Path(classifier_input_csv)
    heartbeat_path = Path(heartbeat_log)
    manifest_path = Path(manifest_csv)

    full = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
    queue = full[tavily_eligible_mask(full)].copy()
    eligible_total = len(queue)

    _preflight_checks(
        queue_eligible_count=eligible_total,
        output_jsonl=out_path,
        state_json=state_path,
        min_free_disk_gb=min_free_disk_gb,
    )

    truncated_bytes = _heal_jsonl_tail(out_path)
    if truncated_bytes:
        print(
            f"[tavily_crawl] healed {truncated_bytes} trailing bytes from {out_path}",
            file=sys.stderr,
            flush=True,
        )

    state = TavilyCrawlState.load(state_path)
    completed_ids = _completed_ids_from_jsonl(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if crawl_rpm is not None and crawl_rpm <= 0:
        effective_rpm: float | None = None
    elif crawl_rpm is not None:
        effective_rpm = float(crawl_rpm)
    else:
        effective_rpm = TAVILY_CRAWL_RPM_DOCUMENTED * float(crawl_rpm_headroom)
    rate_limiter = (
        _CrawlSlidingWindowLimiter(effective_rpm) if effective_rpm and effective_rpm > 0 else None
    )

    api_key = _api_key()

    attempted = 0
    completed_this_run = 0
    fallback_this_run = 0
    failed_this_run = 0
    empty_results_this_run = 0
    skipped_existing = 0
    skipped_invalid_url = 0
    credits_start = state.total_credits
    budget_reached = False
    errors_by_status: Counter[str] = Counter()
    started_monotonic = time.monotonic()
    started_at_iso = datetime.now(timezone.utc).isoformat()
    exit_reason = "completed"

    pending_rows: list[dict[str, Any]] = []
    for row in queue.to_dict(orient="records"):
        org_uuid = str(row.get("org_uuid", "")).strip()
        homepage_url = str(row.get("homepage_url", "")).strip()
        if org_uuid in completed_ids:
            skipped_existing += 1
            continue
        if not is_valid_homepage_url(homepage_url):
            skipped_invalid_url += 1
            state.skipped_invalid_url += 1
            continue
        pending_rows.append(row)

    if max_companies is not None:
        pending_rows = pending_rows[: int(max_companies)]

    # Progress bar + ETA denominator: must match this run's scope (respects --max-companies).
    # Using eligible_total here would inflate ETA ~35× when crawling 1k of 35k rows.
    heartbeat_total = skipped_existing + skipped_invalid_url + len(pending_rows)

    workers = max(1, int(max_concurrent_rows))

    if workers > 1:
        rpm_note = f"{effective_rpm:.1f}" if effective_rpm else "disabled (unlimited)"
        print(
            f"[tavily_crawl] concurrent_rows={workers} crawl_rpm_cap={rpm_note} "
            f"(Tavily crawl endpoint limit is {TAVILY_CRAWL_RPM_DOCUMENTED} RPM; see "
            f"https://docs.tavily.com/documentation/rate-limits)",
            file=sys.stderr,
            flush=True,
        )

    def credits_per_row_estimate() -> float:
        if attempted <= 0:
            return 0.0
        return max(0.0, (state.total_credits - credits_start) / attempted)

    def persist_row(
        out: Any,
        *,
        org_uuid: str,
        outcome: _RowOutcome,
        rows_written: list[int],
    ) -> None:
        """Apply one outcome to ``state``, append JSONL, fsync, and save state."""
        nonlocal completed_this_run, fallback_this_run, empty_results_this_run, failed_this_run

        state.last_org_uuid = org_uuid
        state.total_credits += outcome.credits_added
        if outcome.ok:
            state.completed += 1
            completed_ids.add(org_uuid)
            if outcome.crawl_status == "success":
                completed_this_run += 1
            elif outcome.crawl_status == "success_fallback":
                completed_this_run += 1
                fallback_this_run += 1
            elif outcome.crawl_status == "empty_results":
                empty_results_this_run += 1
                errors_by_status[outcome.crawl_status] += 1
        else:
            state.failed += 1
            failed_this_run += 1
            errors_by_status[outcome.crawl_status] += 1
            if not outcome.retryable:
                completed_ids.add(org_uuid)

        out.write(json.dumps(outcome.record, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
        state.save(state_path)

        # Post-process and append one row to tavily_processed_output.csv.
        if outcome.ok and outcome.crawl_status in ("success", "success_fallback"):
            response = outcome.record.get("response") or {}
            pages_used, evidence = compact_tavily_response(response)
        else:
            pages_used, evidence = "", ""
        _append_processed_row(processed_path, {
            "org_uuid": org_uuid,
            "name": outcome.record.get("name", ""),
            "homepage_url": outcome.record.get("homepage_url", ""),
            "website_pages_used": pages_used,
            "website_evidence": evidence,
        })

        rows_written[0] += 1
        if heartbeat_every > 0 and rows_written[0] % heartbeat_every == 0:
            _emit_heartbeat(
                log_path=heartbeat_path,
                processed=rows_written[0] + skipped_existing + skipped_invalid_url,
                total=heartbeat_total,
                completed=completed_this_run,
                fallbacks=fallback_this_run,
                empty=empty_results_this_run,
                failed=failed_this_run,
                credits=state.total_credits - credits_start,
                elapsed_seconds=time.monotonic() - started_monotonic,
                last_org_uuid=state.last_org_uuid,
            )

    with _GracefulStopController() as stop, out_path.open("a", encoding="utf-8") as out:
        if preflight_dry_call:
            _preflight_dry_call(
                api_key,
                cfg,
                rate_limiter=rate_limiter,
                stop_check=lambda: stop.stop_requested,
            )

        rows_written = [0]
        deque_lock = threading.Lock()
        write_lock = threading.Lock()
        rows_deque: deque[dict[str, Any]] = deque(pending_rows)
        worker_errors: list[BaseException] = []

        def run_worker() -> None:
            nonlocal attempted, budget_reached, exit_reason
            try:
                while True:
                    if stop.stop_requested:
                        return
                    with deque_lock:
                        if not rows_deque:
                            return
                        if state.total_credits + credits_per_row_estimate() > budget_credits:
                            budget_reached = True
                            exit_reason = "budget_reached"
                            return
                        row = rows_deque.popleft()
                        attempted += 1
                    org_uuid = str(row.get("org_uuid", "")).strip()
                    homepage_url = str(row.get("homepage_url", "")).strip()
                    try:
                        outcome = _run_row_with_outage(
                            row=row,
                            org_uuid=org_uuid,
                            homepage_url=homepage_url,
                            cfg=cfg,
                            api_key=api_key,
                            rate_limiter=rate_limiter,
                            stop=stop,
                            max_outage_seconds=max_outage_seconds,
                            outage_backoff_min_seconds=outage_backoff_min_seconds,
                            outage_backoff_max_seconds=outage_backoff_max_seconds,
                        )
                    except TavilyCrawlInterrupted:
                        with deque_lock:
                            rows_deque.appendleft(row)
                            attempted -= 1
                        return
                    except Exception as exc:
                        worker_errors.append(exc)
                        return
                    with write_lock:
                        persist_row(out, org_uuid=org_uuid, outcome=outcome, rows_written=rows_written)
                    if stop.stop_requested:
                        exit_reason = "user_interrupt"
                        return
            except TavilyCrawlInterrupted:
                return
            except Exception as exc:
                worker_errors.append(exc)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_worker) for _ in range(workers)]
            for fut in futures:
                fut.result()
        if worker_errors:
            raise worker_errors[0]

    state.save(state_path)

    if max_companies is not None and exit_reason == "completed" and attempted > 0 and attempted == max_companies:
        exit_reason = "max_companies"

    output_jsonl_size_mb = 0.0
    with suppress(OSError):
        output_jsonl_size_mb = out_path.stat().st_size / (1024 * 1024)

    report = CrawlRunReport(
        attempted=attempted,
        completed=completed_this_run,
        failed=failed_this_run,
        empty_results=empty_results_this_run,
        skipped_existing=skipped_existing,
        skipped_invalid_url=skipped_invalid_url,
        credits_used_this_run=state.total_credits - credits_start,
        total_credits=state.total_credits,
        budget_reached=budget_reached,
        errors_by_status=dict(errors_by_status),
        exit_reason=exit_reason,
    )

    with suppress(OSError):
        _append_run_manifest(
            manifest_path,
            {
                "started_at": started_at_iso,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "attempted": report.attempted,
                "completed": report.completed,
                "failed": report.failed,
                "empty_results": report.empty_results,
                "credits_used": f"{report.credits_used_this_run:.4f}",
                "budget_reached": report.budget_reached,
                "output_jsonl_size_mb": f"{output_jsonl_size_mb:.2f}",
                "last_org_uuid": state.last_org_uuid,
                "exit_reason": report.exit_reason,
            },
        )

    if exit_reason == "completed":
        with suppress(OSError):
            write_classifier_input(
                Path(queue_csv),
                processed_path,
                classifier_input_path,
            )

    return report
