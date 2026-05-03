"""Cost-controlled Tavily Crawl runner for company homepage enrichment."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.enrichment import DEFAULT_CLASSIFIER_INPUT_CSV, is_valid_homepage_url, tavily_eligible_mask
from src.paths import PROJECT_ROOT, TAVILY_DIR

TAVILY_CRAWL_ENDPOINT = "https://api.tavily.com/crawl"
DEFAULT_RAW_RESULTS_JSONL = TAVILY_DIR / "raw_results.jsonl"
CANONICALIZE_USER_AGENT = "ai-native-startup-classification/canonicalize"

# Deprecated name — Tavily reads the same lean classifier CSV and filters rows in memory.
DEFAULT_CRAWL_QUEUE_CSV = DEFAULT_CLASSIFIER_INPUT_CSV
DEFAULT_CRAWL_STATE_JSON = TAVILY_DIR / "crawl_state.json"

DEFAULT_INSTRUCTIONS = (
    "Select up to 5 pages that best explain what this company does, sells, who "
    "it serves, and how the offering works. Prefer homepage, product/platform/"
    "solutions/services/use-case, about, pricing, docs, technical/research "
    "pages with concrete evidence. Capture AI/ML/automation only when explicit "
    "or central. Avoid legal, privacy, login, checkout, support, social, and "
    "blog/news archives."
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


@dataclass(frozen=True)
class TavilyCrawlConfig:
    """Tavily Crawl request defaults tuned for signal and budget control."""

    limit: int = 5
    max_depth: int = 2
    max_breadth: int = 20
    extract_depth: str = "basic"
    chunks_per_source: int = 4
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
    canonicalize_urls: bool = True

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
        self.updated_at = datetime.now(timezone.utc).isoformat()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


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

    def format_report(self) -> str:
        return "\n".join([
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
        ])


def _api_key() -> str:
    load_dotenv(PROJECT_ROOT / "keys" / "tavily.env")
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TAVILY_API_KEY must be set before running Tavily crawl")
    return key


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


def call_tavily_crawl(url: str, config: TavilyCrawlConfig, api_key: str) -> dict[str, Any]:
    """Call Tavily Crawl with stdlib HTTP so no SDK dependency is required."""
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


def resolve_canonical_url(url: str, timeout: float = 15.0) -> str:
    """Follow server redirects to the URL Tavily should start from."""
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": CANONICALIZE_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.geturl()


def _canonical_homepage_url(url: str, config: TavilyCrawlConfig) -> str:
    if not config.canonicalize_urls:
        return url
    try:
        resolved = resolve_canonical_url(url, timeout=min(config.timeout, 15.0))
    except Exception:
        return url
    return resolved if is_valid_homepage_url(resolved) else url


class TavilyCrawlCallError(Exception):
    """Carries a normalized Tavily request error after retry handling."""

    def __init__(self, error: dict[str, Any]):
        super().__init__(str(error))
        self.error = error


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


def _call_tavily_crawl_with_retries(
    url: str,
    config: TavilyCrawlConfig,
    api_key: str,
) -> dict[str, Any]:
    attempts = max(1, config.max_retries + 1)
    last_error: dict[str, Any] | None = None
    for attempt in range(attempts):
        try:
            return call_tavily_crawl(url, config, api_key)
        except Exception as exc:
            error = _error_payload(exc)
            _, retryable = _error_status(error)
            last_error = error
            if not retryable or attempt == attempts - 1:
                break
            time.sleep(config.retry_backoff_seconds * (2 ** attempt))
    raise TavilyCrawlCallError(last_error or {"type": "UnknownError", "message": "unknown"})


def run_tavily_crawl(
    queue_csv: str | Path = DEFAULT_CLASSIFIER_INPUT_CSV,
    output_jsonl: str | Path = DEFAULT_RAW_RESULTS_JSONL,
    state_json: str | Path = DEFAULT_CRAWL_STATE_JSON,
    config: TavilyCrawlConfig | None = None,
    budget_credits: float = 100_000.0,
    max_companies: int | None = None,
    sleep_seconds: float = 0.0,
) -> CrawlRunReport:
    """Run a resumable, budget-capped Tavily crawl over the queue CSV."""
    cfg = config or TavilyCrawlConfig()
    api_key = _api_key()
    state = TavilyCrawlState.load(state_json)
    completed_ids = _completed_ids_from_jsonl(output_jsonl)

    full = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
    queue = full[tavily_eligible_mask(full)].copy()
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    attempted = 0
    completed_this_run = 0
    failed_this_run = 0
    empty_results_this_run = 0
    skipped_existing = 0
    skipped_invalid_url = 0
    credits_start = state.total_credits
    budget_reached = False

    with out_path.open("a", encoding="utf-8") as out:
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
            if state.total_credits >= budget_credits:
                budget_reached = True
                break
            if max_companies is not None and attempted >= max_companies:
                break

            attempted += 1
            record: dict[str, Any] = {
                "org_uuid": org_uuid,
                "name": row.get("name", ""),
                "homepage_url": homepage_url,
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "config": asdict(cfg),
            }

            try:
                crawl_url = _canonical_homepage_url(homepage_url, cfg)
                record["canonical_homepage_url"] = crawl_url
                response = _call_tavily_crawl_with_retries(crawl_url, cfg, api_key)
                credits = extract_usage_credits(response)
                state.total_credits += credits
                state.last_org_uuid = org_uuid

                if _has_usable_results(response):
                    state.completed += 1
                    completed_this_run += 1
                    completed_ids.add(org_uuid)
                    record.update({
                        "ok": True,
                        "crawl_status": "success",
                        "retryable": False,
                        "usage_credits": credits,
                        "response": response,
                    })
                else:
                    fallback_cfg = _fallback_config(cfg)
                    fallback_response = _call_tavily_crawl_with_retries(
                        crawl_url,
                        fallback_cfg,
                        api_key,
                    )
                    fallback_credits = extract_usage_credits(fallback_response)
                    state.total_credits += fallback_credits
                    credits += fallback_credits

                    if _has_usable_results(fallback_response):
                        state.completed += 1
                        completed_this_run += 1
                        completed_ids.add(org_uuid)
                        record.update({
                            "ok": True,
                            "crawl_status": "success_fallback",
                            "retryable": False,
                            "usage_credits": credits,
                            "response": fallback_response,
                            "primary_response": response,
                            "fallback_config": asdict(fallback_cfg),
                        })
                    else:
                        state.completed += 1
                        empty_results_this_run += 1
                        completed_ids.add(org_uuid)
                        record.update({
                            "ok": True,
                            "crawl_status": "empty_results",
                            "retryable": False,
                            "usage_credits": credits,
                            "response": fallback_response,
                            "primary_response": response,
                            "fallback_config": asdict(fallback_cfg),
                        })
            except Exception as exc:
                error = _error_payload(exc)
                crawl_status, retryable = _error_status(error)
                state.failed += 1
                state.last_org_uuid = org_uuid
                failed_this_run += 1
                if not retryable:
                    completed_ids.add(org_uuid)
                record.update({
                    "ok": False,
                    "crawl_status": crawl_status,
                    "retryable": retryable,
                    "usage_credits": 0.0,
                    "error": error,
                })

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            state.save(state_json)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    state.save(state_json)
    return CrawlRunReport(
        attempted=attempted,
        completed=completed_this_run,
        failed=failed_this_run,
        empty_results=empty_results_this_run,
        skipped_existing=skipped_existing,
        skipped_invalid_url=skipped_invalid_url,
        credits_used_this_run=state.total_credits - credits_start,
        total_credits=state.total_credits,
        budget_reached=budget_reached,
    )
