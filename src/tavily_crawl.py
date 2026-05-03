"""Cost-controlled Tavily Crawl runner for company homepage enrichment."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.enrichment import DEFAULT_CRAWL_QUEUE_CSV, OUTPUT_DIR, PROJECT_ROOT, is_valid_homepage_url

TAVILY_CRAWL_ENDPOINT = "https://api.tavily.com/crawl"
DEFAULT_RAW_RESULTS_JSONL = OUTPUT_DIR / "tavily_crawl_raw.jsonl"
DEFAULT_CRAWL_STATE_JSON = OUTPUT_DIR / "tavily_crawl_state.json"

DEFAULT_INSTRUCTIONS = (
    "Select up to 5 pages that best explain what this company actually builds "
    "and sells for AI-native startup classification. Prioritize homepage, "
    "product/platform/solutions pages, about/company pages, careers/team/hiring "
    "pages, and technical/research pages. Prefer pages with evidence about "
    "product mechanism, AI/ML/LLM usage, autonomous agents, proprietary models "
    "or data, target users, and workflow depth. Avoid legal, privacy, terms, "
    "cookie, login, checkout, generic blog/news, support, and social pages."
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
    max_breadth: int = 12
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

    def request_payload(self, url: str) -> dict[str, Any]:
        """Return the JSON body for one Tavily Crawl request."""
        return {
            "url": url,
            "instructions": self.instructions,
            "chunks_per_source": self.chunks_per_source,
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
            if org_uuid and obj.get("ok") is True:
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


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, urllib.error.HTTPError):
        body = exc.read().decode("utf-8", errors="replace")
        return {"type": "HTTPError", "status": exc.code, "body": body}
    return {"type": type(exc).__name__, "message": str(exc)}


def run_tavily_crawl(
    queue_csv: str | Path = DEFAULT_CRAWL_QUEUE_CSV,
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

    queue = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    attempted = 0
    completed_this_run = 0
    failed_this_run = 0
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
                response = call_tavily_crawl(homepage_url, cfg, api_key)
                credits = extract_usage_credits(response)
                state.total_credits += credits
                state.completed += 1
                state.last_org_uuid = org_uuid
                completed_this_run += 1
                completed_ids.add(org_uuid)
                record.update({
                    "ok": True,
                    "usage_credits": credits,
                    "response": response,
                })
            except Exception as exc:
                state.failed += 1
                state.last_org_uuid = org_uuid
                failed_this_run += 1
                record.update({
                    "ok": False,
                    "usage_credits": 0.0,
                    "error": _error_payload(exc),
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
        skipped_existing=skipped_existing,
        skipped_invalid_url=skipped_invalid_url,
        credits_used_this_run=state.total_credits - credits_start,
        total_credits=state.total_credits,
        budget_reached=budget_reached,
    )
