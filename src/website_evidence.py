"""Compact raw Tavily Crawl results into classifier-ready website evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.enrichment import DEFAULT_ENRICHED_CSV, OUTPUT_DIR
from src.tavily_crawl import DEFAULT_RAW_RESULTS_JSONL

DEFAULT_CLASSIFIER_INPUT_CSV = OUTPUT_DIR / "company_44k_with_website_evidence.csv"
DEFAULT_MAX_EVIDENCE_CHARS = 7_000
DEFAULT_MAX_PAGE_CHARS = 1_600


@dataclass(frozen=True)
class EvidenceBuildReport:
    """Summary of Tavily evidence compaction."""

    enriched_rows: int
    crawl_records: int
    rows_with_successful_crawl: int
    rows_with_website_evidence: int
    output_path: Path

    def format_report(self) -> str:
        return "\n".join([
            "WEBSITE EVIDENCE BUILD REPORT",
            f"  Enriched rows:            {self.enriched_rows:,}",
            f"  Crawl records:            {self.crawl_records:,}",
            f"  Successful crawls joined: {self.rows_with_successful_crawl:,}",
            f"  Rows with evidence:       {self.rows_with_website_evidence:,}",
            f"  Output CSV:               {self.output_path}",
        ])


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def _page_kind(url: str) -> str:
    lower = url.lower()
    if any(part in lower for part in ("/product", "/platform", "/solution", "/use-case")):
        return "product"
    if any(part in lower for part in ("/about", "/company", "/team")):
        return "about"
    if any(part in lower for part in ("/career", "/jobs", "/hiring")):
        return "careers"
    if any(part in lower for part in ("/research", "/technology", "/ai", "/ml")):
        return "technical"
    return "homepage_or_other"


def compact_tavily_response(
    response: dict[str, Any],
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_page_chars: int = DEFAULT_MAX_PAGE_CHARS,
) -> tuple[str, str]:
    """Return (`website_pages_used`, `website_evidence`) from one Tavily response."""
    pages = response.get("results")
    if not isinstance(pages, list):
        return "", ""

    evidence_blocks: list[str] = []
    urls: list[str] = []
    for idx, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        url = str(page.get("url", "")).strip()
        raw_content = _clean_text(page.get("raw_content", ""))
        if not url or not raw_content:
            continue
        urls.append(url)
        content = _truncate(raw_content, max_page_chars)
        evidence_blocks.append(f"[Page {idx}: {_page_kind(url)}]\nURL: {url}\n{content}")

    return " | ".join(urls), _truncate("\n\n".join(evidence_blocks), max_evidence_chars)


def _load_latest_successful_records(raw_jsonl: str | Path) -> dict[str, dict[str, Any]]:
    """Load the latest successful Tavily record per `org_uuid`."""
    p = Path(raw_jsonl)
    if not p.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            org_uuid = str(obj.get("org_uuid", "")).strip()
            if org_uuid and obj.get("ok") is True:
                records[org_uuid] = obj
    return records


def build_classifier_input_with_evidence(
    enriched_csv: str | Path = DEFAULT_ENRICHED_CSV,
    raw_jsonl: str | Path = DEFAULT_RAW_RESULTS_JSONL,
    output_csv: str | Path = DEFAULT_CLASSIFIER_INPUT_CSV,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_page_chars: int = DEFAULT_MAX_PAGE_CHARS,
) -> EvidenceBuildReport:
    """Join compact Tavily evidence onto the enriched classifier CSV."""
    enriched = pd.read_csv(enriched_csv, dtype=str, keep_default_na=False)
    records = _load_latest_successful_records(raw_jsonl)

    evidence_by_org: dict[str, dict[str, object]] = {}
    for org_uuid, record in records.items():
        response = record.get("response")
        if not isinstance(response, dict):
            continue
        pages_used, evidence = compact_tavily_response(
            response,
            max_evidence_chars=max_evidence_chars,
            max_page_chars=max_page_chars,
        )
        evidence_by_org[org_uuid] = {
            "website_pages_used": pages_used,
            "website_evidence": evidence,
            "website_crawl_status": "success" if evidence else "success_no_content",
            "website_credit_usage": record.get("usage_credits", 0.0),
        }

    evidence_df = pd.DataFrame.from_dict(evidence_by_org, orient="index")
    evidence_df.index.name = "org_uuid"
    evidence_df = evidence_df.reset_index()

    if evidence_df.empty:
        output = enriched.copy()
        for col in [
            "website_pages_used",
            "website_evidence",
            "website_crawl_status",
            "website_credit_usage",
        ]:
            output[col] = ""
    else:
        output = enriched.merge(evidence_df, on="org_uuid", how="left")
        for col in [
            "website_pages_used",
            "website_evidence",
            "website_crawl_status",
            "website_credit_usage",
        ]:
            output[col] = output[col].fillna("")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    rows_with_success = int(output["website_crawl_status"].eq("success").sum())
    rows_with_evidence = int(output["website_evidence"].astype(str).str.strip().ne("").sum())
    return EvidenceBuildReport(
        enriched_rows=len(enriched),
        crawl_records=len(records),
        rows_with_successful_crawl=rows_with_success,
        rows_with_website_evidence=rows_with_evidence,
        output_path=out_path,
    )
