"""Join compact Tavily crawl results into the lean classifier input CSV."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from src.enrichment import CLASSIFIER_INPUT_COLUMNS, DEFAULT_CLASSIFIER_INPUT_CSV, OUTPUT_DIR
from src.tavily_crawl import DEFAULT_RAW_RESULTS_JSONL

DEFAULT_ENRICHED_CSV = DEFAULT_CLASSIFIER_INPUT_CSV
DEFAULT_MAX_EVIDENCE_CHARS = None
DEFAULT_MAX_PAGE_CHARS = None

_BOILERPLATE_EXACT = {
    "top of page",
    "bottom of page",
    "skip to content",
    "skip to main content",
    "scroll to top",
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "accept cookies",
    "book a demo",
    "book a call",
    "get in touch",
    "contact us",
    "learn more",
    "read more",
}

_BOILERPLATE_CONTAINS = (
    "something went wrong",
    "thank you for joining",
    "i wish to subscribe",
    "please leave your email",
    "our website uses cookies",
    "copyright ",
    "© ",
)

_SIGNAL_TERMS = (
    "about",
    "ai",
    "api",
    "automation",
    "case stud",
    "customer",
    "data",
    "docs",
    "how it works",
    "industry",
    "integration",
    "machine learning",
    "model",
    "platform",
    "pricing",
    "product",
    "research",
    "service",
    "solution",
    "technical",
    "use case",
    "workflow",
)


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
            f"  Classifier rows:          {self.enriched_rows:,}",
            f"  Crawl records:            {self.crawl_records:,}",
            f"  Rows with crawl evidence: {self.rows_with_successful_crawl:,}",
            f"  Rows with evidence text:  {self.rows_with_website_evidence:,}",
            f"  Output CSV:               {self.output_path}",
        ])


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _is_image_or_asset_line(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower()
    if re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", stripped):
        return True
    if lower.startswith("!["):
        return True
    return bool(re.search(r"\.(png|jpe?g|gif|webp|svg|avif)(\)|\?|$)", lower))


def _is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower().strip("[]()#*:_-. ")
    if not lower:
        return True
    if lower in _BOILERPLATE_EXACT:
        return True
    if len(lower) <= 3 and not any(ch.isalpha() for ch in lower):
        return True
    if any(marker in lower for marker in _BOILERPLATE_CONTAINS):
        return True
    if lower.startswith(("tel:", "mailto:")):
        return True
    if lower in {"home", "contact", "login", "sign in", "sign up", "close menu"}:
        return True
    return False


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = re.sub(r"\W+", "", line).lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return out


def _shape_signal_text(text: str) -> str:
    """Remove obvious page chrome while preserving classifier-relevant claims."""
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    kept: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if _is_image_or_asset_line(stripped) or _is_boilerplate_line(stripped):
            continue
        if stripped.startswith("[") and "](" in stripped and len(stripped) < 80:
            continue
        kept.append(stripped)

    deduped = _dedupe_lines(kept)
    if len("\n".join(deduped)) <= 20_000:
        return "\n".join(deduped).strip()

    signal_lines: list[str] = []
    regular_lines: list[str] = []
    for line in deduped:
        lower = line.lower()
        if line.startswith("#") or any(term in lower for term in _SIGNAL_TERMS):
            signal_lines.append(line)
        else:
            regular_lines.append(line)

    shaped: list[str] = []
    total = 0
    for line in signal_lines + regular_lines:
        projected = total + len(line) + 1
        if projected > 20_000:
            break
        shaped.append(line)
        total = projected
    return "\n".join(shaped).strip()


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def _page_kind(url: str) -> str:
    """Label pages by URL route instead of inferred content category."""
    path = unquote(urlparse(url).path).strip("/")
    if not path:
        return "homepage"
    return path.rsplit("/", maxsplit=1)[-1] or "homepage"


def compact_tavily_response(
    response: dict[str, Any],
    max_evidence_chars: int | None = DEFAULT_MAX_EVIDENCE_CHARS,
    max_page_chars: int | None = DEFAULT_MAX_PAGE_CHARS,
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
        raw_content = _shape_signal_text(str(page.get("raw_content", "")))
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
    enriched_csv: str | Path = DEFAULT_CLASSIFIER_INPUT_CSV,
    raw_jsonl: str | Path = DEFAULT_RAW_RESULTS_JSONL,
    output_csv: str | Path = DEFAULT_CLASSIFIER_INPUT_CSV,
    max_evidence_chars: int | None = DEFAULT_MAX_EVIDENCE_CHARS,
    max_page_chars: int | None = DEFAULT_MAX_PAGE_CHARS,
) -> EvidenceBuildReport:
    """Merge Tavily crawl text into ``website_pages_used`` / ``website_evidence`` columns."""
    enriched = pd.read_csv(enriched_csv, dtype=str, keep_default_na=False)
    if "website_alive" not in enriched.columns and "website_live" in enriched.columns:
        enriched["website_alive"] = enriched["website_live"]
    records = _load_latest_successful_records(raw_jsonl)

    evidence_by_org: dict[str, dict[str, str]] = {}
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
        }

    evidence_df = pd.DataFrame.from_dict(evidence_by_org, orient="index")
    evidence_df.index.name = "org_uuid"
    evidence_df = evidence_df.reset_index()

    drop_ev = [c for c in ("website_pages_used", "website_evidence") if c in enriched.columns]
    base = enriched.drop(columns=drop_ev, errors="ignore")

    if evidence_df.empty:
        output = base.copy()
        output["website_pages_used"] = ""
        output["website_evidence"] = ""
    else:
        output = base.merge(evidence_df, on="org_uuid", how="left")
        for c in ("website_pages_used", "website_evidence"):
            output[c] = output[c].fillna("")

    for col in ("website_pages_used", "website_evidence"):
        if col in output.columns:
            output[col] = output[col].map(lambda x: "" if pd.isna(x) else str(x))

    if "website_alive" in output.columns:
        dead = output["website_alive"].astype(str).str.strip().str.lower().eq("false")
        output.loc[dead, "website_pages_used"] = ""
        output.loc[dead, "website_evidence"] = ""

    for col in CLASSIFIER_INPUT_COLUMNS:
        if col not in output.columns:
            output[col] = ""
    output = output[list(CLASSIFIER_INPUT_COLUMNS)].copy()

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    ev_nonempty = output["website_evidence"].astype(str).str.strip().ne("")
    rows_with_success = int(ev_nonempty.sum())
    rows_with_evidence = rows_with_success
    return EvidenceBuildReport(
        enriched_rows=len(enriched),
        crawl_records=len(records),
        rows_with_successful_crawl=rows_with_success,
        rows_with_website_evidence=rows_with_evidence,
        output_path=out_path,
    )
