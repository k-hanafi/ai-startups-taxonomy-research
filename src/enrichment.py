"""Build the single lean classifier input CSV used by GPT + Tavily."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.paths import DATA_DIR, TAVILY_DIR

DEFAULT_SUBSET_CSV = DATA_DIR / "44k_crunchbase_startups.csv"
DEFAULT_MASTER_CSV = DATA_DIR / "company_us_all_var_Khaled.csv"

OUTPUT_DIR = TAVILY_DIR

# One file: columns the formatter reads for GPT, plus homepage + Tavily evidence + gating flag.
DEFAULT_CLASSIFIER_INPUT_CSV = TAVILY_DIR / "classifier_input.csv"

# Back-compat aliases (same path).
DEFAULT_ENRICHED_CSV = DEFAULT_CLASSIFIER_INPUT_CSV
DEFAULT_CRAWL_QUEUE_CSV = DEFAULT_CLASSIFIER_INPUT_CSV

IDENTITY_COLUMNS = [
    "org_uuid",
    "name",
    "homepage_url",
    "short_description",
    "Long description",
    "category_list",
    "category_groups_list",
    "founded_date",
]

RESOURCE_CONTEXT_COLUMNS = [
    "employee_count",
    "total_funding_usd",
]

CLASSIFIER_INPUT_COLUMNS = (
    IDENTITY_COLUMNS
    + RESOURCE_CONTEXT_COLUMNS
    + [
        "website_alive",
        "website_pages_used",
        "website_evidence",
    ]
)

# Internal join pulls master fields needed for RESOURCE_CONTEXT only.
_JOIN_MASTER_COLUMNS = [
    "org_uuid",
    "total_funding_usd",
    "employee_count",
]


@dataclass(frozen=True)
class EnrichmentReport:
    """Summary statistics for classifier input preparation."""

    subset_rows: int
    master_rows: int
    output_rows: int
    matched_rows: int
    unmatched_rows: int
    duplicate_subset_orgs: int
    duplicate_master_orgs: int
    valid_homepage_urls: int
    invalid_homepage_urls: int
    tavily_eligible_rows: int

    def format_report(self) -> str:
        """Return a concise, CLI-friendly preparation report."""
        return "\n".join([
            "CLASSIFIER INPUT PREP REPORT",
            f"  Subset rows:              {self.subset_rows:,}",
            f"  Master rows:              {self.master_rows:,}",
            f"  Output rows:              {self.output_rows:,}",
            f"  Matched by org_uuid:      {self.matched_rows:,}",
            f"  Unmatched subset rows:    {self.unmatched_rows:,}",
            f"  Duplicate subset orgs:    {self.duplicate_subset_orgs:,}",
            f"  Duplicate master orgs:    {self.duplicate_master_orgs:,}",
            f"  Valid homepage URLs:      {self.valid_homepage_urls:,}",
            f"  Invalid homepage URLs:    {self.invalid_homepage_urls:,}",
            f"  Tavily-eligible rows:     {self.tavily_eligible_rows:,}",
        ])


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV as strings to preserve Crunchbase identifiers and dates."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalize_long_description(df: pd.DataFrame) -> pd.DataFrame:
    """Map source `description` into the classifier's `Long description` field."""
    out = df.copy()
    if "Long description" not in out.columns:
        out["Long description"] = ""
    if "description" in out.columns:
        out["Long description"] = out["Long description"].where(
            out["Long description"].astype(str).str.strip().ne(""),
            out["description"],
        )
    return out


def is_valid_homepage_url(value: object) -> bool:
    """Return True for crawlable HTTP(S) homepage URLs."""
    raw = str(value).strip() if value is not None else ""
    if not raw or raw.lower() in {"nan", "none", "nat"}:
        return False
    parsed = urlparse(raw)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _merge_prior_classifier_columns(enriched: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Preserve ``website_alive`` and Tavily evidence from an existing CSV when re-preparing."""
    if not path.is_file():
        return enriched
    prev = _read_csv(path)
    if "org_uuid" not in prev.columns:
        return enriched
    out = enriched.copy()
    orgs = out["org_uuid"].astype(str)

    if "website_alive" in prev.columns:
        m = dict(zip(prev["org_uuid"].astype(str), prev["website_alive"].astype(str).str.strip()))
        out["website_alive"] = orgs.map(m).fillna(out["website_alive"])
    elif "website_live" in prev.columns:
        m = dict(zip(prev["org_uuid"].astype(str), prev["website_live"].astype(str).str.strip()))
        out["website_alive"] = orgs.map(m).fillna(out["website_alive"])

    for col in ("website_pages_used", "website_evidence"):
        if col in prev.columns:
            m = dict(zip(prev["org_uuid"].astype(str), prev[col].astype(str)))
            out[col] = orgs.map(m).fillna(out[col])

    out["website_alive"] = out["website_alive"].fillna("").astype(str)
    for col in ("website_pages_used", "website_evidence"):
        out[col] = out[col].fillna("").astype(str)
    return out


def tavily_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Rows Tavily should crawl: valid URL and confirmed live when the flag exists."""
    valid = df["homepage_url"].map(is_valid_homepage_url)
    if "website_alive" in df.columns:
        col = "website_alive"
    elif "website_live" in df.columns:
        col = "website_live"
    else:
        return valid
    live = df[col].astype(str).str.strip().str.lower().eq("true")
    return valid & live


def _selected_master(master: pd.DataFrame) -> pd.DataFrame:
    return master[[c for c in _JOIN_MASTER_COLUMNS if c in master.columns]].copy()


def build_enriched_dataset(
    subset_csv: str | Path = DEFAULT_SUBSET_CSV,
    master_csv: str | Path = DEFAULT_MASTER_CSV,
) -> tuple[pd.DataFrame, EnrichmentReport]:
    """Join subset to master resource fields and return the lean classifier schema."""
    subset = _normalize_long_description(_read_csv(subset_csv))
    master = _read_csv(master_csv)
    master_selected = _selected_master(master)

    duplicate_subset_orgs = int(subset["org_uuid"].duplicated().sum())
    duplicate_master_orgs = int(master_selected["org_uuid"].duplicated().sum())
    if duplicate_subset_orgs:
        raise ValueError(f"Subset contains {duplicate_subset_orgs} duplicate org_uuid values")
    if duplicate_master_orgs:
        raise ValueError(f"Master contains {duplicate_master_orgs} duplicate org_uuid values")

    joined = subset.merge(
        master_selected,
        on="org_uuid",
        how="left",
        suffixes=("", "_master"),
        indicator=True,
    )
    matched_rows = int(joined["_merge"].eq("both").sum())
    unmatched_rows = int(joined["_merge"].eq("left_only").sum())
    joined = joined.drop(columns=["_merge"])

    body_cols = IDENTITY_COLUMNS + RESOURCE_CONTEXT_COLUMNS
    for col in body_cols:
        if col not in joined.columns:
            joined[col] = ""
    output = joined[body_cols].copy()

    output["website_alive"] = ""
    output["website_pages_used"] = ""
    output["website_evidence"] = ""

    output = output[CLASSIFIER_INPUT_COLUMNS].copy()

    url_valid = output["homepage_url"].map(is_valid_homepage_url)
    eligible = tavily_eligible_mask(output)

    report = EnrichmentReport(
        subset_rows=len(subset),
        master_rows=len(master),
        output_rows=len(output),
        matched_rows=matched_rows,
        unmatched_rows=unmatched_rows,
        duplicate_subset_orgs=duplicate_subset_orgs,
        duplicate_master_orgs=duplicate_master_orgs,
        valid_homepage_urls=int(url_valid.sum()),
        invalid_homepage_urls=int((~url_valid).sum()),
        tavily_eligible_rows=int(eligible.sum()),
    )
    return output, report


def build_crawl_queue(enriched: pd.DataFrame) -> pd.DataFrame:
    """In-memory slice of rows Tavily would crawl (for tests / inspection)."""
    required = ["org_uuid", "name", "homepage_url", "short_description"]
    for col in required:
        if col not in enriched.columns:
            raise ValueError(f"Missing required column: {col}")
    q = enriched[tavily_eligible_mask(enriched)].copy()
    return q[required].reset_index(drop=True)


def write_enrichment_outputs(
    subset_csv: str | Path = DEFAULT_SUBSET_CSV,
    master_csv: str | Path = DEFAULT_MASTER_CSV,
    enriched_csv: str | Path = DEFAULT_CLASSIFIER_INPUT_CSV,
    crawl_queue_csv: str | Path | None = None,
) -> EnrichmentReport:
    """Write the single lean ``classifier_input.csv`` (``crawl_queue_csv`` is ignored)."""
    _ = crawl_queue_csv  # deprecated; kept for API compatibility
    out_path = Path(enriched_csv)
    enriched, report = build_enriched_dataset(subset_csv, master_csv)
    enriched = _merge_prior_classifier_columns(enriched, out_path)
    report = replace(report, tavily_eligible_rows=int(tavily_eligible_mask(enriched).sum()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    return report
