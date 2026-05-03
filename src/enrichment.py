"""Prepare the 44k Tavily enrichment dataset and crawl queue.

The enrichment stage keeps the GPT classifier input narrow: product
descriptions and tags remain primary evidence, while scale/resource fields are
available for RAD confidence without adding contact/geographic noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tavily_enrichment"

DEFAULT_SUBSET_CSV = DATA_DIR / "44k_crunchbase_startups.csv"
DEFAULT_MASTER_CSV = DATA_DIR / "company_us_all_var_Khaled.csv"
DEFAULT_ENRICHED_CSV = OUTPUT_DIR / "company_44k_enriched_for_classifier.csv"
DEFAULT_CRAWL_QUEUE_CSV = OUTPUT_DIR / "tavily_crawl_queue.csv"

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
    "num_funding_rounds",
    "total_funding_usd",
    "last_funding_date",
    "status",
]

AUDIT_ONLY_COLUMNS = [
    "rcid",
    "cb_url",
    "rank",
    "state_code",
    "region",
    "city",
    "linkedin_url",
    "twitter_url",
    "facebook_url",
    "created_date",
    "updated_date",
    "year_created",
    "closed_date",
]

DEFAULT_OUTPUT_COLUMNS = IDENTITY_COLUMNS + RESOURCE_CONTEXT_COLUMNS + AUDIT_ONLY_COLUMNS


@dataclass(frozen=True)
class EnrichmentReport:
    """Summary statistics for the prepared 44k enrichment dataset."""

    subset_rows: int
    master_rows: int
    output_rows: int
    matched_rows: int
    unmatched_rows: int
    duplicate_subset_orgs: int
    duplicate_master_orgs: int
    valid_homepage_urls: int
    invalid_homepage_urls: int
    crawl_queue_rows: int

    def format_report(self) -> str:
        """Return a concise, CLI-friendly preparation report."""
        return "\n".join([
            "TAVILY ENRICHMENT PREP REPORT",
            f"  Subset rows:              {self.subset_rows:,}",
            f"  Master rows:              {self.master_rows:,}",
            f"  Output rows:              {self.output_rows:,}",
            f"  Matched by org_uuid:      {self.matched_rows:,}",
            f"  Unmatched subset rows:    {self.unmatched_rows:,}",
            f"  Duplicate subset orgs:    {self.duplicate_subset_orgs:,}",
            f"  Duplicate master orgs:    {self.duplicate_master_orgs:,}",
            f"  Valid homepage URLs:      {self.valid_homepage_urls:,}",
            f"  Invalid homepage URLs:    {self.invalid_homepage_urls:,}",
            f"  Crawl queue rows:         {self.crawl_queue_rows:,}",
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


def _selected_master(master: pd.DataFrame) -> pd.DataFrame:
    """Keep only master fields with classifier or audit value."""
    keep = [
        "org_uuid",
        "rank",
        "state_code",
        "region",
        "city",
        "status",
        "num_funding_rounds",
        "total_funding_usd",
        "employee_count",
        "linkedin_url",
        "twitter_url",
        "facebook_url",
        "year_created",
        "updated_date",
        "last_funding_date",
        "closed_date",
    ]
    return master[[c for c in keep if c in master.columns]].copy()


def build_enriched_dataset(
    subset_csv: str | Path = DEFAULT_SUBSET_CSV,
    master_csv: str | Path = DEFAULT_MASTER_CSV,
) -> tuple[pd.DataFrame, EnrichmentReport]:
    """Join the 44k subset to selected master fields and validate crawlability."""
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

    for col in DEFAULT_OUTPUT_COLUMNS:
        if col not in joined.columns:
            joined[col] = ""
    output = joined[DEFAULT_OUTPUT_COLUMNS].copy()
    url_valid = output["homepage_url"].map(is_valid_homepage_url)
    output["homepage_url_valid"] = url_valid

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
        crawl_queue_rows=int(url_valid.sum()),
    )
    return output, report


def build_crawl_queue(enriched: pd.DataFrame) -> pd.DataFrame:
    """Build the Tavily queue from enriched rows with valid homepage URLs."""
    required = ["org_uuid", "name", "homepage_url", "status", "short_description"]
    for col in required:
        if col not in enriched.columns:
            raise ValueError(f"Missing required crawl queue column: {col}")

    queue = enriched[enriched["homepage_url"].map(is_valid_homepage_url)].copy()
    return queue[required].reset_index(drop=True)


def write_enrichment_outputs(
    subset_csv: str | Path = DEFAULT_SUBSET_CSV,
    master_csv: str | Path = DEFAULT_MASTER_CSV,
    enriched_csv: str | Path = DEFAULT_ENRICHED_CSV,
    crawl_queue_csv: str | Path = DEFAULT_CRAWL_QUEUE_CSV,
) -> EnrichmentReport:
    """Write the enriched classifier CSV and Tavily crawl queue."""
    enriched, report = build_enriched_dataset(subset_csv, master_csv)
    queue = build_crawl_queue(enriched)

    enriched_path = Path(enriched_csv)
    queue_path = Path(crawl_queue_csv)
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    enriched.to_csv(enriched_path, index=False)
    queue.to_csv(queue_path, index=False)
    return report
