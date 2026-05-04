"""Schema and helpers for the static master_csv.csv and derived pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.paths import DATA_DIR

DEFAULT_MASTER_CSV = DATA_DIR / "master_csv.csv"

# Columns in master_csv.csv (11 fields, static before the crawl).
MASTER_CSV_COLUMNS = [
    "org_uuid",
    "name",
    "homepage_url",
    "short_description",
    "Long description",
    "category_list",
    "category_groups_list",
    "founded_date",
    "employee_count",
    "total_funding_usd",
    "website_alive",
]

# Columns in classifier_input.csv (master + Tavily evidence, produced at run end).
CLASSIFIER_INPUT_COLUMNS = MASTER_CSV_COLUMNS + [
    "website_pages_used",
    "website_evidence",
]


def is_valid_homepage_url(value: object) -> bool:
    """Return True for crawlable HTTP(S) homepage URLs."""
    raw = str(value).strip() if value is not None else ""
    if not raw or raw.lower() in {"nan", "none", "nat"}:
        return False
    parsed = urlparse(raw)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def tavily_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Rows the crawler should visit: valid URL and website_alive=true."""
    valid = df["homepage_url"].map(is_valid_homepage_url)
    if "website_alive" not in df.columns:
        return valid
    live = df["website_alive"].astype(str).str.strip().str.lower().eq("true")
    return valid & live
