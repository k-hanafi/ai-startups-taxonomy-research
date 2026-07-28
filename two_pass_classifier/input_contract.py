"""Stable source columns consumed by the production two-pass classifier."""

from __future__ import annotations

SOURCE_COLUMNS: tuple[str, ...] = (
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
    "website_pages_used",
    "website_evidence",
)

# These values are copied byte-for-byte into model messages. Non-model source
# metadata remains in each manifest row but does not affect input_hash.
MODEL_INPUT_COLUMNS: tuple[str, ...] = (
    "org_uuid",
    "name",
    "short_description",
    "Long description",
    "category_list",
    "category_groups_list",
    "founded_date",
    "employee_count",
    "total_funding_usd",
    "website_pages_used",
    "website_evidence",
)
