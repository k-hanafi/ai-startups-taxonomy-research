"""Stable CSV column contract consumed by the single-pass classifier."""

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

CLASSIFIER_INPUT_COLUMNS = MASTER_CSV_COLUMNS + [
    "website_pages_used",
    "website_evidence",
]
