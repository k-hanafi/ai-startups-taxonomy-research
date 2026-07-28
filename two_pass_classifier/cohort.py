"""Deterministic cohort assignment from the source founding date."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal

from .config import (
    COHORT_BOUNDARY_DAY,
    COHORT_BOUNDARY_MONTH,
    COHORT_BOUNDARY_YEAR,
)

Cohort = Literal["PRE-GENAI", "GENAI-ERA"]
_BOUNDARY = date(
    COHORT_BOUNDARY_YEAR,
    COHORT_BOUNDARY_MONTH,
    COHORT_BOUNDARY_DAY,
)
_MISSING = {"", "nan", "none", "nat", "unknown"}


def compute_cohort(value: Any) -> Cohort:
    """Return the founding cohort without asking the model.

    Full dates use the exact GPT-4 launch date. Month-only values use the
    validated eval convention where all of March 2023 is GENAI-ERA. A bare
    year resolves to January, while missing or unparseable values resolve to
    PRE-GENAI.
    """
    if isinstance(value, datetime):
        return _from_date(value.date())
    if isinstance(value, date):
        return _from_date(value)

    text = str(value).strip() if value is not None else ""
    if text.lower() in _MISSING:
        return "PRE-GENAI"

    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = (int(part) for part in text.split("-"))
        if not 1 <= month <= 12:
            return "PRE-GENAI"
        return (
            "GENAI-ERA"
            if (year, month) >= (COHORT_BOUNDARY_YEAR, COHORT_BOUNDARY_MONTH)
            else "PRE-GENAI"
        )

    if re.fullmatch(r"\d{4}", text):
        return _from_date(date(int(text), 1, 1))

    for fmt in ("%Y-%m-%d", "%d%b%Y", "%d-%b-%y", "%d-%b-%Y", "%m/%d/%Y"):
        try:
            return _from_date(datetime.strptime(text, fmt).date())
        except ValueError:
            continue

    return "PRE-GENAI"


def _from_date(value: date) -> Cohort:
    return "GENAI-ERA" if value >= _BOUNDARY else "PRE-GENAI"
