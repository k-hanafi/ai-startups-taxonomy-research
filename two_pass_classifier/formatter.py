"""Format immutable manifest rows for the two production passes."""

from __future__ import annotations

from typing import Any, Mapping

from single_pass_classifier.formatter import format_user_message

from .cohort import compute_cohort
from .manifest import ManifestRow

FORMATTER_VERSION = 2


def format_input_message(
    row: ManifestRow | Mapping[str, Any],
) -> str:
    """Format the shared classifier inputs through the production package."""
    return format_user_message(_row_inputs(row))


def format_pass_a_message(
    row: ManifestRow | Mapping[str, Any],
) -> str:
    """Format the binary gate input without website page metadata."""
    inputs = _row_inputs(row)
    inputs["website_pages_used"] = ""
    return format_input_message(inputs)


def format_pass_b_message(
    row: ManifestRow | Mapping[str, Any],
    verdict: int,
) -> str:
    """Format the full input plus the fixed family and code-owned cohort."""
    if verdict not in (0, 1):
        raise ValueError(f"verdict must be 0 or 1, got {verdict!r}")
    inputs = _row_inputs(row)
    if isinstance(row, ManifestRow):
        cohort = row.cohort
    else:
        cohort = str(row.get("cohort") or compute_cohort(inputs.get("founded_date")))
    return (
        format_input_message(inputs)
        + f"\nPriorBinaryVerdict: {verdict}"
        + f"\nCohort: {cohort}"
    )


def _row_inputs(row: ManifestRow | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, ManifestRow):
        return dict(row.inputs)
    inputs = row.get("inputs")
    if isinstance(inputs, Mapping):
        return dict(inputs)
    return dict(row)
