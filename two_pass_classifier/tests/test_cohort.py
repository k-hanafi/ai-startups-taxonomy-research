from __future__ import annotations

import pytest

from two_pass_classifier.cohort import compute_cohort


@pytest.mark.parametrize(
    ("founded_date", "expected"),
    [
        ("2023-03-13", "PRE-GENAI"),
        ("2023-03-14", "GENAI-ERA"),
        ("2023-03-15", "GENAI-ERA"),
        ("2023-02", "PRE-GENAI"),
        ("2023-03", "GENAI-ERA"),
        ("2023", "PRE-GENAI"),
        ("2024", "GENAI-ERA"),
        ("01nov2016", "PRE-GENAI"),
        ("14Mar2023", "GENAI-ERA"),
        ("13Mar2023", "PRE-GENAI"),
        ("", "PRE-GENAI"),
        ("nan", "PRE-GENAI"),
        ("Unknown", "PRE-GENAI"),
        ("not-a-date", "PRE-GENAI"),
    ],
)
def test_compute_cohort_edge_cases(founded_date, expected):
    assert compute_cohort(founded_date) == expected
