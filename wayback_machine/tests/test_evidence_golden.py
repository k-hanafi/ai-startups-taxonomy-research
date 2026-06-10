"""Tripwire: the vendored cleaner must match the live src cleaner exactly.

The 2023 vs today comparison is only valid if both evidence sets are cleaned
identically. evidence.py is a frozen copy of src/website_evidence.py; if either
drifts, this test fails and tells us to re-vendor. Skips cleanly if src isn't
importable (e.g. the folder was lifted into its own repo).
"""

from __future__ import annotations

import pytest

from wayback_machine.evidence import compact_tavily_response as vendored

src_evidence = pytest.importorskip("src.website_evidence")
live = src_evidence.compact_tavily_response

CASES = [
    {"results": [{"url": "https://acme.ai/", "raw_content": (
        "# Acme AI\n"
        "We build an AI platform for automation and workflow integration.\n"
        "Privacy Policy\nBook a demo\n"
        "Our product uses machine learning models to serve customers across industries."
    )}]},
    {"results": [{"url": "https://thin.example/", "raw_content": "Home\nLogin\nContact us"}]},
    {"results": []},
    {"no_results_key": True},
    {"results": [
        {"url": "https://multi.example/", "raw_content": "# Heading\nReal product claim about our data API."},
        {"url": "https://multi.example/pricing", "raw_content": "Pricing\nStarter plan with API access and integrations."},
    ]},
]


@pytest.mark.parametrize("response", CASES)
def test_vendored_matches_live(response: dict) -> None:
    assert vendored(response) == live(response)
