from __future__ import annotations

from two_pass_classifier import config
from two_pass_classifier.costing import (
    actual_usage_cost,
    estimate_manifest_cost,
)
from two_pass_classifier.manifest import Manifest, ManifestRow
from two_pass_classifier.request_builder import RequestSettings


def _manifest(count: int = 3) -> Manifest:
    rows = tuple(
        ManifestRow(
            company_id=f"company-{index}",
            company_name=f"Company {index}",
            cohort="GENAI-ERA",
            company_alive="yes",
            website_snapshot_date="2026-05-04",
            evidence_source="live",
            source_row_number=index + 2,
            input_hash=f"hash-{index}",
            inputs={
                "org_uuid": f"company-{index}",
                "name": f"Company {index}",
                "short_description": "AI workflow",
                "Long description": "A detailed workflow.",
                "category_list": "Artificial Intelligence",
                "category_groups_list": "Software",
                "founded_date": "2024-01",
                "employee_count": "1-10",
                "total_funding_usd": "100",
                "website_pages_used": "https://company.test",
                "website_evidence": "Evidence " * (index + 1),
            },
        )
        for index in range(count)
    )
    return Manifest(
        rows=rows,
        sources=(),
        rows_sha256="rows",
        manifest_sha256="manifest",
    )


def test_preview_counts_actual_manifest_requests_and_normal_list_price():
    manifest = _manifest(3)
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")

    preview = estimate_manifest_cost(manifest, settings)

    assert preview.manifest_row_count == 3
    assert preview.pass_a.request_count == 3
    assert preview.pass_b.request_count == 3
    assert preview.pass_a.input_tokens_min > 0
    assert preview.pass_b.input_tokens_min > 0
    assert config.PASS_A_PROVISIONAL_OUTPUT_TOKENS == 320
    assert preview.pass_a.estimated_output_tokens == (
        3 * config.PASS_A_PROVISIONAL_OUTPUT_TOKENS
    )
    assert preview.pass_b.estimated_output_tokens == (
        3 * config.PASS_B_PREVIEW_OUTPUT_TOKENS["low"]
    )
    assert preview.pass_a.one_attempt_cap_tokens == (
        3 * config.PASS_A_MAX_OUTPUT_TOKENS
    )
    expected_pass_a = (
        preview.pass_a.input_tokens_min / 1_000_000 * 0.20
        + preview.pass_a.estimated_output_tokens / 1_000_000 * 1.25
    )
    assert preview.pass_a.estimated_cost_min == expected_pass_a
    assert preview.pricing_per_million["input"] == 0.20


def test_remaining_preview_uses_measured_family_and_skips_completed_pass_a():
    manifest = _manifest(3)
    preview = estimate_manifest_cost(
        manifest,
        RequestSettings(model="gpt-5.4-mini", pass_b_effort="medium"),
        pass_a_company_ids={"company-2"},
        pass_b_families={"company-1": 0, "company-2": None},
    )

    assert preview.pass_a.request_count == 1
    assert preview.pass_b.request_count == 2
    assert preview.known_family_counts == {0: 1, 1: 0}
    assert preview.unknown_family_count == 1


def test_actual_usage_prices_measured_cached_tokens_only():
    events = [
        {
            "event_type": "pass_a_completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        },
        {
            "event_type": "company_completed",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "input_tokens_details": {"cached_tokens": 20},
            },
        },
    ]

    usage = actual_usage_cost(events, model="gpt-5.6-luna")

    assert usage.physical_requests == 2
    assert usage.input_tokens == 220
    assert usage.cached_input_tokens == 60
    assert usage.output_tokens == 50
    assert usage.reasoning_tokens == 5
    assert usage.cost_usd == (
        160 / 1_000_000 * 1.00
        + 60 / 1_000_000 * 0.50
        + 50 / 1_000_000 * 6.00
    )


def test_unknown_pricing_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.delitem(config.MODEL_PRICING, "gpt-5.6-luna")
    settings = RequestSettings(model="gpt-5.6-luna")

    try:
        estimate_manifest_cost(_manifest(1), settings)
    except ValueError as exc:
        assert "unknown pricing" in str(exc)
    else:
        raise AssertionError("missing model pricing must fail")
