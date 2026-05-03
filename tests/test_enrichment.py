import json

import pandas as pd

from src.enrichment import build_crawl_queue, build_enriched_dataset, is_valid_homepage_url
from src.tavily_crawl import TavilyCrawlConfig, extract_usage_credits
from src.website_evidence import build_classifier_input_with_evidence, compact_tavily_response


def test_is_valid_homepage_url():
    assert is_valid_homepage_url("https://example.com")
    assert is_valid_homepage_url("http://example.com/path")
    assert not is_valid_homepage_url("")
    assert not is_valid_homepage_url("example.com")
    assert not is_valid_homepage_url("mailto:hello@example.com")


def test_build_enriched_dataset_joins_master_fields(tmp_path):
    subset = pd.DataFrame([
        {
            "rcid": "1",
            "org_uuid": "org-1",
            "name": "Acme",
            "cb_url": "https://crunchbase.com/acme",
            "homepage_url": "https://acme.test",
            "short_description": "Builds AI agents.",
            "category_list": "Artificial Intelligence,Software",
            "category_groups_list": "Software",
            "created_date": "01jan2024",
            "founded_date": "01jan2024",
            "description": "Acme builds autonomous agents for support teams.",
        }
    ])
    master = pd.DataFrame([
        {
            "org_uuid": "org-1",
            "rank": "42",
            "state_code": "CA",
            "region": "California",
            "city": "San Francisco",
            "status": "operating",
            "num_funding_rounds": "2",
            "total_funding_usd": "5000000",
            "employee_count": "11-50",
            "linkedin_url": "https://linkedin.com/company/acme",
            "twitter_url": "",
            "facebook_url": "",
            "year_created": "2024",
            "updated_date": "01jan2025",
            "last_funding_date": "01jun2024",
            "closed_date": "",
        }
    ])
    subset_path = tmp_path / "subset.csv"
    master_path = tmp_path / "master.csv"
    subset.to_csv(subset_path, index=False)
    master.to_csv(master_path, index=False)

    enriched, report = build_enriched_dataset(subset_path, master_path)

    assert report.matched_rows == 1
    assert report.valid_homepage_urls == 1
    assert enriched.loc[0, "Long description"] == "Acme builds autonomous agents for support teams."
    assert enriched.loc[0, "employee_count"] == "11-50"
    assert enriched.loc[0, "rank"] == "42"


def test_build_crawl_queue_filters_invalid_urls():
    enriched = pd.DataFrame([
        {"org_uuid": "1", "name": "Valid", "homepage_url": "https://valid.test", "status": "operating", "short_description": ""},
        {"org_uuid": "2", "name": "Invalid", "homepage_url": "invalid.test", "status": "operating", "short_description": ""},
    ])

    queue = build_crawl_queue(enriched)

    assert queue["org_uuid"].tolist() == ["1"]


def test_tavily_config_uses_cost_control_defaults():
    payload = TavilyCrawlConfig().request_payload("https://example.com")

    assert payload["limit"] == 5
    assert payload["extract_depth"] == "basic"
    assert payload["include_usage"] is True
    assert payload["allow_external"] is False
    assert "instructions" in payload


def test_extract_usage_credits_accepts_common_shapes():
    assert extract_usage_credits({"usage": {"total_credits": 2}}) == 2.0
    assert extract_usage_credits({"usage": {"credits": "3.5"}}) == 3.5
    assert extract_usage_credits({"usage": {"map": 1, "extract": 2}}) == 3.0


def test_compact_tavily_response_builds_source_linked_evidence():
    pages_used, evidence = compact_tavily_response({
        "results": [
            {"url": "https://acme.test/product", "raw_content": "Product uses LLM agents."},
            {"url": "https://acme.test/about", "raw_content": "About our proprietary AI."},
        ]
    })

    assert pages_used == "https://acme.test/product | https://acme.test/about"
    assert "[Page 1: product]" in evidence
    assert "Product uses LLM agents." in evidence


def test_build_classifier_input_with_evidence(tmp_path):
    enriched = pd.DataFrame([
        {
            "org_uuid": "org-1",
            "name": "Acme",
            "homepage_url": "https://acme.test",
            "short_description": "Builds AI agents.",
        }
    ])
    enriched_path = tmp_path / "enriched.csv"
    raw_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "output.csv"
    enriched.to_csv(enriched_path, index=False)
    raw_path.write_text(
        json.dumps({
            "org_uuid": "org-1",
            "ok": True,
            "usage_credits": 2,
            "response": {
                "results": [
                    {"url": "https://acme.test/product", "raw_content": "AI agent platform."}
                ]
            },
        }) + "\n",
        encoding="utf-8",
    )

    report = build_classifier_input_with_evidence(enriched_path, raw_path, output_path)
    output = pd.read_csv(output_path, dtype=str, keep_default_na=False)

    assert report.rows_with_website_evidence == 1
    assert output.loc[0, "website_crawl_status"] == "success"
    assert "AI agent platform." in output.loc[0, "website_evidence"]
