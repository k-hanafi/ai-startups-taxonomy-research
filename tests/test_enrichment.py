import json

import pandas as pd

from src.enrichment import (
    CLASSIFIER_INPUT_COLUMNS,
    build_crawl_queue,
    build_enriched_dataset,
    is_valid_homepage_url,
    write_enrichment_outputs,
)
from src.tavily_crawl import TavilyCrawlConfig, extract_usage_credits, run_tavily_crawl
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
    assert list(enriched.columns) == list(CLASSIFIER_INPUT_COLUMNS)


def test_build_crawl_queue_filters_invalid_urls():
    enriched = pd.DataFrame([
        {"org_uuid": "1", "name": "Valid", "homepage_url": "https://valid.test", "short_description": ""},
        {"org_uuid": "2", "name": "Invalid", "homepage_url": "invalid.test", "short_description": ""},
    ])

    queue = build_crawl_queue(enriched)

    assert queue["org_uuid"].tolist() == ["1"]
    assert list(queue.columns) == ["org_uuid", "name", "homepage_url", "short_description"]


def test_build_crawl_queue_respects_website_alive_column():
    enriched = pd.DataFrame([
        {
            "org_uuid": "1",
            "name": "Live",
            "homepage_url": "https://live.test",
            "short_description": "",
            "website_alive": "true",
        },
        {
            "org_uuid": "2",
            "name": "Dead",
            "homepage_url": "https://dead.test",
            "short_description": "",
            "website_alive": "false",
        },
        {
            "org_uuid": "3",
            "name": "Unset",
            "homepage_url": "https://unset.test",
            "short_description": "",
            "website_alive": "",
        },
    ])

    queue = build_crawl_queue(enriched)

    assert queue["org_uuid"].tolist() == ["1"]
    assert list(queue.columns) == ["org_uuid", "name", "homepage_url", "short_description"]


def test_write_enrichment_outputs_preserves_website_alive(tmp_path):
    subset = pd.DataFrame([
        {
            "rcid": "1",
            "org_uuid": "org-1",
            "name": "Acme",
            "cb_url": "https://crunchbase.com/acme",
            "homepage_url": "https://acme.test",
            "short_description": "Builds AI agents.",
            "category_list": "AI",
            "category_groups_list": "Software",
            "created_date": "01jan2024",
            "founded_date": "01jan2024",
            "description": "Long text.",
        }
    ])
    master = pd.DataFrame([
        {
            "org_uuid": "org-1",
            "rank": "1",
            "state_code": "CA",
            "region": "CA",
            "city": "SF",
            "status": "operating",
            "num_funding_rounds": "0",
            "total_funding_usd": "0",
            "employee_count": "1-10",
            "linkedin_url": "",
            "twitter_url": "",
            "facebook_url": "",
            "year_created": "2024",
            "updated_date": "",
            "last_funding_date": "",
            "closed_date": "",
        }
    ])
    subset_path = tmp_path / "subset.csv"
    master_path = tmp_path / "master.csv"
    enriched_path = tmp_path / "classifier_input.csv"
    subset.to_csv(subset_path, index=False)
    master.to_csv(master_path, index=False)

    pd.DataFrame([
        {
            "org_uuid": "org-1",
            "name": "Acme",
            "homepage_url": "https://acme.test",
            "short_description": "x",
            "Long description": "",
            "category_list": "",
            "category_groups_list": "",
            "founded_date": "",
            "employee_count": "",
            "total_funding_usd": "",
            "website_alive": "false",
            "website_pages_used": "",
            "website_evidence": "",
        }
    ]).to_csv(enriched_path, index=False)

    report = write_enrichment_outputs(
        subset_csv=subset_path,
        master_csv=master_path,
        enriched_csv=enriched_path,
    )
    out = pd.read_csv(enriched_path, dtype=str, keep_default_na=False)

    assert out.loc[0, "website_alive"] == "false"
    assert report.tavily_eligible_rows == 0
    assert list(out.columns) == list(CLASSIFIER_INPUT_COLUMNS)


def test_tavily_config_uses_cost_control_defaults():
    payload = TavilyCrawlConfig().request_payload("https://example.com")

    assert payload["limit"] == 5
    assert payload["max_breadth"] == 20
    assert payload["chunks_per_source"] == 4
    assert payload["extract_depth"] == "basic"
    assert payload["include_usage"] is True
    assert payload["allow_external"] is False
    assert "instructions" in payload
    assert len(payload["instructions"]) <= 400
    assert "proprietary models" not in payload["instructions"]


def test_tavily_fallback_payload_omits_instruction_only_fields():
    payload = TavilyCrawlConfig(instructions="", chunks_per_source=1).request_payload("https://example.com")

    assert "instructions" not in payload
    assert "chunks_per_source" not in payload


def test_extract_usage_credits_accepts_common_shapes():
    assert extract_usage_credits({"usage": {"total_credits": 2}}) == 2.0
    assert extract_usage_credits({"usage": {"credits": "3.5"}}) == 3.5
    assert extract_usage_credits({"usage": {"map": 1, "extract": 2}}) == 3.0


def test_compact_tavily_response_builds_source_linked_evidence():
    pages_used, evidence = compact_tavily_response({
        "results": [
            {"url": "https://acme.test/product", "raw_content": "Product uses LLM agents."},
            {"url": "https://acme.test/about", "raw_content": "About our proprietary AI."},
            {"url": "https://acme.test/", "raw_content": "Homepage overview."},
        ]
    })

    assert pages_used == "https://acme.test/product | https://acme.test/about | https://acme.test/"
    assert "[Page 1: product]" in evidence
    assert "[Page 2: about]" in evidence
    assert "[Page 3: homepage]" in evidence
    assert "Product uses LLM agents." in evidence


def test_compact_tavily_response_does_not_truncate_by_default():
    long_content = "x" * 10_000
    _, evidence = compact_tavily_response({
        "results": [
            {"url": "https://acme.test/product", "raw_content": long_content},
        ]
    })

    assert long_content in evidence
    assert "[truncated]" not in evidence


def test_compact_tavily_response_removes_common_boilerplate():
    pages_used, evidence = compact_tavily_response({
        "results": [
            {
                "url": "https://acme.test/product",
                "raw_content": "\n".join([
                    "top of page",
                    "![Hero](https://acme.test/hero.png)",
                    "Book a Demo",
                    "AI workflow platform for insurance teams.",
                    "AI workflow platform for insurance teams.",
                    "All rights reserved",
                ]),
            }
        ]
    })

    assert pages_used == "https://acme.test/product"
    assert "AI workflow platform for insurance teams." in evidence
    assert evidence.count("AI workflow platform for insurance teams.") == 1
    assert "hero.png" not in evidence
    assert "Book a Demo" not in evidence
    assert "top of page" not in evidence


def test_run_tavily_crawl_falls_back_on_empty_results(tmp_path, monkeypatch):
    queue = pd.DataFrame([
        {
            "org_uuid": "org-1",
            "name": "Acme",
            "homepage_url": "https://acme.test",
            "short_description": "",
        }
    ])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)
    calls = []

    def fake_call(url, config, api_key):
        calls.append(config.instructions)
        if len(calls) == 1:
            return {"results": [], "usage": {"credits": 0}}
        return {
            "results": [
                {"url": "https://acme.test/", "raw_content": "Recovered homepage content."}
            ],
            "usage": {"credits": 1},
        }

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", fake_call)

    report = run_tavily_crawl(queue_path, output_path, state_path)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert report.completed == 1
    assert report.empty_results == 0
    assert calls[0]
    assert calls[1] == ""
    assert records[0]["crawl_status"] == "success_fallback"
    assert records[0]["usage_credits"] == 1


def test_run_tavily_crawl_records_terminal_empty_results(tmp_path, monkeypatch):
    queue = pd.DataFrame([
        {
            "org_uuid": "org-empty",
            "name": "EmptyCo",
            "homepage_url": "https://empty.test",
            "short_description": "",
        }
    ])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr(
        "src.tavily_crawl.call_tavily_crawl",
        lambda url, config, api_key: {"results": [], "usage": {"credits": 0}},
    )

    report = run_tavily_crawl(queue_path, output_path, state_path)
    report_again = run_tavily_crawl(queue_path, output_path, state_path)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert report.empty_results == 1
    assert report_again.skipped_existing == 1
    assert len(records) == 1
    assert records[0]["crawl_status"] == "empty_results"
    assert records[0]["retryable"] is False


def test_run_tavily_crawl_retries_transient_errors(tmp_path, monkeypatch):
    queue = pd.DataFrame([
        {
            "org_uuid": "org-retry",
            "name": "RetryCo",
            "homepage_url": "https://retry.test",
            "short_description": "",
        }
    ])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)
    calls = 0

    def flaky_call(url, config, api_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out")
        return {
            "results": [
                {"url": "https://retry.test/", "raw_content": "Recovered after retry."}
            ],
            "usage": {"credits": 1},
        }

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", flaky_call)

    report = run_tavily_crawl(
        queue_path,
        output_path,
        state_path,
        config=TavilyCrawlConfig(retry_backoff_seconds=0),
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert calls == 2
    assert report.completed == 1
    assert records[0]["crawl_status"] == "success"


def test_run_tavily_crawl_uses_canonical_url(tmp_path, monkeypatch):
    queue = pd.DataFrame([
        {
            "org_uuid": "org-canon",
            "name": "CanonCo",
            "homepage_url": "https://www.acme.test",
            "short_description": "",
            "website_alive": "true",
        }
    ])
    queue_path = tmp_path / "queue.csv"
    output_path = tmp_path / "raw.jsonl"
    state_path = tmp_path / "state.json"
    queue.to_csv(queue_path, index=False)
    called_urls = []

    def fake_call(url, config, api_key):
        called_urls.append(url)
        return {
            "results": [
                {"url": url, "raw_content": "Canonical homepage content."}
            ],
            "usage": {"credits": 1},
        }

    monkeypatch.setattr("src.tavily_crawl._api_key", lambda: "test-key")
    monkeypatch.setattr("src.tavily_crawl.resolve_canonical_url", lambda url, timeout=15.0: "https://acme.test/")
    monkeypatch.setattr("src.tavily_crawl.call_tavily_crawl", fake_call)

    run_tavily_crawl(queue_path, output_path, state_path)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert called_urls == ["https://acme.test/"]
    assert records[0]["homepage_url"] == "https://www.acme.test"
    assert records[0]["canonical_homepage_url"] == "https://acme.test/"


def test_build_classifier_input_with_evidence(tmp_path):
    enriched = pd.DataFrame([
        {
            "org_uuid": "org-1",
            "name": "Acme",
            "homepage_url": "https://acme.test",
            "short_description": "Builds AI agents.",
            "Long description": "",
            "category_list": "",
            "category_groups_list": "",
            "founded_date": "",
            "employee_count": "",
            "total_funding_usd": "",
            "website_alive": "true",
            "website_pages_used": "",
            "website_evidence": "",
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
    assert "AI agent platform." in output.loc[0, "website_evidence"]
    assert list(output.columns) == list(CLASSIFIER_INPUT_COLUMNS)


def test_build_classifier_input_clears_evidence_when_website_alive_false(tmp_path):
    enriched = pd.DataFrame([
        {
            "org_uuid": "org-dead",
            "name": "DeadCo",
            "homepage_url": "https://dead.test",
            "short_description": "x",
            "Long description": "",
            "category_list": "",
            "category_groups_list": "",
            "founded_date": "",
            "employee_count": "",
            "total_funding_usd": "",
            "website_alive": "false",
            "website_pages_used": "",
            "website_evidence": "",
        }
    ])
    enriched_path = tmp_path / "enriched.csv"
    raw_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "output.csv"
    enriched.to_csv(enriched_path, index=False)
    raw_path.write_text(
        json.dumps({
            "org_uuid": "org-dead",
            "ok": True,
            "usage_credits": 2,
            "response": {
                "results": [
                    {"url": "https://dead.test/", "raw_content": "Should be ignored for dead URL."}
                ]
            },
        }) + "\n",
        encoding="utf-8",
    )

    build_classifier_input_with_evidence(enriched_path, raw_path, output_path)
    output = pd.read_csv(output_path, dtype=str, keep_default_na=False)

    assert output.loc[0, "website_evidence"] == ""
    assert output.loc[0, "website_pages_used"] == ""
