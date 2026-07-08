"""Offline tests for the Stage 3 runner.

No API key and no 249 MB classifier_input.csv are required: request-building,
identity hashing, resume bookkeeping, and record extraction are all exercised
against in-memory fixtures.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evals import runner


@pytest.fixture
def sample_row() -> dict[str, str]:
    return {
        "org_uuid": "abc123-uuid",
        "name": "Acme AI",
        "short_description": "An AI thing.",
        "Long description": "A longer description of the AI thing.",
        "category_list": "Artificial Intelligence (AI)",
        "category_groups_list": "Software",
        "founded_date": "2023-05",
        "employee_count": "1-10",
        "total_funding_usd": "1000000",
        "website_pages_used": "https://acme.ai/",
        "website_evidence": "We build AI agents.",
    }


def test_request_kwargs_are_production_faithful(sample_row):
    from src.builder import _openai_strict_schema, load_system_prompt

    system_prompt = load_system_prompt()
    schema = _openai_strict_schema()
    kwargs = runner.build_request_kwargs(
        sample_row, system_prompt, schema, "gpt-5.4-nano", "medium"
    )

    # Prompt + schema + input come verbatim from production builder.
    assert kwargs["instructions"] == system_prompt
    assert kwargs["input"].startswith("CompanyID: abc123-uuid")
    assert kwargs["text"]["format"]["name"] == "ClassificationResult"
    assert kwargs["store"] is False

    # Experimental deltas are present and match config.
    from evals import config as cfg

    assert kwargs["reasoning"] == {"effort": "medium"}
    assert kwargs["max_output_tokens"] == cfg.MAX_OUTPUT_TOKENS
    # Reasoning models reject temperature, so it is omitted by default.
    assert ("temperature" in kwargs) == cfg.SEND_TEMPERATURE
    # A reasoning effort rejects logprobs, so they must not be sent.
    assert "top_logprobs" not in kwargs
    assert "include" not in kwargs


def test_logprobs_captured_only_when_reasoning_off(sample_row):
    from src.builder import _openai_strict_schema, load_system_prompt
    from evals import config as cfg

    sp = load_system_prompt()
    schema = _openai_strict_schema()

    off = runner.build_request_kwargs(sample_row, sp, schema, "gpt-5.4-nano", cfg.REASONING_OFF)
    assert off["top_logprobs"] == cfg.TOP_LOGPROBS
    assert off["include"] == list(cfg.LOGPROB_INCLUDE)

    on = runner.build_request_kwargs(sample_row, sp, schema, "gpt-5.4-nano", "high")
    assert "top_logprobs" not in on
    assert "include" not in on


def test_identity_hashes_are_stable_and_shaped():
    a = runner.identity_hashes()
    b = runner.identity_hashes()
    assert a == b
    assert set(a) == {"prompt_sha256", "schema_sha256", "formatter_sha256"}
    assert all(len(v) == 64 for v in a.values())


def test_make_run_id_encodes_config():
    rid = runner.make_run_id("gpt-5.4-nano", "high", 2)
    assert rid.endswith("_gpt-5.4-nano_high_r2")


def test_completed_custom_ids_reads_jsonl(tmp_path):
    p = tmp_path / "predictions.jsonl"
    p.write_text(
        json.dumps({"custom_id": "startup-a"}) + "\n"
        + json.dumps({"custom_id": "startup-b"}) + "\n",
        encoding="utf-8",
    )
    assert runner._completed_custom_ids(p) == {"startup-a", "startup-b"}


def test_completed_custom_ids_missing_file(tmp_path):
    assert runner._completed_custom_ids(tmp_path / "nope.jsonl") == set()


def test_completed_custom_ids_tolerates_truncated_final_line(tmp_path):
    p = tmp_path / "predictions.jsonl"
    p.write_text(
        json.dumps({"custom_id": "startup-a"}) + "\n"
        + '{"custom_id": "startup-b", "subcl',  # killed mid-append
        encoding="utf-8",
    )
    assert runner._completed_custom_ids(p) == {"startup-a"}


def test_negative_limit_rejected():
    with pytest.raises(ValueError):
        runner.run(limit=-1, dry_run=True)


def test_resume_config_mismatch_refused(tmp_path, monkeypatch):
    from evals import paths

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(runner, "run_config_path",
                        lambda rid: tmp_path / rid / "config.json")
    (tmp_path / "run1").mkdir()
    # A prior config for a different model must block resume under run1.
    runner.run_config_path("run1").write_text(
        json.dumps({"model": "gpt-5.4-mini", "reasoning_effort": "medium",
                    "prompt_sha256": "x", "schema_sha256": "y",
                    "formatter_sha256": "z"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        runner._ensure_config("run1", "gpt-5.4-nano", "medium", 1, 100)


def test_prediction_record_extracts_labels_and_usage():
    resp = SimpleNamespace(
        status="completed",
        output_text=json.dumps(
            {
                "ai_native": 1,
                "subclass": "1E",
                "rad_score": "RAD-M",
                "cohort": "GENAI-ERA",
                "conf_classification": 4,
                "conf_rad": 3,
            }
        ),
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=350,
            output_tokens_details=SimpleNamespace(reasoning_tokens=180),
        ),
    )
    rec = runner._prediction_record("startup-x", "x", "gpt-5.4-nano", "medium", resp,
                                    latency_s=2.345)
    assert rec["subclass"] == "1E"
    assert rec["ai_native"] == 1
    assert rec["reasoning_tokens"] == 180
    assert rec["input_tokens"] == 1200
    assert rec["org_uuid"] == "x"
    assert rec["latency_s"] == 2.345


def test_prediction_record_survives_empty_output():
    resp = SimpleNamespace(status="incomplete", output_text="", usage=None)
    rec = runner._prediction_record("startup-y", "y", "gpt-5.4-nano", "medium", resp)
    assert rec["subclass"] is None
    assert rec["status"] == "incomplete"
    assert rec["input_tokens"] is None
    assert rec["latency_s"] is None
