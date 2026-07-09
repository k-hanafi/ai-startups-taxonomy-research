"""Offline tests for the Stage 5 two-pass classifier. No API key, no big CSVs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evals import config as cfg
from evals import two_pass


@pytest.fixture
def sample_row() -> dict[str, str]:
    return {
        "org_uuid": "abc123-uuid",
        "name": "Acme AI",
        "short_description": "An AI thing.",
        "Long description": "A longer description.",
        "category_list": "Artificial Intelligence (AI)",
        "category_groups_list": "Software",
        "founded_date": "2023-05",
        "employee_count": "1-10",
        "total_funding_usd": "1000000",
        "website_pages_used": "https://acme.ai/",
        "website_evidence": "We build AI agents.",
    }


# --- cohort ---------------------------------------------------------------

@pytest.mark.parametrize("founded,expected", [
    ("2023-05", "GENAI-ERA"),
    ("2023-03", "GENAI-ERA"),
    ("2023-02", "PRE-GENAI"),
    ("2022-12", "PRE-GENAI"),
    ("01nov2016", "PRE-GENAI"),
    ("2024", "GENAI-ERA"),
    ("2023", "PRE-GENAI"),   # bare year -> January -> before March boundary
    ("", "PRE-GENAI"),
    ("nan", "PRE-GENAI"),
])
def test_compute_cohort(founded, expected):
    assert two_pass.compute_cohort(founded) == expected


# --- messages ---------------------------------------------------------------

def test_pass_a_message_drops_website_pages(sample_row):
    msg = two_pass.pass_a_message(sample_row)
    assert "Website Pages Used" not in msg
    assert "Website Evidence:" in msg
    assert sample_row["website_pages_used"] not in msg


def test_pass_b_message_appends_conditioning_fields(sample_row):
    msg = two_pass.pass_b_message(sample_row, 1, "GENAI-ERA")
    assert msg.endswith("PriorBinaryVerdict: 1\nCohort: GENAI-ERA")
    assert "Website Evidence:" in msg


# --- schemas / family constraint --------------------------------------------

def test_family_schema_selection(sample_row):
    kw1 = two_pass.pass_b_kwargs(sample_row, 1, "GENAI-ERA", "gpt-5.4-nano", "high")
    kw0 = two_pass.pass_b_kwargs(sample_row, 0, "GENAI-ERA", "gpt-5.4-nano", "high")

    subs1 = kw1["text"]["format"]["schema"]["properties"]["subclass"]["enum"]
    subs0 = kw0["text"]["format"]["schema"]["properties"]["subclass"]["enum"]
    assert subs1 == ["1A", "1B", "1C", "1D", "1E", "1F", "1G"]
    assert subs0 == ["0A", "0B", "0C"]

    # RAD fields exist only in the AI-native family schema.
    assert "rad_score" in kw1["text"]["format"]["schema"]["properties"]
    assert "rad_score" not in kw0["text"]["format"]["schema"]["properties"]
    assert "conf_rad" not in kw0["text"]["format"]["schema"]["properties"]


def test_pass_b_prompt_embeds_matching_family_block():
    p1 = two_pass.load_pass_b_prompt(1)
    p0 = two_pass.load_pass_b_prompt(0)
    assert "{family_block}" not in p1 and "{family_block}" not in p0
    assert "1A | Foundation Layer" in p1 and "0A |" not in p1.split("ANALYTICAL")[0][:2000]
    assert "0A | Traditional Tech / SaaS" in p0 and "1A | Foundation Layer" not in p0


def test_strict_schemas_forbid_extra_properties():
    for cls in (two_pass.BinaryResult, two_pass.SubclassResultAI,
                two_pass.SubclassResultNot):
        assert two_pass.strict_schema(cls)["additionalProperties"] is False


# --- request params -----------------------------------------------------------

def test_pass_a_sends_logprobs_pass_b_does_not(sample_row):
    ka = two_pass.pass_a_kwargs(sample_row, "PROMPT", "gpt-5.4-nano")
    kb = two_pass.pass_b_kwargs(sample_row, 1, "GENAI-ERA", "gpt-5.4-nano", "high")

    assert ka["reasoning"] == {"effort": cfg.PASS_A_EFFORT}
    assert ka["top_logprobs"] == cfg.TOP_LOGPROBS
    assert ka["include"] == list(cfg.LOGPROB_INCLUDE)

    assert kb["reasoning"] == {"effort": "high"}
    assert "top_logprobs" not in kb
    assert "include" not in kb
    # Separate cache routes per pass.
    assert ka["prompt_cache_key"] != kb["prompt_cache_key"]


# --- record assembly -----------------------------------------------------------

def _resp(status: str, payload: dict | None, reasoning: int = 0,
          cached: int = 0):
    return SimpleNamespace(
        status=status,
        output_text=json.dumps(payload) if payload is not None else "",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=50,
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


def test_assemble_record_ai_native_family():
    ra = _resp("completed", {"ai_native": 1}, cached=10)
    rb = _resp("completed", {
        "subclass": "1E", "rad_score": "RAD-M", "conf_classification": 4,
        "conf_rad": 3, "reasons_3_points": "a | b | c", "sources_used": "keywords",
        "verification_critique": "ok", "boundary_disagreement": False,
    }, reasoning=500, cached=40)
    rec = two_pass.assemble_record("startup-x", "x", "gpt-5.4-nano", "high",
                                   "GENAI-ERA", ra, rb,
                                   latency_a_s=0.8, latency_b_s=12.4)
    assert rec["status"] == "completed"
    assert rec["ai_native"] == 1 and rec["subclass"] == "1E"
    assert rec["rad_score"] == "RAD-M" and rec["conf_rad"] == 3
    assert rec["cohort"] == "GENAI-ERA"
    assert rec["b_reasoning_tokens"] == 500
    assert rec["a_cached_tokens"] == 10
    assert rec["b_cached_tokens"] == 40
    assert rec["cached_tokens"] == 50
    # Per-pass latencies land separately plus a flat total under the
    # single-pass field name, so the scorer reads one field for both shapes.
    assert rec["a_latency_s"] == 0.8
    assert rec["b_latency_s"] == 12.4
    assert rec["latency_s"] == pytest.approx(13.2)


def test_assemble_record_zero_family_forces_rad_na():
    ra = _resp("completed", {"ai_native": 0})
    rb = _resp("completed", {
        "subclass": "0B", "conf_classification": 5, "reasons_3_points": "a | b | c",
        "sources_used": "keywords", "verification_critique": "ok",
        "boundary_disagreement": True,
    })
    rec = two_pass.assemble_record("startup-y", "y", "gpt-5.4-nano", "high",
                                   "PRE-GENAI", ra, rb)
    assert rec["rad_score"] == "RAD-NA"
    assert rec["conf_rad"] is None
    assert rec["boundary_disagreement"] is True


def test_assemble_record_failed_pass_a_not_completed():
    ra = _resp("incomplete", None)
    rec = two_pass.assemble_record("startup-z", "z", "gpt-5.4-nano", "high",
                                   "PRE-GENAI", ra, None, latency_a_s=1.5)
    assert rec["status"] != "completed"
    assert rec["subclass"] is None and rec["rad_score"] is None
    # No Pass B means no meaningful end-to-end latency: total stays None.
    assert rec["a_latency_s"] == 1.5
    assert rec["latency_s"] is None
    assert "b_latency_s" not in rec


def test_assemble_record_parse_failure_never_marked_completed():
    # Pass A API-completed but output unparseable, Pass B never ran: the row
    # must stay retryable on resume.
    ra = SimpleNamespace(status="completed", output_text="not json", usage=None)
    rec = two_pass.assemble_record("startup-w", "w", "gpt-5.4-nano", "high",
                                   "PRE-GENAI", ra, None)
    assert rec["status"] != "completed"

    # Both passes API-completed but Pass B output unparseable: same rule.
    ra2 = _resp("completed", {"ai_native": 1})
    rb2 = SimpleNamespace(status="completed", output_text="{broken", usage=None)
    rec2 = two_pass.assemble_record("startup-v", "v", "gpt-5.4-nano", "high",
                                    "PRE-GENAI", ra2, rb2)
    assert rec2["status"] == "parse_failed"


def test_resume_config_refuses_repeat_or_row_count_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(two_pass, "run_config_path",
                        lambda rid: tmp_path / rid / "config.json")
    monkeypatch.setattr(two_pass, "identity_hashes", lambda: {
        "prompt_a_sha256": "pa",
        "prompt_b_family1_sha256": "pb1",
        "prompt_b_family0_sha256": "pb0",
        "schema_a_sha256": "sa",
        "schema_b1_sha256": "sb1",
        "schema_b0_sha256": "sb0",
        "formatter_sha256": "fmt",
    })
    monkeypatch.setattr(two_pass, "_git_commit", lambda: "abc123")

    (tmp_path / "repeat-run").mkdir()
    two_pass._ensure_config("repeat-run", "gpt-5.4-nano", "high", repeat=1, n_rows=10)
    with pytest.raises(SystemExit, match="repeat"):
        two_pass._ensure_config("repeat-run", "gpt-5.4-nano", "high", repeat=2, n_rows=10)

    (tmp_path / "limit-run").mkdir()
    two_pass._ensure_config("limit-run", "gpt-5.4-nano", "high", repeat=1, n_rows=10)
    with pytest.raises(SystemExit, match="n_rows"):
        two_pass._ensure_config("limit-run", "gpt-5.4-nano", "high", repeat=1, n_rows=20)
