"""Offline tests for the Stage 6 logprob extractor. No API key, no network.

Two layers:
- Synthetic payloads exercise each failure mode the extractor guards against
  (reconstruction drift, masked sentinels, missing chosen token, variable
  token position, fused punctuation).
- Committed fixtures (evals/tests/fixtures/) pin the REAL gpt-5.4-nano
  tokenization from the banked 2026-07-06 effort=none run, anonymized by
  capture_fixtures.py. If the API's logprob shape or the tokenizer changes,
  these fail before any paid run does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evals import logprob_extract as lpx

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Synthetic payload builders
# ---------------------------------------------------------------------------

def entry(token: str, logprob: float, top: list[tuple[str, float]] | None = None,
          include_self: bool = True) -> dict:
    """One logprobs entry; the chosen token leads its own top list by default."""
    tops = []
    if include_self:
        tops.append({"token": token, "bytes": list(token.encode()), "logprob": logprob})
    for t, lp in top or []:
        tops.append({"token": t, "bytes": list(t.encode()), "logprob": lp})
    return {
        "token": token,
        "bytes": list(token.encode()),
        "logprob": logprob,
        "top_logprobs": tops,
    }


def response_from_entries(entries: list[dict]) -> dict:
    text = b"".join(bytes(e["bytes"]) for e in entries).decode()
    return {
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "logprobs": entries,
            }],
        }],
    }


def pass_a_response(digit: str, decision_entry: dict) -> dict:
    """A minimal Pass A payload: {"ai_native":D} across 4 tokens."""
    assert decision_entry["token"] == digit
    return response_from_entries([
        entry('{"', -0.001, [('{\n', -6.6)]),
        entry("ai_native", 0.0, [("!", -100.0), ('"', -100.0)]),
        entry('":', 0.0),
        decision_entry,
        entry("}", -0.0001),
    ])


# ---------------------------------------------------------------------------
# Byte reconstruction
# ---------------------------------------------------------------------------

def test_reconstruction_mismatch_refuses_extraction():
    resp = pass_a_response("1", entry("1", -0.1, [("0", -2.4)]))
    resp["output"][0]["content"][0]["text"] = '{"ai_native":1} '  # extra byte
    with pytest.raises(lpx.LogprobExtractionError, match="reconstruct"):
        lpx.extract_binary_confidence(resp)


def test_multibyte_text_before_decision_token():
    # Non-ASCII earlier in the output must not desync the byte->char mapping.
    entries = [
        entry('{"', -0.001),
        entry("note", 0.0),
        entry('":"', 0.0),
        entry("caf\u00e9 \u2192 ok", -0.2),
        entry('","', 0.0),
        entry("ai_native", 0.0),
        entry('":', 0.0),
        entry("1", -0.3, [("0", -1.5)]),
        entry("}", 0.0),
    ]
    result = lpx.extract_binary_confidence(response_from_entries(entries))
    assert result.ai_native == 1
    assert result.decision_token_index == 7


# ---------------------------------------------------------------------------
# Structural location (never by token index)
# ---------------------------------------------------------------------------

def test_decision_token_found_at_variable_position():
    # Same decision, shifted by an extra preceding field: index must follow.
    short = pass_a_response("0", entry("0", -0.05, [("1", -3.0)]))
    assert lpx.extract_binary_confidence(short).decision_token_index == 3

    entries = [
        entry('{"', -0.001),
        entry("pad", 0.0), entry('":', 0.0), entry("7", 0.0), entry(',"', 0.0),
        entry("ai_native", 0.0),
        entry('":', 0.0),
        entry("0", -0.05, [("1", -3.0)]),
        entry("}", 0.0),
    ]
    assert lpx.extract_binary_confidence(
        response_from_entries(entries)
    ).decision_token_index == 7


def test_key_lookalike_inside_string_value_is_ignored():
    # Evidence text quoting '"ai_native": 1' must not hijack the locator.
    lure = 'we say \\"ai_native\\": 1 a lot'
    entries = [
        entry('{"', 0.0),
        entry("quote", 0.0),
        entry('":"', 0.0),
        entry(lure, 0.0),
        entry('","', 0.0),
        entry("ai_native", 0.0),
        entry('":', 0.0),
        entry("0", -0.02, [("1", -4.0)]),
        entry("}", 0.0),
    ]
    result = lpx.extract_binary_confidence(response_from_entries(entries))
    assert result.ai_native == 0
    assert result.decision_token_index == 7


def test_first_token_is_never_the_decision():
    # The classic trap: '{"' opens with ~1.0 probability; naive index-0
    # extraction would report certainty regardless of the actual verdict.
    resp = pass_a_response("1", entry("1", -0.693147, [("0", -0.693147)]))
    result = lpx.extract_binary_confidence(resp)
    assert result.decision_token_index != 0
    assert result.top1_prob == pytest.approx(0.5, abs=1e-6)


def test_missing_key_raises():
    entries = [entry('{"', 0.0), entry("other", 0.0), entry('":', 0.0),
               entry("1", 0.0), entry("}", 0.0)]
    with pytest.raises(lpx.LogprobExtractionError, match="ai_native"):
        lpx.extract_binary_confidence(response_from_entries(entries))


# ---------------------------------------------------------------------------
# Masked sentinels + candidate pool
# ---------------------------------------------------------------------------

def test_masked_sentinels_excluded_from_renormalization():
    # A masked '0' at -100.0 must contribute nothing; exp(-100) would still
    # be a nonzero float and silently skew p_one if treated as real.
    masked = entry("1", -0.1, [("0", -100.0), (" ", -14.0)])
    result = lpx.extract_binary_confidence(pass_a_response("1", masked))
    assert result.p_one == 1.0
    assert result.valid_mass == pytest.approx(math.exp(-0.1))

    # The same '0' at a REAL logprob near the sentinel must be kept in the
    # pool (only EXACTLY -100.0 is a mask). Its mass is below float
    # resolution, so assert on the pool, not the renormalized probability.
    real = entry("1", -0.1, [("0", -99.9), (" ", -14.0)])
    assert lpx.binary_candidate_pool(real) == {"1": -0.1, "0": -99.9}
    assert lpx.binary_candidate_pool(masked) == {"1": -0.1}


def test_chosen_token_missing_from_its_top_list_is_merged():
    decision = entry("1", -0.4, [("0", -1.1), (" ", -9.0)], include_self=False)
    assert all(t["token"] != "1" for t in decision["top_logprobs"])
    result = lpx.extract_binary_confidence(pass_a_response("1", decision))
    expected_p1 = math.exp(-0.4) / (math.exp(-0.4) + math.exp(-1.1))
    assert result.p_one == pytest.approx(expected_p1)


def test_chosen_token_duplicated_in_top_list_not_double_counted():
    decision = entry("1", -0.4, [("0", -1.1)], include_self=True)
    result = lpx.extract_binary_confidence(pass_a_response("1", decision))
    assert result.valid_mass == pytest.approx(math.exp(-0.4) + math.exp(-1.1))


def test_fused_punctuation_candidates_count_toward_their_digit():
    decision = entry("1", -0.5, [("0", -1.5), ("1,", -2.5), ("0}", -3.0)])
    result = lpx.extract_binary_confidence(pass_a_response("1", decision))
    mass1 = math.exp(-0.5) + math.exp(-2.5)
    mass0 = math.exp(-1.5) + math.exp(-3.0)
    assert result.p_one == pytest.approx(mass1 / (mass1 + mass0))
    assert result.valid_mass == pytest.approx(mass1 + mass0)


@pytest.mark.parametrize("token,expected", [
    ("0", 0), ("1", 1), (" 1", 1), ("1,", 1), ("0}", 0), ("1 ,", 1),
    ("2", None), ("10", None), (" ", None), ("\t", None), ("true", None),
    ("", None), (",", None),
])
def test_candidate_value(token, expected):
    assert lpx.candidate_value(token) == expected


def test_all_candidates_masked_raises():
    decision = entry("1", -100.0, [("0", -100.0)], include_self=False)
    decision["top_logprobs"].insert(
        0, {"token": "1", "bytes": [49], "logprob": -100.0}
    )
    with pytest.raises(lpx.LogprobExtractionError, match="renormalize"):
        lpx.extract_binary_confidence(pass_a_response("1", decision))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_at_a_perfect_coin_flip():
    lp = math.log(0.5)
    decision = entry("1", lp, [("0", lp)])
    result = lpx.extract_binary_confidence(pass_a_response("1", decision))
    assert result.p_one == pytest.approx(0.5)
    assert result.margin == pytest.approx(0.0)
    assert result.entropy_bits == pytest.approx(1.0)
    assert result.valid_mass == pytest.approx(1.0)


def test_valid_mass_reports_pre_renormalization_leak():
    # 10% of the raw mass leaked to a whitespace token: p_one renormalizes
    # over {0,1} but valid_mass must expose the leak (gate Q3 evidence).
    decision = entry("1", math.log(0.6), [("0", math.log(0.3)),
                                          (" ", math.log(0.1))])
    result = lpx.extract_binary_confidence(pass_a_response("1", decision))
    assert result.valid_mass == pytest.approx(0.9)
    assert result.p_one == pytest.approx(0.6 / 0.9)
    assert result.margin == pytest.approx((0.6 - 0.3) / 0.9)


def test_entropy_saturated_is_zero():
    decision = entry("0", 0.0, [("1", -30.0)])
    result = lpx.extract_binary_confidence(pass_a_response("0", decision))
    assert result.entropy_bits == pytest.approx(0.0, abs=1e-8)
    assert result.top1_prob == pytest.approx(1.0)
    assert result.ai_native == 0


# ---------------------------------------------------------------------------
# Fixture-driven end-to-end (real tokenization, anonymized)
# ---------------------------------------------------------------------------

def fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))


def test_fixtures_exist():
    assert len(fixture_paths()) >= 4, (
        "anonymized fixtures missing; regenerate with "
        "python -m evals.tests.capture_fixtures"
    )


@pytest.mark.parametrize("path", fixture_paths(), ids=lambda p: p.stem)
def test_fixture_end_to_end(path: Path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    expected = fixture["expected"]
    result = lpx.extract_binary_confidence(fixture)
    assert result.ai_native == expected["ai_native"]
    assert result.decision_token in ("0", "1")
    assert result.p_one == pytest.approx(expected["p_one"], abs=1e-9)
    assert result.top1_prob == pytest.approx(expected["top1_prob"], abs=1e-9)
    assert result.margin == pytest.approx(expected["margin"], abs=1e-9)
    assert result.entropy_bits == pytest.approx(expected["entropy_bits"], abs=1e-9)
    assert result.valid_mass == pytest.approx(expected["valid_mass"], abs=1e-9)


@pytest.mark.parametrize("path", fixture_paths(), ids=lambda p: p.stem)
def test_fixture_contains_no_identifying_text(path: Path):
    # Public-repo policy: fixtures carry structure + numbers only. The whole
    # payload must reduce to the Pass-A frame around the decision digit.
    fixture = json.loads(path.read_text(encoding="utf-8"))
    text = fixture["output"][0]["content"][0]["text"]
    assert text in ('{"ai_native":0}', '{"ai_native":1}')


def test_fixture_masked_sentinels_present_and_ignored():
    # Real key tokens carry -100.0 grammar masks; they must survive in the
    # fixture (that's the realism) and be ignored by the pool builder.
    sentinel_seen = False
    for path in fixture_paths():
        fixture = json.loads(path.read_text(encoding="utf-8"))
        for e in fixture["output"][0]["content"][0]["logprobs"]:
            if any(t["logprob"] == -100.0 for t in e["top_logprobs"]):
                sentinel_seen = True
    assert sentinel_seen, "expected at least one real -100.0 mask in fixtures"


# ---------------------------------------------------------------------------
# Run-directory API
# ---------------------------------------------------------------------------

def test_extract_run_prefers_pass_a_files(tmp_path: Path):
    resp = pass_a_response("1", entry("1", -0.2, [("0", -1.7)]))
    (tmp_path / "startup-x_a.json").write_text(json.dumps(resp))
    (tmp_path / "startup-x_b.json").write_text(json.dumps({"output": []}))
    rows = lpx.extract_run(tmp_path)
    assert [r["custom_id"] for r in rows] == ["startup-x"]
    assert rows[0]["ai_native"] == 1


def test_extract_run_falls_back_to_single_pass_files(tmp_path: Path):
    resp = pass_a_response("0", entry("0", -0.2, [("1", -1.7)]))
    (tmp_path / "startup-y.json").write_text(json.dumps(resp))
    rows = lpx.extract_run(tmp_path)
    assert [r["custom_id"] for r in rows] == ["startup-y"]
    assert rows[0]["ai_native"] == 0


# ---------------------------------------------------------------------------
# Scorer connector (pivot 6: confidence in the sampled digit)
# ---------------------------------------------------------------------------

def test_chosen_confidence_follows_the_sampled_digit():
    assert lpx.chosen_confidence({"ai_native": 1, "p_one": 0.9}) == 0.9
    assert lpx.chosen_confidence({"ai_native": 0, "p_one": 0.1}) == pytest.approx(0.9)
    # Minority sampling on both sides: confidence must drop below 0.5,
    # never flip to the argmax mass.
    assert lpx.chosen_confidence({"ai_native": 1, "p_one": 0.28}) == pytest.approx(0.28)
    assert lpx.chosen_confidence({"ai_native": 0, "p_one": 0.7}) == pytest.approx(0.3)


def test_chose_minority_fixture_gets_confidence_below_half():
    fixture = json.loads(
        (FIXTURES_DIR / "chose_minority.json").read_text(encoding="utf-8")
    )
    result = lpx.extract_binary_confidence(fixture)
    confidence = lpx.chosen_confidence(result.as_dict())
    assert result.ai_native == 1  # the sampled digit was the minority token
    assert confidence == pytest.approx(result.p_one)
    assert confidence < 0.5
    assert confidence != pytest.approx(result.top1_prob)  # argmax never substituted


def test_run_confidence_maps_custom_id_to_chosen_digit_mass(tmp_path: Path):
    one = pass_a_response("1", entry("1", -0.2, [("0", -1.7)]))
    zero = pass_a_response("0", entry("0", -0.1, [("1", -2.4)]))
    (tmp_path / "startup-a.json").write_text(json.dumps(one))
    (tmp_path / "startup-b.json").write_text(json.dumps(zero))
    conf = lpx.run_confidence(tmp_path)
    p_one_a = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.7))
    p_zero_b = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.4))
    assert conf.keys() == {"startup-a", "startup-b"}
    assert conf["startup-a"] == pytest.approx(p_one_a)
    assert conf["startup-b"] == pytest.approx(p_zero_b)


def test_run_confidence_refuses_a_run_without_raw_responses(tmp_path: Path):
    # Both a missing dir and an empty one: raw/ is git-ignored, so a fresh
    # clone hits this. The error must be loud, not a silent no-calibration.
    with pytest.raises(lpx.LogprobExtractionError, match="no raw response"):
        lpx.run_confidence(tmp_path / "does-not-exist")
    with pytest.raises(lpx.LogprobExtractionError, match="no raw response"):
        lpx.run_confidence(tmp_path)
