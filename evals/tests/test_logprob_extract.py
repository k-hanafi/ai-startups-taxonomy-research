"""Offline artifact tests over production-owned confidence extraction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evals import logprob_extract as lpx

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _entry(
    token: str,
    logprob: float,
    top: list[tuple[str, float]] | None = None,
) -> dict:
    return {
        "token": token,
        "bytes": list(token.encode()),
        "logprob": logprob,
        "top_logprobs": [
            {
                "token": candidate,
                "bytes": list(candidate.encode()),
                "logprob": candidate_logprob,
            }
            for candidate, candidate_logprob in (
                [(token, logprob), *(top or [])]
            )
        ],
    }


def _response_from_entries(entries: list[dict]) -> dict:
    text = b"".join(bytes(item["bytes"]) for item in entries).decode()
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "logprobs": entries,
                    }
                ],
            }
        ],
    }


def _pass_a_response(digit: str, decision: dict) -> dict:
    return _response_from_entries(
        [
            _entry('{"', 0.0),
            _entry("ai_native", 0.0),
            _entry('":', 0.0),
            decision,
            _entry("}", 0.0),
        ]
    )


def test_production_extractor_locates_variable_decision_position():
    entries = [
        _entry('{"', 0.0),
        _entry("note", 0.0),
        _entry('":"', 0.0),
        _entry("café", 0.0),
        _entry('","', 0.0),
        _entry("ai_native", 0.0),
        _entry('":', 0.0),
        _entry("1", math.log(0.7), [("0", math.log(0.3))]),
        _entry("}", 0.0),
    ]

    result = lpx.extract_binary_confidence(
        _response_from_entries(entries)
    )

    assert result.ai_native == 1
    assert result.decision_token_index == 7
    assert result.sampled_probability == pytest.approx(0.7)


def test_reconstruction_mismatch_refuses_extraction():
    response = _pass_a_response(
        "1",
        _entry("1", math.log(0.7), [("0", math.log(0.3))]),
    )
    response["output"][0]["content"][0]["text"] += " "

    with pytest.raises(lpx.LogprobExtractionError, match="reconstruct"):
        lpx.extract_binary_confidence(response)


def test_masked_opponent_is_not_counted_as_probability():
    response = _pass_a_response(
        "1",
        _entry("1", math.log(0.5), [("0", -100.0), (" ", math.log(0.1))]),
    )

    with pytest.raises(lpx.BinaryConfidenceUnavailable):
        lpx.extract_binary_confidence(response)


def test_censored_opponent_uses_production_midpoint():
    result = lpx.extract_binary_confidence(
        _pass_a_response("0", _entry("0", math.log(0.98)))
    )

    assert result.censored is True
    assert result.p_other_max == pytest.approx(0.02)
    assert result.sampled_probability == pytest.approx(
        0.98 / (0.98 + 0.01)
    )


def test_censored_opponent_refuses_wide_bound():
    response = _pass_a_response(
        "0",
        _entry("0", math.log(0.5), [(" ", math.log(0.1))]),
    )

    with pytest.raises(lpx.BinaryConfidenceUnavailable, match="too wide"):
        lpx.extract_binary_confidence(response)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0", 0),
        ("1", 1),
        (" 1", 1),
        ("1,", 1),
        ("0}", 0),
        ("2", None),
        ("10", None),
        (" ", None),
    ],
)
def test_candidate_value_is_production_owned(token, expected):
    assert lpx.candidate_value(token) == expected


def fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))


def test_fixtures_exist():
    assert len(fixture_paths()) >= 4


@pytest.mark.parametrize("path", fixture_paths(), ids=lambda path: path.stem)
def test_real_anonymized_fixture_end_to_end(path: Path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    expected = fixture["expected"]

    result = lpx.extract_binary_confidence(fixture)

    assert result.ai_native == expected["ai_native"]
    assert result.p_one == pytest.approx(expected["p_one"], abs=1e-9)
    assert result.top1_prob == pytest.approx(
        expected["top1_prob"],
        abs=1e-9,
    )
    assert result.margin == pytest.approx(expected["margin"], abs=1e-9)
    assert result.entropy_bits == pytest.approx(
        expected["entropy_bits"],
        abs=1e-9,
    )
    assert result.valid_mass == pytest.approx(
        expected["valid_mass"],
        abs=1e-9,
    )


@pytest.mark.parametrize("path", fixture_paths(), ids=lambda path: path.stem)
def test_fixture_contains_no_identifying_text(path: Path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    text = fixture["output"][0]["content"][0]["text"]
    assert text in ('{"ai_native":0}', '{"ai_native":1}')


def test_extract_run_prefers_pass_a_files(tmp_path):
    response = _pass_a_response(
        "1",
        _entry("1", math.log(0.8), [("0", math.log(0.2))]),
    )
    (tmp_path / "startup-x_a.json").write_text(
        json.dumps(response),
        encoding="utf-8",
    )
    (tmp_path / "startup-x_b.json").write_text(
        json.dumps({"output": []}),
        encoding="utf-8",
    )

    rows = lpx.extract_run(tmp_path)

    assert [row["custom_id"] for row in rows] == ["startup-x"]
    assert rows[0]["sampled_probability"] == pytest.approx(0.8)


def test_run_confidence_uses_sampled_label_probability(tmp_path):
    minority = _pass_a_response(
        "1",
        _entry("1", math.log(0.3), [("0", math.log(0.7))]),
    )
    (tmp_path / "startup-x_a.json").write_text(
        json.dumps(minority),
        encoding="utf-8",
    )

    confidence = lpx.run_confidence(tmp_path)

    assert confidence["startup-x"] == pytest.approx(0.3)
    assert confidence["startup-x"] < 0.5


def test_valid_mass_summary_remains_eval_specific():
    summary = lpx.valid_mass_summary(
        [
            {"valid_mass": 0.99},
            {"valid_mass": 0.80},
            {"valid_mass": 0.95},
        ],
        threshold=0.90,
        max_below_share=0.05,
    )

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx((0.99 + 0.80 + 0.95) / 3)
    assert summary["n_below_threshold"] == 1
    assert summary["below_share"] == pytest.approx(1 / 3)


def test_run_confidence_refuses_missing_or_unusable_raw(tmp_path):
    with pytest.raises(lpx.LogprobExtractionError, match="no raw response"):
        lpx.run_confidence(tmp_path)

    wide = _pass_a_response(
        "0",
        _entry("0", math.log(0.5), [(" ", math.log(0.1))]),
    )
    (tmp_path / "startup-x_a.json").write_text(
        json.dumps(wide),
        encoding="utf-8",
    )
    with pytest.raises(lpx.LogprobExtractionError, match="none yielded"):
        lpx.run_confidence(tmp_path)
