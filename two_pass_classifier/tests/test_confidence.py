from __future__ import annotations

import math

import pytest

from two_pass_classifier.confidence import (
    BinaryConfidenceUnavailable,
    ai_native_confidence,
    extract_binary_confidence,
)


def _entry(
    token: str,
    logprob: float,
    top: list[tuple[str, float]] | None = None,
) -> dict:
    candidates = [
        {
            "token": token,
            "bytes": list(token.encode()),
            "logprob": logprob,
        }
    ]
    candidates.extend(
        {
            "token": candidate,
            "bytes": list(candidate.encode()),
            "logprob": candidate_logprob,
        }
        for candidate, candidate_logprob in top or []
    )
    return {
        "token": token,
        "bytes": list(token.encode()),
        "logprob": logprob,
        "top_logprobs": candidates,
    }


def _response(digit: str, decision: dict) -> dict:
    entries = [
        _entry('{"', 0.0),
        _entry("ai_native", 0.0),
        _entry('":', 0.0),
        decision,
        _entry("}", 0.0),
    ]
    text = b"".join(bytes(entry["bytes"]) for entry in entries).decode()
    assert text == f'{{"ai_native":{digit}}}'
    return {
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
        ]
    }


@pytest.mark.parametrize(
    ("digit", "chosen_mass", "opponent_mass", "expected"),
    [
        ("1", 0.8, 0.2, 0.8),
        ("0", 0.7, 0.3, 0.7),
        ("1", 0.3, 0.7, 0.3),
        ("0", 0.2, 0.8, 0.2),
    ],
)
def test_confidence_is_probability_of_sampled_digit(
    digit, chosen_mass, opponent_mass, expected
):
    opponent = "0" if digit == "1" else "1"
    response = _response(
        digit,
        _entry(
            digit,
            math.log(chosen_mass),
            [(opponent, math.log(opponent_mass))],
        ),
    )
    result = extract_binary_confidence(response)
    assert result.ai_native == int(digit)
    assert result.sampled_probability == pytest.approx(expected)
    assert result.as_dict()["sampled_probability"] == pytest.approx(expected)
    assert result.top1_prob == pytest.approx(max(result.p_one, 1 - result.p_one))
    assert ai_native_confidence(response) == pytest.approx(expected)


def test_censored_opponent_uses_validated_midpoint_bound():
    response = _response("0", _entry("0", math.log(0.98)))
    result = extract_binary_confidence(response)

    assert result.censored is True
    assert result.p_other_max == pytest.approx(0.02)
    assert result.interval_width == pytest.approx(0.02)
    assert result.sampled_probability == pytest.approx(0.98 / (0.98 + 0.01))
    assert result.sampled_probability < 1.0


def test_unavailable_censored_case_exports_none():
    response = _response(
        "0",
        _entry("0", math.log(0.5), [(" ", math.log(0.1))]),
    )
    with pytest.raises(BinaryConfidenceUnavailable, match="too wide"):
        extract_binary_confidence(response)
    assert ai_native_confidence(response) is None


def test_masked_opponent_is_not_treated_as_real_mass():
    response = _response(
        "1",
        _entry("1", math.log(0.5), [("0", -100.0), (" ", math.log(0.1))]),
    )
    with pytest.raises(BinaryConfidenceUnavailable):
        extract_binary_confidence(response)
    assert ai_native_confidence(response) is None
