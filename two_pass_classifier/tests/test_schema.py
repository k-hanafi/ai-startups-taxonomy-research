from __future__ import annotations

import pytest
from pydantic import ValidationError

from two_pass_classifier import config
from two_pass_classifier.schema import (
    PassAResult,
    PassBAINativeResult,
    PassBNotAINativeResult,
    strict_schema,
)


def test_locked_production_model_defaults():
    assert config.SUPPORTED_MODELS == (
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5.6-luna",
    )
    assert config.DEFAULT_MODEL == "gpt-5.6-luna"
    assert config.PASS_A_REASONING_EFFORT == "none"
    assert config.PASS_A_EFFORT == "none"
    assert config.DEFAULT_PASS_B_REASONING_EFFORT == "low"
    assert config.DEFAULT_PASS_B_EFFORT == "low"
    assert config.PASS_A_TOP_LOGPROBS == 5


def test_exact_schema_field_order_and_family_enums():
    assert list(PassAResult.model_fields) == [
        "ai_native",
        "ai_native_reasoning",
        "sources_used",
        "ai_native_critique",
    ]
    assert list(PassBAINativeResult.model_fields) == [
        "subclass",
        "rad_score",
        "subclass_confidence",
        "rad_confidence",
        "subclass_reasoning",
        "rad_reasoning",
        "sources_used",
        "subclass_critique",
        "rad_critique",
    ]
    assert list(PassBNotAINativeResult.model_fields) == [
        "subclass",
        "subclass_confidence",
        "subclass_reasoning",
        "rad_reasoning",
        "sources_used",
        "subclass_critique",
    ]

    ai_schema = strict_schema(PassBAINativeResult)
    not_ai_schema = strict_schema(PassBNotAINativeResult)
    assert ai_schema["properties"]["subclass"]["enum"] == [
        "1A",
        "1B",
        "1C",
        "1D",
        "1E",
        "1F",
        "1G",
    ]
    assert ai_schema["properties"]["rad_score"]["enum"] == [
        "RAD-H",
        "RAD-M",
        "RAD-L",
    ]
    assert not_ai_schema["properties"]["subclass"]["enum"] == ["0A", "0B", "0C"]
    assert ai_schema["additionalProperties"] is False
    assert not_ai_schema["additionalProperties"] is False


def test_every_reasoning_and_critique_accepts_100_words():
    words = " ".join(f"word{i}" for i in range(100))
    PassAResult(
        ai_native=1,
        ai_native_reasoning=words,
        sources_used=["website_evidence"],
        ai_native_critique=words,
    )
    PassBAINativeResult(
        subclass="1E",
        rad_score="RAD-M",
        subclass_confidence=4,
        rad_confidence=3,
        subclass_reasoning=words,
        rad_reasoning=words,
        sources_used=["website_evidence"],
        subclass_critique=words,
        rad_critique=words,
    )
    PassBNotAINativeResult(
        subclass="0A",
        subclass_confidence=4,
        subclass_reasoning=words,
        rad_reasoning=words,
        sources_used=["website_evidence"],
        subclass_critique=words,
    )


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (PassAResult, "ai_native_reasoning"),
        (PassAResult, "ai_native_critique"),
        (PassBAINativeResult, "subclass_reasoning"),
        (PassBAINativeResult, "rad_reasoning"),
        (PassBAINativeResult, "subclass_critique"),
        (PassBAINativeResult, "rad_critique"),
        (PassBNotAINativeResult, "subclass_reasoning"),
        (PassBNotAINativeResult, "rad_reasoning"),
        (PassBNotAINativeResult, "subclass_critique"),
    ],
)
def test_every_reasoning_and_critique_rejects_101_words(model, field):
    words = " ".join(f"word{i}" for i in range(101))
    if model is PassAResult:
        data = {
            "ai_native": 0,
            "ai_native_reasoning": "valid",
            "sources_used": ["website_evidence"],
            "ai_native_critique": "valid",
        }
    elif model is PassBAINativeResult:
        data = {
            "subclass": "1C",
            "rad_score": "RAD-H",
            "subclass_confidence": 3,
            "rad_confidence": 3,
            "subclass_reasoning": "valid",
            "rad_reasoning": "valid",
            "sources_used": ["website_evidence"],
            "subclass_critique": "valid",
            "rad_critique": "valid",
        }
    else:
        data = {
            "subclass": "0A",
            "subclass_confidence": 3,
            "subclass_reasoning": "valid",
            "rad_reasoning": "valid",
            "sources_used": ["website_evidence"],
            "subclass_critique": "valid",
        }
    data[field] = words
    with pytest.raises(ValidationError, match="at most 100 words"):
        model(**data)


def test_schema_rejects_wrong_family_and_extra_fields():
    with pytest.raises(ValidationError):
        PassBAINativeResult(
            subclass="0A",
            rad_score="RAD-H",
            subclass_confidence=3,
            rad_confidence=3,
            subclass_reasoning="reason",
            rad_reasoning="reason",
            sources_used=[],
            subclass_critique="critique",
            rad_critique="critique",
        )
    with pytest.raises(ValidationError):
        PassBNotAINativeResult(
            subclass="0B",
            subclass_confidence=3,
            subclass_reasoning="reason",
            rad_reasoning="reason",
            sources_used=[],
            subclass_critique="critique",
            unexpected=False,
        )
