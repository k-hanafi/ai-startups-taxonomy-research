"""Parity proofs between eval requests and the production classifier."""

from __future__ import annotations

import json

from evals import classification, config as eval_config, logprob_extract
from two_pass_classifier import config as production_config
from two_pass_classifier.confidence import extract_binary_confidence
from two_pass_classifier.request_builder import (
    RequestSettings,
    build_pass_a_request,
    build_pass_b_request,
    pass_a_request_fingerprint,
    request_fingerprint,
    request_identity,
)
from two_pass_classifier.schema import (
    PassAResult,
    PassBAINativeResult,
    PassBNotAINativeResult,
)


def _row() -> dict[str, str]:
    return {
        "org_uuid": "company-1",
        "name": "Company One",
        "short_description": "AI workflow",
        "Long description": "A longer description",
        "category_list": "Artificial Intelligence",
        "category_groups_list": "Software",
        "founded_date": "2024-01",
        "employee_count": "1-10",
        "total_funding_usd": "100",
        "website_pages_used": "https://company.test",
        "website_evidence": "We build an AI workflow.",
    }


def _bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def test_eval_request_bodies_are_byte_identical_to_production():
    row = _row()
    settings = RequestSettings(
        model="gpt-5.4-mini",
        pass_b_effort="medium",
    )

    assert _bytes(classification.pass_a_kwargs(row, settings.model)) == _bytes(
        build_pass_a_request(row, settings)
    )
    for family in (0, 1):
        assert _bytes(
            classification.pass_b_kwargs(
                row,
                family,
                settings.model,
                settings.pass_b_effort,
            )
        ) == _bytes(build_pass_b_request(row, family, settings))


def test_eval_fingerprints_are_production_fingerprints():
    settings = RequestSettings(
        model="gpt-5.6-luna",
        pass_b_effort="low",
    )
    metadata = classification.production_request_metadata(
        settings.model,
        settings.pass_b_effort,
    )

    assert metadata["semantic_request_fingerprint"] == request_fingerprint(
        settings
    )
    assert metadata["request_identity"] == request_identity(settings)
    assert metadata["pass_a_request_fingerprint"] == (
        pass_a_request_fingerprint(settings)
    )


def test_eval_uses_production_schemas_and_confidence_extractor():
    assert classification.BinaryResult is PassAResult
    assert classification.SubclassResultAI is PassBAINativeResult
    assert classification.SubclassResultNot is PassBNotAINativeResult
    assert logprob_extract.extract_binary_confidence is (
        extract_binary_confidence
    )


def test_eval_model_matrix_and_defaults_are_production_owned():
    assert eval_config.EVAL_MODELS is production_config.SUPPORTED_MODELS
    assert eval_config.MATRIX_PASS_B_EFFORTS is (
        production_config.SUPPORTED_PASS_B_EFFORTS
    )
    assert eval_config.DEFAULT_MODEL == production_config.DEFAULT_MODEL
    assert eval_config.DEFAULT_PASS_B_EFFORT == (
        production_config.DEFAULT_PASS_B_EFFORT
    )
    assert not hasattr(eval_config, "EVAL_MODEL_PRICING")
    assert not hasattr(eval_config, "MAX_OUTPUT_TOKENS")
