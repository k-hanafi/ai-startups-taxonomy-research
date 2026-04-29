"""Tests for ClassificationResult schema enforcement.

Verifies that Pydantic rejects invalid enum values and out-of-range
confidence scores. These are the contracts that have no runtime feedback
until they fail silently at 267K rows.
"""

import pytest
from pydantic import ValidationError

from src.schema import ClassificationResult

# -- Fixtures ------------------------------------------------------------------

_VALID = {
    "CompanyID": "test-001",
    "CompanyName": "Acme AI",
    "ai_native": 1,
    "subclass": "1B",
    "rad_score": "RAD-M",
    "cohort": "PRE-GENAI",
    "conf_classification": 4,
    "conf_rad": 3,
    "reasons_3_points": "Point A | Point B | Point C",
    "sources_used": "short_description, long_description",
    "verification_critique": "Borderline 1B vs 1D.",
}


def _make(**overrides) -> dict:
    return {**_VALID, **overrides}


# -- Valid input ---------------------------------------------------------------


class TestValidInput:
    def test_accepts_valid_ai_native(self):
        result = ClassificationResult.model_validate(_make())
        assert result.ai_native == 1
        assert result.subclass == "1B"

    def test_accepts_all_subclasses(self):
        for sc in ["1A", "1B", "1C", "1D", "1E", "1F", "1G",
                    "0A", "0B", "0E"]:
            result = ClassificationResult.model_validate(_make(subclass=sc))
            assert result.subclass == sc

    def test_rejects_retired_subclasses(self):
        """v2 codes 0C-THIN, 0C-THICK, 0D were removed in v2.1."""
        for sc in ["0C-THIN", "0C-THICK", "0D"]:
            with pytest.raises(ValidationError):
                ClassificationResult.model_validate(_make(subclass=sc))

    def test_accepts_all_rad_scores(self):
        for rad in ["RAD-H", "RAD-M", "RAD-L", "RAD-NA"]:
            result = ClassificationResult.model_validate(_make(rad_score=rad))
            assert result.rad_score == rad

    def test_accepts_both_cohorts(self):
        for c in ["PRE-GENAI", "GENAI-ERA"]:
            result = ClassificationResult.model_validate(_make(cohort=c))
            assert result.cohort == c

    def test_confidence_boundary_values(self):
        for v in [1, 2, 3, 4, 5]:
            result = ClassificationResult.model_validate(
                _make(conf_classification=v, conf_rad=v)
            )
            assert result.conf_classification == v
            assert result.conf_rad == v

    def test_conf_rad_null_for_rad_na(self):
        """conf_rad must be null whenever rad_score is RAD-NA."""
        result = ClassificationResult.model_validate(
            _make(ai_native=0, subclass="0A", rad_score="RAD-NA", conf_rad=None)
        )
        assert result.conf_rad is None

    def test_accepts_insufficient_information_reasons_sentinel(self):
        """Fallback rows use this literal instead of empty reasons_3_points."""
        result = ClassificationResult.model_validate(
            _make(
                ai_native=0,
                subclass="0A",
                rad_score="RAD-NA",
                conf_classification=1,
                conf_rad=None,
                reasons_3_points="Insufficient information",
            )
        )
        assert result.reasons_3_points == "Insufficient information"
        assert result.conf_rad is None


# -- Invalid input (must reject) -----------------------------------------------


class TestInvalidInput:
    def test_rejects_invalid_subclass(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(subclass="2A"))

    def test_rejects_invalid_rad_score(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(rad_score="RAD-X"))

    def test_rejects_invalid_cohort(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(cohort="POST-GENAI"))

    def test_rejects_ai_native_out_of_range(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(ai_native=2))

    def test_rejects_confidence_zero(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(conf_classification=0))

    def test_rejects_confidence_six(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(conf_rad=6))

    def test_rejects_negative_confidence(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(conf_classification=-1))


# -- Schema generation ---------------------------------------------------------


class TestSchemaGeneration:
    def test_schema_has_all_fields(self):
        schema = ClassificationResult.model_json_schema()
        props = schema["properties"]
        expected = [
            "CompanyID", "CompanyName", "ai_native", "subclass", "rad_score",
            "cohort", "conf_classification", "conf_rad", "reasons_3_points",
            "sources_used", "verification_critique",
        ]
        for field_name in expected:
            assert field_name in props, f"Missing field: {field_name}"

    def test_schema_field_count(self):
        schema = ClassificationResult.model_json_schema()
        assert len(schema["properties"]) == 11
