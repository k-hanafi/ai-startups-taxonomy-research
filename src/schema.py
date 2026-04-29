"""Output schema for the v2 two-axis startup classifier.

ClassificationResult is the single source of truth for the 11-field output.
It auto-generates the JSON schema injected into every batch request body via
model_json_schema(), eliminating any risk of the Python types and API schema
diverging.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ClassificationResult(BaseModel):
    """Structured output for one classified startup."""

    CompanyID: str = Field(description="Copied verbatim from input.")
    CompanyName: str = Field(description="Copied verbatim from input.")

    ai_native: Literal[0, 1] = Field(
        description="1 if the startup is AI-native, 0 if not."
    )
    subclass: Literal[
        "1A", "1B", "1C", "1D", "1E", "1F", "1G",
        "0A", "0B", "0E",
    ] = Field(description="Sub-genre within the ai_native dimension.")
    rad_score: Literal["RAD-H", "RAD-M", "RAD-L", "RAD-NA"] = Field(
        description=(
            "Resource-Adjusted AI Dependency score. "
            "RAD-H/M/L for ai_native=1; RAD-NA for ai_native=0."
        )
    )
    cohort: Literal["PRE-GENAI", "GENAI-ERA"] = Field(
        description="PRE-GENAI if founded before 2023, GENAI-ERA if 2023 or later."
    )

    conf_classification: int = Field(
        description="Analyst confidence in the ai_native + subclass assignment (1–5)."
    )
    conf_rad: Optional[int] = Field(
        description=(
            "Analyst confidence in the RAD score assignment (1–5). "
            "null when rad_score is RAD-NA — the question is not applicable."
        )
    )

    reasons_3_points: str = Field(
        description=(
            "Exactly 3 concise bullet points justifying the classification, separated by ' | '. "
            "If inputs are insufficient for substantive reasoning, use the literal "
            "'Insufficient information' instead of an empty string."
        )
    )
    sources_used: str = Field(
        description="Subset of input fields that most influenced the decision."
    )
    verification_critique: str = Field(
        description=(
            "≤40-word self-critique identifying the single biggest uncertainty. "
            "Flag UNCERTAIN if confidence on either axis is 1 or 2."
        )
    )

    @field_validator("conf_classification")
    @classmethod
    def classification_confidence_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError(f"conf_classification must be 1–5, got {v}")
        return v

    @field_validator("conf_rad")
    @classmethod
    def rad_confidence_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not 1 <= v <= 5:
            raise ValueError(f"conf_rad must be 1–5 or null, got {v}")
        return v
