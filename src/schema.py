"""Output schema for the v2 two-axis startup classifier.

ClassificationResult is the single source of truth for the 11-field output.
It auto-generates the JSON schema injected into every batch request body via
model_json_schema(), eliminating any risk of the Python types and API schema
diverging.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClassificationResult(BaseModel):
    """Structured output for one classified startup."""

    CompanyID: str = Field(description="Copied verbatim from input.")
    CompanyName: str = Field(description="Copied verbatim from input.")

    ai_native: Literal[0, 1] = Field(
        description="1 if the startup is AI-native, 0 if not."
    )
    subclass: Literal[
        "1A", "1B", "1C", "1D", "1E",
        "0A", "0B", "0C-THIN", "0C-THICK", "0D", "0E",
    ] = Field(description="Sub-genre within the ai_native dimension.")
    rad_score: Literal["RAD-H", "RAD-M", "RAD-L", "RAD-NA"] = Field(
        description="Resource-Adjusted AI Dependency score."
    )
    cohort: Literal["PRE-GENAI", "GENAI-ERA"] = Field(
        description="PRE-GENAI if founded before 2022, GENAI-ERA if 2022 or later."
    )

    conf_classification: int = Field(
        description="Analyst confidence in the ai_native + subclass assignment (1–5)."
    )
    conf_rad: int = Field(
        description="Analyst confidence in the RAD score assignment (1–5)."
    )

    reasons_3_points: str = Field(
        description="Exactly 3 concise bullet points justifying the classification."
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

    @field_validator("conf_classification", "conf_rad")
    @classmethod
    def must_be_one_to_five(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError(f"Confidence score must be between 1 and 5, got {v}")
        return v
