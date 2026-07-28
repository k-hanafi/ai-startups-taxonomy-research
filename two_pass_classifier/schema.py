"""Strict structured-output contracts for both classifier passes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

PassASource = Literal[
    "company_id",
    "company_name",
    "short_description",
    "long_description",
    "keywords",
    "founded_date",
    "resource_context",
    "website_evidence",
]
PassBSource = Literal[
    "company_id",
    "company_name",
    "short_description",
    "long_description",
    "keywords",
    "founded_date",
    "resource_context",
    "website_pages_used",
    "website_evidence",
]


def _at_most_100_words(value: str) -> str:
    if len(value.split()) > 100:
        raise ValueError("must contain at most 100 words")
    return value


LimitedExplanation = Annotated[str, AfterValidator(_at_most_100_words)]


class _StrictResult(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PassAResult(_StrictResult):
    """Pass A owns the binary verdict and its supporting analysis."""

    ai_native: Literal[0, 1]
    ai_native_reasoning: LimitedExplanation
    sources_used: list[PassASource]
    ai_native_critique: LimitedExplanation


class PassBAINativeResult(_StrictResult):
    """Pass B contract for a fixed AI-native family."""

    subclass: Literal["1A", "1B", "1C", "1D", "1E", "1F", "1G"]
    rad_score: Literal["RAD-H", "RAD-M", "RAD-L"]
    subclass_confidence: int = Field(ge=1, le=5)
    rad_confidence: int = Field(ge=1, le=5)
    subclass_reasoning: LimitedExplanation
    rad_reasoning: LimitedExplanation
    sources_used: list[PassBSource]
    subclass_critique: LimitedExplanation
    rad_critique: LimitedExplanation


class PassBNotAINativeResult(_StrictResult):
    """Pass B contract for a fixed non-AI-native family."""

    subclass: Literal["0A", "0B", "0C"]
    subclass_confidence: int = Field(ge=1, le=5)
    subclass_reasoning: LimitedExplanation
    rad_reasoning: LimitedExplanation
    sources_used: list[PassBSource]
    subclass_critique: LimitedExplanation


# Short aliases keep call sites readable while the explicit class names remain
# available to describe the family in generated JSON schema titles.
PassBAIResult = PassBAINativeResult
PassBNonAIResult = PassBNotAINativeResult


def strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return an OpenAI strict-mode compatible JSON schema."""
    schema = model_cls.model_json_schema()
    _forbid_additional_properties(schema)
    return schema


def _forbid_additional_properties(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for value in node.values():
            _forbid_additional_properties(value)
    elif isinstance(node, list):
        for value in node:
            _forbid_additional_properties(value)
