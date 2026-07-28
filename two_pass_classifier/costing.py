"""Offline token counting and normal Responses API cost accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

import tiktoken

from . import config
from .manifest import Manifest
from .request_builder import (
    RequestSettings,
    build_pass_a_request,
    build_pass_b_request,
)


@dataclass(frozen=True, slots=True)
class StageCost:
    request_count: int
    input_tokens_min: int
    input_tokens_max: int
    estimated_output_tokens: int
    one_attempt_cap_tokens: int
    estimated_cost_min: float
    estimated_cost_max: float
    one_attempt_cap_cost_min: float
    one_attempt_cap_cost_max: float


@dataclass(frozen=True, slots=True)
class CostPreview:
    model: str
    pass_b_effort: str
    manifest_row_count: int
    pass_a: StageCost
    pass_b: StageCost
    pass_b_input_if_family_0: int
    pass_b_input_if_family_1: int
    known_family_counts: dict[int, int]
    unknown_family_count: int
    pricing_per_million: dict[str, float]

    @property
    def estimated_total_min(self) -> float:
        return self.pass_a.estimated_cost_min + self.pass_b.estimated_cost_min

    @property
    def estimated_total_max(self) -> float:
        return self.pass_a.estimated_cost_max + self.pass_b.estimated_cost_max

    @property
    def one_attempt_cap_total_min(self) -> float:
        return (
            self.pass_a.one_attempt_cap_cost_min
            + self.pass_b.one_attempt_cap_cost_min
        )

    @property
    def one_attempt_cap_total_max(self) -> float:
        return (
            self.pass_a.one_attempt_cap_cost_max
            + self.pass_b.one_attempt_cap_cost_max
        )


@dataclass(frozen=True, slots=True)
class ActualUsageCost:
    physical_requests: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    requests_missing_usage: int


def estimate_manifest_cost(
    manifest: Manifest,
    settings: RequestSettings,
    *,
    pass_a_company_ids: Collection[str] | None = None,
    pass_b_families: Mapping[str, int | None] | None = None,
) -> CostPreview:
    """Count production requests and price an all-row or remaining-work plan."""
    pricing = config.require_model_pricing(settings.model)
    rows_by_id = {row.company_id: row for row in manifest.rows}
    pass_a_ids = (
        set(rows_by_id)
        if pass_a_company_ids is None
        else set(pass_a_company_ids)
    )
    family_plan = (
        {company_id: None for company_id in rows_by_id}
        if pass_b_families is None
        else dict(pass_b_families)
    )
    unknown_ids = (pass_a_ids | set(family_plan)) - set(rows_by_id)
    if unknown_ids:
        raise ValueError(
            f"cost plan contains unknown company_id {sorted(unknown_ids)[0]!r}"
        )
    invalid_families = {
        family for family in family_plan.values() if family not in (None, 0, 1)
    }
    if invalid_families:
        raise ValueError(
            f"cost plan contains invalid family {sorted(invalid_families)[0]!r}"
        )

    encoding = _encoding_for_model(settings.model)
    pass_a_input = sum(
        count_request_input_tokens(
            build_pass_a_request(rows_by_id[company_id], settings),
            encoding=encoding,
        )
        for company_id in _manifest_order(manifest, pass_a_ids)
    )

    pass_b_if_0 = 0
    pass_b_if_1 = 0
    known_family_counts = {0: 0, 1: 0}
    unknown_family_count = 0
    for company_id in _manifest_order(manifest, set(family_plan)):
        row = rows_by_id[company_id]
        family = family_plan[company_id]
        input_0 = count_request_input_tokens(
            build_pass_b_request(row, 0, settings),
            encoding=encoding,
        )
        input_1 = count_request_input_tokens(
            build_pass_b_request(row, 1, settings),
            encoding=encoding,
        )
        if family is None:
            pass_b_if_0 += input_0
            pass_b_if_1 += input_1
            unknown_family_count += 1
        elif family == 0:
            pass_b_if_0 += input_0
            pass_b_if_1 += input_0
            known_family_counts[0] += 1
        else:
            pass_b_if_0 += input_1
            pass_b_if_1 += input_1
            known_family_counts[1] += 1

    pass_a_count = len(pass_a_ids)
    pass_b_count = len(family_plan)
    pass_a_output = (
        pass_a_count * config.PASS_A_PROVISIONAL_OUTPUT_TOKENS
    )
    pass_b_output = (
        pass_b_count
        * config.PASS_B_PREVIEW_OUTPUT_TOKENS[settings.pass_b_effort]
    )
    pass_a_cap = pass_a_count * settings.pass_a_max_output_tokens
    pass_b_cap = pass_b_count * int(settings.pass_b_max_output_tokens or 0)

    return CostPreview(
        model=settings.model,
        pass_b_effort=settings.pass_b_effort,
        manifest_row_count=manifest.row_count,
        pass_a=_stage_cost(
            request_count=pass_a_count,
            input_min=pass_a_input,
            input_max=pass_a_input,
            output_estimate=pass_a_output,
            one_attempt_cap=pass_a_cap,
            pricing=pricing,
        ),
        pass_b=_stage_cost(
            request_count=pass_b_count,
            input_min=min(pass_b_if_0, pass_b_if_1),
            input_max=max(pass_b_if_0, pass_b_if_1),
            output_estimate=pass_b_output,
            one_attempt_cap=pass_b_cap,
            pricing=pricing,
        ),
        pass_b_input_if_family_0=pass_b_if_0,
        pass_b_input_if_family_1=pass_b_if_1,
        known_family_counts=known_family_counts,
        unknown_family_count=unknown_family_count,
        pricing_per_million=pricing,
    )


def count_request_input_tokens(
    request: Mapping[str, Any],
    *,
    encoding: tiktoken.Encoding | None = None,
) -> int:
    """Count model-visible request content with a fixed framing allowance."""
    selected_encoding = encoding or _encoding_for_model(str(request["model"]))
    stable_controls = {
        "text": request.get("text"),
        "reasoning": request.get("reasoning"),
        "include": request.get("include"),
        "top_logprobs": request.get("top_logprobs"),
    }
    parts = (
        str(request.get("instructions") or ""),
        str(request.get("input") or ""),
        json.dumps(
            stable_controls,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    return config.INPUT_TOKEN_ESTIMATE_FIXED_OVERHEAD + sum(
        len(selected_encoding.encode(part)) for part in parts
    )


def actual_usage_cost(
    events: Sequence[Mapping[str, Any]],
    *,
    model: str,
) -> ActualUsageCost:
    """Price every recorded physical request from provider usage fields."""
    pricing = config.require_model_pricing(model)
    physical = [
        event
        for event in events
        if event.get("event_type")
        in {"request_error", "pass_a_completed", "company_completed"}
    ]
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    missing = 0
    for event in physical:
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            missing += 1
            continue
        input_tokens += _nonnegative_int(usage.get("input_tokens"))
        output_tokens += _nonnegative_int(usage.get("output_tokens"))
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, Mapping):
            cached_tokens += _nonnegative_int(input_details.get("cached_tokens"))
        output_details = usage.get("output_tokens_details")
        if isinstance(output_details, Mapping):
            reasoning_tokens += _nonnegative_int(
                output_details.get("reasoning_tokens")
            )

    cached_tokens = min(cached_tokens, input_tokens)
    uncached_tokens = input_tokens - cached_tokens
    cost = (
        uncached_tokens / 1_000_000 * pricing["input"]
        + cached_tokens / 1_000_000 * pricing["cached_input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )
    return ActualUsageCost(
        physical_requests=len(physical),
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost,
        requests_missing_usage=missing,
    )


def _stage_cost(
    *,
    request_count: int,
    input_min: int,
    input_max: int,
    output_estimate: int,
    one_attempt_cap: int,
    pricing: Mapping[str, float],
) -> StageCost:
    return StageCost(
        request_count=request_count,
        input_tokens_min=input_min,
        input_tokens_max=input_max,
        estimated_output_tokens=output_estimate,
        one_attempt_cap_tokens=one_attempt_cap,
        estimated_cost_min=_list_cost(input_min, output_estimate, pricing),
        estimated_cost_max=_list_cost(input_max, output_estimate, pricing),
        one_attempt_cap_cost_min=_list_cost(
            input_min, one_attempt_cap, pricing
        ),
        one_attempt_cap_cost_max=_list_cost(
            input_max, one_attempt_cap, pricing
        ),
    )


def _list_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: Mapping[str, float],
) -> float:
    return (
        input_tokens / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


def _encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _manifest_order(
    manifest: Manifest,
    company_ids: set[str],
) -> list[str]:
    return [
        row.company_id for row in manifest.rows if row.company_id in company_ids
    ]


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
