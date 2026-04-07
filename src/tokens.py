"""Pre-flight token counting and cost estimation using tiktoken.

Counts exact tokens for system prompt, schema prefix, and per-row user
messages.  Projects total cost with batch discount and prompt caching
savings so --dry-run can surface a full cost breakdown before any API call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    MAX_OUTPUT_TOKENS,
)
from src.schema import ClassificationResult

# ── Pricing ($ per 1M tokens, sync API, before batch discount) ────────────
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.4":      {"input": 2.50, "output": 10.00},
}

BATCH_DISCOUNT: float = 0.50
CACHE_DISCOUNT: float = 0.50  # cached tokens billed at 50 % of input rate


def get_encoding(model: str) -> tiktoken.Encoding:
    """Return the tiktoken encoding for *model*, falling back to o200k_base."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count tokens in *text* using the tokenizer for *model*."""
    return len(get_encoding(model).encode(text))


# ── Cost estimate dataclass ───────────────────────────────────────────────


@dataclass(frozen=True)
class CostEstimate:
    """Result of a pre-flight cost estimation."""

    model: str
    total_companies: int
    batches_needed: int

    system_prompt_tokens: int
    schema_tokens: int
    prefix_tokens: int
    avg_user_tokens: int
    total_input_tokens: int
    total_output_tokens: int

    cost_input_sync: float
    cost_output_sync: float
    cost_total_sync: float
    cost_total_batch: float
    cost_with_caching: float

    def format_report(self) -> str:
        """Human-readable cost breakdown suitable for --dry-run output."""
        batch_sz = self.total_companies // max(self.batches_needed, 1)
        return "\n".join([
            "=" * 60,
            "PRE-FLIGHT COST ESTIMATE",
            "=" * 60,
            f"Model:              {self.model}",
            f"Companies:          {self.total_companies:,}",
            f"Batches needed:     {self.batches_needed:,}  ({batch_sz:,}/batch)",
            "",
            "TOKEN BREAKDOWN",
            f"  System prompt:    {self.system_prompt_tokens:,} tokens  (cached)",
            f"  Schema prefix:    {self.schema_tokens:,} tokens  (cached)",
            f"  Static prefix:    {self.prefix_tokens:,} tokens  total",
            f"  Avg user message: {self.avg_user_tokens:,} tokens  (varies)",
            f"  Max output cap:   {MAX_OUTPUT_TOKENS:,} tokens  per request",
            "",
            f"  Total input:      ~{self.total_input_tokens / 1e6:.1f}M tokens",
            f"  Total output:     ~{self.total_output_tokens / 1e6:.1f}M tokens  (est.)",
            "",
            "COST PROJECTION",
            f"  Sync API:         ${self.cost_total_sync:>10,.2f}",
            f"  Batch API (50%):  ${self.cost_total_batch:>10,.2f}",
            f"  + caching (est.): ${self.cost_with_caching:>10,.2f}",
            "=" * 60,
        ])


# ── Main estimator ────────────────────────────────────────────────────────


def estimate_cost(
    system_prompt: str,
    user_messages: list[str],
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> CostEstimate:
    """Count tokens across all requests and project costs.

    Args:
        system_prompt: Full text of the classification system prompt.
        user_messages: One formatted user message per company.
        model: Model name for tokenizer selection and pricing lookup.
        batch_size: Requests per batch file.

    Returns:
        CostEstimate with full token and cost breakdown.
    """
    enc = get_encoding(model)
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])

    schema_json = json.dumps(ClassificationResult.model_json_schema())

    system_toks = len(enc.encode(system_prompt))
    schema_toks = len(enc.encode(schema_json))
    prefix_toks = system_toks + schema_toks

    user_tok_counts = [len(enc.encode(msg)) for msg in user_messages]
    avg_user_toks = sum(user_tok_counts) // max(len(user_tok_counts), 1)

    n = len(user_messages)
    total_input = sum(prefix_toks + ut for ut in user_tok_counts)

    # Estimate output conservatively at 60 % of max cap
    est_output_per = int(MAX_OUTPUT_TOKENS * 0.60)
    total_output = n * est_output_per

    batches_needed = (n + batch_size - 1) // batch_size

    # ── sync pricing ──────────────────────────────────────────────────
    cost_in = total_input / 1e6 * pricing["input"]
    cost_out = total_output / 1e6 * pricing["output"]
    cost_sync = cost_in + cost_out
    cost_batch = cost_sync * BATCH_DISCOUNT

    # ── caching estimate (best-case: 90 % of prefix tokens hit cache) ─
    # Cached tokens: batch rate × cache discount (stacks to 25 % of sync)
    prefix_frac = prefix_toks / max(prefix_toks + avg_user_toks, 1)
    cached_toks = int(total_input * prefix_frac * 0.90)
    uncached_toks = total_input - cached_toks

    cost_in_cached = (
        uncached_toks / 1e6 * pricing["input"] * BATCH_DISCOUNT
        + cached_toks / 1e6 * pricing["input"] * BATCH_DISCOUNT * CACHE_DISCOUNT
    )
    cost_with_cache = cost_in_cached + cost_out * BATCH_DISCOUNT

    return CostEstimate(
        model=model,
        total_companies=n,
        batches_needed=batches_needed,
        system_prompt_tokens=system_toks,
        schema_tokens=schema_toks,
        prefix_tokens=prefix_toks,
        avg_user_tokens=avg_user_toks,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        cost_input_sync=cost_in,
        cost_output_sync=cost_out,
        cost_total_sync=cost_sync,
        cost_total_batch=cost_batch,
        cost_with_caching=cost_with_cache,
    )
