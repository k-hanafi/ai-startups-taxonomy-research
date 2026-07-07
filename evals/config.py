"""Eval-harness configuration. All harness tunables live here.

Mirrors the no-magic-numbers convention of src/config.py but is fully
independent: nothing here is imported by (or from) the production pipeline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

# Screened at SCREEN_REASONING_EFFORT first; effort sweep + repeats follow
# only for models on the cost-accuracy frontier (staged matrix design).
EVAL_MODELS: list[str] = [
    "gpt-5.4-nano",   # current production model
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.5",
]

SCREEN_REASONING_EFFORT: str = "medium"
REASONING_EFFORTS: list[str] = ["none", "low", "medium", "high"]
FINALIST_REPEATS: int = 3

# Empirical finding (2026-07-05, gpt-5.4-nano): logprobs are returned ONLY when
# reasoning is fully off. reasoning={"effort":"none"} yields reasoning_tokens=0
# and logprobs; "minimal"/"low"/"medium"/"high" all reject logprobs with a 400
# ("logprobs are not supported with reasoning models"). So token-level
# confidence and reasoning are mutually exclusive, and the none-vs-reasoning
# A/B measures the accuracy cost of getting logprobs. The runner captures
# logprobs only at this effort.
REASONING_OFF: str = "none"

# ---------------------------------------------------------------------------
# Two-pass classifier (Stage 5): Pass A = binary gate, Pass B = subclass + RAD
# ---------------------------------------------------------------------------

# Pass A runs with reasoning off (logprobs on); Pass B defaults to maximum
# reasoning, per the split-reasoning design in the two-pass plan.
PASS_A_EFFORT: str = REASONING_OFF
PASS_B_EFFORT: str = "high"

# Pass A emits a single-field JSON (~6 tokens); 500 is generous headroom.
# Pass B reasons at high effort; observed single-pass high runs peaked at
# ~1,450 output tokens, so the shared 8,000 cap (MAX_OUTPUT_TOKENS) applies.
PASS_A_MAX_OUTPUT_TOKENS: int = 500

# Distinct cache keys per pass: each pass has its own stable instruction
# prefix, and mixing them in one cache route would hurt hit rates.
PASS_A_CACHE_KEY: str = "two-pass-a-binary-gate"
PASS_B_CACHE_KEY: str = "two-pass-b-subclass-rad"

# GPT-4 launch month: the cohort boundary (founded 2023-03 or later = GENAI-ERA).
COHORT_BOUNDARY: tuple[int, int] = (2023, 3)

# ---------------------------------------------------------------------------
# Request parameters (experimental; production does not send these yet)
# ---------------------------------------------------------------------------

TOP_LOGPROBS: int = 15
LOGPROB_INCLUDE: list[str] = ["message.output_text.logprobs"]

# Empirical finding (2026-07-05, gpt-5.4-nano): reasoning models reject the
# `temperature` parameter with a 400 ("not supported with this model"). They
# decode at a fixed internal setting, so temperature is not a lever here and is
# omitted. Determinism instead comes from reasoning effort + the model itself,
# which the finalist-repeat runs measure. Flip SEND_TEMPERATURE only if a
# benchmarked model is a non-reasoning model that accepts it.
SEND_TEMPERATURE: bool = False
DECODING_TEMPERATURE: float = 0.0

# Reasoning tokens count against this cap. Sized ~8x the observed v2 output
# (~210 tokens) to leave room for reasoning=medium; the harness measures the
# real usage that will later size the production MAX_OUTPUT_TOKENS.
MAX_OUTPUT_TOKENS: int = 8_000

# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

GOLDEN_SET_SIZE: int = 100
SAMPLING_SEED: int = 20260705

# Stratification quotas keyed by *predicted* subclass (nano production
# predictions are the strata proxy; true labels don't exist until Stage 2).
# Rare AI-native subclasses are deliberately oversampled vs the population
# (0A alone is ~84% of evidence-bearing rows) so the eval has signal where
# the taxonomy is hard. Rarest evidence-bearing strata: 0B=39, 1A=45, 1C=45.
SUBCLASS_QUOTAS: dict[str, int] = {
    "1A": 8, "1B": 8, "1C": 8, "1D": 8, "1E": 8, "1F": 8, "1G": 8,
    "0A": 24, "0B": 8, "0C": 12,
}

# Within each subclass quota, spread rows across evidence-length terciles
# (short/medium/long) so no model is graded only on evidence-rich rows.
EVIDENCE_TERCILE_LABELS: list[str] = ["short", "medium", "long"]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 20260705
CONFIDENCE_LEVEL: float = 0.95

# ---------------------------------------------------------------------------
# Pricing ($ per 1M tokens, sync API) — verified 2026-07-05 against the
# OpenAI pricing page. src/tokens.py MODEL_PRICING is stale; the harness
# carries its own table so cost numbers in eval reports are trustworthy.
# ---------------------------------------------------------------------------

EVAL_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4":      {"input": 2.50, "output": 15.00},
    "gpt-5.5":      {"input": 5.00, "output": 30.00},
}
