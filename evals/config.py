"""Eval-harness configuration. All harness tunables live here.

Mirrors the no-magic-numbers convention of src/config.py but is fully
independent: nothing here is imported by (or from) the production pipeline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

# Locked Stage 8 screen matrix: nano / mini / luna × Pass B low/medium/high.
# gpt-5.4 / gpt-5.5 stay in EVAL_MODEL_PRICING for scoring older banked runs.
EVAL_MODELS: list[str] = [
    "gpt-5.4-nano",   # current production model
    "gpt-5.4-mini",
    "gpt-5.6-luna",
]

# Pass B effort arms for the locked 9-cell Stage 8 screen (not "none":
# Pass A already owns the logprob/calibration axis at effort=none).
STAGE8_PASS_B_EFFORTS: list[str] = ["low", "medium", "high"]

# Legacy single-pass knobs (kept for scoring older banked runs only).
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
# Classification runner (Stage 5): Pass A = binary gate, Pass B = subclass + RAD
# ---------------------------------------------------------------------------

# Pass A runs with reasoning off (logprobs on); Pass B defaults to maximum
# reasoning, per the split-reasoning design in the classification plan.
PASS_A_EFFORT: str = REASONING_OFF
PASS_B_EFFORT: str = "high"

# Pass A emits a single-field JSON (~6 tokens); 500 is generous headroom.
# Pass B reasons at high effort; observed single-pass high runs peaked at
# ~1,450 output tokens, so the shared 8,000 cap (MAX_OUTPUT_TOKENS) applies.
PASS_A_MAX_OUTPUT_TOKENS: int = 500

# Distinct cache keys per pass: each pass has its own stable instruction
# prefix, and mixing them in one cache route would hurt hit rates.
# Opaque strings kept as the historical "two-pass-*" values so prompt-cache
# identity does not reset when the public CLI/module was renamed.
PASS_A_CACHE_KEY: str = "two-pass-a-binary-gate"
PASS_B_CACHE_KEY: str = "two-pass-b-subclass-rad"

# GPT-4 launch month: the cohort boundary (founded 2023-03 or later = GENAI-ERA).
COHORT_BOUNDARY: tuple[int, int] = (2023, 3)

# ---------------------------------------------------------------------------
# Request parameters (experimental; production does not send these yet)
# ---------------------------------------------------------------------------

# Pass A is a binary {0,1} digit decision. Request depth 2 so both legal
# values can appear in top_logprobs. Legacy TOP_LOGPROBS=15 was a single-pass
# subclass leftover and must not drive Pass A or parity success criteria.
PASS_A_TOP_LOGPROBS: int = 2
# Kept for legacy single-pass runner / older banked runs only.
TOP_LOGPROBS: int = 15
LOGPROB_INCLUDE: list[str] = ["message.output_text.logprobs"]

# Rough Pass B output+reasoning token guesses for dry-run budget preflight.
# Input-only char/4 estimates understate high-effort spend; these are order-of-
# magnitude only (observed single-pass high peaked ~1,450 output tokens).
PASS_B_OUTPUT_TOKEN_ESTIMATE: dict[str, int] = {
    "none": 250,
    "low": 500,
    "medium": 1_000,
    "high": 1_600,
}
PASS_A_OUTPUT_TOKEN_ESTIMATE: int = 8

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

# Calibration (computed only when a per-row binary confidence is available).
CALIBRATION_BINS: int = 10
# Coverage fractions for the selective-prediction curve: accuracy when the
# model only answers on its top-X% most confident rows.
SELECTIVE_COVERAGE_GRID: list[float] = [round(0.1 * k, 1) for k in range(1, 11)]

# ---------------------------------------------------------------------------
# Batch API parity smoke (gate Q4)
# ---------------------------------------------------------------------------

PARITY_ROWS: int = 10
PARITY_POLL_SECONDS: int = 30
# 10 nano rows normally finish in minutes; bail out (with the batch id saved
# for resume) rather than blocking a terminal for the full 24h window.
PARITY_MAX_WAIT_SECONDS: int = 7_200
PARITY_COMPLETION_WINDOW: str = "24h"

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
    # gpt-5.6-luna: OpenAI GPT-5.6 launch pricing (July 2026), $1.00/$6.00 per 1M.
    "gpt-5.6-luna": {"input": 1.00, "output": 6.00},
}


def require_model_pricing(model: str) -> dict[str, float]:
    """Return pricing for *model*, or refuse rather than silently estimate $0."""
    pricing = EVAL_MODEL_PRICING.get(model)
    if pricing is None:
        raise SystemExit(
            f"Unknown model pricing for {model!r}. Add it to EVAL_MODEL_PRICING "
            "before dry-run or cost estimates (refusing a silent $0 figure)."
        )
    return pricing

# ---------------------------------------------------------------------------
# Production cost extrapolation (pivot 8)
# ---------------------------------------------------------------------------
# Same stacking as src/tokens.py / src/merger.py: batch 50% on all tokens,
# then an extra 50% on the cached portion of input (cached input = 25% of
# sync list). Do NOT import from src — evals stays offline-safe without keys.
BATCH_DISCOUNT: float = 0.50
CACHE_DISCOUNT: float = 0.50

# Scale-up N: alive non-empty evidence + dead extractable targets (default).
# Optional named alternatives for later toggles; default is the combo.
N_PROD_ALIVE_EVIDENCE: int = 22_032
N_PROD_DEAD_EXTRACTABLE: int = 19_044
N_PROD_ALIVE_PLUS_DEAD: int = N_PROD_ALIVE_EVIDENCE + N_PROD_DEAD_EXTRACTABLE
N_PROD_DEFAULT: int = N_PROD_ALIVE_PLUS_DEAD

N_PROD_SCALE_OPTIONS: dict[str, int] = {
    "alive_evidence": N_PROD_ALIVE_EVIDENCE,
    "dead_extractable": N_PROD_DEAD_EXTRACTABLE,
    "alive_plus_dead": N_PROD_ALIVE_PLUS_DEAD,
}
