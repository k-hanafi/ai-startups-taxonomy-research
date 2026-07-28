"""Production configuration for the two-pass classifier."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_MODELS: tuple[str, ...] = (
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.6-luna",
)
SUPPORTED_PASS_B_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

DEFAULT_MODEL: str = "gpt-5.6-luna"
PASS_A_REASONING_EFFORT: str = "none"
DEFAULT_PASS_B_REASONING_EFFORT: str = "low"
PASS_A_TOP_LOGPROBS: int = 5
LOGPROB_INCLUDE: tuple[str, ...] = ("message.output_text.logprobs",)
PASS_A_CACHE_KEY: str = "production-v2-pass-a"
PASS_B_CACHE_KEYS: dict[int, str] = {
    0: "production-v2-pass-b-family-0",
    1: "production-v2-pass-b-family-1",
}

# Production-owned caps for the text-heavy strict schemas. The eval harness
# imports these exact values and separately reports measured p95/max usage.
PASS_A_MAX_OUTPUT_TOKENS: int = 1_536
PASS_B_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "low": 4_096,
    "medium": 8_192,
    "high": 16_384,
}
INPUT_TOKEN_ESTIMATE_BYTES_PER_TOKEN: float = 3.0
INPUT_TOKEN_ESTIMATE_FIXED_OVERHEAD: int = 32

REQUEST_FINGERPRINT_VERSION: int = 3
SMOKE_COMPANY_COUNT: int = 10

# Normal Responses API list prices in USD per one million tokens. Preview
# pricing intentionally ignores cache savings. Status can use measured cached
# tokens because the provider reports those after a request completes.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4-nano": {
        "input": 0.20,
        "cached_input": 0.10,
        "output": 1.25,
    },
    "gpt-5.4-mini": {
        "input": 0.75,
        "cached_input": 0.375,
        "output": 4.50,
    },
    "gpt-5.6-luna": {
        "input": 1.00,
        "cached_input": 0.50,
        "output": 6.00,
    },
}

# Provisional Pass A planning estimate pending the next production-aligned paid
# eval. It allows for two explanations of up to 100 words each plus JSON and
# source-list overhead. The one-attempt cap projection is reported separately.
PASS_A_PROVISIONAL_OUTPUT_TOKENS: int = 320
PASS_B_PREVIEW_OUTPUT_TOKENS: dict[str, int] = {
    "low": 350,
    "medium": 550,
    "high": 1_000,
}

# Concise aliases used by request builders in the eval harness and future
# production runner.
PASS_A_EFFORT: str = PASS_A_REASONING_EFFORT
DEFAULT_PASS_B_EFFORT: str = DEFAULT_PASS_B_REASONING_EFFORT

# The completed eval showed that a censored opponent remains useful only while
# its probability interval is narrow. Wider intervals export no confidence.
MAX_CENSORED_INTERVAL_WIDTH: float = 0.05

# GPT-4 launched on March 14, 2023. Partial March dates are classified as
# GENAI-ERA to preserve the validated eval behavior for month-only source data.
COHORT_BOUNDARY_YEAR: int = 2023
COHORT_BOUNDARY_MONTH: int = 3
COHORT_BOUNDARY_DAY: int = 14


@dataclass(frozen=True, slots=True)
class ModelRateLimit:
    requests_per_minute: int
    tokens_per_minute: int


def require_model_pricing(model: str) -> dict[str, float]:
    """Return complete pricing or refuse to produce an unsafe estimate."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        raise ValueError(
            f"unknown pricing for model {model!r}; add verified prices before use"
        )
    required = {"input", "cached_input", "output"}
    if set(pricing) != required:
        raise ValueError(
            f"incomplete pricing for model {model!r}; expected {sorted(required)}"
        )
    return dict(pricing)


# Tier 5 normal Responses API limits. Every model currently shares the same
# published limits, but model-specific records let live headers revise them
# independently if OpenAI changes one route.
MODEL_RATE_LIMITS: dict[str, ModelRateLimit] = {
    model: ModelRateLimit(
        requests_per_minute=30_000,
        tokens_per_minute=180_000_000,
    )
    for model in SUPPORTED_MODELS
}
RATE_LIMIT_TARGET_FRACTION: float = 0.80
RATE_LIMIT_WINDOW_SECONDS: float = 60.0
RATE_LIMIT_IN_FLIGHT_POLL_SECONDS: float = 0.05

INITIAL_CONCURRENCY: int = 128
MAX_CONCURRENCY: int = 1_024
COMPANY_QUEUE_SIZE: int = 256

WRITER_QUEUE_SIZE: int = 1_024
WRITER_GROUP_MAX_EVENTS: int = 64
WRITER_GROUP_MAX_WAIT_SECONDS: float = 0.050

MAX_REQUEST_ATTEMPTS: int = 4
RETRY_BASE_DELAY_SECONDS: float = 0.5
RETRY_MAX_DELAY_SECONDS: float = 8.0
RETRY_JITTER_FRACTION: float = 0.20
DEFAULT_429_RETRY_AFTER_SECONDS: float = 1.0

ADAPTIVE_INCREASE_EVERY_SUCCESSES: int = 32
ADAPTIVE_INCREASE_FRACTION: float = 0.0625
ADAPTIVE_MAX_RATE_UTILIZATION: float = 0.90
ADAPTIVE_MAX_WRITER_ACK_SECONDS: float = 0.100
ADAPTIVE_MAX_LATENCY_SECONDS: float = 60.0
ADAPTIVE_MAX_ERROR_RATE: float = 0.02
ADAPTIVE_ERROR_WINDOW: int = 64
ADAPTIVE_429_REDUCTION: float = 0.50
ADAPTIVE_TRANSIENT_REDUCTION: float = 0.80
