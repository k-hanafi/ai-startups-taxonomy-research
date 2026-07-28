"""Research-only configuration for the golden-set eval harness."""

from __future__ import annotations

from two_pass_classifier import config as production_config

# ---------------------------------------------------------------------------
# Benchmark matrix (locked model × Pass B effort screen)
# ---------------------------------------------------------------------------

# Production owns every classifier setting. The eval harness only defines the
# research matrix assembled from those supported production values.
EVAL_MODELS: tuple[str, ...] = production_config.SUPPORTED_MODELS
MATRIX_PASS_B_EFFORTS: tuple[str, ...] = (
    production_config.SUPPORTED_PASS_B_EFFORTS
)
DEFAULT_MODEL: str = production_config.DEFAULT_MODEL
DEFAULT_PASS_B_EFFORT: str = production_config.DEFAULT_PASS_B_EFFORT

# Per-row floor: below this, most probability went to non-verdict tokens
# (whitespace, punctuation), so renormalized confidence rests on a thin slice.
# The dashboard does not fail on a single outlier. It fails only when the
# mean dips below this floor, or more than VALID_MASS_MAX_BELOW_SHARE of rows
# are thin (so 1/100 is tolerated; 11/100 is not).
VALID_MASS_THRESHOLD: float = 0.90
VALID_MASS_MAX_BELOW_SHARE: float = 0.05

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

# Used only when no immutable production manifest can be supplied or discovered.
OFFLINE_PRODUCTION_ROW_FALLBACK: int = 37_746
