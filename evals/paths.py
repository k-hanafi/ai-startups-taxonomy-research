"""Filesystem paths for the eval harness.

Input artifacts (production predictions + classifier input) are referenced
by literal path rather than imported from src.paths: importing src.config
transitively would require OPENAI_API_KEY at import time, which offline
stages (sampling, scoring) must not need.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Read-only inputs produced by the live pipeline (git-ignored, local only).
PRODUCTION_CLASSIFICATIONS_CSV = (
    PROJECT_ROOT / "outputs" / "production_csvs" / "production_classifications.csv"
)
CLASSIFIER_INPUT_CSV = (
    PROJECT_ROOT / "outputs" / "tavilycrawl" / "processed" / "classifier_input.csv"
)

# Harness artifacts.
EVALS_DIR = PROJECT_ROOT / "evals"
GOLDEN_DIR = EVALS_DIR / "golden"
GOLDEN_SET_CSV = GOLDEN_DIR / "golden_set.csv"          # committed (no evidence text)
RUNS_DIR = EVALS_DIR / "runs"                            # runs/<run_id>/raw/ git-ignored

# Stage 2 labeling artifacts (git-ignored: both embed scraped evidence text).
LABELING_WORKSPACE_DIR = GOLDEN_DIR / "workspace"
REVIEW_PAGE_HTML = GOLDEN_DIR / "review_page.html"
