"""Shared filesystem paths for generated pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TAVILY_DIR = OUTPUTS_DIR / "tavilycrawl"

BATCH_DATA_DIR = OUTPUTS_DIR / "batch_data"
BATCH_REQUESTS_DIR = BATCH_DATA_DIR / "requests"
BATCH_RESULTS_DIR = BATCH_DATA_DIR / "results"
BATCH_ERRORS_DIR = BATCH_DATA_DIR / "errors"
BATCH_OUTPUTS_DIR = BATCH_DATA_DIR / "outputs"
BATCH_STATE_FILE = BATCH_DATA_DIR / "state.json"

PRODUCTION_CSVS_DIR = OUTPUTS_DIR / "production_csvs"
DEFAULT_CLASSIFICATION_OUTPUT_CSV = PRODUCTION_CSVS_DIR / "classified_startups_tavily.csv"

LOGS_DIR = OUTPUTS_DIR / "logs"
