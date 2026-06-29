"""Shared filesystem paths for generated pipeline artifacts."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TAVILY_DIR = OUTPUTS_DIR / "tavilycrawl"
TAVILY_RAW_DIR = TAVILY_DIR / "raw"
TAVILY_PROCESSED_DIR = TAVILY_DIR / "processed"

# An optional run namespace (set CLASSIFY_NS before import) reroutes ALL batch
# state and the output CSV under outputs/<ns>/, so a parallel classify run (e.g.
# the survivorship dead-cohort) physically cannot touch the finished modern
# artifacts. Inputs (DATA_DIR, TAVILY_*) stay shared — only generated outputs
# move. Bound once at import because every consumer imports these constants.
CLASSIFY_NS = os.environ.get("CLASSIFY_NS", "").strip()

if CLASSIFY_NS:
    _CLASSIFY_DIR = OUTPUTS_DIR / CLASSIFY_NS
    BATCH_DATA_DIR = _CLASSIFY_DIR / "batch_data"
    PRODUCTION_CSVS_DIR = _CLASSIFY_DIR
    DEFAULT_CLASSIFICATION_OUTPUT_CSV = _CLASSIFY_DIR / f"{CLASSIFY_NS}_classifications.csv"
else:
    BATCH_DATA_DIR = OUTPUTS_DIR / "batch_data"
    PRODUCTION_CSVS_DIR = OUTPUTS_DIR / "production_csvs"
    DEFAULT_CLASSIFICATION_OUTPUT_CSV = PRODUCTION_CSVS_DIR / "production_classifications.csv"

BATCH_RAW_DIR = BATCH_DATA_DIR / "raw"
BATCH_REQUESTS_DIR = BATCH_RAW_DIR / "requests"
BATCH_RESULTS_DIR = BATCH_RAW_DIR / "results"
BATCH_ERRORS_DIR = BATCH_RAW_DIR / "errors"
BATCH_STATE_FILE = BATCH_DATA_DIR / "state.json"

LOGS_DIR = OUTPUTS_DIR / "logs"
