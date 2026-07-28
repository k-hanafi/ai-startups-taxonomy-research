"""Filesystem paths for production two-pass classifier artifacts."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_LIVE_INPUT = (
    PROJECT_ROOT / "outputs" / "tavilycrawl" / "processed" / "classifier_input.csv"
)
DEFAULT_DEAD_INPUT = (
    PROJECT_ROOT
    / "wayback_machine"
    / "outputs"
    / "processed"
    / "classifier_input_dead.csv"
)
DEFAULT_LIVE_RAW_RESULTS = (
    PROJECT_ROOT / "outputs" / "tavilycrawl" / "raw" / "raw_results.jsonl"
)
DEFAULT_DEAD_SCRAPE_PROCESSED = (
    PROJECT_ROOT
    / "wayback_machine"
    / "outputs"
    / "processed"
    / "scrape_processed_dead.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "two_pass_classifier"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
RUNS_DIR = OUTPUT_ROOT / "runs"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
