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
# Stable Pass A banks: runs/pass_a_banks/<model>/ (git-ignored with runs/).
PASS_A_BANKS_DIRNAME = "pass_a_banks"


def pass_a_bank_run_id(model: str) -> str:
    """Stable run_id for the per-model Pass A bank under evals/runs/."""
    return f"{PASS_A_BANKS_DIRNAME}/{model}"


def run_dir(run_id: str) -> Path:
    """Directory holding one model run's artifacts."""
    return RUNS_DIR / run_id


def run_config_path(run_id: str) -> Path:
    """Config snapshot (model, effort, prompt/schema/formatter hashes, git commit)."""
    return run_dir(run_id) / "config.json"


def run_predictions_path(run_id: str) -> Path:
    """Label-only predictions, one JSON object per row. Drives resume + scoring."""
    return run_dir(run_id) / "predictions.jsonl"


def run_raw_dir(run_id: str) -> Path:
    """Full API responses (incl. logprobs). Git-ignored: large + provider-shaped."""
    return run_dir(run_id) / "raw"


def run_scored_path(run_id: str) -> Path:
    """Offline scoring summary (committed: metrics only, no evidence text)."""
    return run_dir(run_id) / "scored.json"


def parity_report_path(run_id: str) -> Path:
    """Batch-vs-sync parity smoke report (gate Q4)."""
    return run_dir(run_id) / "parity_report.json"

# Stage 2 labeling artifacts (git-ignored: both embed scraped evidence text).
LABELING_WORKSPACE_DIR = GOLDEN_DIR / "workspace"
REVIEW_PAGE_HTML = GOLDEN_DIR / "review_page.html"

# Two-pass classifier prompts (committed, reviewed in PR #14).
PROMPTS_DIR = PROJECT_ROOT / "prompts"
BINARY_GATE_PROMPT = PROMPTS_DIR / "binary_gate_prompt.txt"
SUBCLASS_RAD_PROMPT = PROMPTS_DIR / "subclass_rad_prompt.txt"
FAMILY_BLOCK_AI = PROMPTS_DIR / "family_block_ai_native.txt"
FAMILY_BLOCK_NOT = PROMPTS_DIR / "family_block_not_ai_native.txt"
