"""Shared offline-safe mechanics for eval run orchestration."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from evals.jsonl_io import iter_jsonl
from evals.paths import CLASSIFIER_INPUT_CSV, GOLDEN_SET_CSV

logger = logging.getLogger(__name__)

_RETRIABLE = retry(
    retry=retry_if_exception_type(
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
        )
    ),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def _git_commit() -> str:
    """Return the current commit for artifact provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_golden_rows() -> list[dict[str, Any]]:
    """Load model inputs for golden companies in committed golden-set order."""
    golden = pd.read_csv(
        GOLDEN_SET_CSV,
        dtype=str,
        keep_default_na=False,
    )
    order = list(golden["org_uuid"])
    wanted = set(order)

    matched: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(
        CLASSIFIER_INPUT_CSV,
        dtype=str,
        keep_default_na=False,
        chunksize=5000,
    ):
        hit = chunk[chunk["org_uuid"].isin(wanted)]
        for row in hit.to_dict(orient="records"):
            matched[row["org_uuid"]] = row
        if len(matched) == len(wanted):
            break

    missing = wanted - matched.keys()
    if missing:
        raise AssertionError(
            f"{len(missing)} golden org_uuids not found in classifier input: "
            f"{sorted(missing)[:5]}"
        )
    return [matched[company_id] for company_id in order]


def _completed_custom_ids(predictions_path: Path) -> set[str]:
    """Return IDs whose latest stored record is complete or legacy-complete."""
    if not predictions_path.exists():
        return set()
    completed: set[str] = set()
    for record in iter_jsonl(
        predictions_path,
        tolerate_truncated_final=True,
    ):
        status = record.get("status")
        if status is None or status == "completed":
            custom_id = record.get("custom_id")
            if custom_id:
                completed.add(str(custom_id))
    return completed
