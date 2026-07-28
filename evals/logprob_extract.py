"""Eval artifact adapters for production-owned Pass A confidence extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from two_pass_classifier import config as production_config
from two_pass_classifier.confidence import (
    BinaryConfidence,
    BinaryConfidenceUnavailable,
    LogprobExtractionError,
    candidate_value,
    extract_binary_confidence,
)

from evals import config as cfg


def extract_raw_file(path: Path) -> BinaryConfidence:
    """Extract confidence from one stored production response."""
    return extract_binary_confidence(json.loads(path.read_text(encoding="utf-8")))


def extract_run(raw_dir: Path) -> list[dict[str, Any]]:
    """Extract production confidence fields from every Pass A raw response."""
    files = sorted(raw_dir.glob("*_a.json")) or sorted(raw_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    for path in files:
        custom_id = path.stem.removesuffix("_a")
        try:
            rows.append(
                {
                    "custom_id": custom_id,
                    **extract_raw_file(path).as_dict(),
                }
            )
        except BinaryConfidenceUnavailable:
            continue
    return rows


def chosen_confidence(row: dict[str, Any]) -> float:
    """Return the production probability assigned to the sampled binary label."""
    return float(row["sampled_probability"])


def valid_mass_summary(
    rows: list[dict[str, Any]],
    *,
    threshold: float | None = None,
    max_below_share: float | None = None,
) -> dict[str, Any]:
    """Summarize the research-only valid-mass robustness check."""
    cut = float(cfg.VALID_MASS_THRESHOLD if threshold is None else threshold)
    allow = float(
        cfg.VALID_MASS_MAX_BELOW_SHARE
        if max_below_share is None
        else max_below_share
    )
    masses = [
        float(row["valid_mass"])
        for row in rows
        if row.get("valid_mass") is not None
    ]
    if not masses:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "mean": None,
            "threshold": cut,
            "max_below_share": allow,
            "n_below_threshold": 0,
            "below_share": 0.0,
        }
    ordered = sorted(masses)
    count = len(ordered)
    below = sum(mass < cut for mass in ordered)
    return {
        "n": count,
        "min": ordered[0],
        "p50": ordered[count // 2],
        "mean": sum(ordered) / count,
        "threshold": cut,
        "max_below_share": allow,
        "n_below_threshold": below,
        "below_share": below / count,
    }


def run_confidence(raw_dir: Path) -> dict[str, float]:
    """Map each stored custom ID to sampled-label confidence."""
    rows = extract_confidence_rows(raw_dir)
    return {row["custom_id"]: chosen_confidence(row) for row in rows}


def extract_confidence_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Extract confidence rows and refuse a missing or unusable raw directory."""
    raw_files = sorted(raw_dir.glob("*_a.json")) or sorted(raw_dir.glob("*.json"))
    rows = extract_run(raw_dir)
    if rows:
        return rows
    if not raw_dir.exists() or not raw_files:
        raise LogprobExtractionError(
            f"no raw response files under {raw_dir}; this run cannot supply "
            "production logprob confidence"
        )
    raise LogprobExtractionError(
        f"raw responses under {raw_dir} exist ({len(raw_files)} file(s)) but "
        "none yielded binary confidence because every residual bound exceeded "
        f"{production_config.MAX_CENSORED_INTERVAL_WIDTH}"
    )
