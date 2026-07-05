"""Stage 1: stratified golden-set sampling.

Draws GOLDEN_SET_SIZE companies from the evidence-bearing live cohort,
stratified on the production (nano) predicted subclass crossed with
evidence-length terciles. Nano predictions are a *proxy* for strata —
true labels don't exist until the gold-labeling stage — so quotas are
approximate by design.

The committed golden_set.csv deliberately excludes website_evidence and
description text (public repo, licensing): evidence stays joinable from
the local classifier_input.csv via org_uuid.
"""

from __future__ import annotations

import logging

import pandas as pd

from evals.config import (
    EVIDENCE_TERCILE_LABELS,
    GOLDEN_SET_SIZE,
    SAMPLING_SEED,
    SUBCLASS_QUOTAS,
)
from evals.paths import (
    CLASSIFIER_INPUT_CSV,
    GOLDEN_SET_CSV,
    PRODUCTION_CLASSIFICATIONS_CSV,
)

logger = logging.getLogger(__name__)

# Columns written to the committed golden_set.csv. No evidence or
# description text (public repo). gold_* columns are filled in Stage 2.
GOLDEN_SET_COLUMNS = [
    "org_uuid",
    "name",
    "predicted_subclass",     # nano production prediction (stratum record)
    "predicted_ai_native",
    "predicted_rad",
    "evidence_chars",
    "evidence_tercile",
    "draft_ai_native",
    "draft_subclass",
    "draft_rad",
    "draft_rationale",
    "ambiguity_flag",
    "gold_verdict",
    "gold_ai_native",
    "gold_subclass",
    "gold_rad",
]


def sample_golden_set(
    predictions: pd.DataFrame,
    classifier_input: pd.DataFrame,
    quotas: dict[str, int] | None = None,
    seed: int = SAMPLING_SEED,
) -> pd.DataFrame:
    """Return the stratified golden-set frame (pure; no file I/O).

    Args:
        predictions: production classifications (CompanyID, subclass, ...).
        classifier_input: master + evidence rows (org_uuid, website_evidence, ...).
        quotas: rows to draw per predicted subclass; defaults to config.
        seed: RNG seed making the draw fully deterministic.
    """
    quotas = SUBCLASS_QUOTAS if quotas is None else quotas

    pool = predictions.merge(
        classifier_input[["org_uuid", "name", "website_evidence"]],
        left_on="CompanyID",
        right_on="org_uuid",
        how="inner",
    )
    pool["evidence_chars"] = pool["website_evidence"].fillna("").str.len()
    pool = pool[pool["evidence_chars"] > 0]

    # Global terciles over the evidence-bearing pool, so "short" means the
    # same thing in every stratum.
    tercile_edges = pool["evidence_chars"].quantile([1 / 3, 2 / 3]).tolist()
    pool["evidence_tercile"] = pool["evidence_chars"].map(
        lambda n: _tercile(n, tercile_edges)
    )

    sampled_parts: list[pd.DataFrame] = []
    for subclass, quota in sorted(quotas.items()):
        stratum = pool[pool["subclass"] == subclass]
        if len(stratum) < quota:
            raise ValueError(
                f"Stratum {subclass!r} has {len(stratum)} evidence-bearing rows, "
                f"quota is {quota}. Adjust SUBCLASS_QUOTAS."
            )
        sampled_parts.append(_sample_stratum(stratum, quota, seed))

    golden = pd.concat(sampled_parts, ignore_index=True)
    golden = golden.rename(
        columns={
            "subclass": "predicted_subclass",
            "ai_native": "predicted_ai_native",
            "rad_score": "predicted_rad",
        }
    )
    for col in GOLDEN_SET_COLUMNS:
        if col not in golden.columns:
            golden[col] = ""
    golden = golden[GOLDEN_SET_COLUMNS].sort_values("org_uuid").reset_index(drop=True)

    total = sum(quotas.values())
    if len(golden) != total:
        raise AssertionError(f"Sampled {len(golden)} rows, expected {total}")
    return golden


def _tercile(chars: int, edges: list[float]) -> str:
    if chars <= edges[0]:
        return EVIDENCE_TERCILE_LABELS[0]
    if chars <= edges[1]:
        return EVIDENCE_TERCILE_LABELS[1]
    return EVIDENCE_TERCILE_LABELS[2]


def _sample_stratum(stratum: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    """Draw *quota* rows spread as evenly as possible across terciles.

    Terciles short a row donate their remainder to whichever terciles still
    have supply, keeping the stratum total exactly at quota.
    """
    # Deterministic order before any sampling: pandas sample() is only
    # reproducible if the input row order is itself stable.
    stratum = stratum.sort_values("org_uuid")

    per_tercile = quota // len(EVIDENCE_TERCILE_LABELS)
    remainder = quota % len(EVIDENCE_TERCILE_LABELS)

    taken: list[pd.DataFrame] = []
    deficit = 0
    for i, label in enumerate(EVIDENCE_TERCILE_LABELS):
        want = per_tercile + (1 if i < remainder else 0)
        available = stratum[stratum["evidence_tercile"] == label]
        take = min(want, len(available))
        deficit += want - take
        if take:
            taken.append(available.sample(n=take, random_state=seed))

    if deficit:
        already = pd.concat(taken)["org_uuid"] if taken else pd.Series(dtype=str)
        rest = stratum[~stratum["org_uuid"].isin(already)]
        taken.append(rest.sample(n=deficit, random_state=seed))

    return pd.concat(taken, ignore_index=True)


def build_golden_set() -> pd.DataFrame:
    """I/O wrapper: read production artifacts, write evals/golden/golden_set.csv."""
    predictions = pd.read_csv(PRODUCTION_CLASSIFICATIONS_CSV)
    classifier_input = pd.read_csv(CLASSIFIER_INPUT_CSV)

    golden = sample_golden_set(predictions, classifier_input)

    GOLDEN_SET_CSV.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(GOLDEN_SET_CSV, index=False)
    logger.info("Wrote %d golden-set rows to %s", len(golden), GOLDEN_SET_CSV)

    summary = golden.groupby(["predicted_subclass", "evidence_tercile"]).size()
    logger.info("Strata:\n%s", summary.to_string())
    assert len(golden) == GOLDEN_SET_SIZE
    return golden
