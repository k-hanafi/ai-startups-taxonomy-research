"""Stage 7: offline, re-runnable scorer for eval runs (Metrics v1).

Scores one run's predictions.jsonl against the provisional gold labels (the
Fable draft_* columns in golden_set.csv, per pivots 4 and 5) and writes
evals/runs/<run_id>/scored.json. Because gold may be re-drafted once at the
end of the pipeline, everything here is a pure function of files on disk:
re-scoring every banked run later costs one command.

Metrics: per-axis accuracy (ai_native / subclass / rad), macro-F1, confusion
matrices, bootstrap CIs on accuracy, paired-bootstrap CIs on deltas vs a
baseline run, cost per row from actual token usage, and an output/reasoning
token summary that sizes MAX_OUTPUT_TOKENS and the cost model (gate Q6).

Calibration (reliability bins + selective-prediction curve) is computed ONLY
when a per-row binary confidence is supplied, either as a binary_confidence
field on prediction records or as an external {custom_id|org_uuid: float}
mapping. The logprob extractor (Stage 6) plugs in through that data seam;
this module never imports it.

This module must stay importable without OPENAI_API_KEY: no src.* or
evals.runner imports.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from evals import config as cfg
from evals.paths import (
    GOLDEN_SET_CSV,
    run_config_path,
    run_predictions_path,
    run_scored_path,
)

logger = logging.getLogger(__name__)

# Sentinel for a prediction axis the model never produced (e.g. Pass B never
# ran). Kept as a real label so misses show up in the confusion matrix
# instead of silently shrinking the denominator.
MISSING = "__missing__"

AXES = ("ai_native", "subclass", "rad")

GOLD_SOURCE = "Fable draft_* columns (provisional gold, pivots 4+5)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _norm_label(value: Any) -> str:
    if value is None:
        return MISSING
    text = str(value).strip()
    return text if text else MISSING


def load_gold() -> dict[str, dict[str, str]]:
    """org_uuid -> normalized gold label per axis, from the committed CSV."""
    gold: dict[str, dict[str, str]] = {}
    with GOLDEN_SET_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels = {
                "ai_native": _norm_label(row.get("draft_ai_native")),
                "subclass": _norm_label(row.get("draft_subclass")),
                "rad": _norm_label(row.get("draft_rad")),
            }
            if MISSING in labels.values():
                raise SystemExit(
                    f"Golden row {row.get('org_uuid')} has empty draft_* labels; "
                    "gold must be complete before scoring."
                )
            gold[row["org_uuid"]] = labels
    if not gold:
        raise SystemExit(f"No gold rows found in {GOLDEN_SET_CSV}")
    return gold


def load_predictions(run_id: str) -> list[dict[str, Any]]:
    """Completed prediction records for a run, keyed checks left to callers."""
    path = run_predictions_path(run_id)
    if not path.exists():
        raise SystemExit(f"No predictions found for run {run_id} at {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed predictions line in %s", path)
    return records


def _is_completed(record: dict[str, Any]) -> bool:
    # Records without a status field (older runs) are treated as completed,
    # matching the runner's resume semantics.
    status = record.get("status")
    return status is None or status == "completed"


def completed_by_uuid(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Last completed record per org_uuid (a resumed run may append repairs)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        if _is_completed(rec) and rec.get("org_uuid"):
            out[rec["org_uuid"]] = rec
    return out


def _predicted_labels(record: dict[str, Any]) -> dict[str, str]:
    return {
        "ai_native": _norm_label(record.get("ai_native")),
        "subclass": _norm_label(record.get("subclass")),
        "rad": _norm_label(record.get("rad_score")),
    }


# ---------------------------------------------------------------------------
# Metric math (pure, tested on synthetic fixtures)
# ---------------------------------------------------------------------------

def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        raise ValueError("accuracy of an empty sample is undefined")
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Mean per-class F1 over every label seen in gold or predictions."""
    labels = sorted(set(gold) | set(pred))
    f1s: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def confusion_matrix(gold: list[str], pred: list[str]) -> dict[str, Any]:
    """Sparse gold-row x predicted-column counts, JSON-friendly."""
    matrix: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, pred):
        matrix.setdefault(g, {})
        matrix[g][p] = matrix[g].get(p, 0) + 1
    return {"labels": sorted(set(gold) | set(pred)), "matrix": matrix}


def bootstrap_accuracy_ci(
    correct: list[bool],
    resamples: int = cfg.BOOTSTRAP_RESAMPLES,
    seed: int = cfg.BOOTSTRAP_SEED,
    level: float = cfg.CONFIDENCE_LEVEL,
) -> list[float]:
    """Percentile bootstrap CI on a single run's accuracy."""
    arr = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return [float(lo), float(hi)]


def paired_bootstrap_delta(
    correct_run: list[bool],
    correct_baseline: list[bool],
    resamples: int = cfg.BOOTSTRAP_RESAMPLES,
    seed: int = cfg.BOOTSTRAP_SEED,
    level: float = cfg.CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """CI on (run - baseline) accuracy, resampling the SAME rows for both.

    Pairing matters: both runs answered the same 100 companies, so row
    difficulty is shared noise. Resampling rows jointly cancels it and gives
    a much tighter CI on the delta than two independent bootstraps would.
    """
    if len(correct_run) != len(correct_baseline):
        raise ValueError("paired bootstrap needs aligned samples")
    a = np.asarray(correct_run, dtype=float)
    b = np.asarray(correct_baseline, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return {
        "delta_accuracy": float(a.mean() - b.mean()),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0.0 or hi < 0.0),
    }


# ---------------------------------------------------------------------------
# Cost + token usage (gate Q6)
# ---------------------------------------------------------------------------

def _record_tokens(record: dict[str, Any]) -> dict[str, int]:
    """input/output/reasoning token totals for one record, both run shapes.

    Single-pass records carry flat fields; two-pass records carry a_/b_
    prefixed fields that are summed (a completed two-pass row always has
    both passes).
    """
    if "a_input_tokens" in record:
        keys = {
            "input": ("a_input_tokens", "b_input_tokens"),
            "output": ("a_output_tokens", "b_output_tokens"),
            "reasoning": ("a_reasoning_tokens", "b_reasoning_tokens"),
        }
        return {
            kind: sum(int(record.get(k) or 0) for k in fields)
            for kind, fields in keys.items()
        }
    return {
        "input": int(record.get("input_tokens") or 0),
        "output": int(record.get("output_tokens") or 0),
        "reasoning": int(record.get("reasoning_tokens") or 0),
    }


def _token_stats(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def cost_and_tokens(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Actual-usage cost + token distributions over completed records."""
    tokens = [_record_tokens(r) for r in records]
    total_in = sum(t["input"] for t in tokens)
    total_out = sum(t["output"] for t in tokens)

    pricing = cfg.EVAL_MODEL_PRICING.get(model)
    total_usd = mean_usd = None
    if pricing is not None and records:
        total_usd = total_in / 1e6 * pricing["input"] + total_out / 1e6 * pricing["output"]
        mean_usd = total_usd / len(records)

    return {
        "model": model,
        "n_rows": len(records),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_usd": total_usd,
        "mean_usd_per_row": mean_usd,
        "pricing_per_mtok": pricing,
        "pricing_note": (
            "Sync list price on total input tokens. Cached-input discount not "
            "applied (predictions records do not carry cached_tokens), so this "
            "is an upper bound on actual spend."
        ),
        "output_tokens": _token_stats([t["output"] for t in tokens]),
        "reasoning_tokens": _token_stats([t["reasoning"] for t in tokens]),
        "max_output_tokens_note": (
            "Size MAX_OUTPUT_TOKENS from the output_tokens p95/max above "
            "(reasoning tokens count against the cap)."
        ),
    }


# ---------------------------------------------------------------------------
# Calibration (optional input seam for the Stage 6 extractor)
# ---------------------------------------------------------------------------

def resolve_confidence(
    records: list[dict[str, Any]],
    external: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """org_uuid -> binary confidence, from external mapping or record fields.

    The external mapping may be keyed by org_uuid or custom_id. Returns an
    empty dict when no confidence is available anywhere, which callers treat
    as "skip calibration".
    """
    out: dict[str, float] = {}
    for rec in records:
        uuid = rec.get("org_uuid")
        if not uuid:
            continue
        value = None
        if external is not None:
            value = external.get(uuid)
            if value is None and rec.get("custom_id"):
                value = external.get(rec["custom_id"])
        if value is None:
            value = rec.get("binary_confidence")
        if value is not None:
            out[uuid] = float(value)
    return out


def reliability_bins(
    confidence: list[float],
    correct: list[bool],
    n_bins: int = cfg.CALIBRATION_BINS,
) -> dict[str, Any]:
    """Equal-width reliability bins on [0, 1] plus expected calibration error."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Right-inclusive top bin so confidence == 1.0 is counted.
        mask = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        count = int(mask.sum())
        mean_conf = float(conf[mask].mean()) if count else None
        acc = float(corr[mask].mean()) if count else None
        bins.append({
            "range": [float(lo), float(hi)],
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": acc,
        })
        if count:
            ece += count / len(conf) * abs(acc - mean_conf)
    return {"bins": bins, "ece": float(ece)}


def selective_prediction_curve(
    confidence: list[float],
    correct: list[bool],
    coverage_grid: list[float] = cfg.SELECTIVE_COVERAGE_GRID,
) -> list[dict[str, Any]]:
    """Accuracy when only answering on the top-X% most confident rows.

    A useful confidence signal makes this curve slope down: accuracy on the
    confident head beats accuracy at full coverage.
    """
    order = np.argsort(np.asarray(confidence, dtype=float))[::-1]
    corr = np.asarray(correct, dtype=float)[order]
    points = []
    for coverage in coverage_grid:
        k = max(1, int(round(coverage * len(corr))))
        points.append({
            "coverage": coverage,
            "n": k,
            "accuracy": float(corr[:k].mean()),
        })
    return points


def calibration_report(
    confidence: dict[str, float],
    gold: dict[str, dict[str, str]],
    predictions: dict[str, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Binary-axis calibration, or None when no confidence input exists."""
    if not confidence:
        return None
    conf_list: list[float] = []
    correct_list: list[bool] = []
    for uuid, value in confidence.items():
        if uuid not in gold or uuid not in predictions:
            continue
        pred = _predicted_labels(predictions[uuid])["ai_native"]
        if pred == MISSING:
            continue
        conf_list.append(value)
        correct_list.append(pred == gold[uuid]["ai_native"])
    if not conf_list:
        return None
    return {
        "axis": "ai_native",
        "n": len(conf_list),
        "reliability": reliability_bins(conf_list, correct_list),
        "selective_prediction": selective_prediction_curve(conf_list, correct_list),
    }


# ---------------------------------------------------------------------------
# Scoring a run
# ---------------------------------------------------------------------------

def _axis_correctness(
    gold: dict[str, dict[str, str]],
    predictions: dict[str, dict[str, Any]],
    uuids: list[str],
) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {axis: [] for axis in AXES}
    for uuid in uuids:
        pred = _predicted_labels(predictions[uuid])
        for axis in AXES:
            out[axis].append(pred[axis] == gold[uuid][axis])
    return out


def score_run(
    run_id: str,
    baseline_run_id: Optional[str] = None,
    confidence: Optional[dict[str, float]] = None,
    write: bool = True,
) -> dict[str, Any]:
    """Score one run against gold; optionally compare to a baseline run.

    Returns the scored summary dict and (by default) writes it to
    evals/runs/<run_id>/scored.json.
    """
    gold = load_gold()
    records = load_predictions(run_id)
    predictions = completed_by_uuid(records)

    scored_uuids = sorted(uuid for uuid in predictions if uuid in gold)
    if not scored_uuids:
        raise SystemExit(
            f"Run {run_id} has no completed predictions matching gold rows."
        )

    axes_report: dict[str, Any] = {}
    for axis in AXES:
        gold_labels = [gold[u][axis] for u in scored_uuids]
        pred_labels = [_predicted_labels(predictions[u])[axis] for u in scored_uuids]
        correct = [g == p for g, p in zip(gold_labels, pred_labels)]
        axes_report[axis] = {
            "accuracy": accuracy(gold_labels, pred_labels),
            "accuracy_ci95": bootstrap_accuracy_ci(correct),
            "macro_f1": macro_f1(gold_labels, pred_labels),
            "confusion": confusion_matrix(gold_labels, pred_labels),
        }

    run_config = {}
    config_path = run_config_path(run_id)
    if config_path.exists():
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
    model = run_config.get("model") or predictions[scored_uuids[0]].get("model") or ""

    scored_records = [predictions[u] for u in scored_uuids]
    report: dict[str, Any] = {
        "run_id": run_id,
        "scored_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gold_source": GOLD_SOURCE,
        "n_gold": len(gold),
        "n_prediction_records": len(records),
        "n_scored": len(scored_uuids),
        "bootstrap": {
            "resamples": cfg.BOOTSTRAP_RESAMPLES,
            "seed": cfg.BOOTSTRAP_SEED,
            "confidence_level": cfg.CONFIDENCE_LEVEL,
        },
        "axes": axes_report,
        "cost": cost_and_tokens(scored_records, model),
        "calibration": calibration_report(
            resolve_confidence(scored_records, confidence), gold, predictions
        ),
        "vs_baseline": None,
    }

    if baseline_run_id is not None:
        report["vs_baseline"] = compare_to_baseline(
            gold, predictions, baseline_run_id
        )

    if write:
        out_path = run_scored_path(run_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s", out_path)
    return report


def compare_to_baseline(
    gold: dict[str, dict[str, str]],
    predictions: dict[str, dict[str, Any]],
    baseline_run_id: str,
) -> dict[str, Any]:
    """Paired-bootstrap deltas vs a baseline run, over the shared row set."""
    baseline = completed_by_uuid(load_predictions(baseline_run_id))
    shared = sorted(
        uuid for uuid in predictions if uuid in baseline and uuid in gold
    )
    if not shared:
        raise SystemExit(
            f"No shared completed rows between this run and baseline "
            f"{baseline_run_id}; cannot pair."
        )
    run_correct = _axis_correctness(gold, predictions, shared)
    base_correct = _axis_correctness(gold, baseline, shared)
    return {
        "baseline_run_id": baseline_run_id,
        "n_paired": len(shared),
        "deltas": {
            axis: paired_bootstrap_delta(run_correct[axis], base_correct[axis])
            for axis in AXES
        },
    }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def load_confidence_file(path: str | Path) -> dict[str, float]:
    """External confidence mapping: JSON object of {org_uuid|custom_id: float}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a JSON object mapping ids to floats")
    return {str(k): float(v) for k, v in data.items()}


def score_cli(
    run_id: str,
    baseline: Optional[str],
    confidence: Optional[dict[str, float]],
) -> None:
    report = score_run(run_id, baseline_run_id=baseline, confidence=confidence)
    for axis in AXES:
        ax = report["axes"][axis]
        logger.info(
            "%s: accuracy=%.3f (95%% CI %.3f-%.3f)  macro-F1=%.3f",
            axis, ax["accuracy"], ax["accuracy_ci95"][0],
            ax["accuracy_ci95"][1], ax["macro_f1"],
        )
    cost = report["cost"]
    if cost["mean_usd_per_row"] is not None:
        logger.info("cost: $%.4f/row ($%.2f total, %d rows)",
                    cost["mean_usd_per_row"], cost["total_usd"], cost["n_rows"])
    logger.info("calibration: %s",
                "computed" if report["calibration"] else
                "skipped (no binary confidence input)")
    if report["vs_baseline"]:
        for axis, delta in report["vs_baseline"]["deltas"].items():
            logger.info("delta vs %s on %s: %+.3f (95%% CI %+.3f..%+.3f)",
                        report["vs_baseline"]["baseline_run_id"], axis,
                        delta["delta_accuracy"], delta["ci95"][0], delta["ci95"][1])
