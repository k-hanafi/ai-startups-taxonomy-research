"""Stage 9 eval-dashboard metrics: scored.json (or fixture) → chart-ready dict.

Pure compute. The HTML builder only shapes Plotly traces and the config
filter. No OpenAI imports: offline stages must not require API keys.

Fixture path exists because paid matrix runs are not banked yet. Real
``evals/runs/*/scored.json`` files plug in with the same loader once they
exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from evals.paths import PROJECT_ROOT, RUNS_DIR, run_scored_path

DEFAULT_FIXTURE = (
    PROJECT_ROOT
    / "evals"
    / "tests"
    / "fixtures"
    / "dashboard"
    / "dashboard_mock_runs.json"
)

# Locked screen matrix (labels + fixture). Short group keys drive
# the HTML model-group pills (nano / mini / luna).
MODEL_GROUP_BY_MODEL: dict[str, str] = {
    "gpt-5.4-nano": "nano",
    "gpt-5.4-mini": "mini",
    "gpt-5.6-luna": "luna",
}

MODEL_GROUP_ORDER: tuple[str, ...] = ("nano", "mini", "luna")
# Known Pass B / reasoning efforts first; unknown last so missing metadata
# does not sort as if it were medium.
EFFORT_ORDER: tuple[str, ...] = ("low", "medium", "high", "minimal", "none", "unknown")
UNKNOWN_EFFORT = "unknown"


def _short_model(model: str) -> str:
    group = MODEL_GROUP_BY_MODEL.get(model)
    if group:
        return group
    # Fallback: last hyphen segment (gpt-5.4-nano → nano).
    return model.rsplit("-", 1)[-1] if model else "unknown"


def _config_id(model: str, effort: str) -> str:
    short = _short_model(model)
    effort_key = {"medium": "med"}.get(effort, effort)
    return f"{short}-{effort_key}"


def _parse_effort_from_run_id(run_id: str) -> Optional[str]:
    """Best-effort effort token from run_id (..._low_r1 / ..._medium_r1)."""
    parts = run_id.lower().split("_")
    for token in ("none", "low", "medium", "high", "minimal"):
        if token in parts:
            return token
    return None


def _parse_repeat_from_run_id(run_id: str) -> Optional[int]:
    """Repeat index from run_id suffix (..._r1 → 1)."""
    if not run_id:
        return None
    tail = run_id.rsplit("_", 1)[-1]
    if len(tail) >= 2 and tail[0] == "r" and tail[1:].isdigit():
        return int(tail[1:])
    return None


def _parse_model_from_run_id(run_id: str) -> str:
    """Pull model slug from run_id when scored.json omits an explicit model."""
    # Patterns: 2026-07-05_gpt-5.4-nano_medium_r1, mock_gpt-5.4-mini_low_r1
    for known in MODEL_GROUP_BY_MODEL:
        if known in run_id:
            return known
    return ""


def _infer_kind(scored: dict[str, Any], run_id: str, effort: str) -> Optional[str]:
    """Architecture label for leaderboard captions (classification vs single_pass)."""
    kind = scored.get("kind")
    if kind == "two_pass":
        return "classification"
    if kind:
        return str(kind)
    # New run ids use _classification_; older ones used _2pass_.
    if (
        "classification" in run_id
        or "2pass" in run_id
        or run_id.startswith("mock_")
    ):
        return "classification"
    if effort == "none":
        return "single_pass"
    return None


def _disambiguate_repeat_labels(configs: list[dict[str, Any]]) -> None:
    """When Stage-2 finalist repeats share model×effort, append · rN to labels.

    Stage-1 screens keep plain labels (one row per matrix cell). Only collide
    when multiple runs share the same human label in one metrics payload.
    """
    counts: dict[str, int] = {}
    for c in configs:
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    for c in configs:
        if counts[c["label"]] <= 1:
            continue
        rep = _parse_repeat_from_run_id(str(c.get("run_id") or ""))
        if rep is not None:
            c["label"] = f"{c['label']} · r{rep}"


def _projected_usd(scored: dict[str, Any]) -> Optional[float]:
    est = scored.get("production_cost_estimate") or {}
    if not est:
        return None
    steps = est.get("steps") or {}
    scale = steps.get("4_scale") or {}
    if scale.get("available") and scale.get("estimated_production_usd") is not None:
        return float(scale["estimated_production_usd"])
    screen = scored.get("screen") or {}
    if screen.get("projected_usd") is not None:
        return float(screen["projected_usd"])
    return None


def _ci_half_width(ci: Optional[list[float]], accuracy: float) -> Optional[float]:
    if not ci or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return None
    return float((float(ci[1]) - float(ci[0])) / 2.0)


def _mean_confidence(cal: Optional[dict[str, Any]], screen: dict[str, Any]) -> Optional[float]:
    if screen.get("mean_confidence") is not None:
        return float(screen["mean_confidence"])
    if not cal:
        return None
    rel = cal.get("reliability") or {}
    bins = rel.get("bins") or []
    total = sum(int(b.get("count") or 0) for b in bins)
    if total <= 0:
        return None
    weighted = 0.0
    for b in bins:
        n = int(b.get("count") or 0)
        mc = b.get("mean_confidence")
        if n and mc is not None:
            weighted += n * float(mc)
    return weighted / total if total else None


def _share_above_90(cal: Optional[dict[str, Any]], screen: dict[str, Any]) -> Optional[float]:
    if screen.get("share_above_90") is not None:
        return float(screen["share_above_90"])
    if not cal:
        return None
    rel = cal.get("reliability") or {}
    bins = rel.get("bins") or []
    total = sum(int(b.get("count") or 0) for b in bins)
    if total <= 0:
        return None
    above = 0
    for b in bins:
        rng = b.get("range") or [0, 0]
        if float(rng[0]) >= 0.9:
            above += int(b.get("count") or 0)
    return above / total


def _ece(cal: Optional[dict[str, Any]], screen: dict[str, Any]) -> Optional[float]:
    if screen.get("ece") is not None:
        return float(screen["ece"])
    if not cal:
        return None
    rel = cal.get("reliability") or {}
    ece = rel.get("ece")
    return float(ece) if ece is not None else None


def _selective_at_coverage(
    cal: Optional[dict[str, Any]], coverage: float = 0.5
) -> Optional[float]:
    if not cal:
        return None
    for point in cal.get("selective_prediction") or []:
        if abs(float(point.get("coverage", -1)) - coverage) < 1e-9:
            acc = point.get("accuracy")
            return float(acc) if acc is not None else None
    return None


def _vs_baseline_delta(scored: dict[str, Any], axis: str = "subclass") -> Optional[dict[str, Any]]:
    vs = scored.get("vs_baseline")
    if not vs:
        return None
    delta = (vs.get("deltas") or {}).get(axis)
    if not delta:
        return None
    return {
        "baseline_run_id": vs.get("baseline_run_id"),
        "n_paired": vs.get("n_paired"),
        "delta_accuracy": delta.get("delta_accuracy"),
        "ci95": delta.get("ci95"),
        "significant": delta.get("significant"),
        "axis": axis,
    }


def config_row_from_scored(scored: dict[str, Any]) -> dict[str, Any]:
    """Normalize one scored.json (or stub) into a dashboard config row."""
    run_id = str(scored.get("run_id") or "")
    screen = dict(scored.get("screen") or {})
    model = (
        scored.get("model")
        or screen.get("model")
        or _parse_model_from_run_id(run_id)
    )
    effort = (
        scored.get("effort_b")
        or scored.get("effort")
        or scored.get("reasoning_effort")
        or screen.get("effort_b")
        or _parse_effort_from_run_id(run_id)
        or UNKNOWN_EFFORT
    )
    axes = scored.get("axes") or {}
    subclass = axes.get("subclass") or {}
    ai_native = axes.get("ai_native") or {}
    rad = axes.get("rad") or {}
    cal = scored.get("calibration")
    pbm = scored.get("pass_b_metrics") or {}
    subclass_cond = pbm.get("subclass_family_conditional") or {}
    rad_ai = pbm.get("rad_ai_native_only") or {}
    boundary = pbm.get("boundary_disagreement") or {}
    latency = ((scored.get("latency") or {}).get("latency_s")) or {}
    if not latency and screen:
        latency = {
            "p50": screen.get("latency_p50"),
            "p95": screen.get("latency_p95"),
        }

    subclass_acc = float(subclass.get("accuracy", screen.get("subclass_acc", 0.0)))
    ci = subclass.get("accuracy_ci95")
    half = _ci_half_width(ci, subclass_acc)
    if half is None and screen.get("subclass_ci") is not None:
        half = float(screen["subclass_ci"])

    group = MODEL_GROUP_BY_MODEL.get(str(model), _short_model(str(model)))
    # Filter id must be unique per scored run. model×effort (and screen.id)
    # collide when Stage-2 finalist repeats share the same matrix cell; the
    # HTML toolbar uses a Set of ids, so duplicates make "X of Y visible"
    # disagree with chart/table row count. Labels stay human model×effort
    # until build_metrics disambiguates collisions with · rN.
    if run_id:
        cfg_id = run_id
    else:
        cfg_id = screen.get("id") or _config_id(str(model), str(effort))
    label = screen.get("label") or f"{_short_model(str(model))} / {effort}"
    kind = _infer_kind(scored, run_id, str(effort))
    n_scored = scored.get("n_scored")
    n_expected = scored.get("n_expected")
    is_partial = (
        n_scored is not None
        and n_expected is not None
        and int(n_scored) < int(n_expected)
    )
    rel = (cal or {}).get("reliability") or {}

    return {
        "id": cfg_id,
        "label": label,
        "run_id": run_id,
        "model": model,
        "model_group": group,
        "effort_b": effort,
        "kind": kind,
        "n_scored": n_scored,
        "n_expected": n_expected,
        "is_partial": is_partial,
        "subclass_acc": subclass_acc,
        "subclass_ci": half,
        "subclass_family_conditional_acc": (
            float(subclass_cond["accuracy"])
            if subclass_cond.get("accuracy") is not None
            else screen.get("subclass_family_conditional_acc")
        ),
        "rad_ai_native_only_acc": (
            float(rad_ai["accuracy"])
            if rad_ai.get("accuracy") is not None
            else screen.get("rad_ai_native_only_acc")
        ),
        "boundary_disagreement_rate": (
            float(boundary["rate"])
            if boundary.get("rate") is not None
            else screen.get("boundary_disagreement_rate")
        ),
        "ai_native_acc": float(
            ai_native.get("accuracy", screen.get("ai_native_acc", 0.0))
        ),
        "rad_acc": float(rad.get("accuracy", screen.get("rad_acc", 0.0))),
        "macro_f1": float(
            subclass.get("macro_f1", screen.get("macro_f1", 0.0))
        ),
        "projected_usd": _projected_usd(scored),
        "mean_confidence": _mean_confidence(cal, screen),
        "share_above_90": _share_above_90(cal, screen),
        "ece": _ece(cal, screen),
        "reliability_bins": rel.get("bins") if rel else screen.get("reliability_bins"),
        "selective_acc_50": (
            _selective_at_coverage(cal, 0.5)
            if cal
            else screen.get("selective_acc_50")
        ),
        "vs_baseline": _vs_baseline_delta(scored),
        "latency_p50": (
            float(latency["p50"]) if latency.get("p50") is not None else None
        ),
        "latency_p95": (
            float(latency["p95"]) if latency.get("p95") is not None else None
        ),
        "repeat": _parse_repeat_from_run_id(run_id),
        "is_aggregate": False,
    }


def _model_groups(configs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for g in MODEL_GROUP_ORDER:
        ids = [c["id"] for c in configs if c["model_group"] == g]
        if ids:
            groups[g] = {"label": g, "ids": ids}
    # Any unexpected groups (future models) append after the locked three.
    for c in configs:
        g = c["model_group"]
        if g not in groups:
            groups[g] = {"label": g, "ids": [c["id"]]}
        elif c["id"] not in groups[g]["ids"]:
            groups[g]["ids"].append(c["id"])
    return groups


def _sort_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_rank = {g: i for i, g in enumerate(MODEL_GROUP_ORDER)}
    effort_rank = {e: i for i, e in enumerate(EFFORT_ORDER)}

    def key(c: dict[str, Any]) -> tuple[int, int, str]:
        return (
            group_rank.get(c["model_group"], 99),
            effort_rank.get(c["effort_b"], 99),
            c["id"],
        )

    return sorted(configs, key=key)


def _aggregate_finalist_repeats(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """When r1/r2/r3 share model×effort, append a mean±range aggregate row.

    Individual repeats stay visible (disambiguated labels). The aggregate
    encodes determinism variance the matrix screen asks for.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in configs:
        if c.get("is_aggregate"):
            continue
        rep = c.get("repeat")
        if rep is None:
            continue
        key = (str(c.get("model") or ""), str(c.get("effort_b") or ""))
        groups.setdefault(key, []).append(c)

    extras: list[dict[str, Any]] = []
    for (model, effort), rows in groups.items():
        if len(rows) < 2:
            continue
        accs = [float(r["subclass_acc"]) for r in rows]
        mean_acc = sum(accs) / len(accs)
        lo, hi = min(accs), max(accs)
        eces = [float(r["ece"]) for r in rows if r.get("ece") is not None]
        costs = [
            float(r["projected_usd"])
            for r in rows
            if r.get("projected_usd") is not None
        ]
        group = MODEL_GROUP_BY_MODEL.get(model, _short_model(model))
        extras.append({
            "id": f"agg_{model}_{effort}",
            "label": f"{_short_model(model)} / {effort} · mean±range",
            "run_id": "",
            "model": model,
            "model_group": group,
            "effort_b": effort,
            "kind": rows[0].get("kind"),
            "n_scored": rows[0].get("n_scored"),
            "subclass_acc": mean_acc,
            "subclass_ci": (hi - lo) / 2.0,
            "subclass_acc_range": [lo, hi],
            "subclass_acc_mean": mean_acc,
            "n_repeats": len(rows),
            "ai_native_acc": sum(float(r["ai_native_acc"]) for r in rows) / len(rows),
            "rad_acc": sum(float(r["rad_acc"]) for r in rows) / len(rows),
            "macro_f1": sum(float(r["macro_f1"]) for r in rows) / len(rows),
            "projected_usd": (sum(costs) / len(costs)) if costs else None,
            "mean_confidence": None,
            "share_above_90": None,
            "ece": (sum(eces) / len(eces)) if eces else None,
            "reliability_bins": None,
            "selective_acc_50": None,
            "vs_baseline": None,
            "latency_p50": None,
            "latency_p95": None,
            "repeat": None,
            "is_aggregate": True,
            "subclass_family_conditional_acc": None,
            "rad_ai_native_only_acc": None,
            "boundary_disagreement_rate": None,
        })
    return extras


def build_metrics(
    scored_runs: Iterable[dict[str, Any]],
    *,
    synthetic: bool = False,
    source: str = "",
) -> dict[str, Any]:
    """Build the chart-ready metrics dict (configs + filter keys)."""
    configs = [config_row_from_scored(s) for s in scored_runs]
    _disambiguate_repeat_labels(configs)
    configs.extend(_aggregate_finalist_repeats(configs))
    configs = _sort_configs(configs)
    return {
        "synthetic": synthetic,
        "source": source,
        "n_configs": len(configs),
        "configs": configs,
        "config_ids": [c["id"] for c in configs],
        "model_groups": _model_groups(configs),
        "model_group_order": [
            g for g in MODEL_GROUP_ORDER if g in _model_groups(configs)
        ],
    }


def load_scored_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object (scored.json shape)")
    return data


def load_fixture(path: Path | None = None) -> dict[str, Any]:
    """Load the multi-run mock fixture and return dashboard metrics."""
    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "scored_runs" in raw:
        runs = raw["scored_runs"]
        synthetic = bool(raw.get("synthetic", True))
    elif isinstance(raw, list):
        runs = raw
        synthetic = True
    else:
        raise ValueError(
            f"{fixture_path} must be a list of scored stubs or an object "
            "with a scored_runs array"
        )
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{fixture_path} has no scored_runs")
    return build_metrics(
        runs,
        synthetic=synthetic,
        source=str(fixture_path.relative_to(PROJECT_ROOT))
        if fixture_path.is_relative_to(PROJECT_ROOT)
        else str(fixture_path),
    )


def load_from_run_ids(run_ids: Iterable[str]) -> dict[str, Any]:
    """Load real scored.json files for the given run ids."""
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_id in run_ids:
        path = run_scored_path(run_id)
        if not path.exists():
            missing.append(run_id)
            continue
        runs.append(load_scored_json(path))
    if missing:
        raise FileNotFoundError(
            "Missing scored.json for: " + ", ".join(missing)
        )
    return build_metrics(
        runs,
        synthetic=False,
        source=f"{len(runs)} runs under evals/runs/",
    )


def load_from_scored_paths(paths: Iterable[Path]) -> dict[str, Any]:
    """Load one or more scored.json paths into dashboard metrics."""
    runs = [load_scored_json(Path(p)) for p in paths]
    return build_metrics(
        runs,
        synthetic=False,
        source=f"{len(runs)} scored.json path(s)",
    )


def discover_scored_runs(runs_dir: Path | None = None) -> list[Path]:
    """Return scored.json paths under evals/runs/, newest first."""
    root = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    return sorted(
        root.glob("*/scored.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
