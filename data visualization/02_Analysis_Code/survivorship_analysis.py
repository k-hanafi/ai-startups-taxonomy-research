#!/usr/bin/env python3
"""Compute the survivorship-insights metrics dict consumed by the dashboard.

Pure compute, mirrors the summarize -> build split used by
summarize_death_coverage.py + build_survivorship_dashboard.py: this module owns
the data work and returns one JSON-able dict; the builder only renders it.

It assembles a single analysis frame from the merged survivor-vs-dead dataset
(survivorship_corrected.csv), joins the static metadata it needs (funding,
category, founded date, liveness) plus death timing, defines the survivor / dead
cohorts, and computes the descriptive sections and two logistic-regression
models. If survivorship_corrected.csv is absent it builds a structurally
identical PREVIEW frame from production_classifications.csv tagged by the frozen
dead work list, so the dashboard renders before the classify-dead run lands.

Run directly to print cohort sizes + the metrics JSON (optionally --out FILE).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Verdict fields (mirrors merge_survivorship.VERDICT_COLS); CompanyID is the join key.
VERDICT_COLS = ["ai_native", "subclass", "rad_score", "cohort",
                "conf_classification", "conf_rad"]

SUBCLASS_ORDER = ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "0A", "0B", "0C"]
RAD_ORDER = ["RAD-H", "RAD-M", "RAD-L", "RAD-NA"]
AI_RAD_ORDER = ["RAD-H", "RAD-M", "RAD-L"]

# Defensibility grouping for the RAD-validation story. 1C (thin wrapper) and 1G
# (generative content) are the most commoditizable; 1A/1B/1E carry structural or
# domain moats. 1D/1F sit in between.
SUBCLASS_GROUP = {
    "1A": "Defensible AI", "1B": "Defensible AI", "1E": "Defensible AI",
    "1C": "Commoditizable AI", "1G": "Commoditizable AI",
    "1D": "Other AI", "1F": "Other AI",
    "0A": "Not AI", "0B": "Not AI", "0C": "Not AI",
}
SUBCLASS_GROUP_ORDER = ["Commoditizable AI", "Other AI", "Defensible AI", "Not AI"]

FUNDING_BUCKETS = [
    ("unknown", -np.inf, 0),
    ("<$1M", 1, 1_000_000),
    ("$1-10M", 1_000_001, 10_000_000),
    ("$10-100M", 10_000_001, 100_000_000),
    ("$100M+", 100_000_001, np.inf),
]
FUNDING_BUCKET_ORDER = [b[0] for b in FUNDING_BUCKETS]

# Confident, well-known release dates only. Used as reference lines on the
# deaths-over-time chart; no speculative dates.
MODEL_RELEASES = [
    ("2022-11", "ChatGPT"),
    ("2023-03", "GPT-4"),
    ("2023-11", "GPT-4 Turbo"),
    ("2024-05", "GPT-4o"),
]
GPT4_LAUNCH = "2023-03-14"
TEMPORAL_START = pd.Timestamp("2022-01-01")
TOP_CATEGORY_GROUPS = 8


@dataclass
class Paths:
    """Overridable input paths so the module can run against a fixture."""
    corrected: Path = PROJECT_ROOT / "outputs" / "wayback_dead" / "survivorship_corrected.csv"
    production: Path = PROJECT_ROOT / "outputs" / "production_csvs" / "production_classifications.csv"
    master: Path = PROJECT_ROOT / "data" / "master_csv.csv"
    classifier_input: Path = PROJECT_ROOT / "outputs" / "tavilycrawl" / "processed" / "classifier_input.csv"
    targets_dead: Path = PROJECT_ROOT / "wayback_machine" / "data" / "scrape_targets_dead.csv"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")


def load_frame(paths: Paths) -> tuple[pd.DataFrame, dict]:
    """Build the analysis frame and a meta dict (incl. the preview flag)."""
    if paths.master.exists():
        master = _read(paths.master)
    else:
        raise SystemExit(f"master_csv not found: {paths.master}")

    targets = _read(paths.targets_dead) if paths.targets_dead.exists() else pd.DataFrame()
    dead_ids = set(targets["org_uuid"]) if "org_uuid" in targets else set()

    preview = not paths.corrected.exists()
    if preview:
        if not paths.production.exists():
            raise SystemExit(
                "Neither survivorship_corrected.csv nor production_classifications.csv "
                f"found ({paths.corrected} / {paths.production})."
            )
        base = _read(paths.production)
        base["evidence_source"] = np.where(base["CompanyID"].isin(dead_ids), "wayback_dead", "live")
    else:
        base = _read(paths.corrected)
        if "evidence_source" not in base:
            base["evidence_source"] = np.where(base["CompanyID"].isin(dead_ids), "wayback_dead", "live")

    # Static metadata join (funding, category, founded, liveness).
    meta_cols = ["org_uuid", "founded_date", "employee_count", "total_funding_usd",
                 "category_list", "category_groups_list", "website_alive"]
    master_slim = master[[c for c in meta_cols if c in master.columns]].copy()
    df = base.merge(master_slim, left_on="CompanyID", right_on="org_uuid", how="left")

    # Live evidence presence: a true survivor has non-empty website_evidence.
    if paths.classifier_input.exists():
        ci = _read(paths.classifier_input)[["org_uuid", "website_evidence"]]
        df = df.merge(ci, left_on="CompanyID", right_on="org_uuid", how="left", suffixes=("", "_ci"))
        df["has_live_evidence"] = df["website_evidence"].fillna("").str.strip().ne("")
    else:
        df["has_live_evidence"] = df["evidence_source"].eq("live")

    # Death timing + thin-history provenance (always from the frozen work list).
    if not targets.empty:
        tcols = ["org_uuid", "death_ts", "days_before_death", "closest_ts", "thin_history"]
        tgt = targets[[c for c in tcols if c in targets.columns]].copy()
        df = df.merge(tgt, left_on="CompanyID", right_on="org_uuid", how="left", suffixes=("", "_tgt"))
    for c in ["death_ts", "days_before_death", "closest_ts", "thin_history"]:
        if c not in df.columns:
            df[c] = ""

    df = _derive(df)
    meta = {
        "preview": preview,
        "gpt4_launch": GPT4_LAUNCH,
        "n_total": int(len(df)),
        "n_survivor": int(df["is_survivor"].sum()),
        "n_dead_full": int(df["is_dead_full"].sum()),
        "n_dead_strict": int(df["is_dead_strict"].sum()),
        "n_excluded": int(df["is_excluded"].sum()),
    }
    return df, meta


def _funding_bucket(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    for label, lo, hi in FUNDING_BUCKETS:
        if lo <= v <= hi:
            return label
    return "unknown"


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_native_int"] = pd.to_numeric(df["ai_native"], errors="coerce").fillna(0).astype(int)
    df["funding_usd"] = pd.to_numeric(df["total_funding_usd"], errors="coerce")
    df["log_funding"] = np.log10(df["funding_usd"].fillna(0).clip(lower=0) + 1.0)
    df["funding_bucket"] = df["funding_usd"].apply(_funding_bucket)
    df["founded_year"] = pd.to_numeric(df["founded_date"].str[:4], errors="coerce")
    df["founding_era"] = df["cohort"].where(df["cohort"].isin(["PRE-GENAI", "GENAI-ERA"]), "unknown")
    df["subclass_group"] = df["subclass"].map(SUBCLASS_GROUP).fillna("Not AI")
    df["category_group"] = (
        df["category_groups_list"].fillna("").str.split(",").str[0].str.strip().replace("", "Unknown")
    )
    thin = df["thin_history"].astype(str).str.lower().isin(["true", "1"])
    website_dead = df["website_alive"].astype(str).str.lower().eq("false")

    df["is_dead_full"] = df["evidence_source"].isin(["wayback_dead", "dead_metadata"])
    df["is_survivor"] = df["evidence_source"].eq("live") & df["has_live_evidence"]
    df["is_excluded"] = df["evidence_source"].eq("live") & ~df["has_live_evidence"]
    df["is_dead_strict"] = df["is_dead_full"] & website_dead & ~thin
    return df


# --------------------------------------------------------------------------- #
# Section computations (every helper takes the analysis frame explicitly)
# --------------------------------------------------------------------------- #

def _dist(f: pd.DataFrame, mask: pd.Series, col: str, order: list[str]) -> dict:
    counts = f[mask][col].value_counts().to_dict()
    n = int(mask.sum())
    return {
        "n": n,
        "counts": {k: int(counts.get(k, 0)) for k in order},
        "share": {k: round(counts.get(k, 0) / n * 100, 1) if n else 0.0 for k in order},
    }


def _ai_rate(f: pd.DataFrame, mask: pd.Series) -> dict:
    sub = f[mask]
    n = int(len(sub))
    ai = int((sub["ai_native_int"] == 1).sum())
    return {"n": n, "ai_native": ai, "rate": round(ai / n * 100, 1) if n else 0.0}


def _mortality_by(f: pd.DataFrame, col: str, order: list[str]) -> list[dict]:
    """Per-level mortality = dead / (dead + survivor) within the outcome frame."""
    out = []
    for level in order:
        sel = f[col] == level
        dead = int((sel & f["is_dead_full"]).sum())
        surv = int((sel & f["is_survivor"]).sum())
        tot = dead + surv
        if tot:
            out.append({"label": level, "dead": dead, "survivor": surv, "n": tot,
                        "mortality": round(dead / tot * 100, 1)})
    return out


def _correction(f: pd.DataFrame) -> dict:
    """Survivor-only (biased) vs survivor+dead (corrected) global distributions."""
    biased = f["is_survivor"]
    corrected = f["is_survivor"] | f["is_dead_full"]
    return {
        "biased": {
            "ai_native": _ai_rate(f, biased),
            "subclass": _dist(f, biased, "subclass", SUBCLASS_ORDER),
            "rad": _dist(f, biased, "rad_score", RAD_ORDER),
        },
        "corrected": {
            "ai_native": _ai_rate(f, corrected),
            "subclass": _dist(f, corrected, "subclass", SUBCLASS_ORDER),
            "rad": _dist(f, corrected, "rad_score", RAD_ORDER),
        },
    }


def _ai_vs_survival(f: pd.DataFrame) -> dict:
    surv = _dist(f, f["is_survivor"], "subclass", SUBCLASS_ORDER)
    dead = _dist(f, f["is_dead_full"], "subclass", SUBCLASS_ORDER)
    lift = []
    for s in SUBCLASS_ORDER:
        sv, dv = surv["share"][s], dead["share"][s]
        lift.append({"subclass": s, "survivor": sv, "dead": dv,
                     "lift": round(dv / sv, 2) if sv else None})
    return {
        "ai_rate": {
            "survivor": _ai_rate(f, f["is_survivor"]),
            "dead_full": _ai_rate(f, f["is_dead_full"]),
            "dead_strict": _ai_rate(f, f["is_dead_strict"]),
        },
        "subclass_survivor": surv,
        "subclass_dead": dead,
        "subclass_lift": lift,
    }


def _rad_survival(f: pd.DataFrame) -> dict:
    return {
        "by_rad": _mortality_by(f, "rad_score", AI_RAD_ORDER),
        "by_subclass": _mortality_by(f, "subclass", SUBCLASS_ORDER),
        "by_subclass_group": _mortality_by(f, "subclass_group", SUBCLASS_GROUP_ORDER),
    }


def _dependency_trap(f: pd.DataFrame) -> dict:
    """Mortality grid: funding bucket (rows) x RAD level (cols)."""
    cells = []
    for fb in FUNDING_BUCKET_ORDER:
        row = []
        for rad in AI_RAD_ORDER:
            sel = (f["funding_bucket"] == fb) & (f["rad_score"] == rad)
            dead = int((sel & f["is_dead_full"]).sum())
            surv = int((sel & f["is_survivor"]).sum())
            tot = dead + surv
            row.append({"dead": dead, "survivor": surv, "n": tot,
                        "mortality": round(dead / tot * 100, 1) if tot else None})
        cells.append(row)
    return {"funding_order": FUNDING_BUCKET_ORDER, "rad_order": AI_RAD_ORDER, "cells": cells}


def _vertical(f: pd.DataFrame, top_n: int = TOP_CATEGORY_GROUPS) -> list[dict]:
    """AI-native mortality by primary category group, busiest groups first."""
    ai = f[f["ai_native_int"] == 1]
    rows = []
    for grp, sub in ai.groupby("category_group"):
        dead = int(sub["is_dead_full"].sum())
        surv = int(sub["is_survivor"].sum())
        tot = dead + surv
        if tot >= 1 and grp not in ("", "Unknown"):
            rows.append({"group": grp, "dead": dead, "survivor": surv, "n": tot,
                         "mortality": round(dead / tot * 100, 1)})
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows[:top_n]


def _temporal(f: pd.DataFrame) -> dict:
    dead = f[f["is_dead_full"]].copy()
    ts = pd.to_datetime(dead["death_ts"], format="%Y%m%d%H%M%S", errors="coerce")
    dead = dead.assign(_death=ts)
    dead = dead[dead["_death"] >= TEMPORAL_START]
    months_idx = dead["_death"].dt.to_period("M").astype(str)
    total = months_idx.value_counts().to_dict()
    commod = months_idx[dead["subclass_group"] == "Commoditizable AI"].value_counts().to_dict()
    defens = months_idx[dead["subclass_group"] == "Defensible AI"].value_counts().to_dict()
    months = sorted(total)
    return {
        "months": months,
        "total": [int(total.get(m, 0)) for m in months],
        "commoditizable": [int(commod.get(m, 0)) for m in months],
        "defensible": [int(defens.get(m, 0)) for m in months],
        "releases": [{"month": m, "label": lbl} for m, lbl in MODEL_RELEASES],
    }


def _confidence(f: pd.DataFrame) -> dict:
    def stats(mask: pd.Series) -> dict:
        v = pd.to_numeric(f[mask]["conf_classification"], errors="coerce").dropna()
        dist = {int(k): int(c) for k, c in v.value_counts().sort_index().items()}
        return {"n": int(len(v)), "mean": round(float(v.mean()), 2) if len(v) else None,
                "median": float(v.median()) if len(v) else None, "dist": dist}
    return {"survivor": stats(f["is_survivor"]), "dead_full": stats(f["is_dead_full"])}


def _flips(f: pd.DataFrame, paths: Paths, preview: bool) -> dict:
    """Metadata-only (production) vs recovered-evidence (corrected) verdict flips
    on the dead cohort. Only meaningful once the real corrected CSV exists."""
    if preview or not paths.production.exists():
        return {"available": False, "reason": "preview" if preview else "no production csv"}
    before = _read(paths.production).set_index("CompanyID")
    dead = f[f["is_dead_full"]]
    n = int(len(dead))
    out = {"available": True, "n": n}
    for col in ("ai_native", "subclass", "rad_score"):
        if col not in before.columns:
            continue
        b = before.reindex(dead["CompanyID"])[col].astype(str).values
        a = dead[col].astype(str).values
        changed = int((b != a).sum())
        out[col] = {"changed": changed, "pct": round(changed / n * 100, 1) if n else 0.0}
    return out


def _regression(f: pd.DataFrame) -> dict:
    """Two logistic models. died=1 for the dead cohort, 0 for survivors.

    Model 1 (full outcome frame): does AI-nativeness predict death, net of
    funding, founding era, and vertical. Model 2 (AI-native only): does RAD
    dependency predict death among AI firms (the RAD-validation test).
    """
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:  # statsmodels missing
        unavailable = {"available": False, "error": f"statsmodels unavailable: {exc}"}
        return {"model1": unavailable, "model2": unavailable}

    base = f[f["is_dead_full"] | f["is_survivor"]].copy()
    base["died"] = base["is_dead_full"].astype(int)
    base = base[base["founding_era"] != "unknown"]
    top_cats = base["category_group"].value_counts().head(TOP_CATEGORY_GROUPS).index
    base["cat_grp"] = base["category_group"].where(base["category_group"].isin(top_cats), "Other")

    def fit(frame: pd.DataFrame, formula: str) -> dict:
        frame = frame.dropna(subset=["log_funding"])
        if frame["died"].nunique() < 2 or len(frame) < 30:
            return {"available": False, "error": "insufficient variation"}
        try:
            res = smf.logit(formula, data=frame).fit(disp=0, maxiter=100)
        except Exception as exc:
            return {"available": False, "error": str(exc)}
        ci = res.conf_int()
        terms = []
        for name in res.params.index:
            if name == "Intercept":
                continue
            terms.append({
                "term": name,
                "odds_ratio": round(float(np.exp(res.params[name])), 3),
                "ci_low": round(float(np.exp(ci.loc[name, 0])), 3),
                "ci_high": round(float(np.exp(ci.loc[name, 1])), 3),
                "pvalue": float(res.pvalues[name]),
            })
        return {"available": True, "n": int(res.nobs),
                "pseudo_r2": round(float(res.prsquared), 4), "terms": terms}

    model1 = fit(base, "died ~ ai_native_int + log_funding + C(founding_era) + C(cat_grp)")
    ai = base[base["ai_native_int"] == 1].copy()
    ai = ai[ai["rad_score"].isin(AI_RAD_ORDER)]
    model2 = fit(
        ai,
        "died ~ C(rad_score, Treatment('RAD-L')) + C(subclass_group) + log_funding + C(founding_era)",
    )
    return {"model1": model1, "model2": model2}


def compute_metrics(df: pd.DataFrame, meta: dict, paths: Paths) -> dict:
    return {
        "meta": meta,
        "correction": _correction(df),
        "ai_vs_survival": _ai_vs_survival(df),
        "rad_survival": _rad_survival(df),
        "dependency_trap": _dependency_trap(df),
        "vertical": _vertical(df),
        "temporal": _temporal(df),
        "confidence": _confidence(df),
        "flips": _flips(df, paths, meta["preview"]),
        "regression": _regression(df),
    }


def analyze(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    df, meta = load_frame(paths)
    return compute_metrics(df, meta, paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, default=None)
    parser.add_argument("--production", type=Path, default=None)
    parser.add_argument("--master", type=Path, default=None)
    parser.add_argument("--classifier-input", type=Path, default=None)
    parser.add_argument("--targets-dead", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Also write the metrics JSON here.")
    args = parser.parse_args()

    p = Paths()
    for attr, val in [("corrected", args.corrected), ("production", args.production),
                      ("master", args.master), ("classifier_input", args.classifier_input),
                      ("targets_dead", args.targets_dead)]:
        if val is not None:
            setattr(p, attr, val)

    metrics = analyze(p)
    m = metrics["meta"]
    mode = "PREVIEW (metadata-only dead verdicts)" if m["preview"] else "evidence-based"
    print(f"[{mode}] total={m['n_total']:,}  survivor={m['n_survivor']:,}  "
          f"dead_full={m['n_dead_full']:,}  dead_strict={m['n_dead_strict']:,}  "
          f"excluded={m['n_excluded']:,}", file=sys.stderr)
    blob = json.dumps(metrics, indent=2, ensure_ascii=False, default=str)
    if args.out:
        args.out.write_text(blob + "\n", encoding="utf-8")
    print(blob)


if __name__ == "__main__":
    main()
