"""Production cost extrapolation from a golden-set eval run (pivot 8).

Turns measured golden-set token usage into an interpretable ladder:

1. Golden-set actual — sync list price on measured tokens
2. Cache adjustment — measured cache rate + $ after cache discount
3. Batch adjustment — 50% → production-equivalent $ on the golden n
4. Scale — × (N_prod / n_golden) → full-dataset estimate
5. Assumptions — N, cache source, discounts, two-pass, reasoning in output

Stacking matches ``src/tokens.py`` / ``src/merger.py``: batch discount on
all tokens, then an extra cache discount on the cached portion of input
(cached input = 25% of sync list). Reasoning tokens are billed inside
output (OpenAI Responses usage).

Legacy runs without ``cached_tokens`` fields do NOT invent a production
cache rate: the cache step is marked ``unavailable`` and later steps that
depend on it are omitted or flagged.
"""

from __future__ import annotations

from typing import Any, Optional

from evals import config as cfg


def _sync_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, float],
    *,
    cached_tokens: int = 0,
    apply_cache: bool = False,
) -> float:
    """Sync list $ for input+output. Optionally apply cache discount on input."""
    if apply_cache:
        cached = min(max(cached_tokens, 0), input_tokens)
        uncached = input_tokens - cached
        cost_in = (
            uncached / 1e6 * pricing["input"]
            + cached / 1e6 * pricing["input"] * cfg.CACHE_DISCOUNT
        )
    else:
        cost_in = input_tokens / 1e6 * pricing["input"]
    cost_out = output_tokens / 1e6 * pricing["output"]
    return cost_in + cost_out


def _records_have_cached_field(records: list[dict[str, Any]]) -> bool:
    """True when EVERY record carries an explicit cached_tokens field.

    Distinguishes 'field present, value 0' (real miss) from legacy banked
    runs that never recorded the field. A mixed resume (some rows with the
    field, some without) must NOT count as measured — otherwise missing
    rows silently look like 0% cache hits and inflate the production $.
    """
    if not records:
        return False
    for rec in records:
        if "cached_tokens" in rec:
            continue
        if "a_cached_tokens" in rec or "b_cached_tokens" in rec:
            continue
        return False
    return True


def _records_partial_cached_field(records: list[dict[str, Any]]) -> bool:
    """True when some but not all records carry a cached_tokens field."""
    if not records:
        return False
    present = sum(
        1 for rec in records
        if "cached_tokens" in rec
        or "a_cached_tokens" in rec
        or "b_cached_tokens" in rec
    )
    return 0 < present < len(records)


def _sum_cached(records: list[dict[str, Any]]) -> int:
    from evals.usage import token_totals

    return sum(token_totals(rec)["cached"] for rec in records)


def extrapolate_production_cost(
    *,
    model: str,
    n_golden: int,
    total_input_tokens: int,
    total_output_tokens: int,
    total_cached_tokens: Optional[int],
    cache_field_present: bool,
    n_prod: int = cfg.N_PROD_DEFAULT,
    n_prod_label: str = "alive_plus_dead",
    architecture: str = "two-pass",
) -> dict[str, Any]:
    """Build the five-step production cost ladder as a structured dict.

    ``total_cached_tokens`` may be 0 with ``cache_field_present=True`` (real
    zero hits). When ``cache_field_present=False``, the cache step is
    unavailable and we do not invent a hit rate.
    """
    pricing = cfg.EVAL_MODEL_PRICING.get(model)
    assumptions = {
        "n_prod": n_prod,
        "n_prod_label": n_prod_label,
        "n_golden": n_golden,
        "model": model,
        "architecture": architecture,
        "batch_discount": cfg.BATCH_DISCOUNT,
        "cache_discount": cfg.CACHE_DISCOUNT,
        "cache_source": (
            "measured_from_run"
            if cache_field_present
            else "unavailable_legacy_run_missing_cached_tokens"
        ),
        "reasoning_billed_inside_output": True,
        "stacking": (
            "batch_discount on all tokens; cache_discount additionally on "
            "cached input portion (cached input = sync_input * batch * cache)"
        ),
        "do_not_use_historical_production_cache_rate": True,
    }

    if pricing is None or n_golden <= 0:
        return {
            "available": False,
            "reason": (
                "unknown_model_pricing" if pricing is None else "empty_golden_n"
            ),
            "assumptions": assumptions,
            "steps": {},
        }

    # Step 1: golden sync list (no cache discount)
    sync_usd = _sync_cost(total_input_tokens, total_output_tokens, pricing)
    step1 = {
        "label": "golden_set_actual_sync_list",
        "n_rows": n_golden,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_usd": sync_usd,
        "mean_usd_per_row": sync_usd / n_golden,
        "note": (
            "Sync list price on measured tokens. Reasoning tokens are inside "
            "output_tokens. Cache discount not applied at this step."
        ),
    }

    steps: dict[str, Any] = {"1_golden_sync": step1}

    # Step 2: cache adjustment
    if not cache_field_present:
        step2: dict[str, Any] = {
            "label": "cache_adjustment",
            "available": False,
            "reason": (
                "predictions records lack cached_tokens; re-run to measure. "
                "Do not invent a production cache rate."
            ),
            "total_cached_tokens": None,
            "cache_hit_rate": None,
            "total_usd_after_cache": None,
        }
        steps["2_cache"] = step2
        steps["3_batch"] = {
            "label": "batch_adjustment",
            "available": False,
            "reason": "blocked_on_missing_cache_measurement",
            "total_usd_after_batch": None,
        }
        steps["4_scale"] = {
            "label": "scale_to_production_n",
            "available": False,
            "reason": "blocked_on_missing_cache_measurement",
            "estimated_production_usd": None,
        }
        return {
            "available": False,
            "reason": "cached_tokens_unavailable",
            "assumptions": assumptions,
            "steps": steps,
            # Still expose step-1 dollars so the artifact is useful.
            "golden_sync_usd": sync_usd,
        }

    cached = int(total_cached_tokens or 0)
    hit_rate = cached / total_input_tokens if total_input_tokens > 0 else 0.0
    after_cache = _sync_cost(
        total_input_tokens,
        total_output_tokens,
        pricing,
        cached_tokens=cached,
        apply_cache=True,
    )
    steps["2_cache"] = {
        "label": "cache_adjustment",
        "available": True,
        "total_cached_tokens": cached,
        "cache_hit_rate": hit_rate,
        "cache_discount": cfg.CACHE_DISCOUNT,
        "total_usd_after_cache": after_cache,
        "mean_usd_per_row": after_cache / n_golden,
        "note": (
            "Measured cached_tokens from this run. Cached input billed at "
            f"{cfg.CACHE_DISCOUNT:.0%} of sync input rate."
        ),
    }

    # Step 3: batch adjustment (50% on the after-cache sync total)
    after_batch = after_cache * cfg.BATCH_DISCOUNT
    steps["3_batch"] = {
        "label": "batch_adjustment",
        "available": True,
        "batch_discount": cfg.BATCH_DISCOUNT,
        "total_usd_after_batch": after_batch,
        "mean_usd_per_row": after_batch / n_golden,
        "note": (
            f"Production Batch API at {cfg.BATCH_DISCOUNT:.0%} of sync. "
            "Applied to the cache-adjusted golden total."
        ),
    }

    # Step 4: scale to N_prod
    scale = n_prod / n_golden
    prod_usd = after_batch * scale
    steps["4_scale"] = {
        "label": "scale_to_production_n",
        "available": True,
        "n_prod": n_prod,
        "n_prod_label": n_prod_label,
        "scale_factor": scale,
        "estimated_production_usd": prod_usd,
        "estimated_usd_per_company": prod_usd / n_prod,
        "note": (
            f"Linear scale × ({n_prod} / {n_golden}). Assumes golden-set "
            "token mix and measured cache rate hold at production volume."
        ),
    }

    return {
        "available": True,
        "assumptions": assumptions,
        "steps": steps,
        "summary": {
            "golden_sync_usd": sync_usd,
            "golden_after_cache_usd": after_cache,
            "golden_after_batch_usd": after_batch,
            "estimated_production_usd": prod_usd,
            "cache_hit_rate": hit_rate,
            "n_prod": n_prod,
        },
    }


def production_cost_from_records(
    records: list[dict[str, Any]],
    model: str,
    *,
    n_prod: int = cfg.N_PROD_DEFAULT,
    n_prod_label: str = "alive_plus_dead",
) -> dict[str, Any]:
    """Aggregate token totals from scored records, then extrapolate."""
    from evals.usage import token_totals

    totals = [token_totals(rec) for rec in records]
    total_in = sum(t["input"] for t in totals)
    total_out = sum(t["output"] for t in totals)

    present = _records_have_cached_field(records)
    cached = _sum_cached(records) if present else None
    partial = _records_partial_cached_field(records)

    if any("a_input_tokens" in r for r in records):
        detected_architecture = "two-pass"
    else:
        detected_architecture = "single-pass"

    result = extrapolate_production_cost(
        model=model,
        n_golden=len(records),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cached_tokens=cached,
        cache_field_present=present,
        n_prod=n_prod,
        n_prod_label=n_prod_label,
        architecture=detected_architecture,
    )
    if partial and not present:
        # Override the generic legacy reason with the mixed-resume case.
        step2 = result.get("steps", {}).get("2_cache")
        if step2 is not None:
            step2["reason"] = (
                "mixed predictions: some rows lack cached_tokens (partial "
                "resume over a legacy run). Re-run the full set so every "
                "row is measured; do not invent a cache rate for gaps."
            )
        result["reason"] = "cached_tokens_partial_coverage"
    return result


def format_cost_ladder(estimate: dict[str, Any]) -> str:
    """Human-readable section for CLI report / embeddable artifact."""
    lines = [
        "=" * 60,
        "PRODUCTION COST EXTRAPOLATION",
        "=" * 60,
    ]
    assumptions = estimate.get("assumptions") or {}
    lines.append(
        f"Model: {assumptions.get('model', '?')}  "
        f"Architecture: {assumptions.get('architecture', '?')}"
    )
    lines.append(
        f"N_prod: {assumptions.get('n_prod', '?')} "
        f"({assumptions.get('n_prod_label', '?')})  "
        f"n_golden: {assumptions.get('n_golden', '?')}"
    )
    lines.append(f"Cache source: {assumptions.get('cache_source', '?')}")
    lines.append(
        f"Discounts: batch={assumptions.get('batch_discount')}, "
        f"cache={assumptions.get('cache_discount')} (stacking as production)"
    )
    lines.append("Reasoning tokens: billed inside output")
    lines.append("")

    steps = estimate.get("steps") or {}
    s1 = steps.get("1_golden_sync")
    if s1:
        lines.append(
            f"1. Golden sync list:     ${s1['total_usd']:,.4f}  "
            f"({s1['n_rows']} rows, "
            f"{s1['total_input_tokens']:,} in / {s1['total_output_tokens']:,} out)"
        )
    s2 = steps.get("2_cache")
    if s2:
        if s2.get("available"):
            lines.append(
                f"2. After cache ({s2['cache_hit_rate']:.1%} hit): "
                f"${s2['total_usd_after_cache']:,.4f}  "
                f"(cached_tokens={s2['total_cached_tokens']:,})"
            )
        else:
            lines.append(f"2. Cache adjustment:    UNAVAILABLE — {s2.get('reason')}")
    s3 = steps.get("3_batch")
    if s3:
        if s3.get("available"):
            lines.append(
                f"3. After batch (×{s3['batch_discount']}): "
                f"${s3['total_usd_after_batch']:,.4f}"
            )
        else:
            lines.append(f"3. Batch adjustment:    UNAVAILABLE — {s3.get('reason')}")
    s4 = steps.get("4_scale")
    if s4:
        if s4.get("available"):
            lines.append(
                f"4. Scaled to N={s4['n_prod']:,}: "
                f"${s4['estimated_production_usd']:,.2f}  "
                f"(${s4['estimated_usd_per_company']:.4f}/company)"
            )
        else:
            lines.append(f"4. Scale to production: UNAVAILABLE — {s4.get('reason')}")

    if not estimate.get("available"):
        lines.append("")
        lines.append(f"Full ladder: UNAVAILABLE ({estimate.get('reason')})")

    lines.append("=" * 60)
    return "\n".join(lines)
