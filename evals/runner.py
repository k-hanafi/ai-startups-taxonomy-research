"""Stage 3: sync Responses API runner for the golden set.

One run = one (model, reasoning-effort, repeat) config classifying all 100
golden companies. The request is built from the SAME production code the live
pipeline uses (system prompt, JSON schema, row formatter), so a benchmark
result is a faithful claim about the real classifier. The only additions are
the experimental parameters production does not send yet: logprob capture,
a reasoning-effort knob, and temperature.

Design choices:
- Fidelity over convenience. The request body comes verbatim from
  src.builder.build_request_body; we override only max_output_tokens (reasoning
  needs headroom) and append the experimental params. Every run snapshots the
  SHA-256 of the prompt, schema, and formatter source, so any drift from the
  production artifacts is detectable after the fact.
- Resumable. Each finished row is appended to predictions.jsonl keyed by
  custom_id; a re-run skips rows already present, so a mid-run crash or rate
  limit never re-bills completed work (same checkpoint idea as classify.py).
- Results are matched to inputs by custom_id, never by order.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.builder import (
    _openai_strict_schema,
    build_request_body,
    load_system_prompt,
)
from src.config import OPENAI_API_KEY, PROMPT_CACHE_KEY
from src.formatter import build_custom_id, format_user_message

from evals import config as cfg
from evals.jsonl_io import append_jsonl, iter_jsonl
from evals.paths import (
    CLASSIFIER_INPUT_CSV,
    GOLDEN_SET_CSV,
    run_config_path,
    run_dir,
    run_predictions_path,
    run_raw_dir,
)
from evals.usage import cached_tokens_from_usage

logger = logging.getLogger(__name__)

# Retriable server-side / transport failures. BadRequestError (a malformed
# request) is deliberately not listed: it would fail identically every retry.
_RETRIABLE = retry(
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
    ),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def identity_hashes() -> dict[str, str]:
    """SHA-256 of the three production artifacts a run must stay faithful to."""
    schema = _openai_strict_schema()
    return {
        "prompt_sha256": _sha256(load_system_prompt()),
        "schema_sha256": _sha256(json.dumps(schema, sort_keys=True)),
        "formatter_sha256": _sha256(inspect.getsource(format_user_message)),
    }


def make_run_id(model: str, effort: str, repeat: int) -> str:
    date = datetime.date.today().isoformat()
    return f"{date}_{model}_{effort}_r{repeat}"


def load_golden_rows() -> list[dict[str, Any]]:
    """Golden companies as raw classifier-input rows, in golden-set order.

    The committed golden_set.csv carries no evidence text, so the actual
    request inputs are pulled from classifier_input.csv (filtered to the 100
    golden org_uuids). Reading that 249 MB file in chunks caps memory.
    """
    golden = pd.read_csv(GOLDEN_SET_CSV, dtype=str, keep_default_na=False)
    order = list(golden["org_uuid"])
    wanted = set(order)

    matched: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(CLASSIFIER_INPUT_CSV, dtype=str, keep_default_na=False,
                             chunksize=5000):
        hit = chunk[chunk["org_uuid"].isin(wanted)]
        for row in hit.to_dict(orient="records"):
            matched[row["org_uuid"]] = row
        if len(matched) == len(wanted):
            break

    missing = wanted - matched.keys()
    if missing:
        raise AssertionError(
            f"{len(missing)} golden org_uuids not found in classifier_input: "
            f"{sorted(missing)[:5]}"
        )
    return [matched[uuid] for uuid in order]


def build_request_kwargs(row: dict[str, Any], system_prompt: str, schema: dict,
                         model: str, effort: str) -> dict[str, Any]:
    """Production request body + experimental params, as responses.create kwargs.

    The body (instructions, input, schema, cache key, store) is taken verbatim
    from build_request_body so it is byte-identical to production. Deltas:
    max_output_tokens raised for reasoning headroom, plus reasoning/logprob/
    temperature knobs.
    """
    cid = build_custom_id(row["org_uuid"])
    user_message = format_user_message(row)
    body = build_request_body(user_message, cid, system_prompt, schema, model)["body"]

    body["max_output_tokens"] = cfg.MAX_OUTPUT_TOKENS
    body["reasoning"] = {"effort": effort}
    # Logprobs and reasoning are mutually exclusive on these models: capture
    # token-level confidence only when reasoning is off (see config.REASONING_OFF).
    if effort == cfg.REASONING_OFF:
        body["top_logprobs"] = cfg.TOP_LOGPROBS
        body["include"] = list(cfg.LOGPROB_INCLUDE)
    # Reasoning models reject temperature (see config.SEND_TEMPERATURE).
    if cfg.SEND_TEMPERATURE:
        body["temperature"] = cfg.DECODING_TEMPERATURE
    return body


def _completed_custom_ids(predictions_path: Path) -> set[str]:
    """Resume set: custom_ids whose latest status is completed (or legacy).

    Only rows with ``status`` missing (older runs) or ``status == "completed"``
    count as done. Tolerates a truncated final JSONL line; fails on interior
    corruption.
    """
    if not predictions_path.exists():
        return set()
    done: set[str] = set()
    for rec in iter_jsonl(predictions_path, tolerate_truncated_final=True):
        status = rec.get("status")
        if status is None or status == "completed":
            cid = rec.get("custom_id")
            if cid:
                done.add(cid)
    return done


def _prediction_record(custom_id: str, org_uuid: str, model: str, effort: str,
                       resp: Any, latency_s: float | None = None) -> dict[str, Any]:
    """Label-only row for predictions.jsonl (no scraped evidence text)."""
    parsed: dict[str, Any] = {}
    text = getattr(resp, "output_text", "") or ""
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}

    usage = getattr(resp, "usage", None)
    reasoning_tokens = None
    if usage is not None and getattr(usage, "output_tokens_details", None) is not None:
        reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", None)

    return {
        "custom_id": custom_id,
        "org_uuid": org_uuid,
        "model": model,
        "effort": effort,
        "status": getattr(resp, "status", None),
        "ai_native": parsed.get("ai_native"),
        "subclass": parsed.get("subclass"),
        "rad_score": parsed.get("rad_score"),
        "cohort": parsed.get("cohort"),
        "conf_classification": parsed.get("conf_classification"),
        "conf_rad": parsed.get("conf_rad"),
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "reasoning_tokens": reasoning_tokens,
        # 0 when usage/details absent (API omits the field on a miss); never
        # invent a production cache rate for missing data.
        "cached_tokens": cached_tokens_from_usage(usage),
        "latency_s": latency_s,
    }


def run(model: str = cfg.EVAL_MODELS[0],
        effort: str = cfg.SCREEN_REASONING_EFFORT,
        repeat: int = 1,
        limit: int | None = None,
        dry_run: bool = False,
        run_id: str | None = None) -> str:
    """Classify the golden set with one model config. Returns the run_id.

    limit caps rows (cheap smoke tests). dry_run prints the plan and cost
    estimate without any API call. run_id overrides the default date-derived id
    so an interrupted run can be resumed explicitly (e.g. across midnight).
    """
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be a positive row cap, got {limit}")

    rows = load_golden_rows()
    if limit is not None:
        rows = rows[:limit]

    run_id = run_id or make_run_id(model, effort, repeat)
    system_prompt = load_system_prompt()
    schema = _openai_strict_schema()

    if dry_run:
        _print_dry_run(rows, system_prompt, schema, model, effort, run_id)
        return run_id

    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_raw_dir(run_id).mkdir(parents=True, exist_ok=True)
    predictions_path = run_predictions_path(run_id)

    _ensure_config(run_id, model, effort, repeat, len(rows))

    done = _completed_custom_ids(predictions_path)
    if done:
        logger.info("Resuming %s: %d rows already complete", run_id, len(done))

    client = OpenAI(api_key=OPENAI_API_KEY)
    todo = [r for r in rows if build_custom_id(r["org_uuid"]) not in done]
    logger.info("Run %s: %d rows to classify (%s, effort=%s)",
                run_id, len(todo), model, effort)

    for i, row in enumerate(todo, start=1):
        cid = build_custom_id(row["org_uuid"])
        kwargs = build_request_kwargs(row, system_prompt, schema, model, effort)
        # Wall-clock latency around the API call; retry backoff is included,
        # so this is the honest per-row cost a production caller would feel.
        started = time.monotonic()
        resp = _create(client, kwargs)
        latency_s = round(time.monotonic() - started, 3)

        (run_raw_dir(run_id) / f"{cid}.json").write_text(
            json.dumps(resp.model_dump(), ensure_ascii=False), encoding="utf-8"
        )
        record = _prediction_record(cid, row["org_uuid"], model, effort, resp,
                                    latency_s)
        append_jsonl(predictions_path, record)

        logger.info("  [%d/%d] %s -> %s (%s)",
                    i, len(todo), row.get("name", "")[:28],
                    record.get("subclass"), record.get("status"))

    logger.info("Run %s complete: %s", run_id, predictions_path)
    return run_id


@_RETRIABLE
def _create(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    return client.responses.create(**kwargs)


# A resume must reproduce the exact request identity, or the accumulated
# predictions would mix incompatible configs and corrupt the benchmark.
_RESUME_INVARIANTS = (
    "model", "reasoning_effort", "repeat", "n_rows",
    "max_output_tokens", "temperature", "top_logprobs",
    "prompt_sha256", "schema_sha256", "formatter_sha256",
)


def _build_config(run_id: str, model: str, effort: str, repeat: int,
                  n_rows: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "kind": "single_pass",
        "model": model,
        "reasoning_effort": effort,
        "repeat": repeat,
        "n_rows": n_rows,
        "temperature": cfg.DECODING_TEMPERATURE if cfg.SEND_TEMPERATURE else None,
        "logprobs_captured": effort == cfg.REASONING_OFF,
        "top_logprobs": cfg.TOP_LOGPROBS if effort == cfg.REASONING_OFF else None,
        "max_output_tokens": cfg.MAX_OUTPUT_TOKENS,
        "logprob_include": list(cfg.LOGPROB_INCLUDE) if effort == cfg.REASONING_OFF else [],
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "git_commit": _git_commit(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **identity_hashes(),
    }


def _ensure_config(run_id: str, model: str, effort: str, repeat: int,
                   n_rows: int) -> None:
    """Write config.json on first run; on resume, verify it still matches.

    Refuses to append to a run whose prompt/schema/formatter/model/effort has
    changed since it started — a fresh run_id is required instead.
    """
    config = _build_config(run_id, model, effort, repeat, n_rows)
    path = run_config_path(run_id)
    if not path.exists():
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return

    prior = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [k for k in _RESUME_INVARIANTS if prior.get(k) != config[k]]
    if mismatched:
        raise SystemExit(
            f"Cannot resume run {run_id}: {mismatched} changed since it started. "
            "Start a fresh run with a new --run-id."
        )


def _print_dry_run(rows: Iterable[dict[str, Any]], system_prompt: str,
                   schema: dict, model: str, effort: str, run_id: str) -> None:
    rows = list(rows)
    pricing = cfg.require_model_pricing(model)
    # Rough input-token proxy: 1 token ~= 4 chars of prompt + formatted row.
    prompt_chars = len(system_prompt) + len(json.dumps(schema))
    row_chars = sum(len(format_user_message(r)) for r in rows)
    est_input_tokens = (prompt_chars * len(rows) + row_chars) / 4
    in_cost = est_input_tokens / 1e6 * pricing["input"]
    logger.info("DRY RUN %s", run_id)
    logger.info("  model=%s effort=%s rows=%d", model, effort, len(rows))
    logger.info("  est input tokens ~%d, est input cost ~$%.4f "
                "(output/reasoning excluded)", int(est_input_tokens), in_cost)
    logger.info("  identity hashes: %s", identity_hashes())
