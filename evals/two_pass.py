"""Stage 5: the two-pass split-reasoning classifier over the golden set.

Pass A (reasoning off, logprobs on) answers the binary "AI-native or not?"
with a one-field JSON, putting the whole confidence signal on a handful of
tokens. Pass B (reasoning high, no logprobs) assigns the fine-grained
subclass — hard-constrained by the response schema to the family Pass A
chose — plus RAD when the family is AI-native. Cohort never touches an LLM:
it is a pure function of founded_date.

Empirical basis (see the two-pass plan): logprobs and reasoning are mutually
exclusive per request, binary accuracy survives without reasoning (93% vs
Fable either way), and 10-way subclass accuracy does not (41% vs 66%).

Reuses the Stage 3 runner's reliability harness: tenacity retries, per-row
resume keyed by custom_id, config snapshot with prompt hashes, raw responses
banked per pass. A row is complete only when BOTH passes succeeded; a crash
between passes re-runs Pass A on resume (a deliberate simplicity trade —
Pass A costs a fraction of a cent).
"""

from __future__ import annotations

import datetime
import inspect
import json
import logging
from typing import Any, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from src.builder import _add_additional_properties_false
from src.config import OPENAI_API_KEY
from src.formatter import _normalize_founded_date, format_user_message

from evals import config as cfg
from evals.paths import (
    BINARY_GATE_PROMPT,
    FAMILY_BLOCK_AI,
    FAMILY_BLOCK_NOT,
    SUBCLASS_RAD_PROMPT,
    run_config_path,
    run_dir,
    run_predictions_path,
    run_raw_dir,
)
from evals.runner import (
    _RETRIABLE,
    _completed_custom_ids,
    _git_commit,
    _sha256,
    load_golden_rows,
)

logger = logging.getLogger(__name__)

FAMILY_BLOCK_PLACEHOLDER = "{family_block}"


# ---------------------------------------------------------------------------
# Output schemas (one per request shape)
# ---------------------------------------------------------------------------

class BinaryResult(BaseModel):
    """Pass A output: the binary verdict, nothing else."""

    ai_native: Literal[0, 1]


class SubclassResultAI(BaseModel):
    """Pass B output when the gate said AI-native (family 1A-1G, RAD applies)."""

    subclass: Literal["1A", "1B", "1C", "1D", "1E", "1F", "1G"]
    rad_score: Literal["RAD-H", "RAD-M", "RAD-L"]
    conf_classification: int = Field(ge=1, le=5)
    conf_rad: int = Field(ge=1, le=5)
    reasons_3_points: str
    sources_used: str
    verification_critique: str
    boundary_disagreement: bool


class SubclassResultNot(BaseModel):
    """Pass B output when the gate said not-AI-native (family 0A-0C, no RAD)."""

    subclass: Literal["0A", "0B", "0C"]
    conf_classification: int = Field(ge=1, le=5)
    reasons_3_points: str
    sources_used: str
    verification_critique: str
    boundary_disagreement: bool


def strict_schema(model_cls: type[BaseModel]) -> dict:
    """OpenAI strict-mode JSON schema for a Pydantic model."""
    schema = model_cls.model_json_schema()
    _add_additional_properties_false(schema)
    return schema


def _text_format(model_cls: type[BaseModel]) -> dict:
    return {
        "format": {
            "type": "json_schema",
            "name": model_cls.__name__,
            "strict": True,
            "schema": strict_schema(model_cls),
        }
    }


# ---------------------------------------------------------------------------
# Prompts and messages
# ---------------------------------------------------------------------------

def load_pass_a_prompt() -> str:
    return BINARY_GATE_PROMPT.read_text(encoding="utf-8").strip()


def load_pass_b_prompt(family: int) -> str:
    """Pass B instructions with the family block substituted in."""
    template = SUBCLASS_RAD_PROMPT.read_text(encoding="utf-8").strip()
    block_path = FAMILY_BLOCK_AI if family == 1 else FAMILY_BLOCK_NOT
    block = block_path.read_text(encoding="utf-8").strip()
    if FAMILY_BLOCK_PLACEHOLDER not in template:
        raise AssertionError(
            f"{SUBCLASS_RAD_PROMPT.name} is missing the {FAMILY_BLOCK_PLACEHOLDER} placeholder"
        )
    return template.replace(FAMILY_BLOCK_PLACEHOLDER, block)


def compute_cohort(founded_date: Any) -> str:
    """PRE-GENAI / GENAI-ERA from founded_date; deterministic, no LLM.

    Unknown or year-only dates resolve conservatively: an unknown date maps to
    PRE-GENAI (most Crunchbase rows predate 2023), and a bare year uses
    January, so year-2023 rows without a month count as PRE-GENAI (Jan < Mar).
    """
    normalized = _normalize_founded_date(founded_date)  # YYYY-MM | YYYY | Unknown
    if normalized == "Unknown":
        return "PRE-GENAI"
    try:
        year = int(normalized[:4])
        month = int(normalized[5:7]) if len(normalized) >= 7 else 1
    except ValueError:
        return "PRE-GENAI"
    return "GENAI-ERA" if (year, month) >= cfg.COHORT_BOUNDARY else "PRE-GENAI"


def pass_a_message(row: dict[str, Any]) -> str:
    """Pass A user message: the production format minus Website Pages Used.

    The gate prompt dropped that field as redundant with Website Evidence, so
    the message must not carry it either.
    """
    trimmed = dict(row)
    trimmed["website_pages_used"] = ""
    return format_user_message(trimmed)


def pass_b_message(row: dict[str, Any], verdict: int, cohort: str) -> str:
    """Pass B user message: full production format + the two conditioning fields."""
    return (
        format_user_message(row)
        + f"\nPriorBinaryVerdict: {verdict}"
        + f"\nCohort: {cohort}"
    )


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def pass_a_kwargs(row: dict[str, Any], prompt_a: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": prompt_a,
        "input": pass_a_message(row),
        "prompt_cache_key": cfg.PASS_A_CACHE_KEY,
        "max_output_tokens": cfg.PASS_A_MAX_OUTPUT_TOKENS,
        "store": False,
        "text": _text_format(BinaryResult),
        "reasoning": {"effort": cfg.PASS_A_EFFORT},
        "top_logprobs": cfg.TOP_LOGPROBS,
        "include": list(cfg.LOGPROB_INCLUDE),
    }


def pass_b_kwargs(row: dict[str, Any], verdict: int, cohort: str,
                  model: str, effort_b: str) -> dict[str, Any]:
    result_cls = SubclassResultAI if verdict == 1 else SubclassResultNot
    return {
        "model": model,
        "instructions": load_pass_b_prompt(verdict),
        "input": pass_b_message(row, verdict, cohort),
        "prompt_cache_key": cfg.PASS_B_CACHE_KEY,
        "max_output_tokens": cfg.MAX_OUTPUT_TOKENS,
        "store": False,
        "text": _text_format(result_cls),
        "reasoning": {"effort": effort_b},
    }


def identity_hashes() -> dict[str, str]:
    """SHA-256 of every artifact that defines the two-pass request identity."""
    return {
        "prompt_a_sha256": _sha256(load_pass_a_prompt()),
        "prompt_b_family1_sha256": _sha256(load_pass_b_prompt(1)),
        "prompt_b_family0_sha256": _sha256(load_pass_b_prompt(0)),
        "schema_a_sha256": _sha256(json.dumps(strict_schema(BinaryResult), sort_keys=True)),
        "schema_b1_sha256": _sha256(json.dumps(strict_schema(SubclassResultAI), sort_keys=True)),
        "schema_b0_sha256": _sha256(json.dumps(strict_schema(SubclassResultNot), sort_keys=True)),
        "formatter_sha256": _sha256(inspect.getsource(format_user_message)),
    }


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def _usage_fields(resp: Any, prefix: str) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    reasoning = None
    if usage is not None and getattr(usage, "output_tokens_details", None) is not None:
        reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", None)
    return {
        f"{prefix}_input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        f"{prefix}_output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        f"{prefix}_reasoning_tokens": reasoning,
    }


def _parse_output(resp: Any) -> Optional[dict[str, Any]]:
    text = getattr(resp, "output_text", "") or ""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def assemble_record(custom_id: str, org_uuid: str, model: str, effort_b: str,
                    cohort: str, resp_a: Any, resp_b: Any) -> dict[str, Any]:
    """One predictions.jsonl line combining both passes.

    status is 'completed' only when both passes completed AND parsed; resume
    treats anything else as unfinished and re-runs the row.
    """
    a = _parse_output(resp_a)
    b = _parse_output(resp_b) if resp_b is not None else None

    status_a = getattr(resp_a, "status", None)
    status_b = getattr(resp_b, "status", None) if resp_b is not None else None
    ok = status_a == "completed" and status_b == "completed" and a is not None and b is not None

    if ok:
        status = "completed"
    else:
        # Never let a failed row carry status="completed" (resume would skip it
        # forever): an API-completed pass whose JSON did not parse, or a row
        # where Pass B never ran, must remain retryable.
        status = status_b or status_a or "failed"
        if status == "completed":
            status = "parse_failed"

    verdict = a.get("ai_native") if a else None
    record: dict[str, Any] = {
        "custom_id": custom_id,
        "org_uuid": org_uuid,
        "model": model,
        "effort_a": cfg.PASS_A_EFFORT,
        "effort_b": effort_b,
        "status": status,
        "status_a": status_a,
        "status_b": status_b,
        "ai_native": verdict,
        "subclass": b.get("subclass") if b else None,
        # RAD is structural for the zero family: not a model opinion.
        "rad_score": (b.get("rad_score") if verdict == 1 else "RAD-NA") if b else None,
        "cohort": cohort,
        "conf_classification": b.get("conf_classification") if b else None,
        "conf_rad": (b.get("conf_rad") if verdict == 1 else None) if b else None,
        "boundary_disagreement": b.get("boundary_disagreement") if b else None,
        "reasons_3_points": b.get("reasons_3_points") if b else None,
        "verification_critique": b.get("verification_critique") if b else None,
    }
    record.update(_usage_fields(resp_a, "a"))
    if resp_b is not None:
        record.update(_usage_fields(resp_b, "b"))
    return record


# ---------------------------------------------------------------------------
# The run engine
# ---------------------------------------------------------------------------

def make_run_id(model: str, effort_b: str, repeat: int) -> str:
    date = datetime.date.today().isoformat()
    return f"{date}_2pass_{model}_{effort_b}_r{repeat}"


@_RETRIABLE
def _create(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    return client.responses.create(**kwargs)


# Resume refuses to mix changed prompts/schemas/model into an existing run.
_RESUME_INVARIANTS = (
    "model", "effort_b",
    "prompt_a_sha256", "prompt_b_family1_sha256", "prompt_b_family0_sha256",
    "schema_a_sha256", "schema_b1_sha256", "schema_b0_sha256",
    "formatter_sha256",
)


def _ensure_config(run_id: str, model: str, effort_b: str, repeat: int,
                   n_rows: int) -> None:
    config = {
        "run_id": run_id,
        "kind": "two_pass",
        "model": model,
        "effort_a": cfg.PASS_A_EFFORT,
        "effort_b": effort_b,
        "repeat": repeat,
        "n_rows": n_rows,
        "top_logprobs": cfg.TOP_LOGPROBS,
        "pass_a_max_output_tokens": cfg.PASS_A_MAX_OUTPUT_TOKENS,
        "pass_b_max_output_tokens": cfg.MAX_OUTPUT_TOKENS,
        "pass_a_cache_key": cfg.PASS_A_CACHE_KEY,
        "pass_b_cache_key": cfg.PASS_B_CACHE_KEY,
        "git_commit": _git_commit(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **identity_hashes(),
    }
    path = run_config_path(run_id)
    if not path.exists():
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [k for k in _RESUME_INVARIANTS if prior.get(k) != config[k]]
    if mismatched:
        raise SystemExit(
            f"Cannot resume two-pass run {run_id}: {mismatched} changed since it "
            "started. Start a fresh run with a new --run-id."
        )


def run_two_pass(model: str = cfg.EVAL_MODELS[0],
                 effort_b: str = cfg.PASS_B_EFFORT,
                 repeat: int = 1,
                 limit: int | None = None,
                 dry_run: bool = False,
                 run_id: str | None = None) -> str:
    """Run Pass A + Pass B over the golden set. Returns the run_id."""
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be a positive row cap, got {limit}")

    rows = load_golden_rows()
    if limit is not None:
        rows = rows[:limit]

    run_id = run_id or make_run_id(model, effort_b, repeat)
    prompt_a = load_pass_a_prompt()

    if dry_run:
        _print_dry_run(rows, prompt_a, model, effort_b, run_id)
        return run_id

    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_raw_dir(run_id).mkdir(parents=True, exist_ok=True)
    predictions_path = run_predictions_path(run_id)
    _ensure_config(run_id, model, effort_b, repeat, len(rows))

    done = _completed_custom_ids(predictions_path)
    if done:
        logger.info("Resuming %s: %d rows already complete", run_id, len(done))

    client = OpenAI(api_key=OPENAI_API_KEY)
    todo = [r for r in rows if f"startup-{r['org_uuid']}" not in done]
    logger.info("Two-pass run %s: %d rows (%s, B effort=%s)",
                run_id, len(todo), model, effort_b)

    for i, row in enumerate(todo, start=1):
        cid = f"startup-{row['org_uuid']}"
        cohort = compute_cohort(row.get("founded_date", ""))

        resp_a = _create(client, pass_a_kwargs(row, prompt_a, model))
        (run_raw_dir(run_id) / f"{cid}_a.json").write_text(
            json.dumps(resp_a.model_dump(), ensure_ascii=False), encoding="utf-8"
        )

        a = _parse_output(resp_a)
        resp_b = None
        if a is not None and a.get("ai_native") in (0, 1):
            resp_b = _create(
                client,
                pass_b_kwargs(row, a["ai_native"], cohort, model, effort_b),
            )
            (run_raw_dir(run_id) / f"{cid}_b.json").write_text(
                json.dumps(resp_b.model_dump(), ensure_ascii=False), encoding="utf-8"
            )
        else:
            logger.warning("Pass A gave no usable verdict for %s (status=%s); "
                           "row will retry on resume", cid, getattr(resp_a, "status", None))

        record = assemble_record(
            cid, row["org_uuid"], model, effort_b, cohort, resp_a, resp_b
        )
        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("  [%d/%d] %s -> A=%s B=%s/%s (%s)",
                    i, len(todo), str(row.get("name", ""))[:24],
                    record.get("ai_native"), record.get("subclass"),
                    record.get("rad_score"), record.get("status"))

    logger.info("Two-pass run %s complete: %s", run_id, predictions_path)
    return run_id


def _print_dry_run(rows: list[dict[str, Any]], prompt_a: str, model: str,
                   effort_b: str, run_id: str) -> None:
    pricing = cfg.EVAL_MODEL_PRICING.get(model, {})
    prompt_b1 = load_pass_b_prompt(1)
    prompt_b0 = load_pass_b_prompt(0)
    # ~4 chars/token; Pass B prompt size depends on the family, use the mean.
    a_chars = sum(len(prompt_a) + len(pass_a_message(r)) for r in rows)
    b_prompt_mean = (len(prompt_b1) + len(prompt_b0)) / 2
    b_chars = sum(b_prompt_mean + len(format_user_message(r)) + 40 for r in rows)
    est_tokens = (a_chars + b_chars) / 4
    logger.info("DRY RUN %s", run_id)
    logger.info("  model=%s pass A effort=%s, pass B effort=%s, rows=%d",
                model, cfg.PASS_A_EFFORT, effort_b, len(rows))
    logger.info("  est input tokens ~%d, est input cost ~$%.4f "
                "(output/reasoning excluded)",
                int(est_tokens), est_tokens / 1e6 * pricing.get("input", 0.0))
    logger.info("  identity hashes: %s", identity_hashes())
