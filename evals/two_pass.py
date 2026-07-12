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
banked per pass. A row is complete only when BOTH passes succeeded.

Stage 8 Pass B effort sweeps bank Pass A once per model by default into
``evals/runs/pass_a_banks/<model>/``. Later efforts auto-reuse that bank so
effort deltas are not confounded by resampling the gate. Escape hatches:
``--rerun-pass-a`` (invalidate / rebuild) and ``--pass-a-from`` (pin a
historical run). Without a bank, a crash between passes still re-runs Pass A
on resume of the cell run.
"""

from __future__ import annotations

import datetime
import inspect
import json
import logging
import shutil
import time
from types import SimpleNamespace
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
    pass_a_bank_run_id,
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
        "top_logprobs": cfg.PASS_A_TOP_LOGPROBS,
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
    from evals.usage import cached_tokens_from_usage

    usage = getattr(resp, "usage", None)
    reasoning = None
    if usage is not None and getattr(usage, "output_tokens_details", None) is not None:
        reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", None)
    return {
        f"{prefix}_input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        f"{prefix}_output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        f"{prefix}_reasoning_tokens": reasoning,
        # 0 when usage/details absent; same semantics as single-pass runner.
        f"{prefix}_cached_tokens": cached_tokens_from_usage(usage),
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
                    cohort: str, resp_a: Any, resp_b: Any,
                    latency_a_s: float | None = None,
                    latency_b_s: float | None = None) -> dict[str, Any]:
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
    record["a_latency_s"] = latency_a_s
    if resp_b is not None:
        record.update(_usage_fields(resp_b, "b"))
        record["b_latency_s"] = latency_b_s
    # Flat totals under the single-pass field names so the scorer reads one
    # field for both run shapes. A row missing Pass B has no meaningful
    # end-to-end latency (it will be re-run), so the total stays None.
    if latency_a_s is not None and latency_b_s is not None:
        record["latency_s"] = round(latency_a_s + latency_b_s, 3)
    else:
        record["latency_s"] = None
    # Cached tokens always sum A+B (0 when a pass is missing).
    record["cached_tokens"] = int(record.get("a_cached_tokens") or 0) + int(
        record.get("b_cached_tokens") or 0
    )
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
    "model", "effort_b", "repeat", "n_rows",
    "prompt_a_sha256", "prompt_b_family1_sha256", "prompt_b_family0_sha256",
    "schema_a_sha256", "schema_b1_sha256", "schema_b0_sha256",
    "formatter_sha256",
)


def _ensure_config(run_id: str, model: str, effort_b: str, repeat: int,
                   n_rows: int,
                   pass_a_bank_run_id: str | None = None) -> None:
    config = {
        "run_id": run_id,
        "kind": "two_pass",
        "model": model,
        "effort_a": cfg.PASS_A_EFFORT,
        "effort_b": effort_b,
        "repeat": repeat,
        "n_rows": n_rows,
        "top_logprobs": cfg.PASS_A_TOP_LOGPROBS,
        "pass_a_max_output_tokens": cfg.PASS_A_MAX_OUTPUT_TOKENS,
        "pass_b_max_output_tokens": cfg.MAX_OUTPUT_TOKENS,
        "pass_a_cache_key": cfg.PASS_A_CACHE_KEY,
        "pass_b_cache_key": cfg.PASS_B_CACHE_KEY,
        "pass_a_bank_run_id": pass_a_bank_run_id,
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
    if prior.get("pass_a_bank_run_id") != pass_a_bank_run_id:
        raise SystemExit(
            f"Cannot resume two-pass run {run_id}: pass_a_bank_run_id changed "
            f"({prior.get('pass_a_bank_run_id')!r} -> {pass_a_bank_run_id!r})."
        )


def load_pass_a_bank(bank_run_id: str) -> dict[str, dict[str, Any]]:
    """Load banked Pass A verdicts + raw payloads keyed by custom_id.

    Returns custom_id -> {ai_native, raw_a (dict), usage latency fields from
    the banked prediction record}. Raises SystemExit if the bank is missing
    completed Pass A rows or raw ``*_a.json`` files.
    """
    preds_path = run_predictions_path(bank_run_id)
    if not preds_path.exists():
        raise SystemExit(
            f"Pass A bank {bank_run_id!r} has no predictions at {preds_path}"
        )
    bank_config: dict[str, Any] = {}
    config_path = run_config_path(bank_run_id)
    if config_path.exists():
        bank_config = json.loads(config_path.read_text(encoding="utf-8"))

    by_cid: dict[str, dict[str, Any]] = {}
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("status") != "completed":
            continue
        cid = rec.get("custom_id")
        verdict = rec.get("ai_native")
        if not cid or verdict not in (0, 1):
            continue
        by_cid[cid] = rec

    if not by_cid:
        raise SystemExit(
            f"Pass A bank {bank_run_id!r} has no completed rows with ai_native "
            "in {{0,1}}"
        )

    raw_root = run_raw_dir(bank_run_id)
    out: dict[str, dict[str, Any]] = {}
    missing_raw: list[str] = []
    for cid, rec in by_cid.items():
        raw_path = raw_root / f"{cid}_a.json"
        if not raw_path.exists():
            missing_raw.append(cid)
            continue
        out[cid] = {
            "ai_native": int(rec["ai_native"]),
            "raw_a": json.loads(raw_path.read_text(encoding="utf-8")),
            "record": rec,
            "bank_model": bank_config.get("model") or rec.get("model"),
        }
    if missing_raw:
        raise SystemExit(
            f"Pass A bank {bank_run_id!r} missing raw Pass A files for "
            f"{len(missing_raw)} row(s), e.g. {missing_raw[0]}_a.json "
            "(raw/ is machine-local; copy it with the bank run)."
        )
    return out


def _assert_bank_model(bank: dict[str, dict[str, Any]], bank_id: str,
                       model: str) -> None:
    bank_model = next(iter(bank.values()))["bank_model"]
    if not bank_model:
        raise SystemExit(
            f"Pass A bank {bank_id!r} has no model recorded in "
            "config.json or prediction rows. Cannot verify same-model "
            "reuse. Re-bank Pass A with --rerun-pass-a or fix the bank "
            "metadata."
        )
    if bank_model != model:
        raise SystemExit(
            f"Pass A bank {bank_id!r} was run with model "
            f"{bank_model!r}, but this run requested {model!r}. "
            "Bank Pass A once per model."
        )


def pass_a_bank_covers(bank_run_id: str, custom_ids: list[str]) -> bool:
    """True when bank_run_id has completed Pass A + raw for every custom_id."""
    preds_path = run_predictions_path(bank_run_id)
    if not preds_path.exists():
        return False
    have: set[str] = set()
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("status") != "completed":
            continue
        cid = rec.get("custom_id")
        if cid and rec.get("ai_native") in (0, 1):
            raw = run_raw_dir(bank_run_id) / f"{cid}_a.json"
            if raw.exists():
                have.add(cid)
    return all(cid in have for cid in custom_ids)


def clear_pass_a_bank(model: str) -> str:
    """Remove the stable per-model Pass A bank directory. Returns bank run_id."""
    bank_id = pass_a_bank_run_id(model)
    path = run_dir(bank_id)
    if path.exists():
        shutil.rmtree(path)
    return bank_id


def _ensure_pass_a_bank_config(bank_id: str, model: str, n_rows: int) -> None:
    """Write or validate the stable Pass A bank config snapshot."""
    hashes = identity_hashes()
    config = {
        "run_id": bank_id,
        "kind": "pass_a_bank",
        "model": model,
        "effort_a": cfg.PASS_A_EFFORT,
        "n_rows": n_rows,
        "top_logprobs": cfg.PASS_A_TOP_LOGPROBS,
        "pass_a_max_output_tokens": cfg.PASS_A_MAX_OUTPUT_TOKENS,
        "pass_a_cache_key": cfg.PASS_A_CACHE_KEY,
        "git_commit": _git_commit(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt_a_sha256": hashes["prompt_a_sha256"],
        "schema_a_sha256": hashes["schema_a_sha256"],
        "formatter_sha256": hashes["formatter_sha256"],
    }
    path = run_config_path(bank_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "n_rows", "prompt_a_sha256", "schema_a_sha256",
                "formatter_sha256"):
        if prior.get(key) != config[key]:
            raise SystemExit(
                f"Cannot extend Pass A bank {bank_id!r}: {key} changed "
                f"({prior.get(key)!r} -> {config[key]!r}). Pass --rerun-pass-a "
                "to rebuild the bank."
            )


def _persist_pass_a_bank_row(bank_id: str, model: str, cid: str, org_uuid: str,
                             resp_a: Any, latency_a_s: float | None,
                             raw_a: dict[str, Any]) -> None:
    """Append one completed Pass A row into the stable bank (idempotent skip)."""
    preds_path = run_predictions_path(bank_id)
    if preds_path.exists():
        for line in preds_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("custom_id") == cid and rec.get("status") == "completed":
                return
    a = _parse_output(resp_a)
    if a is None or a.get("ai_native") not in (0, 1):
        return
    run_raw_dir(bank_id).mkdir(parents=True, exist_ok=True)
    (run_raw_dir(bank_id) / f"{cid}_a.json").write_text(
        json.dumps(raw_a, ensure_ascii=False), encoding="utf-8"
    )
    record = {
        "custom_id": cid,
        "org_uuid": org_uuid,
        "model": model,
        "status": "completed",
        "ai_native": a["ai_native"],
        "a_latency_s": latency_a_s,
        **_usage_fields(resp_a, "a"),
    }
    with preds_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _assert_bank_identity(bank_id: str) -> None:
    """Refuse a stable bank whose Pass A prompt/schema no longer match."""
    path = run_config_path(bank_id)
    if not path.exists():
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    if prior.get("kind") != "pass_a_bank":
        # Historical cell runs used as --pass-a-from pins skip this check.
        return
    hashes = identity_hashes()
    for key in ("prompt_a_sha256", "schema_a_sha256", "formatter_sha256"):
        if prior.get(key) is None:
            continue
        if prior.get(key) != hashes[key]:
            raise SystemExit(
                f"Pass A bank {bank_id!r} was built with a different {key}. "
                "Pass --rerun-pass-a to rebuild (refusing silent reuse of a "
                "stale bank)."
            )


def resolve_pass_a_source(
    model: str,
    custom_ids: list[str],
    *,
    pass_a_from: str | None = None,
    rerun_pass_a: bool = False,
) -> tuple[str, dict[str, dict[str, Any]] | None, bool]:
    """Decide which Pass A bank to use.

    Returns (bank_id, bank_or_None, creating).
    bank is loaded when reusing; None when this run must call Pass A and
    persist into the stable bank. creating is True only for the stable-bank
    write path (not for --pass-a-from pins).
    """
    if pass_a_from and rerun_pass_a:
        raise SystemExit(
            "Pass only one of --pass-a-from and --rerun-pass-a "
            "(cannot pin a historical bank and rebuild at once)."
        )
    if pass_a_from:
        bank = load_pass_a_bank(pass_a_from)
        _assert_bank_model(bank, pass_a_from, model)
        missing = [cid for cid in custom_ids if cid not in bank]
        if missing:
            raise SystemExit(
                f"Pass A bank {pass_a_from!r} has no completed row for "
                f"{missing[0]}; refuse partial reuse (science confound)."
            )
        return pass_a_from, bank, False

    bank_id = pass_a_bank_run_id(model)
    if rerun_pass_a:
        clear_pass_a_bank(model)
        return bank_id, None, True

    if pass_a_bank_covers(bank_id, custom_ids):
        _assert_bank_identity(bank_id)
        bank = load_pass_a_bank(bank_id)
        _assert_bank_model(bank, bank_id, model)
        return bank_id, bank, False

    return bank_id, None, True


class _BankedPassAResponse:
    """Duck-typed Responses API object assembled from a banked raw dump."""

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw
        self.status = raw.get("status", "completed")
        self.output_text = ""
        for item in raw.get("output") or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") == "output_text" and content.get("text"):
                    self.output_text = content["text"]
                    break
        usage = raw.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        cached_details = usage.get("input_tokens_details") or {}
        self.usage = SimpleNamespace(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=details.get("reasoning_tokens"),
            ),
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached_details.get("cached_tokens"),
            ),
        )

    def model_dump(self) -> dict[str, Any]:
        return self._raw


def run_two_pass(model: str = cfg.EVAL_MODELS[0],
                 effort_b: str = cfg.PASS_B_EFFORT,
                 repeat: int = 1,
                 limit: int | None = None,
                 dry_run: bool = False,
                 run_id: str | None = None,
                 pass_a_from: str | None = None,
                 rerun_pass_a: bool = False,
                 reuse_pass_a_from: str | None = None) -> str:
    """Run Pass A + Pass B over the golden set. Returns the run_id.

    Pass A is banked once per model under ``pass_a_banks/<model>/`` and
    reused by default for later Pass B efforts. ``--rerun-pass-a`` rebuilds
    that bank. ``--pass-a-from`` pins a historical bank (override).
    ``reuse_pass_a_from`` is a deprecated alias for ``pass_a_from``.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be a positive row cap, got {limit}")

    if reuse_pass_a_from and pass_a_from:
        raise SystemExit(
            "Pass only one of --pass-a-from and --reuse-pass-a-from "
            "(the latter is a deprecated alias)."
        )
    if reuse_pass_a_from:
        logger.warning(
            "--reuse-pass-a-from is deprecated. Pass A banks auto-reuse per "
            "model by default. Use --pass-a-from only to pin a historical bank."
        )
        pass_a_from = reuse_pass_a_from

    rows = load_golden_rows()
    if limit is not None:
        rows = rows[:limit]

    run_id = run_id or make_run_id(model, effort_b, repeat)
    prompt_a = load_pass_a_prompt()
    custom_ids = [f"startup-{r['org_uuid']}" for r in rows]

    bank_id, bank, creating_bank = resolve_pass_a_source(
        model, custom_ids,
        pass_a_from=pass_a_from,
        rerun_pass_a=rerun_pass_a,
    )

    if dry_run:
        _print_dry_run(
            rows, prompt_a, model, effort_b, run_id,
            pass_a_bank_id=bank_id if bank is not None else None,
        )
        return run_id

    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_raw_dir(run_id).mkdir(parents=True, exist_ok=True)
    predictions_path = run_predictions_path(run_id)
    _ensure_config(
        run_id, model, effort_b, repeat, len(rows),
        pass_a_bank_run_id=bank_id,
    )

    if creating_bank:
        run_dir(bank_id).mkdir(parents=True, exist_ok=True)
        run_raw_dir(bank_id).mkdir(parents=True, exist_ok=True)
        _ensure_pass_a_bank_config(bank_id, model, len(rows))

    done = _completed_custom_ids(predictions_path)
    if done:
        logger.info("Resuming %s: %d rows already complete", run_id, len(done))

    client = OpenAI(api_key=OPENAI_API_KEY)
    todo = [r for r in rows if f"startup-{r['org_uuid']}" not in done]
    if bank is not None:
        logger.info(
            "Two-pass run %s: %d rows (%s, B effort=%s, Pass A reused from %s)",
            run_id, len(todo), model, effort_b, bank_id,
        )
    else:
        logger.info(
            "Two-pass run %s: %d rows (%s, B effort=%s, banking Pass A to %s)",
            run_id, len(todo), model, effort_b, bank_id,
        )

    for i, row in enumerate(todo, start=1):
        cid = f"startup-{row['org_uuid']}"
        cohort = compute_cohort(row.get("founded_date", ""))

        if bank is not None:
            banked = bank.get(cid)
            if banked is None:
                raise SystemExit(
                    f"Pass A bank {bank_id!r} has no completed row "
                    f"for {cid}; refuse partial reuse (science confound)."
                )
            resp_a = _BankedPassAResponse(banked["raw_a"])
            latency_a_s = banked["record"].get("a_latency_s")
            (run_raw_dir(run_id) / f"{cid}_a.json").write_text(
                json.dumps(banked["raw_a"], ensure_ascii=False), encoding="utf-8"
            )
            a = {"ai_native": banked["ai_native"]}
        else:
            # Prefer a row already in the (partial) stable bank on resume.
            if pass_a_bank_covers(bank_id, [cid]):
                partial = load_pass_a_bank(bank_id)
                banked = partial[cid]
                resp_a = _BankedPassAResponse(banked["raw_a"])
                latency_a_s = banked["record"].get("a_latency_s")
                raw_a = banked["raw_a"]
            else:
                # Wall-clock latency around each API call; retry backoff is
                # included, so this is the honest per-pass cost a production
                # caller would feel.
                started_a = time.monotonic()
                resp_a = _create(client, pass_a_kwargs(row, prompt_a, model))
                latency_a_s = round(time.monotonic() - started_a, 3)
                raw_a = resp_a.model_dump()
            (run_raw_dir(run_id) / f"{cid}_a.json").write_text(
                json.dumps(
                    raw_a if isinstance(raw_a, dict) else resp_a.model_dump(),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            a = _parse_output(resp_a)
            if creating_bank and a is not None and a.get("ai_native") in (0, 1):
                _persist_pass_a_bank_row(
                    bank_id, model, cid, row["org_uuid"], resp_a, latency_a_s,
                    raw_a if isinstance(raw_a, dict) else resp_a.model_dump(),
                )

        resp_b = None
        latency_b_s = None
        if a is not None and a.get("ai_native") in (0, 1):
            started_b = time.monotonic()
            resp_b = _create(
                client,
                pass_b_kwargs(row, a["ai_native"], cohort, model, effort_b),
            )
            latency_b_s = round(time.monotonic() - started_b, 3)
            (run_raw_dir(run_id) / f"{cid}_b.json").write_text(
                json.dumps(resp_b.model_dump(), ensure_ascii=False), encoding="utf-8"
            )
        else:
            logger.warning("Pass A gave no usable verdict for %s (status=%s); "
                           "row will retry on resume", cid, getattr(resp_a, "status", None))

        record = assemble_record(
            cid, row["org_uuid"], model, effort_b, cohort, resp_a, resp_b,
            latency_a_s, latency_b_s,
        )
        record["pass_a_bank_run_id"] = bank_id
        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("  [%d/%d] %s -> A=%s B=%s/%s (%s)",
                    i, len(todo), str(row.get("name", ""))[:24],
                    record.get("ai_native"), record.get("subclass"),
                    record.get("rad_score"), record.get("status"))

    logger.info("Two-pass run %s complete: %s", run_id, predictions_path)
    return run_id


def stage8_matrix_cells() -> list[tuple[str, str]]:
    """Locked Stage 8 (model, Pass B effort) pairs in screen order."""
    return [
        (model, effort)
        for model in cfg.EVAL_MODELS
        for effort in cfg.STAGE8_PASS_B_EFFORTS
    ]


def validate_stage8_cell(model: str, effort_b: str) -> None:
    """Refuse unknown models/efforts so a typo cannot create an off-matrix paid run."""
    if model not in cfg.EVAL_MODELS:
        raise SystemExit(
            f"Unknown Stage 8 model {model!r}. Locked EVAL_MODELS = "
            f"{cfg.EVAL_MODELS}"
        )
    if effort_b not in cfg.STAGE8_PASS_B_EFFORTS:
        raise SystemExit(
            f"Unknown Stage 8 Pass B effort {effort_b!r}. Locked "
            f"STAGE8_PASS_B_EFFORTS = {cfg.STAGE8_PASS_B_EFFORTS}"
        )


def _print_dry_run(rows: list[dict[str, Any]], prompt_a: str, model: str,
                   effort_b: str, run_id: str,
                   pass_a_bank_id: str | None = None) -> None:
    pricing = cfg.require_model_pricing(model)
    prompt_b1 = load_pass_b_prompt(1)
    prompt_b0 = load_pass_b_prompt(0)
    # ~4 chars/token; Pass B prompt size depends on the family, use the mean.
    a_chars = sum(len(prompt_a) + len(pass_a_message(r)) for r in rows)
    b_prompt_mean = (len(prompt_b1) + len(prompt_b0)) / 2
    b_chars = sum(b_prompt_mean + len(format_user_message(r)) + 40 for r in rows)
    n = len(rows)
    if pass_a_bank_id:
        est_input = b_chars / 4
        est_out = n * cfg.PASS_B_OUTPUT_TOKEN_ESTIMATE.get(effort_b, 1_000)
        logger.info("DRY RUN %s (Pass A reused from %s; input+output for Pass B only)",
                    run_id, pass_a_bank_id)
    else:
        est_input = (a_chars + b_chars) / 4
        est_out = (
            n * cfg.PASS_A_OUTPUT_TOKEN_ESTIMATE
            + n * cfg.PASS_B_OUTPUT_TOKEN_ESTIMATE.get(effort_b, 1_000)
        )
        logger.info("DRY RUN %s (Pass A + Pass B; no bank yet)", run_id)
    est_in_cost = est_input / 1e6 * pricing["input"]
    est_out_cost = est_out / 1e6 * pricing["output"]
    logger.info("  model=%s pass A effort=%s, pass B effort=%s, rows=%d",
                model, cfg.PASS_A_EFFORT, effort_b, n)
    logger.info(
        "  est input tokens ~%d (~$%.4f) + rough output/reasoning ~%d (~$%.4f) "
        "→ total ~$%.4f (output estimate is order-of-magnitude only)",
        int(est_input), est_in_cost, int(est_out), est_out_cost,
        est_in_cost + est_out_cost,
    )
    if effort_b in ("medium", "high"):
        logger.info(
            "  WARNING: Pass B effort=%s can dominate spend via reasoning "
            "tokens; treat the output estimate as a floor, not a cap.",
            effort_b,
        )
    logger.info("  identity hashes: %s", identity_hashes())
