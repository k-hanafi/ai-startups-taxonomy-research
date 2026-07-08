"""Stage 7: Batch-vs-sync parity smoke for Pass A requests (gate Q4).

The eval harness runs everything through the sync Responses API, but the
production pipeline lives on the Batch API. Before trusting either side, this
smoke submits a handful of Pass A rows BOTH ways with byte-identical request
bodies (built by evals.two_pass.pass_a_kwargs in both cases) and asserts that
the Batch API returns the same logprob payload shape and honors the same
parameters (top_logprobs, reasoning effort, temperature, include). If parity
fails, logprob confidence measured sync would not transfer to production.

Paid step (tiny: PARITY_ROWS nano requests x2). The parity checks themselves
are pure functions over response dicts, so the assertion logic is fully
testable offline against fixtures.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from typing import Any

from openai import OpenAI

from src.config import OPENAI_API_KEY

from evals import config as cfg
from evals.paths import parity_report_path, run_dir, run_raw_dir
from evals.runner import _RETRIABLE, _git_commit, load_golden_rows
from evals.two_pass import _parse_output, load_pass_a_prompt, pass_a_kwargs

logger = logging.getLogger(__name__)

BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


# ---------------------------------------------------------------------------
# Pure payload inspection + parity checks (offline-testable)
# ---------------------------------------------------------------------------

def batch_input_line(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """One Batch API JSONL line wrapping the exact sync request body."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def extract_logprob_entries(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    """The token-level logprob list from a Responses API payload dict."""
    for item in response_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            entries = content.get("logprobs")
            if entries:
                return entries
    return []


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _binary_verdict(response_body: dict[str, Any]) -> Any:
    """Parse the Pass A output text of a payload dict into the 0/1 verdict."""
    text = ""
    for item in response_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("text"):
                text = content["text"]
    if not text:
        return None
    try:
        return json.loads(text).get("ai_native")
    except json.JSONDecodeError:
        return None


def _logprob_shape_checks(
    side: str, entries: list[dict[str, Any]], requested_top: int
) -> list[dict[str, Any]]:
    checks = [_check(f"{side}_logprobs_present", bool(entries),
                     f"{len(entries)} token entries")]
    if not entries:
        return checks
    entry_keys = {"token", "bytes", "logprob", "top_logprobs"}
    checks.append(_check(
        f"{side}_logprob_entry_keys",
        all(entry_keys <= set(e) for e in entries),
        f"expected keys {sorted(entry_keys)}",
    ))
    top_keys = {"token", "bytes", "logprob"}
    checks.append(_check(
        f"{side}_top_logprob_entry_keys",
        all(top_keys <= set(t) for e in entries for t in e.get("top_logprobs") or []),
        f"expected keys {sorted(top_keys)}",
    ))
    # Every token must carry the requested depth. Tolerance of one: the API
    # omits the chosen token from its own alternatives list on some positions
    # (observed live on BOTH sync and batch: first token returns 14 of 15).
    # A payload where only some tokens reach full depth still fails.
    lengths = [len(e.get("top_logprobs") or []) for e in entries]
    checks.append(_check(
        f"{side}_top_logprobs_honored",
        max(lengths) == requested_top and min(lengths) >= requested_top - 1,
        f"per-token alternatives min {min(lengths)} / max {max(lengths)}, "
        f"requested {requested_top}",
    ))
    return checks


def parity_checks(
    request_body: dict[str, Any],
    sync_body: dict[str, Any],
    batch_body: dict[str, Any],
) -> list[dict[str, Any]]:
    """All parity assertions for one row, over plain payload dicts."""
    checks = [
        _check("sync_completed", sync_body.get("status") == "completed",
               f"sync status={sync_body.get('status')}"),
        _check("batch_completed", batch_body.get("status") == "completed",
               f"batch status={batch_body.get('status')}"),
    ]

    # Parameter honoring: both payloads echo the request's experimental
    # params. temperature is compared sync-vs-batch (we do not send it, so
    # both sides must land on the same server default).
    requested_top = request_body.get("top_logprobs")
    requested_effort = (request_body.get("reasoning") or {}).get("effort")
    for side, body in (("sync", sync_body), ("batch", batch_body)):
        checks.append(_check(
            f"{side}_echoes_top_logprobs",
            body.get("top_logprobs") == requested_top,
            f"echoed {body.get('top_logprobs')}, requested {requested_top}",
        ))
        checks.append(_check(
            f"{side}_echoes_reasoning_effort",
            (body.get("reasoning") or {}).get("effort") == requested_effort,
            f"echoed {(body.get('reasoning') or {}).get('effort')}, "
            f"requested {requested_effort}",
        ))
    checks.append(_check(
        "temperature_parity",
        sync_body.get("temperature") == batch_body.get("temperature"),
        f"sync {sync_body.get('temperature')} vs batch {batch_body.get('temperature')}",
    ))

    # Reasoning genuinely off on both sides (logprobs depend on it).
    for side, body in (("sync", sync_body), ("batch", batch_body)):
        details = (body.get("usage") or {}).get("output_tokens_details") or {}
        checks.append(_check(
            f"{side}_reasoning_tokens_zero",
            details.get("reasoning_tokens") == 0,
            f"reasoning_tokens={details.get('reasoning_tokens')}",
        ))

    # Logprob payload shape parity (the include param is honored iff present).
    sync_entries = extract_logprob_entries(sync_body)
    batch_entries = extract_logprob_entries(batch_body)
    checks.extend(_logprob_shape_checks("sync", sync_entries, requested_top))
    checks.extend(_logprob_shape_checks("batch", batch_entries, requested_top))

    # Both sides produce a usable binary verdict.
    for side, body in (("sync", sync_body), ("batch", batch_body)):
        verdict = _binary_verdict(body)
        checks.append(_check(
            f"{side}_binary_verdict_valid", verdict in (0, 1),
            f"ai_native={verdict}",
        ))
    return checks


def build_parity_report(
    rows: dict[str, dict[str, Any]],
    sync_bodies: dict[str, dict[str, Any]],
    batch_bodies: dict[str, dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Aggregate per-row checks into the gate Q4 report."""
    per_row: dict[str, Any] = {}
    all_ok = True
    for cid, request_body in rows.items():
        sync_body = sync_bodies.get(cid)
        batch_body = batch_bodies.get(cid)
        if sync_body is None or batch_body is None:
            per_row[cid] = {
                "ok": False,
                "checks": [_check("responses_present", False,
                                  f"sync={sync_body is not None}, "
                                  f"batch={batch_body is not None}")],
            }
            all_ok = False
            continue
        checks = parity_checks(request_body, sync_body, batch_body)
        row_ok = all(c["ok"] for c in checks)
        per_row[cid] = {"ok": row_ok, "checks": checks}
        all_ok = all_ok and row_ok
    return {
        "gate": "Q4",
        "verdict": "PASS" if all_ok else "FAIL",
        "model": model,
        "n_rows": len(rows),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "rows": per_row,
    }


# ---------------------------------------------------------------------------
# Paid execution (sync side + batch side)
# ---------------------------------------------------------------------------

@_RETRIABLE
def _create_sync(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    return client.responses.create(**kwargs)


def _wait_for_batch(client: OpenAI, batch_id: str) -> tuple[Any, str | None]:
    """Poll until the batch reaches a terminal status or the wait cap."""
    deadline = time.monotonic() + cfg.PARITY_MAX_WAIT_SECONDS
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in BATCH_TERMINAL_STATUSES:
            return batch, None
        if time.monotonic() >= deadline:
            return batch, (
                f"Batch {batch_id} still {batch.status} after "
                f"{cfg.PARITY_MAX_WAIT_SECONDS}s. Retrieve it later from the "
                "OpenAI dashboard; the sync responses are already saved."
            )
        logger.info("Batch %s: %s, polling again in %ds",
                    batch_id, batch.status, cfg.PARITY_POLL_SECONDS)
        time.sleep(cfg.PARITY_POLL_SECONDS)


def _download_batch_bodies(
    client: OpenAI, batch: Any
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """custom_id -> response payload dict from the batch output file."""
    if not batch.output_file_id:
        return {}, (
            f"Batch {batch.id} finished {batch.status} with no "
            f"output file (errors: {batch.error_file_id})"
        )
    content = client.files.content(batch.output_file_id).text
    bodies: dict[str, dict[str, Any]] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        response = rec.get("response") or {}
        bodies[rec["custom_id"]] = response.get("body") or {}
    return bodies, None


def _write_parity_report(
    run_id: str,
    requests: dict[str, dict[str, Any]],
    sync_bodies: dict[str, dict[str, Any]],
    batch_bodies: dict[str, dict[str, Any]],
    model: str,
    *,
    batch_error: str | None = None,
) -> dict[str, Any]:
    """Build and persist the gate Q4 report; always returns the report dict.

    A batch_error (timeout or missing output file) forces a FAIL verdict, so
    the CLI exits nonzero even though the sync results are preserved on disk.
    """
    report = build_parity_report(requests, sync_bodies, batch_bodies, model)
    if batch_error:
        report["batch_error"] = batch_error
        report["verdict"] = "FAIL"
    report_path = parity_report_path(run_id)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Gate Q4 verdict: %s (%s)", report["verdict"], report_path)
    if batch_error:
        logger.error("Batch phase failed: %s", batch_error)
    for cid, row in report["rows"].items():
        if not row["ok"]:
            for c in row["checks"]:
                if not c["ok"]:
                    logger.warning("  %s: %s (%s)", cid, c["name"], c["detail"])
    return report


def run_parity(model: str = cfg.EVAL_MODELS[0]) -> dict[str, Any]:
    """Submit PARITY_ROWS Pass A rows sync AND batch; write the parity report."""
    rows = load_golden_rows()[:cfg.PARITY_ROWS]
    prompt_a = load_pass_a_prompt()
    requests = {
        f"startup-{row['org_uuid']}": pass_a_kwargs(row, prompt_a, model)
        for row in rows
    }

    run_id = f"{datetime.date.today().isoformat()}_parity_{model}"
    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    raw = run_raw_dir(run_id)
    raw.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)

    logger.info("Parity smoke %s: %d rows, sync side first", run_id, len(requests))
    sync_bodies: dict[str, dict[str, Any]] = {}
    for i, (cid, body) in enumerate(requests.items(), start=1):
        resp = _create_sync(client, body)
        sync_bodies[cid] = resp.model_dump()
        (raw / f"{cid}_sync.json").write_text(
            json.dumps(sync_bodies[cid], ensure_ascii=False), encoding="utf-8"
        )
        logger.info("  sync [%d/%d] %s -> %s (ai_native=%s)",
                    i, len(requests), cid, resp.status,
                    (_parse_output(resp) or {}).get("ai_native"))

    # Inside raw/: each line embeds the prompt + scraped evidence text, which
    # must stay out of the public repo (raw/ is git-ignored).
    batch_file = raw / "batch_input.jsonl"
    batch_file.write_text(
        "".join(json.dumps(batch_input_line(cid, body), ensure_ascii=False) + "\n"
                for cid, body in requests.items()),
        encoding="utf-8",
    )
    with batch_file.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=cfg.PARITY_COMPLETION_WINDOW,
        metadata={"run_id": run_id, "purpose": "gate-q4-parity-smoke"},
    )
    logger.info("Submitted batch %s, waiting for completion", batch.id)

    batch, wait_error = _wait_for_batch(client, batch.id)
    batch_error = wait_error
    batch_bodies: dict[str, dict[str, Any]] = {}
    if wait_error is None:
        batch_bodies, download_error = _download_batch_bodies(client, batch)
        if download_error:
            batch_error = download_error
        else:
            for cid, body in batch_bodies.items():
                (raw / f"{cid}_batch.json").write_text(
                    json.dumps(body, ensure_ascii=False), encoding="utf-8"
                )

    return _write_parity_report(
        run_id, requests, sync_bodies, batch_bodies, model,
        batch_error=batch_error,
    )
