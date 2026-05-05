"""Batch result downloader with error-file recovery and cache hit reporting.

Batch output order is not guaranteed to match input order. Positional
matching would silently corrupt the dataset at 267K rows. Results are
matched to inputs exclusively by custom_id. error_file_id is downloaded
separately so failed and expired requests are never lost.

Per-response cached_tokens from the usage object are aggregated so the
final report can show actual dollars saved from prompt caching.

Results are appended to production_classifications.csv directly during
download so no separate merge step is required.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.paths import (
    BATCH_ERRORS_DIR,
    BATCH_RESULTS_DIR,
    DEFAULT_CLASSIFICATION_OUTPUT_CSV,
)
from src.schema import ClassificationResult
from src.state import PipelineState
from src.submitter import get_client

logger = logging.getLogger(__name__)

RESULTS_DIR = BATCH_RESULTS_DIR
ERRORS_DIR = BATCH_ERRORS_DIR


def _download_file(client, file_id: str, dest: Path) -> Path:
    """Download a file from OpenAI and write it to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    dest.write_bytes(content.read())
    logger.info("Downloaded %s -> %s", file_id, dest.name)
    return dest


def _assistant_json_from_batch_body(body: dict) -> str | None:
    """Extract assistant JSON text from a batch line's `response.body`.

    Supports **Responses API** (`output` with `output_text` blocks) and legacy
    **Chat Completions** (`choices[0].message.content`) for older batch files.
    """
    out_items = body.get("output")
    if out_items is not None:
        parts: list[str] = []
        for item in out_items:
            if item.get("type") != "message":
                continue
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    parts.append(block.get("text") or "")
        text = "".join(parts).strip()
        if text:
            return text

    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content

    return None


def _usage_from_batch_body(body: dict) -> dict[str, int]:
    """Normalize per-response usage for cost aggregation (batch discount math)."""
    usage = body.get("usage") or {}
    if "input_tokens" in usage:
        inp_details = usage.get("input_tokens_details") or {}
        return {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "cached_tokens": int(inp_details.get("cached_tokens") or 0),
        }
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
    }


def _parse_result_line(line: dict) -> dict | None:
    """Extract classification fields and usage stats from one JSONL result line.

    Returns None if the line represents an error response.
    """
    custom_id = line.get("custom_id", "")
    response = line.get("response", {})
    body = response.get("body", {})

    if response.get("status_code") != 200:
        error = line.get("error", response.get("error", {}))
        logger.warning(
            "Non-200 for %s: %s",
            custom_id, error.get("message", "unknown error"),
        )
        return None

    content_str = _assistant_json_from_batch_body(body)
    if not content_str:
        logger.warning("No assistant output in response for %s", custom_id)
        return None

    try:
        parsed = json.loads(content_str)
        ClassificationResult.model_validate(parsed)
    except Exception:
        logger.warning("Validation failed for %s", custom_id, exc_info=True)
        return None

    u = _usage_from_batch_body(body)

    return {
        "custom_id": custom_id,
        "classification": parsed,
        "usage": {
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "cached_tokens": u["cached_tokens"],
        },
    }


def _append_to_output_csv(records: list[dict], output_path: Path) -> int:
    """Append parsed classification results directly to the production CSV.

    Writes the header only when the file does not yet exist or is empty.
    Returns the number of rows written.
    """
    if not records:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ClassificationResult.model_fields.keys())
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow(rec["classification"])

    return len(records)


def download_completed(
    state: PipelineState,
    output_path: Path = DEFAULT_CLASSIFICATION_OUTPUT_CSV,
) -> None:
    """Download results for all completed batches and append to production CSV.

    Skips batches that have already been fully processed (rows_written > 0).
    Handles crash-recovery: if the raw JSONL exists but rows_written is 0,
    re-parses and appends without re-downloading.
    """
    client = get_client()
    completed = state.completed_batches()

    if not completed:
        logger.info("No completed batches to download.")
        return

    for rec in completed:
        if rec.error_file_id:
            error_path = ERRORS_DIR / f"batch_{rec.batch_number:04d}_errors.jsonl"
            if not error_path.exists():
                _download_file(client, rec.error_file_id, error_path)

        if not rec.output_file_id:
            logger.warning("Batch %d completed but no output_file_id", rec.batch_number)
            continue

        result_path = RESULTS_DIR / f"batch_{rec.batch_number:04d}.jsonl"

        if result_path.exists() and rec.rows_written > 0:
            logger.info(
                "Batch %d already downloaded and written (%d rows). Skipping.",
                rec.batch_number, rec.rows_written,
            )
            continue

        if not result_path.exists():
            _download_file(client, rec.output_file_id, result_path)

        parsed_records: list[dict] = []
        batch_prompt_toks = 0
        batch_completion_toks = 0
        batch_cached_toks = 0

        with open(result_path, encoding="utf-8") as f:
            for line_str in f:
                line = json.loads(line_str.strip())
                result = _parse_result_line(line)
                if result:
                    parsed_records.append(result)
                    batch_prompt_toks += result["usage"]["prompt_tokens"]
                    batch_completion_toks += result["usage"]["completion_tokens"]
                    batch_cached_toks += result["usage"]["cached_tokens"]

        rows_written = _append_to_output_csv(parsed_records, output_path)
        rec.rows_written = rows_written

        state.total_prompt_tokens += batch_prompt_toks
        state.total_completion_tokens += batch_completion_toks
        state.total_cached_tokens += batch_cached_toks

        cache_rate = (
            batch_cached_toks / batch_prompt_toks * 100
            if batch_prompt_toks > 0 else 0.0
        )
        logger.info(
            "Batch %d: %d rows -> %s, %d prompt toks, %d cached (%.1f%% hit rate)",
            rec.batch_number, rows_written, output_path.name,
            batch_prompt_toks, batch_cached_toks, cache_rate,
        )

    state.save()


def collect_failed_custom_ids(state: PipelineState) -> list[str]:
    """Read all error files and collect custom_ids that need retry."""
    failed_ids: list[str] = []

    for rec in state.completed_batches() + state.failed_batches():
        error_path = ERRORS_DIR / f"batch_{rec.batch_number:04d}_errors.jsonl"
        if not error_path.exists():
            continue
        with open(error_path, encoding="utf-8") as f:
            for line_str in f:
                line = json.loads(line_str.strip())
                cid = line.get("custom_id", "")
                if cid:
                    failed_ids.append(cid)

    logger.info("Collected %d failed custom_ids for retry", len(failed_ids))
    return failed_ids
