"""Batch result downloader with error-file recovery and cache hit reporting.

Batch output order is not guaranteed to match input order. Positional
matching would silently corrupt the dataset at 267K rows. Results are
matched to inputs exclusively by custom_id. error_file_id is downloaded
separately so failed and expired requests are never lost.

Per-response cached_tokens from the usage object are aggregated so the
final report can show actual dollars saved from prompt caching.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.schema import ClassificationResult
from src.state import PipelineState
from src.submitter import get_client

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = _PROJECT_ROOT / "outputs" / "batch_results"
ERRORS_DIR = _PROJECT_ROOT / "outputs" / "batch_errors"
OUTPUTS_DIR = _PROJECT_ROOT / "outputs" / "batch_outputs"


def _download_file(client, file_id: str, dest: Path) -> Path:
    """Download a file from OpenAI and write it to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    dest.write_bytes(content.read())
    logger.info("Downloaded %s -> %s", file_id, dest.name)
    return dest


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

    choices = body.get("choices", [])
    if not choices:
        logger.warning("No choices in response for %s", custom_id)
        return None

    message = choices[0].get("message", {})
    content_str = message.get("content", "")

    try:
        parsed = json.loads(content_str)
        ClassificationResult.model_validate(parsed)
    except Exception:
        logger.warning("Validation failed for %s", custom_id, exc_info=True)
        return None

    usage = body.get("usage", {})
    prompt_details = usage.get("prompt_tokens_details", {})

    return {
        "custom_id": custom_id,
        "classification": parsed,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": prompt_details.get("cached_tokens", 0),
        },
    }


def _write_batch_csv(records: list[dict], batch_num: int) -> Path:
    """Write parsed classification results to a per-batch CSV."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"batch_{batch_num:04d}.csv"

    if not records:
        logger.warning("No valid records for batch %d. Skipping CSV.", batch_num)
        return path

    fieldnames = list(ClassificationResult.model_fields.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec["classification"])

    logger.info("Wrote %d rows -> %s", len(records), path.name)
    return path


def download_completed(state: PipelineState) -> None:
    """Download results for all completed batches and update usage stats.

    Skips batches whose output files have already been downloaded (based on
    whether the local result file exists).
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

        if result_path.exists():
            logger.info("Batch %d already downloaded. Skipping.", rec.batch_number)
            continue

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

        _write_batch_csv(parsed_records, rec.batch_number)

        state.total_prompt_tokens += batch_prompt_toks
        state.total_completion_tokens += batch_completion_toks
        state.total_cached_tokens += batch_cached_toks

        cache_rate = (
            batch_cached_toks / batch_prompt_toks * 100
            if batch_prompt_toks > 0 else 0.0
        )
        logger.info(
            "Batch %d: %d results, %d prompt toks, %d cached (%.1f%% hit rate)",
            rec.batch_number, len(parsed_records),
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
