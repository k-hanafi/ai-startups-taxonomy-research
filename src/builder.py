"""Build JSONL batch files with structured output schema injection.

Each JSONL line is a complete OpenAI Responses API request (`POST /v1/responses`)
with an identical static prefix (instructions + Pydantic-generated JSON schema in
`text.format` + prompt_cache_key) followed by one variable user string. The
identical prefix is the structural prerequisite for prompt caching. It is
guaranteed by construction, not by developer discipline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    MAX_FILE_SIZE_MB,
    MAX_OUTPUT_TOKENS,
    PROMPT_CACHE_KEY,
)
from src.formatter import build_custom_id, format_user_message
from src.schema import ClassificationResult

logger = logging.getLogger(__name__)

PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "system_classifier_prompt.txt"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "batch_requests"


def load_system_prompt() -> str:
    """Read the classification system prompt from disk."""
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def _openai_strict_schema() -> dict:
    """Generate an OpenAI-compatible JSON schema with strict mode requirements.

    Pydantic's model_json_schema() omits additionalProperties, but OpenAI's
    strict mode requires it on every object. This post-processes the schema
    to satisfy that contract.
    """
    schema = ClassificationResult.model_json_schema()
    _add_additional_properties_false(schema)
    return schema


def _add_additional_properties_false(node: dict) -> None:
    """Recursively set additionalProperties: false on all object nodes."""
    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
    for value in node.values():
        if isinstance(value, dict):
            _add_additional_properties_false(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _add_additional_properties_false(item)


def responses_text_format_json_schema(schema: dict) -> dict:
    """Responses API structured-output config (`text` parameter body fragment)."""
    return {
        "format": {
            "type": "json_schema",
            "name": "ClassificationResult",
            "strict": True,
            "schema": schema,
        }
    }


def build_request_body(
    user_message: str,
    custom_id: str,
    system_prompt: str,
    schema: dict,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Build one JSONL line for the Batch API (Responses endpoint).

    The body includes the Pydantic-generated JSON schema under text.format,
    plus prompt_cache_key so all requests share cache routing.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": system_prompt,
            "input": user_message,
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "text": responses_text_format_json_schema(schema),
        },
    }


def build_batch_files(
    csv_path: str | Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    row_slice: slice | None = None,
) -> list[Path]:
    """Read the dataset CSV and write JSONL batch files to outputs/batch_requests/.

    Args:
        csv_path: Path to the input CSV.
        model: Model name for request bodies.
        batch_size: Number of requests per JSONL file.
        row_slice: Optional slice to process a subset of rows.

    Returns:
        List of paths to the written JSONL files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if row_slice is not None:
        df = df.iloc[row_slice]

    system_prompt = load_system_prompt()
    schema = _openai_strict_schema()
    written_files: list[Path] = []
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    for batch_start in range(0, len(df), batch_size):
        batch_df = df.iloc[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        file_path = OUTPUT_DIR / f"batch_{batch_num:04d}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            for row_tuple in batch_df.itertuples(index=False):
                row_dict = row_tuple._asdict()
                user_msg = format_user_message(row_dict)
                cid = build_custom_id(str(row_dict.get("org_uuid", "")))
                body = build_request_body(user_msg, cid, system_prompt, schema, model)
                f.write(json.dumps(body, ensure_ascii=False) + "\n")

        file_size = file_path.stat().st_size
        if file_size > max_bytes:
            logger.warning(
                "%s is %.1f MB (limit %d MB). Reduce batch_size.",
                file_path.name, file_size / 1024 / 1024, MAX_FILE_SIZE_MB,
            )

        written_files.append(file_path)
        logger.info("Wrote %s  (%d requests)", file_path.name, len(batch_df))

    return written_files
