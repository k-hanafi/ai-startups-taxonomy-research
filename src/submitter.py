"""Fault-tolerant batch file upload and batch job creation.

All OpenAI API calls are wrapped in tenacity's @retry decorator with random
exponential backoff. The jitter prevents thundering herd. Concurrent uploads
hitting a rate limit won't retry simultaneously and re-trigger the same
limit. Each batch is tagged with metadata for traceability in the OpenAI
dashboard.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from openai import BadRequestError, OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.config import DEFAULT_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)


class BillingLimitError(RuntimeError):
    """OpenAI returned billing_hard_limit_reached; fix org/project budget then resume."""

    pass


def _bad_request_error_code(exc: BadRequestError) -> str | None:
    """Extract OpenAI error.code from a BadRequestError body, if present."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, str):
                return code
    # SDK / version fallbacks
    text = str(exc).lower()
    if "billing_hard_limit_reached" in text or "billing hard limit" in text:
        return "billing_hard_limit_reached"
    return None


def get_client() -> OpenAI:
    """Create an OpenAI client using the configured API key."""
    return OpenAI(api_key=OPENAI_API_KEY)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def upload_batch_file(client: OpenAI, file_path: str | Path) -> str:
    """Upload a JSONL file for batch processing.

    Returns the file ID.
    Retries up to 6 times with random exponential backoff (1-60 s)
    to survive transient 429s and server errors.
    """
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="batch")
    logger.info("Uploaded %s -> file_id=%s", Path(file_path).name, response.id)
    return response.id


@retry(
    retry=retry_if_not_exception_type(BillingLimitError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def create_batch(
    client: OpenAI,
    file_id: str,
    *,
    run_id: str,
    batch_number: int,
    total_batches: int,
    row_range: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Create a batch job from an uploaded file.

    Returns the batch ID.
    Metadata tags attach run context. Visible in the OpenAI dashboard
    and useful for debugging overnight runs.

    Raises:
        BillingLimitError: Monthly billing hard limit reached — not retried.
    """
    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "run_id": run_id,
                "batch_number": f"{batch_number}/{total_batches}",
                "row_range": row_range,
                "model": model,
            },
        )
    except BadRequestError as e:
        if _bad_request_error_code(e) == "billing_hard_limit_reached":
            raise BillingLimitError(
                "OpenAI billing hard limit reached (monthly budget cap). "
                "Raise the limit at https://platform.openai.com/settings/organization/limits "
                "(and check project budgets), then resume with: python classify.py submit"
            ) from e
        raise

    logger.info(
        "Created batch %s  (%s)  [%s/%s]",
        batch.id, row_range, batch_number, total_batches,
    )
    return batch.id


def generate_run_id(model: str = DEFAULT_MODEL) -> str:
    """Generate a run ID from the model name and current UTC timestamp."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H%M")
    return f"v2-{model}-{ts}"


