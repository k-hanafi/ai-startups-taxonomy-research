"""Normalize Responses API usage fields for eval run records.

Matches the mapping in ``src/downloader._usage_from_batch_body`` so eval
cost math and production cost math read the same cached-token signal.
Missing cache details are treated as 0 (not null): the API omits the
field when nothing was cached, and inventing a production hit rate would
be worse than under-counting.
"""

from __future__ import annotations

from typing import Any


def cached_tokens_from_usage(usage: Any) -> int:
    """Return ``cached_tokens`` from a Responses (or Chat) usage object.

    Offline-safe: ``None`` usage or missing details → ``0``.
    """
    if usage is None:
        return 0

    # Responses API (SDK object or dict): input_tokens_details.cached_tokens
    details = _get(usage, "input_tokens_details")
    if details is not None:
        return int(_get(details, "cached_tokens") or 0)

    # Chat Completions fallback: prompt_tokens_details.cached_tokens
    prompt_details = _get(usage, "prompt_tokens_details")
    if prompt_details is not None:
        return int(_get(prompt_details, "cached_tokens") or 0)

    return 0


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
