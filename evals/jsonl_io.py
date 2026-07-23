"""Canonical JSONL read/append for eval run artifacts.

Policy: tolerate only a malformed *final* line (truncated mid-append). Fail
loudly on malformed *interior* lines so silent data loss cannot hide
corruption in the middle of predictions.jsonl or a Pass A bank.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class MalformedJSONLError(ValueError):
    """A non-final JSONL line could not be parsed."""


def iter_jsonl(
    path: Path,
    *,
    tolerate_truncated_final: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file.

    Empty / whitespace-only lines are skipped. When
    ``tolerate_truncated_final`` is True, a malformed final non-empty line
    is logged and skipped (crash mid-append). A malformed interior line
    always raises ``MalformedJSONLError``.
    """
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    # Drop trailing blank lines so "final" means the last content line.
    while lines and not lines[-1].strip():
        lines.pop()
    n = len(lines)
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            is_final = i == n - 1
            if tolerate_truncated_final and is_final:
                logger.warning(
                    "Skipping truncated final JSONL line in %s (%s)",
                    path, exc,
                )
                return
            raise MalformedJSONLError(
                f"Malformed JSONL line {i + 1} in {path}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise MalformedJSONLError(
                f"JSONL line {i + 1} in {path} is not a JSON object"
            )
        yield obj


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as a single JSONL line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
