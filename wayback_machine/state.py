"""Crash-safe resume primitives for the scrape.

Three things make a multi-hour paid run safe to interrupt and resume:

1. ``ExtractState`` — a tiny JSON checkpoint written atomically (temp file +
   ``os.replace``) so a crash mid-write can never corrupt it.
2. ``heal_jsonl_tail`` — on startup, repair a half-flushed final line in the
   append-only raw log left by a power loss between ``write`` and ``fsync``.
3. ``completed_ids_from_jsonl`` + ``reconcile_extract_state`` — read which
   companies are already done and rebuild counters from JSONL so budget state
   cannot drift from the append-only log.

Adapted from the proven helpers in ``src/tavily_crawl.py``.
"""

from __future__ import annotations

import json
import os
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExtractState:
    """Persistent resume + budget state for one extract run.

    Tavily ``/extract`` returns no usage field, so spend is tracked by counting
    successful extractions (basic billing is 1 credit per 5).
    """

    successful: int = 0
    empty: int = 0
    failed: int = 0
    last_org_uuid: str = ""
    updated_at: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "ExtractState":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        """Persist via temp file + atomic rename so a crash cannot corrupt it."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self), indent=2).encode("utf-8")
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)


def heal_jsonl_tail(path: str | Path) -> int:
    """Truncate any unterminated partial line at the tail of a JSONL file.

    Returns the number of bytes removed (0 if the file was already clean or just
    missing its final newline, which we append).
    """
    p = Path(path)
    if not p.exists():
        return 0
    size = p.stat().st_size
    if size == 0:
        return 0

    chunk_size = 4096
    last_newline = -1
    position = size
    with p.open("rb") as fh:
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            fh.seek(position)
            idx = fh.read(read_size).rfind(b"\n")
            if idx >= 0:
                last_newline = position + idx
                break
        tail_start = last_newline + 1
        fh.seek(tail_start)
        tail = fh.read()

    if not tail:
        return 0

    try:
        json.loads(tail.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        with p.open("rb+") as fh:
            fh.truncate(tail_start)
        return len(tail)

    with p.open("ab") as fh:
        fh.write(b"\n")
    return 0


@dataclass
class OutcomeTally:
    """Aggregate counters derived from the append-only snapshots JSONL."""

    successful: int = 0
    empty: int = 0
    failed: int = 0
    completed_ids: set[str] = field(default_factory=set)


def tally_outcomes_from_jsonl(path: str | Path) -> OutcomeTally:
    """Single pass over JSONL to rebuild counters and the completed-id set."""
    p = Path(path)
    tally = OutcomeTally()
    if not p.exists():
        return tally

    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            org_uuid = str(obj.get("org_uuid", "")).strip()
            ok = obj.get("ok")
            retryable = obj.get("retryable")
            status = str(obj.get("status", ""))

            if org_uuid and (ok is True or retryable is False):
                tally.completed_ids.add(org_uuid)

            if ok is True:
                if status == "success":
                    tally.successful += 1
                elif status in ("empty_results", "thin_evidence"):
                    tally.empty += 1
            elif ok is False:
                tally.failed += 1

    return tally


def reconcile_extract_state(state: ExtractState, jsonl_path: str | Path) -> OutcomeTally:
    """Overwrite counter fields from JSONL so resume state cannot drift."""
    tally = tally_outcomes_from_jsonl(jsonl_path)
    state.successful = tally.successful
    state.empty = tally.empty
    state.failed = tally.failed
    return tally


def processed_ids_from_csv(path: str | Path) -> set[str]:
    """Return org_uuids already present in the processed evidence CSV."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return set()

    ids: set[str] = set()
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            org_uuid = str(row.get("org_uuid", "")).strip()
            if org_uuid:
                ids.add(org_uuid)
    return ids


def completed_ids_from_jsonl(path: str | Path) -> set[str]:
    """Return org_uuids already finished (ok, or a terminal non-retryable error).

    Retryable failures are intentionally excluded so a resumed run retries them.
    """
    p = Path(path)
    if not p.exists():
        return set()

    completed: set[str] = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            org_uuid = str(obj.get("org_uuid", "")).strip()
            if org_uuid and (obj.get("ok") is True or obj.get("retryable") is False):
                completed.add(org_uuid)
    return completed
