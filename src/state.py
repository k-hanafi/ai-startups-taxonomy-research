"""JSON checkpoint for pipeline state.

state.json tracks every batch through its full lifecycle so any subcommand
can resume exactly where it left off. Even after a crash, terminal close,
or overnight interruption. Without this, any failure means re-submitting
completed batches and paying twice.

Lifecycle stages per batch:
  prepared -> submitted -> completed | failed | expired

The file is rewritten atomically (write-to-temp then rename) so a crash
mid-write never corrupts the checkpoint.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_STATE_DIR = Path(__file__).resolve().parents[1] / "outputs"
STATE_FILE = _STATE_DIR / "state.json"

BatchStatus = Literal[
    "prepared",
    "submitted",
    "in_progress",
    "completed",
    "failed",
    "expired",
    "cancelled",
]

_BATCH_RECORD_FIELDS: set[str] = set()  # populated after class definition


@dataclass
class BatchRecord:
    """One batch's progress through the pipeline."""

    batch_number: int
    file_path: str
    row_range: str
    estimated_tokens: int = 0
    status: BatchStatus = "prepared"
    file_id: str = ""
    batch_id: str = ""
    output_file_id: str = ""
    error_file_id: str = ""
    request_count: int = 0
    completed_count: int = 0
    failed_count: int = 0


_BATCH_RECORD_FIELDS.update(f.name for f in fields(BatchRecord))


@dataclass
class PipelineState:
    """Full pipeline state, serialised to outputs/state.json."""

    run_id: str = ""
    model: str = ""
    total_companies: int = 0
    batches: dict[str, BatchRecord] = field(default_factory=dict)

    # Aggregate usage (populated during download)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0

    # -- Persistence -----------------------------------------------------------

    def save(self) -> None:
        """Atomically write state to disk."""
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        fd, tmp = tempfile.mkstemp(dir=_STATE_DIR, suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            Path(tmp).replace(STATE_FILE)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.debug("State saved to %s", STATE_FILE)

    @classmethod
    def load(cls) -> PipelineState:
        """Load state from disk, or return a fresh state if no file exists.

        Extra keys in state.json (from future versions or manual edits) are
        silently dropped so the load never crashes on unknown fields.
        """
        if not STATE_FILE.exists():
            logger.info("No existing state file. Starting fresh.")
            return cls()

        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state = cls(
            run_id=raw.get("run_id", ""),
            model=raw.get("model", ""),
            total_companies=raw.get("total_companies", 0),
            total_prompt_tokens=raw.get("total_prompt_tokens", 0),
            total_completion_tokens=raw.get("total_completion_tokens", 0),
            total_cached_tokens=raw.get("total_cached_tokens", 0),
        )
        for key, rec in raw.get("batches", {}).items():
            filtered = {k: v for k, v in rec.items() if k in _BATCH_RECORD_FIELDS}
            state.batches[key] = BatchRecord(**filtered)

        logger.info(
            "Loaded state: run_id=%s, %d batches tracked",
            state.run_id, len(state.batches),
        )
        return state

    # -- Convenience queries ---------------------------------------------------

    def pending_batches(self) -> list[BatchRecord]:
        """Batches that still need submission."""
        return [b for b in self.batches.values() if b.status == "prepared"]

    def in_flight_batches(self) -> list[BatchRecord]:
        """Batches submitted but not yet terminal."""
        return [
            b for b in self.batches.values()
            if b.status in ("submitted", "in_progress")
        ]

    def completed_batches(self) -> list[BatchRecord]:
        """Batches that finished successfully."""
        return [b for b in self.batches.values() if b.status == "completed"]

    def failed_batches(self) -> list[BatchRecord]:
        """Batches that failed or expired."""
        return [
            b for b in self.batches.values()
            if b.status in ("failed", "expired")
        ]

    def estimated_queued_tokens(self) -> int:
        """Sum of estimated tokens across all in-flight batches."""
        return sum(b.estimated_tokens for b in self.in_flight_batches())
