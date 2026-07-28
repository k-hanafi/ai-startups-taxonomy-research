"""Immutable run discovery, smoke selection, and resume context."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from . import config
from .journal import (
    JournalState,
    RunArtifactPaths,
    RunLock,
    load_journal_state,
    read_journal_events,
    rebuild_derived_artifacts,
    replay_journal_events,
)
from .manifest import Manifest, ManifestRow, load_manifest, select_manifest_rows
from .paths import MANIFESTS_DIR, PROJECT_ROOT, RUNS_DIR
from .request_builder import (
    RequestSettings,
    request_fingerprint,
)

RUN_CONFIG_VERSION = 1
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class WorkflowError(ValueError):
    """A CLI workflow invariant is missing or unsafe."""


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    paths: RunArtifactPaths
    header: dict[str, Any]
    run_config: dict[str, Any]
    manifest_path: Path
    manifest: Manifest
    settings: RequestSettings
    state: JournalState
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SmokeGate:
    run_id: str
    run_dir: Path
    completed_at: str


def resolve_manifest_path(value: str | Path | None) -> Path:
    """Resolve an explicit manifest or discover the newest valid artifact."""
    if value is not None:
        path = _resolve_project_path(value)
        if not path.is_file():
            raise WorkflowError(f"manifest not found: {path}")
        load_manifest(path)
        return path

    candidates = sorted(
        MANIFESTS_DIR.glob("manifest_*.jsonl"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not candidates:
        raise WorkflowError(
            "no production manifest found; run "
            "'python -m two_pass_classifier build-manifest' first"
        )
    failures: list[str] = []
    for candidate in candidates:
        try:
            load_manifest(candidate)
        except Exception as exc:
            failures.append(f"{candidate.name}: {exc}")
            continue
        return candidate
    raise WorkflowError(
        "no valid production manifest found; newest error: "
        + (failures[0] if failures else "unknown")
    )


def make_run_id(
    kind: str,
    model: str,
    pass_b_effort: str,
    *,
    now: datetime | None = None,
    entropy: str | None = None,
) -> str:
    """Create a readable timestamped ID with collision-safe entropy."""
    if kind not in {"smoke", "run"}:
        raise ValueError("run kind must be smoke or run")
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S_%f")
    suffix = entropy or secrets.token_hex(3)
    model_short = model.removeprefix("gpt-").replace(".", "")
    return f"{kind}_{timestamp}_{model_short}_{pass_b_effort}_{suffix}"


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise WorkflowError(
            "run ID must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return run_id


def new_run_dir(run_id: str) -> Path:
    """Return a new run path or refuse an existing identity."""
    validated = validate_run_id(run_id)
    path = RUNS_DIR / validated
    if path.exists():
        journal = path / "events.jsonl"
        if journal.is_file():
            raise WorkflowError(
                f"run ID {validated!r} already exists at {path}.\n"
                "Continue it with: "
                f"'python -m two_pass_classifier resume {validated}'"
            )
        raise WorkflowError(
            f"run ID {validated!r} already has a directory at {path}, "
            "but no journal yet (an earlier attempt stopped before the run "
            "started). Remove that directory or choose a different --run-id."
        )
    return path


def run_dir(run_id: str) -> Path:
    validated = validate_run_id(run_id)
    path = RUNS_DIR / validated
    if not path.is_dir():
        raise WorkflowError(
            f"run {validated!r} was not found under {RUNS_DIR}"
        )
    return path


def select_smoke_manifest(
    parent: Manifest,
    *,
    count: int = config.SMOKE_COMPANY_COUNT,
) -> Manifest:
    """Select a stable source-balanced and evidence-length-spread sample."""
    if parent.row_count < count:
        raise WorkflowError(
            f"smoke requires {count} companies, but the manifest has "
            f"{parent.row_count}"
        )
    live = [row for row in parent.rows if row.evidence_source == "live"]
    dead = [row for row in parent.rows if row.evidence_source == "dead"]
    live_count, dead_count = _balanced_source_counts(
        len(live),
        len(dead),
        count,
    )
    selected_live = _length_spread(live, live_count, parent.manifest_sha256)
    selected_dead = _length_spread(dead, dead_count, parent.manifest_sha256)
    selected: list[ManifestRow] = []
    for index in range(max(len(selected_live), len(selected_dead))):
        if index < len(selected_live):
            selected.append(selected_live[index])
        if index < len(selected_dead):
            selected.append(selected_dead[index])
    if len(selected) != count:
        raise AssertionError("smoke selection did not produce the locked count")
    return select_manifest_rows(
        parent,
        [row.company_id for row in selected],
    )


def build_run_metadata(
    *,
    kind: str,
    run_id: str,
    manifest_path: Path,
    manifest: Manifest,
    settings: RequestSettings,
    parent_manifest_path: Path | None = None,
    parent_manifest: Manifest | None = None,
) -> dict[str, Any]:
    """Build the immutable user-facing run configuration."""
    if kind not in {"smoke", "full"}:
        raise ValueError("run metadata kind must be smoke or full")
    parent = parent_manifest or manifest
    parent_path = parent_manifest_path or manifest_path
    metadata: dict[str, Any] = {
        "run_config_version": RUN_CONFIG_VERSION,
        "kind": kind,
        "run_id": validate_run_id(run_id),
        "manifest_path": _stored_path(manifest_path),
        "parent_manifest_path": _stored_path(parent_path),
        "parent_manifest_sha256": parent.manifest_sha256,
        "semantic_request_fingerprint": request_fingerprint(settings),
        "model": settings.model,
        "pass_a_effort": config.PASS_A_EFFORT,
        "pass_b_effort": settings.pass_b_effort,
        "pass_a_max_output_tokens": settings.pass_a_max_output_tokens,
        "pass_b_max_output_tokens": settings.pass_b_max_output_tokens,
    }
    if kind == "smoke":
        metadata.update(
            {
                "selection_company_ids": [
                    row.company_id for row in manifest.rows
                ],
                "selection_source_counts": manifest.source_counts,
                "selection_strategy": (
                    "balanced evidence source, then spread by evidence length"
                ),
                "ai_family_stratified": False,
                "ai_family_note": (
                    "The manifest has no ground-truth AI-family labels."
                ),
            }
        )
    return metadata


def load_run_context(run_id: str) -> RunContext:
    """Load one consistent, read-only snapshot of a CLI run."""
    paths = RunArtifactPaths.from_run_dir(run_dir(run_id))
    events = tuple(read_journal_events(paths.journal))
    initial_state = replay_journal_events(events)
    header = initial_state.header
    if header is None:
        raise WorkflowError(
            f"run {run_id!r} has no journal header; it cannot be resumed safely"
        )
    raw_config = header.get("run_config")
    if not isinstance(raw_config, Mapping):
        raise WorkflowError(
            f"run {run_id!r} predates the production CLI configuration; "
            "start a new run"
        )
    run_config = dict(raw_config)
    if run_config.get("run_id") != run_id:
        raise WorkflowError(
            f"journal run ID {run_config.get('run_id')!r} does not match "
            f"directory {run_id!r}"
        )
    if run_config.get("run_config_version") != RUN_CONFIG_VERSION:
        raise WorkflowError(
            f"unsupported run configuration version "
            f"{run_config.get('run_config_version')!r}"
        )

    identity = header.get("request_identity")
    if not isinstance(identity, Mapping):
        raise WorkflowError("journal header is missing request_identity")
    settings = RequestSettings(
        model=str(identity.get("model") or ""),
        pass_b_effort=str(identity.get("pass_b_effort") or ""),
        pass_a_max_output_tokens=int(
            identity.get("pass_a_max_output_tokens") or 0
        ),
        pass_b_max_output_tokens=int(
            identity.get("pass_b_max_output_tokens") or 0
        ),
    )
    fingerprint = request_fingerprint(settings)
    if fingerprint != header.get("request_fingerprint"):
        raise WorkflowError(
            "current prompts, schemas, formatter, model, effort, or output caps "
            "do not match this run; start a new smoke and full run"
        )
    if run_config.get("semantic_request_fingerprint") != fingerprint:
        raise WorkflowError(
            "run metadata and journal request fingerprints disagree"
        )

    manifest_path = _resolve_project_path(str(run_config.get("manifest_path") or ""))
    if not manifest_path.is_file():
        raise WorkflowError(
            f"immutable run manifest is missing: {manifest_path}"
        )
    manifest = load_manifest(manifest_path)
    state = replay_journal_events(
        events,
        manifest=manifest,
        expected_fingerprint=fingerprint,
    )
    return RunContext(
        run_id=run_id,
        paths=paths,
        header=header,
        run_config=run_config,
        manifest_path=manifest_path,
        manifest=manifest,
        settings=settings,
        state=state,
        events=events,
    )


def repair_run_artifacts(context: RunContext) -> dict[str, Any]:
    """Heal and rebuild derived files under the run's exclusive lock."""
    fingerprint = request_fingerprint(context.settings)
    with RunLock(context.paths.lock):
        state = load_journal_state(
            context.paths.journal,
            manifest=context.manifest,
            expected_fingerprint=fingerprint,
            replay_mode="repair",
        )
        return rebuild_derived_artifacts(
            context.manifest,
            state,
            context.paths,
            stopped=False,
        )


def find_matching_smoke(
    *,
    parent_manifest: Manifest,
    settings: RequestSettings,
) -> SmokeGate | None:
    """Find the newest complete smoke with an identical semantic identity."""
    expected_fingerprint = request_fingerprint(settings)
    expected_selection = [
        row.company_id for row in select_smoke_manifest(parent_manifest).rows
    ]
    matches: list[SmokeGate] = []
    if not RUNS_DIR.is_dir():
        return None
    for candidate in RUNS_DIR.iterdir():
        if not candidate.is_dir():
            continue
        journal = candidate / "events.jsonl"
        if not journal.is_file():
            continue
        try:
            with journal.open("r", encoding="utf-8") as handle:
                header = json.loads(handle.readline())
            metadata = header.get("run_config")
            if not isinstance(metadata, Mapping):
                continue
            if metadata.get("kind") != "smoke":
                continue
            if (
                metadata.get("parent_manifest_sha256")
                != parent_manifest.manifest_sha256
            ):
                continue
            if (
                metadata.get("semantic_request_fingerprint")
                != expected_fingerprint
            ):
                continue
            if metadata.get("selection_company_ids") != expected_selection:
                continue
            context = load_run_context(candidate.name)
        except Exception:
            continue
        if context.manifest.row_count != config.SMOKE_COMPANY_COUNT:
            continue
        if len(context.state.completed) != config.SMOKE_COMPANY_COUNT:
            continue
        if context.state.latest_errors or not context.paths.final_csv.is_file():
            continue
        matches.append(
            SmokeGate(
                run_id=context.run_id,
                run_dir=context.paths.run_dir,
                completed_at=_latest_finished_at(context),
            )
        )
    if not matches:
        return None
    return max(matches, key=lambda match: (match.completed_at, match.run_id))


def _balanced_source_counts(
    live_count: int,
    dead_count: int,
    total: int,
) -> tuple[int, int]:
    target_live = total // 2
    target_dead = total - target_live
    selected_live = min(live_count, target_live)
    selected_dead = min(dead_count, target_dead)
    remaining = total - selected_live - selected_dead
    if remaining:
        add_live = min(remaining, live_count - selected_live)
        selected_live += add_live
        remaining -= add_live
    if remaining:
        add_dead = min(remaining, dead_count - selected_dead)
        selected_dead += add_dead
        remaining -= add_dead
    if remaining:
        raise WorkflowError(
            f"manifest cannot supply {total} distinct smoke companies"
        )
    return selected_live, selected_dead


def _length_spread(
    rows: list[ManifestRow],
    count: int,
    parent_hash: str,
) -> list[ManifestRow]:
    if count == 0:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            len(row.inputs.get("website_evidence", "")),
            _stable_rank(parent_hash, row.company_id),
        ),
    )
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    ]
    return [ordered[position] for position in positions]


def _stable_rank(parent_hash: str, company_id: str) -> str:
    return hashlib.sha256(
        f"{parent_hash}:{company_id}".encode("utf-8")
    ).hexdigest()


def _latest_finished_at(context: RunContext) -> str:
    timestamps = [
        str(event.get("finished_at"))
        for event in context.events
        if event.get("finished_at")
    ]
    return max(timestamps, default=str(context.header.get("created_at") or ""))


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _stored_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
