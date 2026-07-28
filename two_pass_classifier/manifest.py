"""Build one immutable, evidence-only production input manifest.

The artifact is JSON Lines. Its first line is a metadata record containing
source hashes and measured counts. Every later line is one manifest row. JSON
escaping preserves parsed CSV field values, including embedded commas, quotes,
newlines, and surrounding whitespace.

Professor-facing provenance fields ``company_alive`` and
``website_snapshot_date`` are frozen here as non-model metadata. They are
joined at build time from local scrape artifacts and never enter
``MODEL_INPUT_COLUMNS`` or ``input_hash``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .cohort import Cohort, compute_cohort
from .input_contract import MODEL_INPUT_COLUMNS, SOURCE_COLUMNS
from .paths import (
    DEFAULT_DEAD_INPUT,
    DEFAULT_DEAD_SCRAPE_PROCESSED,
    DEFAULT_LIVE_INPUT,
    DEFAULT_LIVE_RAW_RESULTS,
    MANIFESTS_DIR,
)

EvidenceSource = Literal["live", "dead"]
CompanyAlive = Literal["yes", "no"]
MANIFEST_VERSION = 2

_ISO_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_WAYBACK_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})\d{6}$")


class ManifestValidationError(ValueError):
    """Source data violates an immutable-manifest invariant."""


class ManifestCollisionError(FileExistsError):
    """A manifest identity path already contains different bytes."""


@dataclass(frozen=True, slots=True)
class SourceSummary:
    evidence_source: EvidenceSource
    filename: str
    file_sha256: str
    input_row_count: int
    included_row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_source": self.evidence_source,
            "filename": self.filename,
            "file_sha256": self.file_sha256,
            "input_row_count": self.input_row_count,
            "included_row_count": self.included_row_count,
        }


@dataclass(frozen=True, slots=True)
class ManifestRow:
    company_id: str
    company_name: str
    cohort: Cohort
    company_alive: CompanyAlive
    website_snapshot_date: str
    evidence_source: EvidenceSource
    source_row_number: int
    input_hash: str
    inputs: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "cohort": self.cohort,
            "company_alive": self.company_alive,
            "website_snapshot_date": self.website_snapshot_date,
            "evidence_source": self.evidence_source,
            "source_row_number": self.source_row_number,
            "input_hash": self.input_hash,
            "inputs": dict(self.inputs),
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    rows: tuple[ManifestRow, ...]
    sources: tuple[SourceSummary, ...]
    rows_sha256: str
    manifest_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def source_counts(self) -> dict[str, int]:
        return {
            source.evidence_source: source.included_row_count
            for source in self.sources
        }

    def header_dict(self) -> dict[str, Any]:
        return {
            "record_type": "manifest",
            "manifest_version": MANIFEST_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "rows_sha256": self.rows_sha256,
            "row_count": self.row_count,
            "source_counts": self.source_counts,
            "sources": [source.to_dict() for source in self.sources],
        }


def build_manifest(
    live_csv: str | Path = DEFAULT_LIVE_INPUT,
    dead_csv: str | Path = DEFAULT_DEAD_INPUT,
    *,
    live_raw_results: str | Path = DEFAULT_LIVE_RAW_RESULTS,
    dead_scrape_processed: str | Path = DEFAULT_DEAD_SCRAPE_PROCESSED,
) -> Manifest:
    """Build and validate the combined manifest without writing it.

    Blank live evidence is an eligibility filter. Blank dead evidence is an
    upstream contract failure because the dead input should already contain
    evidence-bearing rows. Every included row must join to a scrape or archive
    snapshot date; missing joins fail loudly.
    """
    live_dates = _load_live_snapshot_dates(Path(live_raw_results))
    dead_dates = _load_dead_snapshot_dates(Path(dead_scrape_processed))

    live_rows, live_summary = _read_source(
        Path(live_csv),
        "live",
        filter_blank_evidence=True,
        snapshot_dates=live_dates,
    )
    dead_rows, dead_summary = _read_source(
        Path(dead_csv),
        "dead",
        filter_blank_evidence=False,
        snapshot_dates=dead_dates,
    )

    live_ids = {row.company_id for row in live_rows}
    dead_ids = {row.company_id for row in dead_rows}
    overlap = sorted(live_ids & dead_ids)
    if overlap:
        raise ManifestValidationError(
            "live and dead inputs overlap on company_id "
            f"{overlap[0]!r} ({len(overlap)} overlapping ID(s))"
        )

    rows = tuple((*live_rows, *dead_rows))
    sources = (live_summary, dead_summary)
    rows_sha256 = _rows_sha256(rows)
    identity = {
        "manifest_version": MANIFEST_VERSION,
        "rows_sha256": rows_sha256,
        "sources": [source.to_dict() for source in sources],
    }
    return Manifest(
        rows=rows,
        sources=sources,
        rows_sha256=rows_sha256,
        manifest_sha256=_sha256_json(identity),
    )


def write_manifest(
    manifest: Manifest,
    output_dir: str | Path = MANIFESTS_DIR,
) -> Path:
    """Publish a content-addressed manifest with an atomic same-dir replace."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"manifest_{manifest.manifest_sha256}.jsonl"
    if _existing_artifact_matches(path, manifest):
        return path
    if path.exists() and _existing_artifact_is_valid(path):
        raise ManifestCollisionError(
            f"refusing to overwrite different manifest artifact: {path}"
        )

    temporary = directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(_compact_json(manifest.header_dict()) + "\n")
            for row in manifest.rows:
                handle.write(
                    _compact_json({"record_type": "row", **row.to_dict()}) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

        # Recheck after the temporary write in case another builder published
        # while this process was serializing the same manifest.
        if _existing_artifact_matches(path, manifest):
            return path
        if path.exists() and _existing_artifact_is_valid(path):
            raise ManifestCollisionError(
                f"refusing to overwrite different manifest artifact: {path}"
            )
        os.replace(temporary, path)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_and_write_manifest(
    live_csv: str | Path = DEFAULT_LIVE_INPUT,
    dead_csv: str | Path = DEFAULT_DEAD_INPUT,
    output_dir: str | Path = MANIFESTS_DIR,
    *,
    live_raw_results: str | Path = DEFAULT_LIVE_RAW_RESULTS,
    dead_scrape_processed: str | Path = DEFAULT_DEAD_SCRAPE_PROCESSED,
) -> Path:
    """Build, validate, and immutably write one combined manifest."""
    return write_manifest(
        build_manifest(
            live_csv,
            dead_csv,
            live_raw_results=live_raw_results,
            dead_scrape_processed=dead_scrape_processed,
        ),
        output_dir,
    )


def select_manifest_rows(
    manifest: Manifest,
    company_ids: list[str] | tuple[str, ...],
) -> Manifest:
    """Build a deterministic immutable manifest from selected parent rows."""
    selected_ids = tuple(company_ids)
    if not selected_ids:
        raise ManifestValidationError("manifest selection cannot be empty")
    if len(selected_ids) != len(set(selected_ids)):
        raise ManifestValidationError("manifest selection contains duplicate IDs")

    by_id = {row.company_id: row for row in manifest.rows}
    missing = [company_id for company_id in selected_ids if company_id not in by_id]
    if missing:
        raise ManifestValidationError(
            f"manifest selection contains unknown company_id {missing[0]!r}"
        )
    rows = tuple(by_id[company_id] for company_id in selected_ids)
    selected_counts = {
        source: sum(row.evidence_source == source for row in rows)
        for source in ("live", "dead")
    }
    sources = tuple(
        SourceSummary(
            evidence_source=source.evidence_source,
            filename=source.filename,
            file_sha256=source.file_sha256,
            input_row_count=source.input_row_count,
            included_row_count=selected_counts[source.evidence_source],
        )
        for source in manifest.sources
    )
    rows_sha256 = _rows_sha256(rows)
    identity = {
        "manifest_version": MANIFEST_VERSION,
        "rows_sha256": rows_sha256,
        "sources": [source.to_dict() for source in sources],
    }
    return Manifest(
        rows=rows,
        sources=sources,
        rows_sha256=rows_sha256,
        manifest_sha256=_sha256_json(identity),
    )


def load_manifest(path: str | Path) -> Manifest:
    """Load an artifact and verify its counts and content hashes."""
    artifact = Path(path)
    with artifact.open("r", encoding="utf-8", newline="") as handle:
        first = handle.readline()
        if not first:
            raise ManifestValidationError(f"manifest is empty: {artifact}")
        header = json.loads(first)
        if header.get("record_type") != "manifest":
            raise ManifestValidationError("first JSONL record is not manifest metadata")
        if header.get("manifest_version") != MANIFEST_VERSION:
            raise ManifestValidationError(
                f"unsupported manifest version {header.get('manifest_version')!r}"
            )

        rows: list[ManifestRow] = []
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.pop("record_type", None) != "row":
                raise ManifestValidationError(
                    f"JSONL line {line_number} is not a row record"
                )
            rows.append(ManifestRow(**raw))

    sources = tuple(SourceSummary(**source) for source in header["sources"])
    loaded = Manifest(
        rows=tuple(rows),
        sources=sources,
        rows_sha256=str(header["rows_sha256"]),
        manifest_sha256=str(header["manifest_sha256"]),
    )
    _verify_loaded_manifest(loaded, header)
    return loaded


def sha256_file(path: str | Path) -> str:
    """Hash a source or artifact file as raw bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _company_alive_for(evidence_source: EvidenceSource) -> CompanyAlive:
    return "yes" if evidence_source == "live" else "no"


def _load_live_snapshot_dates(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"live raw crawl results not found: {path}")

    dates: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"live raw_results.jsonl line {line_number} is not valid JSON"
                ) from exc
            if not raw.get("ok"):
                continue
            org_uuid = str(raw.get("org_uuid") or "").strip()
            if not org_uuid:
                continue
            requested_at = raw.get("requested_at")
            if not isinstance(requested_at, str) or not requested_at.strip():
                raise ManifestValidationError(
                    f"live raw_results.jsonl line {line_number} org_uuid "
                    f"{org_uuid!r} has blank requested_at"
                )
            # Last successful crawl wins: classifier_input keeps the last
            # processed row per org_uuid, so the professor date must match.
            dates[org_uuid] = _date_from_requested_at(
                requested_at,
                context=(
                    f"live raw_results.jsonl line {line_number} org_uuid "
                    f"{org_uuid!r}"
                ),
            )
    return dates


def _load_dead_snapshot_dates(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"dead scrape processed CSV not found: {path}")

    dates: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "org_uuid" not in fieldnames or "snapshot_ts" not in fieldnames:
            raise ManifestValidationError(
                "dead scrape processed CSV must include org_uuid and snapshot_ts"
            )
        for row_number, raw in enumerate(reader, start=2):
            org_uuid = str(raw.get("org_uuid") or "").strip()
            if not org_uuid or org_uuid in dates:
                continue
            snapshot_ts = str(raw.get("snapshot_ts") or "").strip()
            if not snapshot_ts:
                raise ManifestValidationError(
                    f"dead scrape processed row {row_number} org_uuid "
                    f"{org_uuid!r} has blank snapshot_ts"
                )
            dates[org_uuid] = _date_from_snapshot_ts(
                snapshot_ts,
                context=(
                    f"dead scrape processed row {row_number} org_uuid "
                    f"{org_uuid!r}"
                ),
            )
    return dates


def _date_from_requested_at(value: str, *, context: str) -> str:
    match = _ISO_DATE_RE.match(value.strip())
    if match is None:
        raise ManifestValidationError(
            f"{context} has unparseable requested_at {value!r}"
        )
    return match.group(1)


def _date_from_snapshot_ts(value: str, *, context: str) -> str:
    match = _WAYBACK_TS_RE.match(value.strip())
    if match is None:
        raise ManifestValidationError(
            f"{context} has unparseable snapshot_ts {value!r}"
        )
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def _read_source(
    path: Path,
    evidence_source: EvidenceSource,
    *,
    filter_blank_evidence: bool,
    snapshot_dates: dict[str, str],
) -> tuple[list[ManifestRow], SourceSummary]:
    if not path.is_file():
        raise FileNotFoundError(f"{evidence_source} classifier input not found: {path}")

    before = path.stat()
    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    input_row_count = 0
    company_alive = _company_alive_for(evidence_source)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in SOURCE_COLUMNS if column not in fieldnames]
        if missing:
            raise ManifestValidationError(
                f"{evidence_source} input is missing required columns: {missing}"
            )
        if len(fieldnames) != len(set(fieldnames)):
            raise ManifestValidationError(
                f"{evidence_source} input contains duplicate column names"
            )

        for source_row_number, raw in enumerate(reader, start=2):
            input_row_count += 1
            values = _source_values(raw, evidence_source, source_row_number)
            evidence = values["website_evidence"]
            if not evidence.strip():
                if filter_blank_evidence:
                    continue
                raise ManifestValidationError(
                    f"{evidence_source} row {source_row_number} has blank "
                    "website_evidence"
                )

            company_id = values["org_uuid"]
            if not company_id.strip():
                raise ManifestValidationError(
                    f"{evidence_source} row {source_row_number} has blank org_uuid"
                )
            if company_id in seen_ids:
                raise ManifestValidationError(
                    f"{evidence_source} input contains duplicate company_id "
                    f"{company_id!r}"
                )
            seen_ids.add(company_id)

            snapshot_date = snapshot_dates.get(company_id)
            if snapshot_date is None:
                raise ManifestValidationError(
                    f"{evidence_source} row {source_row_number} company_id "
                    f"{company_id!r} has no website_snapshot_date join"
                )

            rows.append(
                ManifestRow(
                    company_id=company_id,
                    company_name=values["name"],
                    cohort=compute_cohort(values["founded_date"]),
                    company_alive=company_alive,
                    website_snapshot_date=snapshot_date,
                    evidence_source=evidence_source,
                    source_row_number=source_row_number,
                    input_hash=_input_hash(values),
                    inputs=values,
                )
            )

    file_sha256 = sha256_file(path)
    after = path.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise ManifestValidationError(
            f"{evidence_source} input changed while the manifest was being built"
        )

    summary = SourceSummary(
        evidence_source=evidence_source,
        filename=path.name,
        file_sha256=file_sha256,
        input_row_count=input_row_count,
        included_row_count=len(rows),
    )
    return rows, summary


def _source_values(
    raw: dict[str | None, str | None],
    evidence_source: EvidenceSource,
    source_row_number: int,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for column in SOURCE_COLUMNS:
        value = raw.get(column)
        if value is None:
            raise ManifestValidationError(
                f"{evidence_source} row {source_row_number} has no value for "
                f"required column {column!r}"
            )
        values[column] = value
    return values


def _input_hash(values: dict[str, str]) -> str:
    ordered_values = [[column, values[column]] for column in MODEL_INPUT_COLUMNS]
    return _sha256_json(ordered_values)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _rows_sha256(rows: tuple[ManifestRow, ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        _update_row_digest(digest, row.to_dict())
    return digest.hexdigest()


def _update_row_digest(
    digest: Any,
    row: dict[str, Any],
) -> None:
    digest.update(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _existing_artifact_matches(path: Path, manifest: Manifest) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = json.loads(handle.readline())
            if header != manifest.header_dict():
                return False

            digest = hashlib.sha256()
            row_count = 0
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.pop("record_type", None) != "row":
                    return False
                _update_row_digest(digest, raw)
                row_count += 1
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return (
        row_count == manifest.row_count
        and digest.hexdigest() == manifest.rows_sha256
    )


def _existing_artifact_is_valid(path: Path) -> bool:
    try:
        load_manifest(path)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return False
    return True


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_loaded_manifest(manifest: Manifest, header: dict[str, Any]) -> None:
    if manifest.row_count != header.get("row_count"):
        raise ManifestValidationError(
            f"manifest row count mismatch: header={header.get('row_count')}, "
            f"actual={manifest.row_count}"
        )
    if manifest.source_counts != header.get("source_counts"):
        raise ManifestValidationError("manifest source counts do not match metadata")

    rows_sha256 = _rows_sha256(manifest.rows)
    if rows_sha256 != manifest.rows_sha256:
        raise ManifestValidationError("manifest rows_sha256 does not match row content")
    identity = {
        "manifest_version": MANIFEST_VERSION,
        "rows_sha256": rows_sha256,
        "sources": [source.to_dict() for source in manifest.sources],
    }
    if _sha256_json(identity) != manifest.manifest_sha256:
        raise ManifestValidationError(
            "manifest_sha256 does not match source and row metadata"
        )
