"""Assemble and write the exact professor-facing classification CSV.

``sources_used`` is a compact JSON array stored inside one CSV cell, for
example ``["website_evidence","short_description"]``. The standard CSV writer
quotes that cell as needed, so commas or quotes cannot make the representation
ambiguous. Sources keep first-use order from Pass A followed by Pass B, with
duplicates removed.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifest import ManifestRow
from .schema import PassAResult, PassBAINativeResult, PassBNotAINativeResult

PROFESSOR_CSV_COLUMNS: tuple[str, ...] = (
    "company_id",
    "company_name",
    "cohort",
    "company_alive",
    "website_snapshot_date",
    "ai_native",
    "subclass",
    "rad_score",
    "ai_native_confidence",
    "subclass_confidence",
    "rad_confidence",
    "sources_used",
    "ai_native_reasoning",
    "subclass_reasoning",
    "rad_reasoning",
    "ai_native_critique",
    "subclass_critique",
    "rad_critique",
)
FINAL_CSV_COLUMNS = PROFESSOR_CSV_COLUMNS


def assemble_professor_row(
    manifest_row: ManifestRow | Mapping[str, Any],
    pass_a: PassAResult | Mapping[str, Any],
    pass_b: PassBAINativeResult | PassBNotAINativeResult | Mapping[str, Any],
    confidence: float | None,
) -> dict[str, Any]:
    """Combine validated pass outputs into exactly the 18 public fields."""
    result_a = PassAResult.model_validate(_as_mapping(pass_a))
    if result_a.ai_native == 1:
        result_b = PassBAINativeResult.model_validate(_as_mapping(pass_b))
        rad_score = result_b.rad_score
        rad_confidence: int | str = result_b.rad_confidence
        rad_critique = result_b.rad_critique
    else:
        result_b = PassBNotAINativeResult.model_validate(_as_mapping(pass_b))
        rad_score = "RAD-NA"
        rad_confidence = ""
        rad_critique = ""

    (
        company_id,
        company_name,
        cohort,
        company_alive,
        website_snapshot_date,
    ) = _manifest_identity(manifest_row)
    confidence_value: float | str = _validated_confidence(confidence)
    sources = stable_source_union(result_a.sources_used, result_b.sources_used)

    return {
        "company_id": company_id,
        "company_name": company_name,
        "cohort": cohort,
        "company_alive": company_alive,
        "website_snapshot_date": website_snapshot_date,
        "ai_native": result_a.ai_native,
        "subclass": result_b.subclass,
        "rad_score": rad_score,
        "ai_native_confidence": confidence_value,
        "subclass_confidence": result_b.subclass_confidence,
        "rad_confidence": rad_confidence,
        "sources_used": sources,
        "ai_native_reasoning": result_a.ai_native_reasoning,
        "subclass_reasoning": result_b.subclass_reasoning,
        "rad_reasoning": result_b.rad_reasoning,
        "ai_native_critique": result_a.ai_native_critique,
        "subclass_critique": result_b.subclass_critique,
        "rad_critique": rad_critique,
    }


def stable_source_union(
    pass_a_sources: Iterable[str],
    pass_b_sources: Iterable[str],
) -> str:
    """Return a compact JSON array with stable first-use deduplication."""
    ordered: list[str] = []
    seen: set[str] = set()
    for source in (*pass_a_sources, *pass_b_sources):
        value = str(source)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def write_professor_csv(
    rows: Iterable[Mapping[str, Any]],
    output_csv: str | Path,
) -> Path:
    """Write only the locked public columns, discarding operational metadata."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(PROFESSOR_CSV_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            missing = [column for column in PROFESSOR_CSV_COLUMNS if column not in row]
            if missing:
                raise ValueError(
                    f"professor row {index} is missing required columns: {missing}"
                )
            writer.writerow(
                {
                    column: "" if row[column] is None else row[column]
                    for column in PROFESSOR_CSV_COLUMNS
                }
            )
    return path


def export_professor_csv(
    rows: Iterable[Mapping[str, Any]],
    output_csv: str | Path,
) -> Path:
    """Public name for writing the locked professor-facing artifact."""
    return write_professor_csv(rows, output_csv)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"expected a Pydantic result or mapping, got {type(value).__name__}")


def _manifest_identity(
    row: ManifestRow | Mapping[str, Any],
) -> tuple[str, str, str, str, str]:
    if isinstance(row, ManifestRow):
        return (
            row.company_id,
            row.company_name,
            row.cohort,
            row.company_alive,
            row.website_snapshot_date,
        )

    if "company_id" in row:
        return (
            str(row["company_id"]),
            str(row["company_name"]),
            str(row["cohort"]),
            str(row["company_alive"]),
            str(row["website_snapshot_date"]),
        )
    inputs = row.get("inputs")
    if isinstance(inputs, Mapping):
        return (
            str(inputs["org_uuid"]),
            str(inputs["name"]),
            str(row["cohort"]),
            str(row["company_alive"]),
            str(row["website_snapshot_date"]),
        )
    raise ValueError("manifest row does not expose company identity and cohort")


def _validated_confidence(value: float | None) -> float | str:
    if value is None:
        return ""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"ai_native confidence must be between 0 and 1, got {value!r}")
    return result
