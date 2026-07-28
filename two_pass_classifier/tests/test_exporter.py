from __future__ import annotations

import csv
import json

import pytest

from two_pass_classifier.exporter import (
    PROFESSOR_CSV_COLUMNS,
    assemble_professor_row,
    export_professor_csv,
)
from two_pass_classifier.manifest import ManifestRow


def _pass_a(ai_native: int) -> dict:
    return {
        "ai_native": ai_native,
        "ai_native_reasoning": "AI is or is not the core product mechanism.",
        "sources_used": ["website_evidence", "short_description"],
        "ai_native_critique": "The website evidence may omit implementation detail.",
    }


def _manifest_row(**overrides: object) -> ManifestRow:
    values = {
        "company_id": "company-1",
        "company_name": "Company One",
        "cohort": "GENAI-ERA",
        "company_alive": "yes",
        "website_snapshot_date": "2026-05-04",
        "evidence_source": "live",
        "source_row_number": 2,
        "input_hash": "hash",
        "inputs": {
            "org_uuid": "company-1",
            "name": "Company One",
            "short_description": "desc",
            "Long description": "long",
            "category_list": "AI",
            "category_groups_list": "Software",
            "founded_date": "2024-01",
            "employee_count": "1-10",
            "total_funding_usd": "1",
            "website_pages_used": "https://example.test",
            "website_evidence": "Evidence",
        },
    }
    values.update(overrides)
    return ManifestRow(**values)  # type: ignore[arg-type]


def test_professor_column_order_is_exactly_eighteen_fields():
    assert PROFESSOR_CSV_COLUMNS == (
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
    assert len(PROFESSOR_CSV_COLUMNS) == 18


def test_ai_native_row_has_exact_columns_and_stable_source_union():
    row = assemble_professor_row(
        _manifest_row(),
        _pass_a(1),
        {
            "subclass": "1E",
            "rad_score": "RAD-M",
            "subclass_confidence": 4,
            "rad_confidence": 3,
            "subclass_reasoning": "The product is deep vertical AI.",
            "rad_reasoning": "Proprietary data offsets model dependency.",
            "sources_used": [
                "website_evidence",
                "resource_context",
                "founded_date",
            ],
            "subclass_critique": "A thick integrator is the closest alternative.",
            "rad_critique": "The exact model ownership is not explicit.",
        },
        0.82,
    )

    assert tuple(row) == PROFESSOR_CSV_COLUMNS
    assert row["company_alive"] == "yes"
    assert row["website_snapshot_date"] == "2026-05-04"
    assert json.loads(row["sources_used"]) == [
        "website_evidence",
        "short_description",
        "resource_context",
        "founded_date",
    ]
    assert row["ai_native_confidence"] == 0.82


def test_non_ai_row_forces_rad_na_and_blank_non_applicable_fields():
    row = assemble_professor_row(
        _manifest_row(
            company_id="company-0",
            company_name="Company Zero",
            cohort="PRE-GENAI",
            company_alive="no",
            website_snapshot_date="2024-09-02",
            evidence_source="dead",
        ),
        _pass_a(0),
        {
            "subclass": "0B",
            "subclass_confidence": 5,
            "subclass_reasoning": "AI augments a durable SaaS core.",
            "rad_reasoning": "RAD is not applicable outside the AI-native family.",
            "sources_used": ["website_evidence"],
            "subclass_critique": "The shipped feature is visible but secondary.",
        },
        None,
    )

    assert row["company_alive"] == "no"
    assert row["website_snapshot_date"] == "2024-09-02"
    assert row["rad_score"] == "RAD-NA"
    assert row["ai_native_confidence"] == ""
    assert row["rad_confidence"] == ""
    assert row["rad_critique"] == ""


def test_csv_header_is_exact_and_operational_fields_never_export(tmp_path):
    row = assemble_professor_row(
        _manifest_row(
            company_id="company-0",
            company_name="Company, Zero",
            cohort="PRE-GENAI",
            company_alive="no",
            website_snapshot_date="2023-03-14",
            evidence_source="dead",
        ),
        _pass_a(0),
        {
            "subclass": "0A",
            "subclass_confidence": 4,
            "subclass_reasoning": "Conventional software remains without AI.",
            "rad_reasoning": "RAD is RAD-NA for the fixed non-AI family.",
            "sources_used": ["company_name", "website_evidence"],
            "subclass_critique": "A visible AI feature could move it to 0B.",
        },
        None,
    )
    row["worker_id"] = "internal"
    row["latency_seconds"] = 3.2
    row["evidence_source"] = "dead"

    output = export_professor_csv([row], tmp_path / "professor.csv")
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        exported = list(reader)

    assert tuple(reader.fieldnames or ()) == PROFESSOR_CSV_COLUMNS
    assert len(exported) == 1
    assert set(exported[0]) == set(PROFESSOR_CSV_COLUMNS)
    assert "worker_id" not in exported[0]
    assert "latency_seconds" not in exported[0]
    assert "evidence_source" not in exported[0]
    assert exported[0]["company_alive"] == "no"
    assert exported[0]["website_snapshot_date"] == "2023-03-14"
    assert exported[0]["rad_score"] == "RAD-NA"
    assert exported[0]["rad_confidence"] == ""
    assert exported[0]["rad_critique"] == ""
    assert json.loads(exported[0]["sources_used"]) == [
        "website_evidence",
        "short_description",
        "company_name",
    ]


def test_mapping_manifest_row_requires_alive_and_snapshot_date():
    with pytest.raises(KeyError):
        assemble_professor_row(
            {
                "company_id": "company-1",
                "company_name": "Company One",
                "cohort": "GENAI-ERA",
            },
            _pass_a(1),
            {
                "subclass": "1E",
                "rad_score": "RAD-M",
                "subclass_confidence": 4,
                "rad_confidence": 3,
                "subclass_reasoning": "Vertical AI.",
                "rad_reasoning": "Some data moat.",
                "sources_used": ["website_evidence"],
                "subclass_critique": "Integrator alternative.",
                "rad_critique": "Ownership unclear.",
            },
            0.5,
        )
