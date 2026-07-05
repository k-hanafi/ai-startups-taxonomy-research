"""Draft-validation tests for the Stage 2 labeling module (no I/O against
production data; golden CSV and drafts are temp files)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import evals.labeling as labeling


@pytest.fixture()
def golden_csv(tmp_path, monkeypatch):
    path = tmp_path / "golden_set.csv"
    frame = pd.DataFrame(
        {
            "org_uuid": ["u1", "u2"],
            "name": ["Alpha", "Beta"],
            "predicted_subclass": ["1E", "0A"],
            "draft_ai_native": ["", ""],
            "draft_subclass": ["", ""],
            "draft_rad": ["", ""],
            "draft_rationale": ["", ""],
            "ambiguity_flag": ["", ""],
        }
    )
    frame.to_csv(path, index=False)
    monkeypatch.setattr(labeling, "GOLDEN_SET_CSV", path)
    return path


def _write_drafts(tmp_path, payload) -> str:
    p = tmp_path / "drafts.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def _valid_draft(**overrides):
    draft = {
        "draft_ai_native": "1",
        "draft_subclass": "1E",
        "draft_rad": "RAD-M",
        "draft_rationale": "Vertical AI with proprietary pipeline.",
        "ambiguity_flag": "",
    }
    draft.update(overrides)
    return draft


def test_apply_valid_draft(golden_csv, tmp_path):
    drafts = _write_drafts(tmp_path, {"u1": _valid_draft()})
    assert labeling.apply_drafts(drafts) == 1
    saved = pd.read_csv(golden_csv, dtype=str, keep_default_na=False)
    row = saved[saved["org_uuid"] == "u1"].iloc[0]
    assert row["draft_subclass"] == "1E"
    assert row["draft_rad"] == "RAD-M"
    # Untouched row stays empty.
    assert saved[saved["org_uuid"] == "u2"].iloc[0]["draft_subclass"] == ""


def test_unknown_uuid_rejected(golden_csv, tmp_path):
    drafts = _write_drafts(tmp_path, {"nope": _valid_draft()})
    with pytest.raises(KeyError):
        labeling.apply_drafts(drafts)


def test_native_flag_must_match_subclass(golden_csv, tmp_path):
    drafts = _write_drafts(tmp_path, {"u1": _valid_draft(draft_ai_native="0")})
    with pytest.raises(ValueError, match="contradicts"):
        labeling.apply_drafts(drafts)


def test_rad_na_required_for_non_native(golden_csv, tmp_path):
    bad = _valid_draft(draft_ai_native="0", draft_subclass="0A", draft_rad="RAD-M")
    drafts = _write_drafts(tmp_path, {"u1": bad})
    with pytest.raises(ValueError, match="contradicts"):
        labeling.apply_drafts(drafts)


def test_invalid_subclass_rejected(golden_csv, tmp_path):
    drafts = _write_drafts(tmp_path, {"u1": _valid_draft(draft_subclass="9Z")})
    with pytest.raises(ValueError, match="invalid subclass"):
        labeling.apply_drafts(drafts)


def test_empty_rationale_rejected(golden_csv, tmp_path):
    drafts = _write_drafts(tmp_path, {"u1": _valid_draft(draft_rationale="  ")})
    with pytest.raises(ValueError, match="rationale"):
        labeling.apply_drafts(drafts)


def test_nothing_written_when_batch_invalid(golden_csv, tmp_path):
    payload = {"u1": _valid_draft(), "u2": _valid_draft(draft_subclass="9Z")}
    drafts = _write_drafts(tmp_path, payload)
    with pytest.raises(ValueError):
        labeling.apply_drafts(drafts)
    saved = pd.read_csv(golden_csv, dtype=str, keep_default_na=False)
    assert (saved["draft_subclass"] == "").all()
