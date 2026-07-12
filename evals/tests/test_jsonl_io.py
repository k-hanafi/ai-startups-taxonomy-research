"""Tests for the canonical evals JSONL reader/writer policy."""

from __future__ import annotations

import json

import pytest

from evals.jsonl_io import MalformedJSONLError, append_jsonl, iter_jsonl


def test_iter_jsonl_yields_objects(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"a": 1}) + "\n"
        + "\n"
        + json.dumps({"b": 2}) + "\n",
        encoding="utf-8",
    )
    assert list(iter_jsonl(path)) == [{"a": 1}, {"b": 2}]


def test_iter_jsonl_tolerates_truncated_final(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"custom_id": "ok"}) + "\n"
        + '{"custom_id": "trunc',
        encoding="utf-8",
    )
    assert list(iter_jsonl(path, tolerate_truncated_final=True)) == [
        {"custom_id": "ok"}
    ]


def test_iter_jsonl_fails_on_interior_malformed(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"custom_id": "a"}) + "\n"
        + "{not-json\n"
        + json.dumps({"custom_id": "c"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedJSONLError, match="line 2"):
        list(iter_jsonl(path, tolerate_truncated_final=True))


def test_iter_jsonl_fails_on_truncated_final_when_not_tolerated(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{trunc', encoding="utf-8")
    with pytest.raises(MalformedJSONLError):
        list(iter_jsonl(path, tolerate_truncated_final=False))


def test_append_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    append_jsonl(path, {"x": 1})
    append_jsonl(path, {"x": 2})
    assert list(iter_jsonl(path)) == [{"x": 1}, {"x": 2}]
