"""Tests for CDX coverage probe edge cases."""

from __future__ import annotations

from wayback_machine.scripts import probe_coverage as probe_mod


def test_probe_window_with_unparseable_timestamps(monkeypatch) -> None:
    def fake_cdx_get(_query: str) -> list:
        return [["urlkey", "not-a-timestamp", "mimetype", "statuscode", "digest", "length"]]

    monkeypatch.setattr(probe_mod, "_cdx_get", fake_cdx_get)

    result = probe_mod.probe_one({
        "org_uuid": "u1",
        "name": "Acme",
        "homepage_url": "https://acme.ai/",
        "founded_date": "2020-01",
    })

    assert result["status"] == "error:bad_timestamp"
    assert result["has_2023"] != "True"
    assert result["closest_ts"] == ""
