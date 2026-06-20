"""Unit tests for the cohort contracts and Wayback helpers."""

from __future__ import annotations

from wayback_machine.cohort import (
    CLASSIFIER_INPUT_COLUMNS,
    build_snapshot_url,
    existed_by,
    is_retrievable,
    is_valid_homepage_url,
)


def test_snapshot_url_uses_id_suffix_and_keeps_scheme() -> None:
    url = build_snapshot_url("20230314120000", "https://stripe.com/")
    assert url == "http://web.archive.org/web/20230314120000id_/https://stripe.com/"


def test_snapshot_url_adds_scheme_when_missing() -> None:
    url = build_snapshot_url("20230314120000", "stripe.com")
    assert url == "http://web.archive.org/web/20230314120000id_/https://stripe.com"


def test_is_retrievable_requires_ok_hit_and_timestamp() -> None:
    assert is_retrievable({"status": "ok", "has_2023": "True", "closest_ts": "20230314"})
    assert not is_retrievable({"status": "ok", "has_2023": "False", "closest_ts": ""})
    assert not is_retrievable({"status": "error:URLError", "has_2023": "True", "closest_ts": "x"})
    assert not is_retrievable({"status": "ok", "has_2023": "True", "closest_ts": ""})


def test_is_valid_homepage_url() -> None:
    assert is_valid_homepage_url("https://example.com")
    assert is_valid_homepage_url("http://example.com/path")
    assert not is_valid_homepage_url("")
    assert not is_valid_homepage_url("nan")
    assert not is_valid_homepage_url("ftp://example.com")
    assert not is_valid_homepage_url("not a url")


def test_existed_by_cutoff_is_inclusive_of_march_2023() -> None:
    assert existed_by("2020-01")
    assert existed_by("2023-03")          # March 2023 included
    assert not existed_by("2023-04")      # founded after GPT-4 launch
    assert not existed_by("2024-01")
    assert not existed_by("")             # unknown founding -> excluded


def test_classifier_columns_end_with_evidence_pair() -> None:
    assert CLASSIFIER_INPUT_COLUMNS[-2:] == ["website_pages_used", "website_evidence"]
