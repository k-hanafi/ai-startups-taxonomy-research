"""Offline tests for the numbered eval instance archive.

No OpenAI key required: evals.instances is filesystem + formatting only.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from evals.instances import (
    ArchivedInstance,
    archive_instance,
    format_run_headline,
    instance_filename,
    load_registry,
    next_instance_number,
    render_index,
    run_meta_bits,
)

UTC = datetime.timezone.utc


def _metrics(
    started: str | None = "2026-07-24T18:47:11+00:00",
    *,
    last: str | None = None,
    synthetic: bool = False,
    config_ids: list[str] | None = None,
) -> dict:
    return {
        "synthetic": synthetic,
        "source": "9 runs under evals/runs/",
        "n_configs": len(config_ids or ["a", "b"]),
        "config_ids": config_ids or ["a", "b"],
        "run_instance": {
            "synthetic": synthetic,
            "n_runs": 9,
            "started_first": started,
            "started_last": last or started,
            "models": ["gpt-5.4-nano"],
            "model_groups": ["nano"],
            "n_gold": 100,
            "git_commit": "acb70ac",
        },
    }


def _archive(tmp_path: Path, metrics: dict, minute: int = 0) -> ArchivedInstance:
    return archive_instance(
        f"<html>{metrics['n_configs']}</html>",
        metrics,
        directory=tmp_path,
        now=datetime.datetime(2026, 7, 24, 20, minute, tzinfo=UTC),
    )


def test_archive_writes_numbered_page_registry_and_index(tmp_path):
    archived = _archive(tmp_path, _metrics())

    assert archived.number == 1
    assert archived.path == tmp_path / "eval_instance_01.html"
    assert archived.path.read_text(encoding="utf-8") == "<html>2</html>"
    assert archived.replaced is False
    assert (tmp_path / "index.html").exists()

    entries = load_registry(tmp_path)
    assert [e["n"] for e in entries] == [1]
    assert entries[0]["file"] == "eval_instance_01.html"
    assert entries[0]["synthetic"] is False
    assert entries[0]["archived_utc"].startswith("2026-07-24T20:00")


def test_distinct_runs_get_consecutive_numbers(tmp_path):
    first = _archive(tmp_path, _metrics("2026-07-24T18:00:00+00:00"))
    second = _archive(tmp_path, _metrics("2026-07-25T09:30:00+00:00"), minute=5)

    assert (first.number, second.number) == (1, 2)
    assert {p.name for p in tmp_path.glob("eval_instance_*.html")} == {
        "eval_instance_01.html",
        "eval_instance_02.html",
    }


def test_rebuilding_the_same_runs_replaces_that_instance(tmp_path):
    """A styling fix must not mint a second instance for one sweep."""
    _archive(tmp_path, _metrics())
    again = _archive(tmp_path, _metrics(), minute=40)

    assert again.number == 1
    assert again.replaced is True
    entries = load_registry(tmp_path)
    assert len(entries) == 1
    assert entries[0]["archived_utc"].startswith("2026-07-24T20:00")
    assert entries[0]["rebuilt_utc"].startswith("2026-07-24T20:40")


def test_same_start_time_but_different_cells_is_a_new_instance(tmp_path):
    """Identity includes which configs were loaded, not just when they ran."""
    _archive(tmp_path, _metrics(config_ids=["nano/low"]))
    other = _archive(tmp_path, _metrics(config_ids=["nano/low", "mini/low"]), minute=5)

    assert other.number == 2
    assert other.replaced is False


def test_undated_runs_never_overwrite_each_other(tmp_path):
    """Without a recorded start we cannot tell two sweeps apart, so keep both."""
    first = _archive(tmp_path, _metrics(None))
    second = _archive(tmp_path, _metrics(None), minute=5)

    assert (first.number, second.number) == (1, 2)
    assert load_registry(tmp_path)[0]["identity"] is None


def test_numbering_survives_a_lost_registry(tmp_path):
    """Filenames on disk are consulted so an old page cannot be overwritten."""
    _archive(tmp_path, _metrics())
    (tmp_path / "instances.json").unlink()

    recovered = _archive(tmp_path, _metrics("2026-07-25T10:00:00+00:00"), minute=5)
    assert recovered.number == 2


def test_damaged_registry_reads_as_empty(tmp_path):
    (tmp_path / "instances.json").write_text("{ not json", encoding="utf-8")
    assert load_registry(tmp_path) == []


def test_registry_ignores_entries_without_a_number(tmp_path):
    (tmp_path / "instances.json").write_text(
        json.dumps({"instances": [{"file": "stray.html"}, {"n": 3}]}), encoding="utf-8"
    )
    assert [e["n"] for e in load_registry(tmp_path)] == [3]
    assert next_instance_number(tmp_path, load_registry(tmp_path)) == 4


def test_instance_filename_zero_pads_then_grows():
    assert instance_filename(1) == "eval_instance_01.html"
    assert instance_filename(7) == "eval_instance_07.html"
    assert instance_filename(142) == "eval_instance_142.html"


def test_index_lists_newest_first_with_run_details(tmp_path):
    _archive(tmp_path, _metrics("2026-07-24T18:00:00+00:00"))
    _archive(tmp_path, _metrics("2026-07-25T09:30:00+00:00"), minute=5)

    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert page.index("eval_instance_02.html") < page.index("eval_instance_01.html")
    assert "9 eval runs" in page
    assert "100 golden companies" in page
    assert "commit acb70ac" in page
    assert 'href="../eval_dashboard.html"' in page


def test_index_marks_synthetic_builds(tmp_path):
    _archive(tmp_path, _metrics(None, synthetic=True))
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Synthetic data (mock matrix)" in page
    assert "start time not recorded" not in page


def test_rebuilding_the_same_synthetic_preview_replaces(tmp_path):
    """--save-instance on the mock must not pile up identical previews."""
    first = _archive(tmp_path, _metrics(None, synthetic=True))
    again = _archive(tmp_path, _metrics(None, synthetic=True), minute=40)
    assert first.number == again.number == 1
    assert again.replaced is True
    assert len(load_registry(tmp_path)) == 1


def test_index_without_instances_says_so():
    page = render_index([])
    assert "No instances archived yet" in page
    assert "<table" not in page


def test_index_escapes_run_text():
    page = render_index(
        [
            {
                "n": 1,
                "file": "eval_instance_01.html",
                "archived_utc": "2026-07-24T20:00:00+00:00",
                "n_configs": 1,
                "run": {"n_runs": 1, "model_groups": ["<script>"]},
            }
        ]
    )
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_run_headline_and_meta_read_as_english():
    run = {
        "n_runs": 9,
        "started_first": "2026-07-24T18:47:11+00:00",
        "started_last": "2026-07-24T19:59:11+00:00",
        "model_groups": ["nano", "mini", "luna"],
        "n_gold": 100,
        "git_commit": "acb70ac",
    }
    headline = format_run_headline(run)
    assert headline.startswith("9 eval runs, Jul 24, 2026")
    assert " to " in headline
    assert run_meta_bits(run) == ["nano, mini, luna", "100 golden companies", "commit acb70ac"]


def test_run_headline_without_times_says_not_recorded():
    assert format_run_headline({"n_runs": 1}) == "Eval run, start time not recorded"
