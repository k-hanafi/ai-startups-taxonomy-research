"""Tests for the run-evals orchestrator (no network, no real API key)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from evals import config as cfg
from evals import orchestrate as orch
from evals.jsonl_io import append_jsonl
from evals.paths import pass_a_bank_run_id


def _fake_rows(n: int = 2) -> list[dict]:
    return [
        {
            "org_uuid": f"u{i}",
            "name": f"Co{i}",
            "short_description": "x",
            "website_evidence": "y",
            "founded_on": "2024-01-01",
        }
        for i in range(n)
    ]


def test_build_job_plan_shape():
    jobs = orch.build_job_plan(n_rows=100, date="2026-07-25")
    kinds = [j.kind for j in jobs]
    assert kinds.count("pass_a") == 3
    assert kinds.count("cell") == 9
    assert kinds.count("dashboard") == 1
    cells = [j for j in jobs if j.kind == "cell"]
    assert cells[0].run_id == (
        f"2026-07-25_classification_{cfg.EVAL_MODELS[0]}_low_r1"
    )


def test_phase1_complete_requires_all_banks_green():
    jobs = orch.build_job_plan(2, date="2026-07-25")
    assert not orch.phase1_complete(jobs)
    for j in jobs:
        if j.kind == "pass_a":
            j.status = "done"
    assert orch.phase1_complete(jobs)
    jobs[0].status = "failed"
    assert not orch.phase1_complete(jobs)
    assert orch.phase1_failed(jobs)


def test_should_block_dashboard_on_failed_cell():
    jobs = orch.build_job_plan(2, date="2026-07-25")
    for j in jobs:
        if j.kind in ("pass_a", "cell"):
            j.status = "done"
    assert not orch.should_block_dashboard(jobs)
    cells = [j for j in jobs if j.kind == "cell"]
    cells[0].status = "failed"
    assert orch.should_block_dashboard(jobs)


def test_count_completed_and_spend_from_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "run_predictions_path", lambda rid: tmp_path / rid / "predictions.jsonl")
    run_id = "cell1"
    (tmp_path / run_id).mkdir()
    path = tmp_path / run_id / "predictions.jsonl"
    model = cfg.EVAL_MODELS[0]
    append_jsonl(path, {
        "custom_id": "startup-u0",
        "status": "completed",
        "a_input_tokens": 1_000_000,
        "a_output_tokens": 0,
        "b_input_tokens": 500_000,
        "b_output_tokens": 0,
    })
    append_jsonl(path, {
        "custom_id": "startup-u1",
        "status": "failed",
        "a_input_tokens": 1_000_000,
        "a_output_tokens": 0,
        "b_input_tokens": 0,
        "b_output_tokens": 0,
    })
    assert orch.count_completed_predictions(run_id) == 1
    # Default sums a+b: 2.5M input at nano $0.20 / 1M = $0.50
    assert orch.spend_from_predictions(run_id, model) == pytest.approx(0.50)
    # Cell footer must not re-bill Pass A (bank job already counted it).
    assert orch.spend_from_predictions(
        run_id, model, prefixes=("b",),
    ) == pytest.approx(0.10)
    assert orch.spend_from_predictions(
        run_id, model, prefixes=("a",),
    ) == pytest.approx(0.40)


def test_cell_already_scored_requires_exact_row_count(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "run_scored_path", lambda rid: tmp_path / rid / "scored.json")
    monkeypatch.setattr(orch, "run_config_path", lambda rid: tmp_path / rid / "config.json")
    run = tmp_path / "r1"
    run.mkdir()
    (run / "scored.json").write_text(json.dumps({"n_scored": 100}), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"n_rows": 100}), encoding="utf-8")
    assert orch.cell_already_scored("r1", 100)
    # Full matrix must not satisfy a --limit 1 smoke skip.
    assert not orch.cell_already_scored("r1", 1)
    assert not orch.cell_already_scored("r1", 101)
    assert not orch.cell_already_scored("missing", 10)


def test_build_status_table_renders_statuses():
    jobs = orch.build_job_plan(5, date="2026-07-25")
    jobs[0].status = "done"
    jobs[0].rows_done = 5
    jobs[1].status = "running"
    jobs[1].rows_done = 2
    jobs[1].started_at = 100.0
    table = orch.build_status_table(jobs, now=110.0)
    # Rich Table stores rows; ensure it builds without error and caption exists.
    assert table.caption is not None
    assert "steps green" in table.caption
    rendered = Console(record=True, width=120).render_str("")  # smoke import path
    _ = rendered
    con = Console(record=True, width=140)
    con.print(table)
    text = con.export_text()
    assert "Pass A bank" in text
    assert "done" in text


def test_seed_skipped_marks_scored_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "run_scored_path", lambda rid: tmp_path / "scored" / f"{rid}.json")
    # Also redirect config.json: a real run dir of the same id on disk would
    # otherwise supply an n_rows that fails the exact-match check.
    monkeypatch.setattr(orch, "run_config_path", lambda rid: tmp_path / "conf" / f"{rid}.json")
    monkeypatch.setattr(
        orch, "bank_already_complete", lambda model, cids: False
    )
    jobs = orch.build_job_plan(2, date="2026-07-25")
    cell = next(j for j in jobs if j.kind == "cell")
    scored_dir = tmp_path / "scored"
    scored_dir.mkdir()
    (scored_dir / f"{cell.run_id}.json").write_text(
        json.dumps({"n_scored": 2}), encoding="utf-8"
    )
    orch._seed_skipped(jobs, ["startup-u0", "startup-u1"])
    assert cell.status == "skipped"
    pending_cells = [j for j in jobs if j.kind == "cell" and j.status == "pending"]
    assert len(pending_cells) == 8


def test_find_cell_run_id_resumes_prior_day(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RUNS_DIR", tmp_path)
    model = cfg.EVAL_MODELS[0]
    effort = "low"
    old_id = f"2026-07-20_classification_{model}_{effort}_r1"
    run = tmp_path / old_id
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps({"model": model, "effort_b": effort, "n_rows": 100}),
        encoding="utf-8",
    )
    (run / "scored.json").write_text(
        json.dumps({"n_scored": 100}), encoding="utf-8",
    )
    found = orch.find_cell_run_id(model, effort, 100, date="2026-07-25")
    assert found == old_id
    # Different row count must mint a fresh id for the requested day.
    fresh = orch.find_cell_run_id(model, effort, 1, date="2026-07-25")
    assert fresh == f"2026-07-25_classification_{model}_{effort}_r1"


def test_open_dashboard_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "EVAL_INSTANCES_DIR", tmp_path / "empty_instances")
    with pytest.raises(SystemExit, match="run-evals"):
        orch.open_dashboard_index()


def test_open_dashboard_index_opens_index(tmp_path, monkeypatch):
    inst = tmp_path / "eval_instances"
    inst.mkdir()
    (inst / "eval_instance_01.html").write_text("<html>ok</html>", encoding="utf-8")
    (inst / "instances.json").write_text(
        json.dumps({
            "instances": [{
                "n": 1,
                "file": "eval_instance_01.html",
                "archived_utc": "2026-07-24T20:00:00+00:00",
                "n_configs": 1,
                "run": {},
            }]
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(orch, "EVAL_INSTANCES_DIR", inst)
    opened: list[str] = []
    monkeypatch.setattr(
        orch.webbrowser, "open", lambda uri: opened.append(uri)
    )
    path = orch.open_dashboard_index()
    assert path == inst / "index.html"
    assert path.exists()
    assert opened and opened[0].startswith("file:")


def test_require_openai_key_refuses_placeholder(monkeypatch, tmp_path):
    env = tmp_path / "openai.env"
    env.write_text("OPENAI_API_KEY=placeholder\n", encoding="utf-8")
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    # load_dotenv reads PROJECT_ROOT / keys / openai.env
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "openai.env").write_text("OPENAI_API_KEY=placeholder\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        orch.require_openai_key()


class _FakeProc:
    """Minimal Popen stand-in: finishes on first poll after N ticks."""

    def __init__(self, ticks: int = 1, rc: int = 0):
        self._ticks = ticks
        self._rc = rc
        self.stdout = None

    def poll(self):
        if self._ticks > 0:
            self._ticks -= 1
            return None
        return self._rc

    def wait(self):
        return self._rc


def test_run_evals_phase2_waits_for_banks(tmp_path, monkeypatch):
    """Phase 2 cell spawns must not happen until all 3 Pass A banks finish."""
    rows = _fake_rows(2)
    monkeypatch.setattr(orch, "load_golden_rows", lambda: rows)
    monkeypatch.setattr(orch, "require_openai_key", lambda: None)
    monkeypatch.setattr(orch, "print_matrix_preview", lambda r: MagicMock(total_cost=1.23))
    monkeypatch.setattr(orch, "bank_already_complete", lambda m, c: False)
    monkeypatch.setattr(orch, "cell_already_scored", lambda r, n: False)
    monkeypatch.setattr(orch, "count_completed_predictions", lambda r: 0)
    monkeypatch.setattr(orch, "spend_from_predictions", lambda r, m: 0.0)
    monkeypatch.setattr(orch, "refresh_job_progress", lambda j: None)

    # Avoid writing under real evals/runs.
    monkeypatch.setattr(
        orch, "run_dir", lambda rid: tmp_path / "runs" / rid.replace("/", "_")
    )
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    spawn_log: list[str] = []
    bank_ticks = {"n": 2}  # banks take 2 poll cycles

    def fake_spawn(cmd, log_path):
        # cmd = [python, "-m", "evals", <subcommand>, ...]
        sub = cmd[3] if len(cmd) > 3 else "?"
        spawn_log.append(sub)
        # bank-pass-a finishes after a couple polls; cells finish immediately
        if "bank-pass-a" in cmd:
            proc = _FakeProc(ticks=bank_ticks["n"], rc=0)
        else:
            proc = _FakeProc(ticks=0, rc=0)
        return proc, MagicMock()

    monkeypatch.setattr(orch, "_spawn", fake_spawn)

    # Capture whether any cell was spawned before all banks done.
    original_start_cell = orch._start_cell_classify
    phase1_was_complete: list[bool] = []

    def wrapped_start_cell(job, limit):
        jobs_snapshot = wrapped_start_cell.jobs  # type: ignore[attr-defined]
        phase1_was_complete.append(orch.phase1_complete(jobs_snapshot))
        return original_start_cell(job, limit)

    # Hook into run_evals by wrapping after jobs exist: patch _start_cell_classify
    # to record phase1 state. We need access to jobs list — patch build_job_plan
    # to stash it.
    real_build = orch.build_job_plan
    held: dict = {}

    def build_and_hold(*a, **k):
        jobs = real_build(*a, **k)
        held["jobs"] = jobs
        wrapped_start_cell.jobs = jobs  # type: ignore[attr-defined]
        return jobs

    monkeypatch.setattr(orch, "build_job_plan", build_and_hold)
    monkeypatch.setattr(orch, "_start_cell_classify", wrapped_start_cell)

    # Dashboard spawn: succeed immediately
    def fake_start_dash(job, cell_ids):
        orch._mark_running(job)
        job.process = _FakeProc(ticks=0, rc=0)
        job.log_handle = MagicMock()

    monkeypatch.setattr(orch, "_start_dashboard", fake_start_dash)

    # Score stage also uses _spawn via _start_cell_score
    rc = orch.run_evals(
        yes=True,
        limit=2,
        console=Console(quiet=True),
        poll_seconds=0.01,
    )
    assert rc == 0
    assert phase1_was_complete
    assert all(phase1_was_complete), (
        "a Pass B cell launched before all Pass A banks were green"
    )
    assert spawn_log.count("bank-pass-a") == 3
    assert spawn_log.count("run-classification") == 9
    assert spawn_log.count("score") == 9


def test_run_evals_failed_cell_blocks_dashboard(tmp_path, monkeypatch):
    rows = _fake_rows(1)
    monkeypatch.setattr(orch, "load_golden_rows", lambda: rows)
    monkeypatch.setattr(orch, "require_openai_key", lambda: None)
    monkeypatch.setattr(orch, "print_matrix_preview", lambda r: MagicMock(total_cost=0.5))
    monkeypatch.setattr(orch, "bank_already_complete", lambda m, c: True)
    monkeypatch.setattr(orch, "cell_already_scored", lambda r, n: False)
    monkeypatch.setattr(orch, "count_completed_predictions", lambda r: 0)
    monkeypatch.setattr(orch, "spend_from_predictions", lambda r, m: 0.0)
    monkeypatch.setattr(orch, "refresh_job_progress", lambda j: None)
    monkeypatch.setattr(
        orch, "run_dir", lambda rid: tmp_path / "runs" / rid.replace("/", "_")
    )
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    fail_once = {"done": False}

    def fake_spawn(cmd, log_path):
        if "run-classification" in cmd and not fail_once["done"]:
            fail_once["done"] = True
            return _FakeProc(ticks=0, rc=2), MagicMock()
        return _FakeProc(ticks=0, rc=0), MagicMock()

    monkeypatch.setattr(orch, "_spawn", fake_spawn)
    dash_started = {"n": 0}

    def track_dash(job, cell_ids):
        dash_started["n"] += 1
        orch._mark_running(job)
        job.process = _FakeProc(ticks=0, rc=0)
        job.log_handle = MagicMock()

    monkeypatch.setattr(orch, "_start_dashboard", track_dash)

    rc = orch.run_evals(
        yes=True,
        limit=1,
        console=Console(quiet=True),
        poll_seconds=0.01,
    )
    assert rc == 1
    assert dash_started["n"] == 0  # real dashboard spawn never called
