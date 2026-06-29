#!/usr/bin/env python3
"""Overnight supervisor for the dead-cohort Tavily crawl.

Every ``--interval-minutes`` (default 30), inspects ``crawl_dead.jsonl`` and the
runner process, applies recovery actions, and ensures ``run_crawl_dead.py`` is
moving forward at safe settings (concurrency 4).

Recovery actions:
  * Start the crawl if it exited but work remains.
  * Stop + purge post-pilot ``empty_results`` when the recent empty rate is
    catastrophically high (classic signature of Archive/Tavily choke at high
    concurrency). Purged rows are backed up and become eligible for retry.
  * Restart if the JSONL has not grown for ``--stuck-minutes`` while incomplete.

Run once inside tmux (outside Cursor sandbox):

    caffeinate -ims python3 wayback_machine/scripts/supervise_crawl_dead.py

First health check runs immediately; then every 30 minutes until the cohort is
fully attempted or you Ctrl-C this supervisor.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.paths import (  # noqa: E402
    CRAWL_DEAD_JSONL,
    CRAWL_DEAD_LOG,
    LOGS_DIR,
    SCRAPE_TARGETS_DEAD_CSV,
)

SUPERVISOR_LOG = LOGS_DIR / "supervise_crawl_dead.log"
RUNNER_SCRIPT = PROJECT_ROOT / "wayback_machine" / "scripts" / "run_crawl_dead.py"

# Rows 1..PILOT_ROWS were validated at concurrency 4; never purge these.
PILOT_ROWS = 100
RECENT_WINDOW = 40
MAX_RECENT_EMPTY_RATE = 0.50
MIN_RECENT_SAMPLE = 20


@dataclass(frozen=True)
class Snapshot:
    jsonl_lines: int
    target_count: int
    post_pilot_empty_rate: float
    recent_empty_rate: float
    recent_sample: int
    successes: int
    seconds_since_write: float
    runner_pids: tuple[int, ...]
    runner_concurrency: int | None


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _count_targets() -> int:
    with SCRAPE_TARGETS_DEAD_CSV.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in f) - 1


def _load_jsonl() -> list[dict]:
    if not CRAWL_DEAD_JSONL.exists():
        return []
    out: list[dict] = []
    with CRAWL_DEAD_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _runner_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "wayback_machine/scripts/run_crawl_dead.py"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [int(p) for p in out.split() if p.strip()]


def _runner_concurrency(pids: list[int]) -> int | None:
    if not pids:
        return None
    try:
        out = subprocess.check_output(["ps", "-p", str(pids[0]), "-o", "args="], text=True)
    except subprocess.CalledProcessError:
        return None
    if "--concurrency" in out:
        parts = out.split()
        idx = parts.index("--concurrency")
        if idx + 1 < len(parts):
            return int(parts[idx + 1])
    return 4


def _snapshot() -> Snapshot:
    rows = _load_jsonl()
    post = rows[PILOT_ROWS:]
    post_emp = sum(1 for r in post if r.get("status") == "empty_results")
    post_rate = post_emp / len(post) if post else 0.0

    recent = rows[-RECENT_WINDOW:] if rows else []
    recent_emp = sum(1 for r in recent if r.get("status") == "empty_results")
    recent_rate = recent_emp / len(recent) if recent else 0.0

    succ = sum(1 for r in rows if r.get("status") in ("success", "success_fallback"))
    mtime = CRAWL_DEAD_JSONL.stat().st_mtime if CRAWL_DEAD_JSONL.exists() else 0.0
    pids = _runner_pids()
    return Snapshot(
        jsonl_lines=len(rows),
        target_count=_count_targets(),
        post_pilot_empty_rate=post_rate,
        recent_empty_rate=recent_rate,
        recent_sample=len(recent),
        successes=succ,
        seconds_since_write=time.time() - mtime if mtime else float("inf"),
        runner_pids=tuple(pids),
        runner_concurrency=_runner_concurrency(pids),
    )


def _stop_runner(pids: list[int], *, wait_s: float = 180.0) -> None:
    if not pids:
        return
    _log(f"stopping runner pids={pids} (SIGINT)")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            pass
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not _runner_pids():
            _log("runner stopped cleanly")
            return
        time.sleep(2.0)
    remaining = _runner_pids()
    if remaining:
        _log(f"runner still alive after {wait_s:.0f}s; sending SIGTERM to {remaining}")
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        time.sleep(5.0)


def _purge_post_pilot_empties() -> int:
    if not CRAWL_DEAD_JSONL.exists():
        return 0
    lines = CRAWL_DEAD_JSONL.read_text(encoding="utf-8").splitlines()
    keep: list[str] = []
    removed = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if i < PILOT_ROWS:
            keep.append(line)
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            keep.append(line)
            continue
        if obj.get("status") in ("success", "success_fallback"):
            keep.append(line)
            continue
        removed += 1
    backup = CRAWL_DEAD_JSONL.with_suffix(
        f".jsonl.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    backup.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    CRAWL_DEAD_JSONL.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    _log(f"purged {removed} post-pilot non-success rows; backup={backup.name}")
    return removed


def _start_runner(*, concurrency: int, heartbeat_every: int) -> None:
    if _runner_pids():
        _log("start skipped: runner already running")
        return
    cmd = [
        "caffeinate", "-ims",
        sys.executable,
        str(RUNNER_SCRIPT),
        "--heartbeat-every", str(heartbeat_every),
        "--concurrency", str(concurrency),
    ]
    _log(f"starting runner: {' '.join(cmd)}")
    log_out = LOGS_DIR / "crawl_dead_runner.stdout.log"
    with log_out.open("a", encoding="utf-8") as fh:
        fh.write(f"\n--- supervisor start {datetime.now(timezone.utc).isoformat()} ---\n")
        subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _needs_purge(s: Snapshot) -> bool:
    if s.jsonl_lines <= PILOT_ROWS:
        return False
    if s.recent_sample >= MIN_RECENT_SAMPLE and s.recent_empty_rate >= MAX_RECENT_EMPTY_RATE:
        return True
    if len(_load_jsonl()) > PILOT_ROWS and s.post_pilot_empty_rate >= 0.75:
        return True
    return False


def _tick(*, concurrency: int, heartbeat_every: int, stuck_seconds: float) -> str:
    s = _snapshot()
    done = s.jsonl_lines >= s.target_count
    _log(
        f"check lines={s.jsonl_lines}/{s.target_count} successes={s.successes} "
        f"post_pilot_empty={s.post_pilot_empty_rate:.0%} recent_empty={s.recent_empty_rate:.0%}"
        f"(n={s.recent_sample}) runner={s.runner_pids or 'none'}"
        f" conc={s.runner_concurrency} stale={s.seconds_since_write:.0f}s"
    )

    if done:
        return "complete"

    bad_conc = s.runner_concurrency is not None and s.runner_concurrency > concurrency
    purge = _needs_purge(s)
    stuck = (
        bool(s.runner_pids)
        and s.seconds_since_write > stuck_seconds
        and s.jsonl_lines < s.target_count
    )

    if purge or bad_conc or stuck:
        reason = []
        if purge:
            reason.append("high_empty_rate")
        if bad_conc:
            reason.append(f"concurrency>{concurrency}")
        if stuck:
            reason.append("stuck")
        _log(f"recovery triggered: {', '.join(reason)}")
        _stop_runner(list(s.runner_pids))
        if purge:
            _purge_post_pilot_empties()
        _start_runner(concurrency=concurrency, heartbeat_every=heartbeat_every)
        return "recovered"

    if not s.runner_pids:
        if purge:
            _purge_post_pilot_empties()
        _start_runner(concurrency=concurrency, heartbeat_every=heartbeat_every)
        return "started"

    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval-minutes", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--heartbeat-every", type=int, default=10)
    ap.add_argument("--stuck-minutes", type=int, default=45)
    ap.add_argument("--once", action="store_true", help="Run one check then exit.")
    args = ap.parse_args()

    stuck_seconds = float(args.stuck_minutes * 60)
    _log(
        f"supervisor online interval={args.interval_minutes}m concurrency={args.concurrency}"
        f" stuck={args.stuck_minutes}m pilot_rows={PILOT_ROWS}"
    )

    while True:
        outcome = _tick(
            concurrency=args.concurrency,
            heartbeat_every=args.heartbeat_every,
            stuck_seconds=stuck_seconds,
        )
        if outcome == "complete":
            _log("cohort fully attempted — supervisor exiting")
            break
        if args.once:
            break
        time.sleep(max(60, args.interval_minutes * 60))


if __name__ == "__main__":
    main()
