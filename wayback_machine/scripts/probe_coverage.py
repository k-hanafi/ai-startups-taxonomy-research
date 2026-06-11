#!/usr/bin/env python3
"""Estimate March-2023 Wayback retrievability for the frozen cohort.

Samples N companies from wayback_cohort.csv and asks the Internet Archive's CDX
Server API whether each homepage has a 200/HTML capture near 2023-03-14. This is
a feasibility measurement, not the real scrape: we sample (not census) so a few
hundred queries give a tight estimate of the coverage percentage before we
commit to building the full historical pipeline.

Resumable: re-running skips org_uuids already resolved (status=ok), including
confirmed no-March-2023 misses — those never hit the API again.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.config import (  # noqa: E402
    CDX_429_MIN_PAUSE_SECONDS,
    CDX_DEFAULT_CONCURRENCY,
    CDX_DEFAULT_RETRIES,
    CDX_DEFAULT_TIMEOUT_SECONDS,
    CDX_SAFE_RPM,
    WINDOW_FROM,
    WINDOW_TO,
)

COHORT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "wayback_cohort.csv"
OUTPUT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "coverage_sample.csv"

CDX_ENDPOINT = "http://web.archive.org/cdx/search/cdx"
TARGET = datetime(2023, 3, 14)
USER_AGENT = "wayback-coverage-probe/2.0 (research; contact: batchkit)"

DO_EVER_CALL = True
_LIMITER: CdxRateLimiter | None = None
_CDX_RETRIES = CDX_DEFAULT_RETRIES
_CDX_TIMEOUT = CDX_DEFAULT_TIMEOUT_SECONDS

OUTPUT_FIELDS = [
    "org_uuid", "name", "homepage_url", "founded_date", "host",
    "has_2023", "closest_ts", "days_from_target", "n_window_captures",
    "has_any_ever", "status",
]


class CdxRateLimiter:
    """Global CDX throttle: one slot every (60/rpm) seconds, freeze all threads on 429.

    IA documents /cdx/* at 60 req/min; clients should target 48/min (80% headroom).
    On 429, every worker must pause ≥60s or risk an hour-long IP block.
    """

    def __init__(self, rpm: float, *, pause_on_429: float = CDX_429_MIN_PAUSE_SECONDS) -> None:
        self._min_interval = 60.0 / rpm
        self._pause_on_429 = pause_on_429
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._frozen_until = 0.0

    def wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            wake = max(self._next_slot, self._frozen_until)
            if now < wake:
                time.sleep(wake - now)
                now = time.monotonic()
            self._next_slot = now + self._min_interval

    def freeze_for_429(self, retry_after: float | None) -> float:
        with self._lock:
            pause = max(self._pause_on_429, retry_after or 0.0)
            self._frozen_until = max(self._frozen_until, time.monotonic() + pause)
            return pause


def to_host(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    net = urlparse(u).netloc.lower()
    return net[4:] if net.startswith("www.") else net


def _cdx_get(params: str) -> list[list[str]]:
    """GET the CDX endpoint with global rate limiting and patient retries."""
    if _LIMITER is None:
        msg = "CdxRateLimiter not initialized"
        raise RuntimeError(msg)

    url = f"{CDX_ENDPOINT}?{params}"
    last_exc: Exception | None = None
    for attempt in range(_CDX_RETRIES):
        _LIMITER.wait_turn()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=_CDX_TIMEOUT) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip()
            if not body:
                return []
            data = json.loads(body)
            return data[1:] if len(data) > 1 else []
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                retry_hdr = exc.headers.get("Retry-After") if exc.headers else None
                retry_after = float(retry_hdr) if retry_hdr and retry_hdr.isdigit() else None
                pause = _LIMITER.freeze_for_429(retry_after)
                print(f"  CDX 429 — pausing all requests {pause:.0f}s", flush=True)
                time.sleep(pause)
            elif exc.code in {500, 502, 503, 504}:
                time.sleep(min(30.0, 2.0 * (2**attempt)))
            else:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_exc = exc
            time.sleep(min(30.0, 2.0 * (2**attempt)))
    if last_exc:
        raise last_exc
    return []


def probe_one(row: dict[str, str]) -> dict[str, str]:
    host = to_host(row.get("homepage_url", ""))
    result = {
        "org_uuid": row.get("org_uuid", ""),
        "name": row.get("name", ""),
        "homepage_url": row.get("homepage_url", ""),
        "founded_date": row.get("founded_date", ""),
        "host": host,
        "has_2023": "", "closest_ts": "", "days_from_target": "",
        "n_window_captures": "", "has_any_ever": "", "status": "",
    }
    if not host:
        result["status"] = "no_host"
        return result

    try:
        window = _cdx_get(
            f"url={quote(host)}&from={WINDOW_FROM}&to={WINDOW_TO}"
            "&filter=statuscode:200&filter=mimetype:text/html"
            "&collapse=timestamp:8&limit=400&output=json"
        )
    except Exception as exc:  # noqa: BLE001 - record, don't crash the sweep
        result["status"] = f"error:{type(exc).__name__}"
        return result

    result["n_window_captures"] = str(len(window))
    if window:
        best_ts, best_days = "", None
        for cap in window:
            ts = cap[1]
            try:
                days = abs((datetime.strptime(ts, "%Y%m%d%H%M%S") - TARGET).days)
            except ValueError:
                continue
            if best_days is None or days < best_days:
                best_days, best_ts = days, ts
        result.update(
            has_2023="True",
            closest_ts=best_ts,
            days_from_target=str(best_days if best_days is not None else ""),
            has_any_ever="True",
            status="ok",
        )
        return result

    result["has_2023"] = "False"
    if not DO_EVER_CALL:
        result["has_any_ever"] = ""
        result["status"] = "ok"
        return result
    try:
        ever = _cdx_get(
            f"url={quote(host)}&filter=statuscode:200&collapse=urlkey&limit=1&output=json"
        )
        result["has_any_ever"] = "True" if ever else "False"
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"error_ever:{type(exc).__name__}"
    return result


def load_resolved_ids(path: Path) -> set[str]:
    """Return org_uuids already resolved (status ok) — never query the API again.

    Includes confirmed no-March-2023 misses (ok + has_2023=False). Error rows are
    intentionally NOT counted, so a resumed run retries throttle failures only.
    """
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("org_uuid") and r.get("status") == "ok":
                out.add(r["org_uuid"])
    return out


def main() -> None:
    global DO_EVER_CALL, _LIMITER, _CDX_RETRIES, _CDX_TIMEOUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=300,
                        help="Companies to probe. Use 0 (or >= cohort) for the full census.")
    parser.add_argument("--concurrency", type=int, default=CDX_DEFAULT_CONCURRENCY,
                        help="Worker threads (keep at 1 unless rpm is raised).")
    parser.add_argument("--rpm", type=float, default=float(CDX_SAFE_RPM),
                        help="Max CDX requests/min globally (IA hard cap 60; safe 48).")
    parser.add_argument("--retries", type=int, default=CDX_DEFAULT_RETRIES,
                        help="Per-request retries before recording an error row.")
    parser.add_argument("--timeout", type=float, default=CDX_DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--skip-ever-call", action="store_true",
                        help="Skip the second 'archived ever?' call on misses (faster for big runs).")
    args = parser.parse_args()

    if args.rpm > 60:
        parser.error("--rpm cannot exceed IA hard cap of 60/min for /cdx/*")
    if args.concurrency > 1 and args.rpm >= 48:
        print("  note: concurrency>1 with rpm≤48 rarely helps; workers share one rate limit",
              flush=True)

    _LIMITER = CdxRateLimiter(args.rpm)
    _CDX_RETRIES = args.retries
    _CDX_TIMEOUT = args.timeout
    DO_EVER_CALL = not args.skip_ever_call

    with COHORT_CSV.open(encoding="utf-8", newline="") as f:
        cohort = list(csv.DictReader(f))

    if args.sample_size <= 0 or args.sample_size >= len(cohort):
        targets = list(cohort)
        random.Random(args.seed).shuffle(targets)
        mode = f"FULL census ({len(cohort):,}), shuffled seed {args.seed}"
    else:
        rng = random.Random(args.seed)
        targets = rng.sample(cohort, args.sample_size)
        mode = f"sample={args.sample_size} (seed {args.seed})"

    done = load_resolved_ids(args.output)
    todo = [r for r in targets if r.get("org_uuid") not in done]
    print(
        f"Cohort={len(cohort):,}  {mode}  already_resolved={len(done)}  to_probe={len(todo)}  "
        f"concurrency={args.concurrency}  rpm={args.rpm:.0f}  retries={args.retries}  "
        f"ever_call={DO_EVER_CALL}",
        flush=True,
    )
    if not todo:
        print("Nothing to do.", flush=True)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    lock = threading.Lock()
    counter = {"n": 0, "hit": 0, "err": 0}
    started = time.monotonic()

    with args.output.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
            out.flush()

        def work(row: dict[str, str]) -> None:
            res = probe_one(row)
            with lock:
                writer.writerow(res)
                out.flush()
                counter["n"] += 1
                if res.get("status") != "ok":
                    counter["err"] += 1
                elif res.get("has_2023") == "True":
                    counter["hit"] += 1
                if counter["n"] % 100 == 0:
                    n, hit, err = counter["n"], counter["hit"], counter["err"]
                    ok = n - err
                    pct = hit / ok * 100 if ok else 0.0
                    rate = n / (time.monotonic() - started) * 60
                    eta_h = (len(todo) - n) / (rate / 60) / 3600 if rate else float("inf")
                    print(
                        f"  probed={n:,}/{len(todo):,}  resolved={ok:,}  err={err:,}  "
                        f"has_2023={hit:,} ({pct:.1f}%)  {rate:.0f}/min  ETA={eta_h:.1f}h",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))

    n, hit, err = counter["n"], counter["hit"], counter["err"]
    ok = n - err
    pct = hit / ok * 100 if ok else 0.0
    print(
        f"DONE  probed={n:,}  resolved={ok:,}  errors={err:,}  "
        f"has_2023={hit:,} ({pct:.1f}% of resolved)  -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
