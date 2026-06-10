#!/usr/bin/env python3
"""Estimate March-2023 Wayback retrievability for the frozen cohort.

Samples N companies from wayback_cohort.csv and asks the Internet Archive's CDX
Server API whether each homepage has a 200/HTML capture near 2023-03-14. This is
a feasibility measurement, not the real scrape: we sample (not census) so a few
hundred queries give a tight estimate of the coverage percentage before we
commit to building the full historical pipeline.

Resumable: re-running skips org_uuids already in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "wayback_cohort.csv"
OUTPUT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "coverage_sample.csv"

CDX_ENDPOINT = "http://web.archive.org/cdx/search/cdx"
TARGET = datetime(2023, 3, 14)
WINDOW_FROM = "20221201"
WINDOW_TO = "20230630"

# Tunables set from CLI args in main(); read by probe_one in worker threads.
JITTER = (0.3, 0.9)
DO_EVER_CALL = True

OUTPUT_FIELDS = [
    "org_uuid", "name", "homepage_url", "founded_date", "host",
    "has_2023", "closest_ts", "days_from_target", "n_window_captures",
    "has_any_ever", "status",
]


def to_host(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    net = urlparse(u).netloc.lower()
    return net[4:] if net.startswith("www.") else net


def _cdx_get(params: str, *, retries: int = 2, timeout: float = 40.0) -> list[list[str]]:
    """GET the CDX endpoint, retrying transient failures. Returns parsed rows minus header."""
    url = f"{CDX_ENDPOINT}?{params}"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wayback-coverage-probe/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip()
            if not body:
                return []
            data = json.loads(body)
            return data[1:] if len(data) > 1 else []
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 6.0 * (attempt + 1)
                time.sleep(wait)
            elif exc.code in {500, 502, 503, 504}:
                time.sleep(2.0 * (attempt + 1))
            else:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            # Connection drops = throttling; a cheap single retry, then move on (errors retried next run).
            time.sleep(2.0 * (attempt + 1))
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

    time.sleep(random.uniform(*JITTER))  # politeness jitter — keep us under the CDX rate limit
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
        result.update(has_2023="True", closest_ts=best_ts,
                      days_from_target=str(best_days if best_days is not None else ""),
                      has_any_ever="True", status="ok")
        return result

    # No 2023 window capture. Optionally check whether the domain is archived at all.
    # Skipped on the full-census run to halve calls on misses (we already learned
    # from the 300 sample that "never archived" is ~0).
    result["has_2023"] = "False"
    if not DO_EVER_CALL:
        result["has_any_ever"] = ""
        result["status"] = "ok"
        return result
    try:
        ever = _cdx_get(f"url={quote(host)}&filter=statuscode:200&collapse=urlkey&limit=1&output=json")
        result["has_any_ever"] = "True" if ever else "False"
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"error_ever:{type(exc).__name__}"
    return result


def load_resolved_ids(path: Path) -> set[str]:
    """Return org_uuids already resolved (status ok) in a prior run, so we skip them.

    Error rows are intentionally NOT counted, so a resumed run retries them.
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
    global JITTER, DO_EVER_CALL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=300,
                        help="Companies to probe. Use 0 (or >= cohort) for the full census.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--skip-ever-call", action="store_true",
                        help="Skip the second 'archived ever?' call on misses (faster for big runs).")
    parser.add_argument("--jitter-min", type=float, default=0.3)
    parser.add_argument("--jitter-max", type=float, default=0.9)
    args = parser.parse_args()

    JITTER = (args.jitter_min, args.jitter_max)
    DO_EVER_CALL = not args.skip_ever_call

    with COHORT_CSV.open(encoding="utf-8", newline="") as f:
        cohort = list(csv.DictReader(f))

    if args.sample_size <= 0 or args.sample_size >= len(cohort):
        # Full census, but shuffled (fixed seed) so an interrupted overnight run leaves an
        # UNBIASED random subset rather than a non-random prefix of the cohort file.
        targets = list(cohort)
        random.Random(args.seed).shuffle(targets)
        mode = f"FULL census ({len(cohort):,}), shuffled seed {args.seed}"
    else:
        rng = random.Random(args.seed)
        targets = rng.sample(cohort, args.sample_size)
        mode = f"sample={args.sample_size} (seed {args.seed})"

    done = load_resolved_ids(args.output)
    todo = [r for r in targets if r.get("org_uuid") not in done]
    print(f"Cohort={len(cohort):,}  {mode}  already_resolved={len(done)}  to_probe={len(todo)}  "
          f"concurrency={args.concurrency}  ever_call={DO_EVER_CALL}", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    # Incremental append + flush per row: an interruption (sleep, crash) loses at most
    # the in-flight rows, and a resumed run skips everything already written.
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
                    print(f"  probed={n:,}/{len(todo):,}  resolved={ok:,}  err={err:,}  "
                          f"has_2023={hit:,} ({pct:.1f}%)  {rate:.0f}/min  ETA={eta_h:.1f}h",
                          flush=True)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))

    n, hit, err = counter["n"], counter["hit"], counter["err"]
    ok = n - err
    pct = hit / ok * 100 if ok else 0.0
    print(f"DONE  probed={n:,}  resolved={ok:,}  errors={err:,}  "
          f"has_2023={hit:,} ({pct:.1f}% of resolved)  -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
