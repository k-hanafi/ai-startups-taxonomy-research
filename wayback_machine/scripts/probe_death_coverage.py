#!/usr/bin/env python3
"""Death-anchored Wayback coverage probe for the survivorship-bias cohort.

For each 'Tavily not found' company, query the Internet Archive CDX API for the
full history of 200/HTML captures, treat the most recent capture as the site's
"death", and pick the captured day closest to (death - ~6 months). Scraping that
pre-death snapshot avoids the parked/dead tail a final snapshot often contains.

Free (CDX is unauthenticated), resumable (skip already-resolved org_uuids), and
rate-limited under the Archive's 60/min per-IP cap. One CDX call per company;
concurrency only fills request latency and never exceeds the shared limit.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.cdx import CdxClient, RateLimiter, to_host  # noqa: E402
from wayback_machine.config import (  # noqa: E402
    CDX_429_MIN_PAUSE_SECONDS,
    CDX_DEATH_CONCURRENCY,
    CDX_DEFAULT_RETRIES,
    CDX_DEFAULT_TIMEOUT_SECONDS,
    CDX_FAST_RPM,
    CDX_HARD_LIMIT_RPM,
    DEATH_LOOKBACK_DAYS,
)

COHORT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "not_found_cohort.csv"
OUTPUT_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "death_coverage.csv"

# Daily-collapsed captures fetched per host. Far more than any startup accrues,
# so a single call always contains the true first and last (death) captures.
# (Querying the bare host is sufficient for the homepage: Wayback's SURT key
# strips ``www.``, so ``example.com/`` and ``www.example.com/`` share one key,
# while subdomains and subpaths get distinct keys and are correctly excluded.)
HISTORY_LIMIT = 20000

OUTPUT_FIELDS = [
    "org_uuid", "name", "homepage_url", "website_alive", "founded_date", "host",
    "n_captures", "first_ts", "death_ts", "target_ts", "closest_ts",
    "latest_url", "target_url", "earliest_url",
    "days_from_target", "days_before_death", "lifespan_days",
    "has_pre_death_snapshot", "thin_history", "status",
]

# A row is "resolved" (never re-queried on resume) once we have a definitive
# answer. Error rows are intentionally excluded so a re-run retries throttles.
RESOLVED_STATUSES = {"ok", "no_snapshots", "no_host"}

_CLIENT: CdxClient | None = None


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S")
    except (ValueError, TypeError):
        return None


def _archive_url(timestamp: str, homepage_url: str) -> str:
    """Human-viewable Wayback URL (with toolbar) for a given capture timestamp.

    No ``id_`` suffix: this renders the normal Wayback viewer for auditing. The
    raw byte URL used for Tavily extraction (``id_``) is built later by
    ``cohort.build_snapshot_url`` from the same ``closest_ts``.
    """
    url = (homepage_url or "").strip()
    if "://" not in url:
        url = "https://" + url
    return f"https://web.archive.org/web/{timestamp}/{url}"


def probe_one(row: dict[str, str]) -> dict[str, str]:
    if _CLIENT is None:
        msg = "CdxClient not initialized"
        raise RuntimeError(msg)

    host = to_host(row.get("homepage_url", ""))
    result = {k: "" for k in OUTPUT_FIELDS}
    result.update({
        "org_uuid": row.get("org_uuid", ""),
        "name": row.get("name", ""),
        "homepage_url": row.get("homepage_url", ""),
        "website_alive": row.get("website_alive", ""),
        "founded_date": row.get("founded_date", ""),
        "host": host,
    })
    if not host:
        result["status"] = "no_host"
        return result

    try:
        rows = _CLIENT.get(
            f"url={quote(host)}"
            "&filter=statuscode:200&filter=mimetype:text/html"
            f"&collapse=timestamp:8&fl=timestamp&limit={HISTORY_LIMIT}&output=json"
        )
    except Exception as exc:  # noqa: BLE001 - record, don't crash the sweep
        result["status"] = f"error:{type(exc).__name__}"
        return result

    # With fl=timestamp each row is a single-field list: [timestamp].
    captures = sorted(
        dt for dt in (_parse_ts(r[0]) for r in rows if r) if dt is not None
    )
    result["n_captures"] = str(len(captures))
    if not captures:
        # Empty body => never archived; non-empty but unparseable => bad data.
        result["status"] = "error:bad_timestamp" if rows else "no_snapshots"
        return result

    first_dt, death_dt = captures[0], captures[-1]
    target_dt = death_dt - timedelta(days=DEATH_LOOKBACK_DAYS)
    # Prefer the most recent capture at or before the 6-months-before-death mark.
    # This guarantees we never land in the parked/dead tail while staying as fresh
    # as possible. Only when the whole history is younger than 6 months (nothing
    # at or before target) do we fall back to the earliest capture and flag it.
    pre_target = [dt for dt in captures if dt <= target_dt]
    closest_dt = pre_target[-1] if pre_target else first_dt

    first_ts = first_dt.strftime("%Y%m%d%H%M%S")
    death_ts = death_dt.strftime("%Y%m%d%H%M%S")
    closest_ts = closest_dt.strftime("%Y%m%d%H%M%S")
    homepage = result["homepage_url"]
    result.update({
        "first_ts": first_ts,
        "death_ts": death_ts,
        "target_ts": target_dt.strftime("%Y%m%d%H%M%S"),
        "closest_ts": closest_ts,
        "latest_url": _archive_url(death_ts, homepage),
        "target_url": _archive_url(closest_ts, homepage),
        "earliest_url": _archive_url(first_ts, homepage),
        "days_from_target": str(abs((closest_dt - target_dt).days)),
        "days_before_death": str((death_dt - closest_dt).days),
        "lifespan_days": str((death_dt - first_dt).days),
        "has_pre_death_snapshot": "True" if pre_target else "False",
        "thin_history": "False" if pre_target else "True",
        "status": "ok",
    })
    return result


def load_resolved_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("org_uuid") and r.get("status") in RESOLVED_STATUSES:
                out.add(r["org_uuid"])
    return out


def main() -> None:
    global _CLIENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Probe first N after shuffle (smoke test); 0 = full cohort.")
    parser.add_argument("--rpm", type=float, default=float(CDX_FAST_RPM),
                        help=f"Global CDX requests/min (IA hard cap {CDX_HARD_LIMIT_RPM}).")
    parser.add_argument("--concurrency", type=int, default=CDX_DEATH_CONCURRENCY,
                        help="Worker threads sharing the one global rate limit.")
    parser.add_argument("--retries", type=int, default=CDX_DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=float, default=CDX_DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rpm > CDX_HARD_LIMIT_RPM:
        parser.error(f"--rpm cannot exceed the IA hard cap of {CDX_HARD_LIMIT_RPM}/min for /cdx/*")
    if not args.cohort.exists():
        parser.error(f"cohort not found: {args.cohort} (run build_not_found_cohort.py first)")

    limiter = RateLimiter(args.rpm, min_pause_on_429=CDX_429_MIN_PAUSE_SECONDS)
    _CLIENT = CdxClient(limiter, retries=args.retries, timeout=args.timeout)

    with args.cohort.open(encoding="utf-8", newline="") as f:
        cohort = list(csv.DictReader(f))
    random.Random(args.seed).shuffle(cohort)
    if 0 < args.sample_size < len(cohort):
        cohort = cohort[: args.sample_size]
        mode = f"sample={args.sample_size}"
    else:
        mode = f"FULL cohort ({len(cohort):,})"

    done = load_resolved_ids(args.output)
    todo = [r for r in cohort if r.get("org_uuid") not in done]
    print(
        f"cohort={len(cohort):,}  {mode}  already_resolved={len(done):,}  to_probe={len(todo):,}  "
        f"rpm={args.rpm:.0f}  concurrency={args.concurrency}  lookback={DEATH_LOOKBACK_DAYS}d",
        flush=True,
    )
    if not todo:
        print("Nothing to do.", flush=True)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    lock = threading.Lock()
    counter = {"n": 0, "hit": 0, "thin": 0, "none": 0, "err": 0}
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
                status = res.get("status", "")
                if status == "ok":
                    counter["hit" if res.get("has_pre_death_snapshot") == "True" else "thin"] += 1
                elif status == "no_snapshots":
                    counter["none"] += 1
                else:
                    counter["err"] += 1
                if counter["n"] % 100 == 0:
                    n = counter["n"]
                    rate = n / (time.monotonic() - started) * 60
                    eta_h = (len(todo) - n) / (rate / 60) / 3600 if rate else float("inf")
                    print(
                        f"  probed={n:,}/{len(todo):,}  pre_death={counter['hit']:,}  "
                        f"thin={counter['thin']:,}  none={counter['none']:,}  err={counter['err']:,}  "
                        f"{rate:.0f}/min  ETA={eta_h:.1f}h",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))

    n = counter["n"]
    print(
        f"DONE  probed={n:,}  pre_death={counter['hit']:,}  thin={counter['thin']:,}  "
        f"none={counter['none']:,}  err={counter['err']:,}  -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
