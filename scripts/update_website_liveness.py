#!/usr/bin/env python3
"""Populate ``website_alive`` in data/master_csv.csv by probing each homepage.

Runs a parallel HTTP GET against every row whose ``website_alive`` is blank
(or every row when --force-recheck is passed). Writes true/false back to
master_csv.csv in-place. Safe to re-run — already-checked rows are skipped
by default.

Usage:
    python scripts/update_website_liveness.py            # check blank rows
    python scripts/update_website_liveness.py --force-recheck  # re-check all
    python scripts/update_website_liveness.py --workers 64 --timeout 10
"""

from __future__ import annotations

import argparse
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.master_csv import DEFAULT_MASTER_CSV, MASTER_CSV_COLUMNS, is_valid_homepage_url

USER_AGENT = "ai-native-startup-classification/liveness-check (+https://github.com)"

_PARKED_MARKERS = (
    "domain is for sale",
    "domain for sale",
    "this domain may be for sale",
    "this domain is parked",
    "parked free",
    "sedo domain parking",
    "forsale",
    "buy this domain",
    "hugedomains",
    "afternic",
    "dan.com/inquire",
    "domain has expired",
    "renew your domain",
    "under construction",
    "site not found",
    "default web site page",
    "welcome to nginx",
    "it works!",
)


def _looks_parked(html_snippet: str) -> bool:
    lower = html_snippet.lower()
    return any(m in lower for m in _PARKED_MARKERS)


def probe_website_alive(url: str, *, timeout: float, ssl_context: ssl.SSLContext) -> str:
    """Return ``'true'`` or ``'false'`` based on a stdlib HTTP GET."""
    raw = str(url).strip() if url is not None else ""
    if not is_valid_homepage_url(raw):
        return "false"

    req = request.Request(raw, headers={"User-Agent": USER_AGENT})
    try:
        resp = request.urlopen(req, timeout=timeout, context=ssl_context)
    except Exception:
        return "false"
    else:
        try:
            code = resp.getcode()
            snippet = resp.read(32_768).decode("utf-8", errors="ignore")
        finally:
            resp.close()

        if code is not None and not (200 <= int(code) < 400):
            return "false"
        if _looks_parked(snippet):
            return "false"
        return "true"


def _path(value: str) -> Path:
    return Path(value).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=_path, default=DEFAULT_MASTER_CSV,
                        help="Path to master_csv.csv (updated in-place).")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel HTTP workers (default 32).")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Per-request timeout in seconds (default 15).")
    parser.add_argument("--force-recheck", action="store_true",
                        help="Re-probe all rows, not just blank website_alive rows.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of rows to probe (for testing).")
    args = parser.parse_args()

    path: Path = args.input
    if not path.is_file():
        raise SystemExit(f"master_csv.csv not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = [c for c in MASTER_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing expected columns: {missing}")

    if args.max_rows is not None:
        df = df.iloc[: int(args.max_rows)].copy()

    to_check: list[int] = []
    for i in df.index:
        alive_val = str(df.at[i, "website_alive"]).strip().lower()
        if args.force_recheck or alive_val not in {"true", "false"}:
            to_check.append(int(i))

    print(f"Rows to probe: {len(to_check):,}  (total {len(df):,})", flush=True)
    if not to_check:
        print("Nothing to do.")
        return

    ssl_context = ssl.create_default_context()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {
            ex.submit(probe_website_alive, df.at[i, "homepage_url"],
                      timeout=args.timeout, ssl_context=ssl_context): i
            for i in to_check
        }
        done = 0
        true_count = 0
        false_count = 0
        for fut in as_completed(futs):
            i = futs[fut]
            result = fut.result()
            df.loc[i, "website_alive"] = result
            if result == "true":
                true_count += 1
            else:
                false_count += 1
            done += 1
            if done % 500 == 0 or done == len(futs):
                print(f"  {done:,} / {len(futs):,}  alive={true_count:,}  dead={false_count:,}",
                      flush=True)

    # Write back in MASTER_CSV_COLUMNS order so the schema stays locked.
    for col in MASTER_CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[MASTER_CSV_COLUMNS].to_csv(path, index=False)
    print(f"\nWrote {path}")
    print(f"  website_alive=true:  {true_count:,}")
    print(f"  website_alive=false: {false_count:,}")


if __name__ == "__main__":
    main()
