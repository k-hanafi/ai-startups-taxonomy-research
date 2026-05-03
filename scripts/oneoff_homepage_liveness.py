#!/usr/bin/env python3
"""One-off: set ``website_alive`` on classifier_input.csv and refresh website evidence.

Safe to delete or leave untracked after the dataset has been annotated and verified.
"""

from __future__ import annotations

import argparse
import socket
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.enrichment import CLASSIFIER_INPUT_COLUMNS, DEFAULT_CLASSIFIER_INPUT_CSV, is_valid_homepage_url
from src.tavily_crawl import DEFAULT_RAW_RESULTS_JSONL
from src.website_evidence import build_classifier_input_with_evidence

USER_AGENT = "ai-native-startup-classification/homepage-liveness (+https://github.com)"

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
    """Return ``\"true\"`` or ``\"false\"`` (stdlib HTTP GET, first 32KiB for parking heuristics)."""
    raw = str(url).strip() if url is not None else ""
    if not is_valid_homepage_url(raw):
        return "false"

    req = request.Request(raw, headers={"User-Agent": USER_AGENT})
    try:
        resp = request.urlopen(req, timeout=timeout, context=ssl_context)
    except (ssl.SSLError, error.HTTPError, error.URLError, TimeoutError, OSError):
        return "false"
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=_path, default=DEFAULT_CLASSIFIER_INPUT_CSV)
    p.add_argument("--raw-jsonl", type=_path, default=DEFAULT_RAW_RESULTS_JSONL)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--force-recheck",
        action="store_true",
        help="Re-probe every row (default: only rows with empty website_alive).",
    )
    p.add_argument("--skip-evidence", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    path: Path = args.input
    if not path.is_file():
        raise SystemExit(f"Missing classifier CSV: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "homepage_url" not in df.columns:
        raise SystemExit("CSV must contain homepage_url")

    keep = [c for c in CLASSIFIER_INPUT_COLUMNS if c in df.columns]
    df = df[keep].copy()
    for c in CLASSIFIER_INPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    if args.max_rows is not None:
        df = df.iloc[: int(args.max_rows)].copy()

    ssl_context = ssl.create_default_context()
    to_check: list[int] = []
    for i in df.index:
        if args.force_recheck or not str(df.at[i, "website_alive"]).strip():
            to_check.append(int(i))

    print(f"Rows to probe: {len(to_check):,} (total {len(df):,})", flush=True)

    if to_check:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {
                ex.submit(probe_website_alive, df.at[i, "homepage_url"], timeout=args.timeout, ssl_context=ssl_context): i
                for i in to_check
            }
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                df.loc[i, "website_alive"] = fut.result()
                done += 1
                if done % 500 == 0 or done == len(futs):
                    print(f"  completed {done:,} / {len(futs):,}", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df[CLASSIFIER_INPUT_COLUMNS].to_csv(path, index=False)
    print(f"Wrote {path} ({len(df):,} rows)")

    if not args.skip_evidence:
        report = build_classifier_input_with_evidence(
            enriched_csv=path,
            raw_jsonl=args.raw_jsonl,
            output_csv=path,
        )
        print(report.format_report())


if __name__ == "__main__":
    main()
