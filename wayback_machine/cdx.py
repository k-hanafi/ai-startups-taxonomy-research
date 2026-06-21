"""Minimal Internet Archive CDX Server API client (shared by the death probe).

The Archive hard-caps ``/cdx/*`` at 60 requests/minute per IP (averaged over a
5-minute window). Exceeding it returns HTTP 429; ignoring 429s for >1 minute
triggers an hour-long IP firewall block that doubles on repeat. This client
paces every request through one shared limiter and freezes all callers on a 429
(honoring ``Retry-After``), so concurrency can saturate the rate without ever
exceeding it.

Kept separate from ``scripts/probe_coverage.py`` on purpose: that historical
probe and its test stay untouched. The small overlap can be DRYed later.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

CDX_ENDPOINT = "http://web.archive.org/cdx/search/cdx"
USER_AGENT = "wayback-death-coverage-probe/1.0 (research; contact: batchkit)"


def to_host(url: str) -> str:
    """Normalize a homepage URL to a bare host (no scheme, no path, no ``www.``)."""
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    net = urlparse(u).netloc.lower()
    return net[4:] if net.startswith("www.") else net


class RateLimiter:
    """Global throttle: release one slot every ``60/rpm`` seconds; freeze on 429.

    Shared by every worker thread so the aggregate request rate stays under the
    Archive's per-IP budget regardless of how many workers are running.
    """

    def __init__(self, rpm: float, *, min_pause_on_429: float = 60.0) -> None:
        self._min_interval = 60.0 / rpm
        self._min_pause = min_pause_on_429
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._frozen_until = 0.0

    def wait_turn(self) -> None:
        # Reserve the next slot under the lock, then sleep OUTSIDE it. If we slept
        # while holding the lock, workers would serialize on it instead of running
        # requests in parallel; reserving-then-releasing lets ~N requests stay in
        # flight while the min-interval spacing still caps the aggregate rate.
        #
        # Loop so a 429 freeze installed by another worker DURING our sleep is
        # honored: without the re-check, slots reserved just before the freeze
        # would fire inside the back-off window and risk the IA firewall ban.
        while True:
            with self._lock:
                wake = max(self._next_slot, self._frozen_until, time.monotonic())
                self._next_slot = wake + self._min_interval
            delay = wake - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            with self._lock:
                if time.monotonic() >= self._frozen_until:
                    return

    def freeze_for_429(self, retry_after: float | None) -> float:
        with self._lock:
            pause = max(self._min_pause, retry_after or 0.0)
            self._frozen_until = max(self._frozen_until, time.monotonic() + pause)
            return pause


class CdxClient:
    """Rate-limited CDX GET with patient retries on 429 / 5xx / transient errors."""

    def __init__(self, limiter: RateLimiter, *, retries: int = 8, timeout: float = 60.0) -> None:
        self._limiter = limiter
        self._retries = retries
        self._timeout = timeout

    def get(self, params: str) -> list[list[str]]:
        """Return CDX rows (header row stripped) for a query string, or raise."""
        url = f"{CDX_ENDPOINT}?{params}"
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            self._limiter.wait_turn()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
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
                    pause = self._limiter.freeze_for_429(retry_after)
                    print(f"  CDX 429 - pausing all requests {pause:.0f}s", flush=True)
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
