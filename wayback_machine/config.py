"""Tunables for the historical scrape. No magic numbers anywhere else.

The engine here calls Tavily ``/extract`` (not ``/crawl``): we already know the
exact archive snapshot URL per company, so there is nothing to crawl, just one
page to fetch and clean. The request shape is therefore much simpler than the
live crawl config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Discovery target (kept for reference; discovery already ran)
# ---------------------------------------------------------------------------

TARGET_DATE = "20230314"          # GPT-4 launch, the anchor for "March 2023"
WINDOW_FROM = "20221201"
WINDOW_TO = "20230630"

# CDX Server API rate limits (Internet Archive staff, wayback-py #137):
#   /cdx/* hard cap ≈ 60 requests/minute (averaged over 5 minutes).
#   Exceeding → HTTP 429; ignoring 429s >1 min → IP firewall block (1h, doubles).
#   Recommended client default: 80% of hard cap = 48/min.
CDX_HARD_LIMIT_RPM = 60
CDX_SAFE_RPM = 48
CDX_429_MIN_PAUSE_SECONDS = 60.0
CDX_DEFAULT_RETRIES = 8
CDX_DEFAULT_TIMEOUT_SECONDS = 60.0
CDX_DEFAULT_CONCURRENCY = 1

# Cohort existence cutoff: only study companies that existed at GPT-4 launch.
# founded_date is YYYY-MM in our data; inclusive of March 2023. A company founded
# after this could still have a (later) capture in the probe window, so this is a
# real filter, not a no-op.
COHORT_FOUNDED_CUTOFF = "2023-03"

# The ``id_`` suffix on the timestamp tells Wayback to return the raw archived
# bytes WITHOUT its navigation toolbar — verified in the discovery spike.
SNAPSHOT_HOST = "http://web.archive.org"
SNAPSHOT_SUFFIX = "id_"

# ---------------------------------------------------------------------------
# Tavily Extract endpoint + request defaults
# ---------------------------------------------------------------------------

TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"

# Tavily documents the extract endpoint separately from the default API RPM.
# We keep a conservative cap with headroom; the real bottleneck is the Internet
# Archive throttling Tavily's fetches, which surfaces as empty/failed results.
TAVILY_EXTRACT_RPM_DOCUMENTED = 100
DEFAULT_EXTRACT_RPM_HEADROOM = 0.8
DEFAULT_MAX_CONCURRENT_ROWS = 4

# Per-call retries handle 30s blips; the outage layer handles multi-minute
# Archive/Tavily outages without losing the row.
DEFAULT_MAX_OUTAGE_SECONDS = 1800.0
DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS = 60.0
DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS = 600.0

DEFAULT_HEARTBEAT_EVERY = 100

# ---------------------------------------------------------------------------
# Budget control (call-count based — extract returns no usage field)
# ---------------------------------------------------------------------------

# Tavily bills extract in blocks of 5 successful URL extractions (failed rows free).
# Basic: 1 credit / 5 successes. Advanced: 2 credits / 5 successes.
# https://docs.tavily.com/documentation/api-credits
EXTRACTIONS_PER_CREDIT_BLOCK = 5
CREDITS_PER_BLOCK_BASIC = 1
CREDITS_PER_BLOCK_ADVANCED = 2
# Back-compat alias used in comments elsewhere.
EXTRACTIONS_PER_CREDIT = EXTRACTIONS_PER_CREDIT_BLOCK
# Generous ceiling: ~16k retrievable companies ≈ ~3,200 credits. The cap is a
# guardrail against a runaway loop, not an expected limit.
DEFAULT_BUDGET_CREDITS = 50_000.0


@dataclass(frozen=True)
class ExtractConfig:
    """One Tavily ``/extract`` request's parameters.

    No ``query``/``chunks_per_source``: we want the full page content so the
    cleaner sees the same raw markdown the live crawl produced, keeping the
    2023 vs today comparison fair.
    """

    extract_depth: str = "basic"
    format: str = "markdown"
    include_images: bool = False
    include_favicon: bool = False
    timeout: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0

    def request_payload(self, url: str) -> dict[str, Any]:
        """Return the JSON body for one extract request (single URL)."""
        return {
            "urls": [url],
            "extract_depth": self.extract_depth,
            "format": self.format,
            "include_images": self.include_images,
            "include_favicon": self.include_favicon,
        }


def estimate_credits(successful_extractions: int, *, extract_depth: str = "basic") -> float:
    """Estimate API credits spent for N successful extractions at the given depth."""
    if successful_extractions <= 0:
        return 0.0
    credits_per_block = (
        CREDITS_PER_BLOCK_ADVANCED if extract_depth == "advanced" else CREDITS_PER_BLOCK_BASIC
    )
    return successful_extractions * credits_per_block / float(EXTRACTIONS_PER_CREDIT_BLOCK)
