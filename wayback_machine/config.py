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

# Basic extraction bills 1 API credit per 5 successful extractions.
EXTRACTIONS_PER_CREDIT = 5
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


def estimate_credits(successful_extractions: int) -> float:
    """Estimate API credits spent for N successful basic extractions."""
    if successful_extractions <= 0:
        return 0.0
    return successful_extractions / float(EXTRACTIONS_PER_CREDIT)
