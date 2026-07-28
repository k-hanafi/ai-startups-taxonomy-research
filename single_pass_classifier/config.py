"""Pipeline configuration: Tier 5 rate limits, model defaults, and env loading.

All Tier 5 limits are sourced from the OpenAI rate limits documentation.
No magic numbers should appear in any other module. Import from here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parents[1] / "keys" / "openai.env"
load_dotenv(_ENV_FILE)

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "gpt-5.4-nano"

# Shared across all 267K requests. Improves cache routing to the same server.
# In-memory caching only for gpt-5.4-nano (5-10 min idle, up to 1h max).
# Switch to gpt-5.4 or gpt-5.1 to unlock 24h extended prompt caching.
PROMPT_CACHE_KEY: str = "v2-classifier-system-prompt"

# ---------------------------------------------------------------------------
# Tier 5 rate limits (source: platform.openai.com/docs/guides/rate-limits)
# ---------------------------------------------------------------------------

# Sync API limits — not consumed by the Batch API (separate pool).
# Kept here for reference; do not use these to gate batch submissions.
MAX_REQUESTS_PER_MINUTE: int = 30_000
MAX_TOKENS_PER_MINUTE: int = 180_000_000        # 180M TPM

# Batch API limits (separate pool from sync).
# Pending batch tokens count against this limit until each batch completes.
MAX_BATCH_QUEUE_TOKENS: int = 15_000_000_000    # 15B enqueued prompt tokens
MAX_REQUESTS_PER_BATCH: int = 50_000            # OpenAI hard limit per batch file
BATCH_CREATION_PER_HOUR: int = 2_000            # OpenAI hard limit
# OpenAI hard limit is 200 MB. Calibrated from the 1k pilot: 1k requests = 36 MB,
# so the safe ceiling is ~5,000 requests/file (180 MB), with 20 MB headroom.
MAX_FILE_SIZE_MB: int = 190                     # 200 MB hard limit minus 10 MB margin

# ---------------------------------------------------------------------------
# Per-request budget
# ---------------------------------------------------------------------------

# Measured from full 44k dataset (500-row sample): 9,593 avg prompt tokens.
# P95 = 12,503; max = 28,068 (outlier with very large website evidence).
# Used by the sliding-window queue pressure control to stay under 15B queue limit.
ESTIMATED_TOKENS_PER_REQUEST: int = 10_000

# Hard cap on model output per request. The v2 output is ~150-300 tokens.
# 1000 gives 4-5× headroom over observed avg (210) without risk of truncation.
MAX_OUTPUT_TOKENS: int = 1000

# ---------------------------------------------------------------------------
# Batch construction defaults
# ---------------------------------------------------------------------------

# Calibrated from full 44k dataset: avg 51 KB/row (website evidence increases size
# vs the 1k pilot which had 36 KB/row). 3500 rows = ~170 MB avg, 176 MB max.
# Stays comfortably under the 190 MB soft cap and 200 MB OpenAI hard limit.
DEFAULT_BATCH_SIZE: int = 3_500
