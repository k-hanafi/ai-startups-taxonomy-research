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

MAX_REQUESTS_PER_MINUTE: int = 30_000
MAX_TOKENS_PER_MINUTE: int = 180_000_000        # 180M TPM

# Pending batch tokens count against this limit until each batch completes.
# Submitting all batches simultaneously risks exhausting the queue.
MAX_BATCH_QUEUE_TOKENS: int = 15_000_000_000    # 15B

MAX_REQUESTS_PER_BATCH: int = 50_000            # OpenAI hard limit per batch file
BATCH_CREATION_PER_HOUR: int = 2_000            # OpenAI hard limit
MAX_FILE_SIZE_MB: int = 190                     # OpenAI hard limit is 200 MB (10 MB margin)

# ---------------------------------------------------------------------------
# Per-request budget
# ---------------------------------------------------------------------------

# ~2K tokens (system prompt + schema prefix) + ~300 user message + ~200 output.
# Used by the token estimator and sliding-window queue pressure control.
ESTIMATED_TOKENS_PER_REQUEST: int = 2_500

# Hard cap on model output per request. The v2 output is ~150-300 tokens.
# This cap prevents runaway cost from any malformed response.
MAX_OUTPUT_TOKENS: int = 450

# ---------------------------------------------------------------------------
# Batch construction defaults
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 7_000     # requests per JSONL file, well under 50K limit
