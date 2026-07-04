---
name: V1→V2 Classifier Migration
overview: "Rebuild the startup classifier from a fragile v1 monolith into a world-class v2 CLI pipeline using every relevant OpenAI best practice: Pydantic structured outputs, tiktoken pre-flight cost estimation, error-file retry logic, prompt caching awareness, configurable concurrency, and a fully typed, logged, tested codebase."
todos:
  - id: scaffold-structure
    content: Create src/ package, outputs/ subdirs, tests/, update .gitignore, create pyproject.toml
    status: pending
  - id: config-schema
    content: Write src/config.py (Tier 5 limits, dotenv loading) and src/schema.py (Pydantic model → auto-generates JSON schema, no divergence)
    status: pending
  - id: token-cost
    content: Write src/tokens.py (tiktoken-based token counter + cost estimator with --dry-run support)
    status: pending
  - id: formatter-builder
    content: Write src/formatter.py (row → user message, truncation guards) and src/builder.py (JSONL builder with structured output + batch metadata)
    status: pending
  - id: submitter-monitor
    content: Write src/submitter.py (upload + create batches with metadata) and src/monitor.py (async concurrent polling with rich progress)
    status: pending
  - id: downloader-merger
    content: Write src/downloader.py (output file + error_file_id handling, custom_id matching, usage logging) and src/merger.py (final merge + stats report)
    status: pending
  - id: state-checkpoint
    content: Write src/state.py (JSON checkpoint tracking each batch through all lifecycle stages) and src/logger.py (structured logging to file + console)
    status: pending
  - id: cli-entry
    content: "Write classify.py CLI: prepare, submit, status, download, merge, retry, run subcommands; rich tables for all output"
    status: pending
  - id: tests
    content: Write tests/ covering schema validation, formatter output, token estimation, and custom_id round-trip
    status: pending
  - id: cleanup
    content: Archive old model-specific folders, update README with architecture, quickstart, research context, and cost breakdown
    status: pending
isProject: false
---

# V1 → V2 Classifier Migration Plan

## Core v1 → v2 Changes

- **Taxonomy**: Binary 7-field output → two-axis 11-field output (`ai_native`, `subclass`, `rad_score`, `cohort`, `conf_classification`, `conf_rad`, `reasons_3_points`, `sources_used`, `verification_critique`)
- **Prompt**: `system_prompt.txt` (binary) → `[prompts/Multiclassification_prompt.txt](prompts/Multiclassification_prompt.txt)`
- **Response format**: Brittle CSV text parsing → **Pydantic + OpenAI Structured Outputs** (`json_schema`, `strict: true`) — zero parse failures, type-safe
- **Model name**: `gpt-5-nano` (old) → `**gpt-5.4-nano`** (confirmed from OpenAI docs)
- **Folder structure**: Two model-specific folders → single `src/` package, model switchable via `--model`
- **Concurrency**: Sequential → `--concurrency N` (default 1, test with 1, scale to max for full run)
- **Tier 5 limits**: Updated to 30k RPM, 180M TPM, 15B batch queue
- **API key**: Per-folder `api_key.txt` → `keys/openai.env` loaded via `python-dotenv`

## New Project Structure

```
ai-native-startup-classification/
├── src/
│   ├── __init__.py
│   ├── config.py        # Pydantic Settings; loads keys/openai.env + CLI overrides
│   ├── schema.py        # Pydantic BaseModel for 11-field output → auto-generates JSON schema
│   ├── tokens.py        # tiktoken token counter + cost estimator (--dry-run)
│   ├── formatter.py     # CSV row → user message string (with length guards)
│   ├── builder.py       # Build JSONL batch files (structured output body + batch metadata)
│   ├── submitter.py     # Upload JSONL + create batches with metadata tags
│   ├── monitor.py       # asyncio concurrent batch polling with rich progress display
│   ├── downloader.py    # Download output_file + error_file_id; match by custom_id
│   ├── merger.py        # Merge per-batch CSVs; emit usage stats + classification report
│   ├── state.py         # outputs/state.json checkpoint (survives any crash/restart)
│   └── logger.py        # Structured logging: rotating file + rich console handler
├── tests/
│   ├── test_schema.py   # Pydantic model validation and enum enforcement
│   ├── test_formatter.py
│   └── test_tokens.py
├── classify.py          # CLI entry point (argparse subcommands)
├── pyproject.toml       # Modern Python packaging + dev dependencies
├── outputs/             # Runtime-generated, gitignored
│   ├── batch_requests/  # JSONL input files
│   ├── batch_results/   # Raw JSONL responses
│   ├── batch_errors/    # Error files from error_file_id
│   ├── batch_outputs/   # Parsed per-batch CSVs
│   ├── state.json        # Checkpoint
│   ├── run.log           # Structured log file
│   └── classified_startups_v2.csv
├── data/                # gitignored
├── keys/                # gitignored
└── prompts/
    └── Multiclassification_prompt.txt
```

## CLI Design

```bash
# Pre-flight: count tokens, estimate cost, show batch plan — no API calls
python classify.py prepare --dry-run

# Build JSONL batch files (creates outputs/batch_requests/)
python classify.py prepare [--model gpt-5.4-nano] [--batch-size 7000] [--rows 0:50000]

# Submit N batches at a time; adds metadata tags per batch for traceability
python classify.py submit [--concurrency 3]

# Live status table of all in-flight batches (requests: total/completed/failed)
python classify.py status

# Download completed output files + error files; log per-batch token usage
python classify.py download

# Retry all failed/expired custom_ids from error files as a new batch
python classify.py retry

# Merge all batch outputs → final CSV + print classification distribution
python classify.py merge [--output outputs/classified_v2.csv]

# All-in-one pipeline
python classify.py run [--model gpt-5.4-nano] [--concurrency 5] [--dry-run]
```

## Best Practices from OpenAI Docs — Applied Here

### 1. Pydantic Structured Outputs (no schema divergence)

Define one `ClassificationResult(BaseModel)` — the SDK generates the JSON schema automatically. No risk of the Python dataclass and JSON schema drifting apart. Per the docs: *"strongly recommend using the native Pydantic SDK support"*.

```python
from pydantic import BaseModel
from typing import Literal

class ClassificationResult(BaseModel):
    CompanyID: str
    CompanyName: str
    ai_native: Literal[0, 1]
    subclass: Literal["1A","1B","1C","1D","1E","0A","0B","0C-THIN","0C-THICK","0D","0E"]
    rad_score: Literal["RAD-H","RAD-M","RAD-L","RAD-NA"]
    cohort: Literal["PRE-GENAI","GENAI-ERA"]
    conf_classification: int  # 1–5
    conf_rad: int             # 1–5
    reasons_3_points: str
    sources_used: str
    verification_critique: str
```

The schema is injected into each JSONL request body via `ClassificationResult.model_json_schema()`.

### 2. tiktoken Pre-flight Token Counting + Cost Estimation

Before any API call, `src/tokens.py` counts exact tokens using `tiktoken` (same tokenizer OpenAI uses). The `--dry-run` flag prints a full cost breakdown without submitting anything:

```
Model:          gpt-5.4-nano
Companies:      267,790
Input tokens:   ~621M  ($0.20/MTok)   → ~$124
Output tokens:  ~107M  ($1.25/MTok)   → ~$134
Batch discount: 50%                   → ~$129 total
Batches needed: 39 (at 7,000/batch)
```

Per the docs, Batch API gives **50% cost discount** vs synchronous API.

### 3. `max_tokens` Guard

Set `max_tokens=450` on every request body. The v2 output is ~150–300 tokens; this cap prevents runaway cost from any malformed model response while giving enough headroom.

### 4. Prompt Caching — Structural Design + `prompt_cache_key`

All 267K requests share an **identical prefix**: the JSON schema (injected first by OpenAI) + system prompt (~2,400 tokens combined). Prompt caching is automatic and free — up to **90% reduction in input token cost** and 80% latency reduction on cache hits.

Two concrete actions to maximize this:

**Structural rule**: the schema and system prompt are always first in every JSONL request body, never modified between requests. User message (which varies) always goes last. This is already how the batch is built.

`**prompt_cache_key` parameter** in each request body — the docs explicitly recommend this for workloads where many requests share the same prefix. It influences cache routing so requests are sent to the same server that already has the prefix cached:

```python
"body": {
    "model": "gpt-5.4-nano",
    "prompt_cache_key": "v2-classifier-system-prompt",  # same for all 267K requests
    "messages": [...],
    "response_format": {...},
    "max_tokens": 450,
}
```

**Important constraint from the docs**: extended prompt caching (24-hour retention) is **not available for `gpt-5.4-nano`**. Nano gets in-memory caching only — 5 to 10 minutes of inactivity, up to 1 hour max. Cache hits primarily happen within the same batch window while the server is actively processing. Switching to `gpt-5.4` or `gpt-5.1` would unlock 24h extended caching.

**Cache performance logging**: the `usage` object in each result line contains `prompt_tokens_details.cached_tokens`. `src/downloader.py` logs this per batch and the final report shows: total cached tokens, estimated dollars saved from caching, cache hit rate %.

### 5. Batch Metadata for Traceability

Each `client.batches.create()` call includes metadata:

```python
metadata={
    "run_id": "v2-2026-04-04",
    "batch_number": "3/39",
    "row_range": "14000-20999",
    "model": "gpt-5.4-nano"
}
```

### 6. Error File Handling (`error_file_id`)

After each batch completes, `src/downloader.py` checks `batch.error_file_id` and downloads it separately to `outputs/batch_errors/`. The `retry` subcommand reads all error files, collects failed `custom_id`s, and re-submits them as a new batch. Per the docs, expired requests also appear here with `"code": "batch_expired"`.

### 7. `custom_id`-Based Result Matching

Batch output order is **not guaranteed** to match input order. Every request uses `custom_id: "startup-{org_uuid}"` and results are matched by `custom_id`, never by position.

### 8. Usage Tracking Per Batch

Parse `response.body.usage` from each JSONL result line and aggregate: total prompt tokens, completion tokens, cached tokens. Log these per-batch and sum in the final report.

### 9. Structured Logging (`src/logger.py`)

Replace all `print()` with Python's `logging` module:

- **Console**: `rich` handler — colored, human-readable
- **File**: rotating `outputs/run.log` — machine-readable, persists across runs

### 10. Async Concurrent Monitoring (`src/monitor.py`)

`asyncio` + `asyncio.gather()` polls all in-flight batches in parallel. A `rich` live table refreshes every 30 seconds showing batch ID, status, progress counts, and elapsed time.

### 11. Exponential Backoff with Jitter — `tenacity` (`src/submitter.py`)

The rate limits docs recommend `tenacity` specifically for exponential backoff on API calls. Upload and batch creation calls in `src/submitter.py` use the `@retry` decorator:

```python
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def upload_batch_file(client: OpenAI, path: str) -> str:
    with open(path, "rb") as f:
        return client.files.create(file=f, purpose="batch").id
```

The random jitter (`wait_random_exponential`) prevents thundering herd — if multiple batches are submitted concurrently and all hit a rate limit, they don't retry at the exact same time. `tenacity` is added to `pyproject.toml` dependencies.

### 12. Flex Processing for Single-Company Validation (`classify.py test`)

A new `test` subcommand uses `service_tier: "flex"` for synchronous calls — priced at batch API rates (50% off) with prompt caching on top. Useful for validating one company's classification before committing to a 267K run:

```bash
python classify.py test --company-id "abc-123"
python classify.py test --company-name "Anthropic"
```

This lets you rapidly iterate on the prompt and verify edge cases without spinning up a batch job. The `flex` tier occasionally returns `429 Resource Unavailable` — the `test` command retries automatically with standard processing as fallback (`service_tier: "auto"`).

### 13. Queue Pressure Awareness in Concurrency Logic

From the rate limits docs: **pending batch tokens count against the 15B queue limit until each batch completes**. This means submitting all 39 batches simultaneously could exhaust the queue before any complete and release capacity.

`src/monitor.py` tracks estimated queued tokens across all in-flight batches and the `--concurrency` flag acts as a sliding window: it only submits the next batch when an in-flight batch completes and its tokens are released from the queue. This is more precise than a fixed concurrency number.

## Tier 5 Limits (`src/config.py`)

```python
MAX_BATCH_QUEUE_TOKENS   = 15_000_000_000  # 15B enqueued; pending batches count against this
MAX_REQUESTS_PER_MINUTE  = 30_000
MAX_TOKENS_PER_MINUTE    = 180_000_000     # 180M
MAX_FILE_SIZE_MB         = 190             # <200MB OpenAI hard limit
MAX_REQUESTS_PER_BATCH   = 50_000          # OpenAI hard limit
BATCH_CREATION_PER_HOUR  = 2_000           # OpenAI hard limit
ESTIMATED_TOKENS_PER_REQUEST = 2_500       # ~2K system/schema + ~300 user + ~200 output
MAX_OUTPUT_TOKENS        = 450             # cost guard per request
DEFAULT_MODEL            = "gpt-5.4-nano"
PROMPT_CACHE_KEY         = "v2-classifier-system-prompt"  # improves cache routing
# Prompt caching: in-memory only for nano (5-10 min); 24h extended requires gpt-5.4+
```

## Portfolio-Quality Details

- `**pyproject.toml**`: proper packaging with `[project]`, `[tool.pytest]`, dev extras (`tenacity`, `rich`, `tiktoken`)
- **Full type hints** on every function signature
- **Docstrings** on every public function
- `**tests/`**: unit tests for schema validation, formatter, token estimator, cache key consistency
- **README**: architecture diagram, dataset description, research context, quickstart, cost table (with caching discount breakdown), sample output rows
- **Final run report** printed by `classify.py merge`: classification distribution, total tokens used, cached tokens, actual dollars saved from caching, error rate, retry count

## What Gets Removed

- `GPT-5-nano batch API processing/` → replaced by `src/` + `classify.py`
- `GPT-5-mini batch API processing/` → model is now `--model gpt-5.4-mini`
- `system_prompt.txt` → v1 binary prompt, no longer used
- `requirements.txt` → replaced by `pyproject.toml`

---

## Commit History

Each commit is a coherent, independently reviewable unit of progress. The sequence tells a story: researcher first, engineer second, pipeline built piece by piece, then hardened for reliability and validated before scale.

---

### Commit 1 — Data hygiene

```
Secure dataset and API credentials outside version control

A 267K-row CSV and a live API key reaching a public repository would
compromise both research data privacy and billing before a single line
of pipeline code exists. Gitignore rules for data/, keys/, and the
runtime outputs/ directory are the contract — established before the
files existed, not retrofitted after.
```

**Files**: `.gitignore` (data/, keys/, outputs/), `data/`, `keys/openai.env`

---

### Commit 3 — Research framework (already committed as f11f974)

```
Replace binary prompt with v2 two-axis taxonomy and input/output contract

The binary AI-native label (0/1) lacks the resolution needed for
publishable research. This commit formalizes the v2 framework: 11
subclasses distinguishing Foundation Layer from Thin Wrappers, plus
an independent RAD score measuring structural dependency on third-party
GenAI APIs.
```

**Files**: `prompts/Multiclassification_prompt.txt`, `data visualization/01_Presentation_Materials/Multiclassification_proposal.html`

---

### Commit 3b — Prompt refinement

```
Refine v2 prompt to enforce input contract and strip CSV formatting instructions

Output structure was previously enforced by strongly-worded prompt
instructions: field order, pipe separators, CSV line format. This is
the fragile v1 approach — the model can ignore it, hallucinate
delimiters, or wrap output in markdown. With Pydantic structured
outputs injecting the schema at the API level, those instructions
become redundant noise. This commit removes them and adds an explicit
INPUT FORMAT block so the model knows exactly what fields to expect
and copies CompanyID/CompanyName verbatim.
```

**Files**: `prompts/Multiclassification_prompt.txt`

---

### Commit 4 — Project scaffold

```
Scaffold src/ package with pyproject.toml, replacing model-specific folders

Two model-specific folders make the model a hardcoded architectural
choice — switching from gpt-5.4-nano to gpt-5.4 for 24h extended
prompt caching would require a new folder, not a flag. The src/
package makes it one CLI argument. pyproject.toml replaces
requirements.txt with proper packaging, dev extras, and pytest
configuration that scales with the project.
```

**Files**: `src/__init__.py`, `pyproject.toml`, `.gitignore` (outputs/ subdirs added), remove old batch processing folders

---

### Commit 5 — Schema and configuration

```
Define Pydantic output schema and Tier 5 constants as single sources of truth

A hand-written JSON schema alongside a Python dataclass will drift.
When they drift, the batch produces wrong output silently at 267K
rows. ClassificationResult(BaseModel) is one definition that
auto-generates the JSON schema injected into every request body —
zero divergence by construction. Config centralises all Tier 5 limits
with source comments so no magic numbers appear downstream.
```

**Files**: `src/schema.py`, `src/config.py`

---

### Commit 6 — Cost awareness

```
Add tiktoken pre-flight cost estimator with --dry-run safeguard

At 267K companies, a misconfigured batch could burn $100+ before a
single result returns. The estimator uses tiktoken — OpenAI's own
tokenizer — to project exact token counts, applies the 50% batch
discount, and models prompt caching savings. --dry-run makes a full
cost breakdown the mandatory first step of every run, not an
afterthought.
```

**Files**: `src/tokens.py`

---

### Commit 7 — Batch file construction

```
Build cache-optimized JSONL generator with structured output schema injection

Prompt caching can reduce input token costs by up to 90%, but only if
every request shares an identical static prefix. The formatter maps
each row to a user message; the builder assembles JSONL files with
the Pydantic-generated schema and a consistent prompt_cache_key
injected into every request body. The static prefix is structurally
guaranteed — not dependent on discipline at call time.
```

**Files**: `src/formatter.py`, `src/builder.py`

---

### Commit 8 — Fault-tolerant submission

```
Add fault-tolerant batch submitter with tenacity exponential backoff

A batch upload that crashes on a transient 429 halfway through loses
all progress and requires manual restart. Upload and batch creation
calls are wrapped in tenacity's @retry decorator with random
exponential backoff. The jitter is deliberate — without it, concurrent
uploads hitting a rate limit would retry simultaneously and re-trigger
the same limit. Each batch is tagged with run_id, batch number, and
row range for traceability in the OpenAI dashboard.
```

**Files**: `src/submitter.py`

---

### Commit 9 — Pipeline reliability infrastructure

```
Add structured logging and JSON checkpoint for overnight pipeline reliability

A 267K-row batch run spans 24+ hours across multiple terminal
sessions. Without a checkpoint, any crash means re-submitting
completed batches and paying twice. state.json tracks every batch
through its full lifecycle so any subcommand resumes exactly where it
left off. All runtime output routes through Python's logging module
— rich handler for the console, rotating file for persistent run.log
— replacing every print() in the pipeline.
```

**Files**: `src/logger.py`, `src/state.py`

---

### Commit 10 — Concurrent monitoring

```
Add async batch monitor with sliding-window queue pressure control

Submitting all 39 batches simultaneously can exhaust the 15B token
queue before any complete, deadlocking the pipeline with no way
forward until batches expire. asyncio.gather() polls all in-flight
batches in parallel; the --concurrency flag acts as a sliding window
tied to queue token pressure, only submitting the next batch when a
running one completes and releases capacity. A rich Live table
refreshes every 30 seconds with batch ID, status, and request counts.
```

**Files**: `src/monitor.py`

---

### Commit 11 — Result retrieval and error recovery

```
Add batch downloader with error-file recovery and cache hit reporting

Batch output order is not guaranteed to match input order — positional
matching would silently corrupt the dataset at 267K rows. Results are
matched to inputs exclusively by custom_id. error_file_id is
downloaded separately so failed and expired requests are never lost;
the retry subcommand re-submits them as a new batch. Per-response
cached_tokens from the usage object are logged so the final report
can show actual dollars saved from prompt caching.
```

**Files**: `src/downloader.py`

---

### Commit 12 — Research deliverable

```
Add result merger with classification distribution and cost breakdown report

Raw per-batch CSVs scattered across outputs/ are not a research
deliverable. The merger produces classified_startups_v2.csv and prints
the numbers that feed the paper: subclass and RAD score distribution
across 267K companies, total tokens consumed, cache hit rate, dollars
saved from caching, and retry count from error files.
```

**Files**: `src/merger.py`

---

### Commit 13 — CLI interface

```
Wire seven-subcommand CLI with model switching and --dry-run validation

A monolithic script makes every stage dependent on every other — a
download failure forces a full re-run from submission. Each subcommand
does exactly one thing and reads state.json to know where to start.
--dry-run on prepare and run prints the full cost plan without
touching the API. Every subcommand has --help text that describes
its effect, not just its parameters.
```

**Files**: `classify.py`

---

### Commit 14 — Prompt validation workflow

```
Add flex-processing test command for single-company prompt validation

Iterating on the prompt by submitting a 267K-row batch job is a
$100+ feedback loop with a 24-hour wait. classify.py test classifies
one company synchronously using service_tier: "flex" — priced at
batch API rates with prompt caching on top — closing that loop to
seconds and near-zero cost. Falls back to standard processing
automatically on 429 Resource Unavailable.
```

**Files**: `classify.py` (test subcommand addition)

---

### Commit 15 — Test coverage

```
Add unit tests for schema enforcement, formatter, and token estimation

The contracts that matter most have no runtime feedback until they
fail silently at scale: Pydantic must reject invalid enum values and
out-of-range confidence scores; the formatter must produce the exact
expected string from a known input row; the token estimator must stay
within 5% of actual tiktoken counts. All tests run offline with no
API calls — fast, deterministic, and safe to run in CI before any
batch is submitted.
```

**Files**: `tests/test_schema.py`, `tests/test_formatter.py`, `tests/test_tokens.py`

---

### Commit 16 — Documentation and cleanup

```
Remove v1 processing folders and document pipeline architecture in README

Leaving the v1 GPT-5-mini and GPT-5-nano folders alongside src/
creates ambiguity about which code runs — a reader shouldn't have to
infer it. The README documents the two-axis taxonomy, data flow,
CLI quickstart, output schema, and a cost table showing the token
math with batch discount and caching savings. The project is now
legible to someone who opens it cold.
```

**Files**: removed `GPT-5-mini batch API processing/`, removed `GPT-5-nano batch API processing/`, removed `system_prompt.txt`, `README.md`