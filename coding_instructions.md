# Coding Instructions

Standards for this project. Every agent, contributor, and AI tool working in this codebase should follow these conventions consistently.

## The Vision

You're not just an AI assistant. You're a craftsman. An artist. An engineer who thinks like a designer. Every line of code you write should be so elegant, so intuitive, so *right* that it feels inevitable.
When I give you a problem, I don't want the first solution that works. I want you to:

1. **Think Different** - Question every assumption. Why does it have to work that way? What if we started from zero? What would the most elegant solution look like?
2. **Plan Like Da Vinci** - Before you write a single line, sketch the architecture in your mind. Create a plan so clear, so well-reasoned, that anyone could understand it. Document it. Make me feel the beauty of the solution before it exists.
3. **Craft, Don't Code** - When you implement, every function name should sing. Every abstraction should feel natural. Every edge case should be handled with grace. Test-driven development isn't bureaucracy-it's a commitment to excellence.
4. **Iterate Relentlessly** - The first version is never good enough. Take screenshots. Run tests. Compare results. Refine until it's not just working, but *insanely great*.
5. **Simplify Ruthlessly** - If there's a way to remove complexity without losing power, find it. Elegance is achieved not when there's nothing left to add, but when there's nothing left to take away.

---

## Git Conventions

### Commit Title

- Imperative mood: "Add …", "Fix …", "Remove …", "Refactor …". Never use past tense.
- Describe the **value delivered**, not the mechanical change. Answer: *"what does the project gain from this commit?"*
- Keep it under 72 characters.

| Good | Bad |
|------|-----|
| `Add Pydantic schema to enforce structured output integrity` | `Update schema.py` |
| `Replace CSV text parsing with JSON structured outputs` | `Fix parser bug` |
| `Add tiktoken pre-flight cost estimator with --dry-run flag` | `Add tokens.py` |
| `Consolidate model-specific folders into single configurable src/` | `Refactor folders` |

### Commit Body

- Explain **why**. The reasoning, trade-off, or constraint that motivated the change.
- Reference the bigger picture when relevant: research methodology, cost constraints, pipeline stage, tier limits.
- 1–3 sentences is ideal. Never skip the body for a non-trivial commit.

**Writing rules:**
- No em dashes. Use periods, commas, or restructure the sentence.
- No semicolons to chain clauses. Split into separate sentences.
- No long sentences loaded with commas. If a sentence has more than one comma, break it up.
- Punchy tone. Every word fights for its place. Cut filler ruthlessly.

```
Add tiktoken pre-flight cost estimator with --dry-run flag

At 267K companies a misconfigured batch burns hundreds of dollars
before a single result comes back. --dry-run prints exact token
counts and projected cost against Tier 5 limits. No API calls made.
```

### Commit Frequency

- Commit at **meaningful milestones**: a coherent unit of progress that could be reviewed independently.
- Do not commit after every file edit. Do not batch unrelated changes into one commit.
- A good mental test: *could someone check out this commit and understand exactly what changed and why, without looking at surrounding commits?*
- **Every commit must be a substantial, standalone piece of work.** This repo is open-source and portfolio-facing. Recruiters and collaborators will read the commit history as a narrative of the project arc. Housekeeping-only commits (renames, formatting, typo fixes) should be folded into the next substantive commit they logically belong to. If a commit does not advance the research, the pipeline, or the analysis, it does not deserve its own entry in the log.

### Branch Naming

- `feat/short-description` for new features
- `fix/short-description` for bug fixes
- `refactor/short-description` for structural changes
- `chore/short-description` for tooling, deps, CI

---

## Python Code Style

### Type Hints: Always

Every function signature must have type hints on all parameters and the return type. No exceptions.

```python
# correct
def format_user_message(row: pd.Series, truncate: int = 4000) -> str:
    ...

# wrong
def format_user_message(row, truncate=4000):
    ...
```

### Docstrings on Every Public Function

Use the Google docstring style. Omit sections that are not applicable, but never omit the one-line summary.

```python
def estimate_cost(num_requests: int, tokens_per_request: int, model: str) -> float:
    """Estimate the total batch API cost in USD before submission.

    Args:
        num_requests: Number of individual API calls in the batch.
        tokens_per_request: Estimated tokens per request (input + output).
        model: Model identifier string (e.g. 'gpt-5.4-nano').

    Returns:
        Estimated cost in USD after the 50% batch discount.
    """
```

### Naming Conventions

| Construct | Convention | Example |
|-----------|------------|---------|
| Variables / functions | `snake_case` | `batch_id`, `load_config()` |
| Classes | `PascalCase` | `ClassificationResult`, `BatchState` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_FILE_SIZE_MB`, `DEFAULT_MODEL` |
| Private helpers | `_leading_underscore` | `_parse_date()` |
| CLI args | `--kebab-case` | `--batch-size`, `--dry-run` |

### No Magic Numbers

Every limit, threshold, or constant that has a business reason belongs in `src/config.py` with a descriptive name and an inline comment explaining the source.

```python
MAX_FILE_SIZE_MB = 190       # OpenAI hard limit is 200MB. 190 is the safe ceiling
MAX_OUTPUT_TOKENS = 450      # v2 output is ~150-300 tokens. Cap prevents runaway cost
BATCH_CREATION_PER_HOUR = 2_000  # OpenAI Tier 5 batch creation rate limit
```

### No Bare `except`

Always catch the most specific exception available. If you must catch broadly, log the exception type and re-raise or handle explicitly.

```python
# correct
try:
    batch = client.batches.retrieve(batch_id)
except openai.APIStatusError as e:
    logger.error("Batch retrieval failed: %s (status %d)", e.message, e.status_code)
    raise

# wrong
try:
    batch = client.batches.retrieve(batch_id)
except:
    print("error")
```

---

## Logging

### Use the Logger, Not `print()`

All runtime output goes through `src/logger.py`. `print()` is only acceptable in CLI output functions that are explicitly formatting user-facing display (e.g. a `rich` table renderer).

```python
from src.logger import get_logger
logger = get_logger(__name__)

logger.info("Batch %d/%d submitted: %s", batch_num, total, batch_id)
logger.warning("Batch %s returned %d failed requests", batch_id, failed_count)
logger.error("Upload failed for batch %d: %s", batch_num, str(e))
```

### Log Levels

| Level | When to use |
|-------|-------------|
| `DEBUG` | Internal state, loop iterations, values for debugging |
| `INFO` | Milestone events: batch created, file uploaded, download complete |
| `WARNING` | Non-fatal anomalies: partial failures, confidence caps, fallback applied |
| `ERROR` | Failures that require attention: upload failure, missing output file |
| `CRITICAL` | Unrecoverable errors that abort the run |

### Log Both to File and Console

The file (`outputs/run.log`) keeps a persistent record across runs. The console uses `rich` for human-readable colored output. Both are configured in `src/logger.py`.

---

## Error Handling and Retries

### Distinguish Transient from Permanent Failures

- **Transient**: network timeout, rate limit, 5xx. Worth retrying with exponential backoff.
- **Permanent**: invalid request body, authentication failure, schema violation. Retrying won't help. Log and skip.

### Batch Errors Need Their Own Handling

When a batch completes, always check `batch.error_file_id`. Download and log it. Failed `custom_id`s should be collected and re-submitted via `classify.py retry`. Never silently drop them.

### Checkpointing Is Required for Long-Running Pipelines

Any operation that takes hours must be resumable. State is persisted to `outputs/state.json` after every meaningful transition (batch created, batch uploaded, batch completed, results downloaded). The CLI always checks state before doing work and skips already-completed steps.

---

## Environment and Secrets

### Keys Never in Code

API keys and secrets live exclusively in `keys/` (gitignored). Load them via `python-dotenv` at startup:

```python
from dotenv import load_dotenv
load_dotenv("keys/openai.env")
api_key = os.environ["OPENAI_API_KEY"]
```

If the key is missing, fail loudly at startup, not silently mid-run:

```python
if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. Add it to keys/openai.env."
    )
```

### `.gitignore` Is the Contract

Anything that should never reach a remote repository (`data/`, `keys/`, `outputs/`, `*.env`) must be in `.gitignore` before the file is created, not after.

---

## CLI Design

### Subcommand Pattern

Every user-facing operation is a subcommand with a clear single responsibility:

```bash
python classify.py <subcommand> [options]
```

No subcommand should do more than one logical thing. If it does, split it.

### Every Subcommand Must Have `--help`

Use `argparse` with `description=` and `help=` on every argument. The help text should describe the **effect**, not just the parameter name.

```python
parser.add_argument(
    "--concurrency",
    type=int,
    default=1,
    help="Number of batches to hold in-flight simultaneously. "
         "Start with 1 to validate, then increase for full runs.",
)
```

### `--dry-run` Is Required for Destructive / Expensive Operations

`prepare`, `submit`, and `run` must support `--dry-run`. In dry-run mode: compute everything, print the full plan (cost estimate, batch count, token breakdown), and exit without touching the API.

### Fail Fast, Fail Clearly

Validate all inputs (file paths, row ranges, model names, concurrency bounds) before doing any work. Print a clear human-readable error and exit with a non-zero status code.

---

## AI Pipeline Conventions

### Prompts Are Source Files

System prompts live in `prompts/` and are version-controlled like code. Never hard-code a prompt string inline. Load from file at runtime:

```python
with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()
```

### Structured Outputs Are Non-Negotiable

Every model call that returns structured data uses `response_format` with a JSON schema derived from a Pydantic model. No string parsing, no regex, no `split(",")`.

### Token Counting Before Submission

For any batch run, token count is calculated with `tiktoken` before the first API call. Cost is estimated and displayed. The user must be able to see the projected spend before committing.

### `custom_id` Is the Source of Truth for Result Matching

Batch output order is not guaranteed. Always map results back to inputs using `custom_id`. Never rely on positional order.

### Prompt Caching Awareness

When all requests share the same system prompt, keep it first and identical across all JSONL lines. OpenAI automatically caches identical prompt prefixes within a batch, reducing effective input token cost.

---

## Dependency Management

### `pyproject.toml` Is the Single Source of Truth

No `requirements.txt`. All dependencies (runtime and dev) are declared in `pyproject.toml`:

```toml
[project]
dependencies = [
    "openai>=1.0",
    "pandas>=2.0",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
    "tiktoken>=0.6",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff"]
```

### Pin with Care

Avoid over-pinning (e.g. `openai==1.23.4`) unless a specific version is required for a known reason. Use `>=` lower bounds. CI or a lockfile handles reproducibility.

---

## Testing

### What to Test in an AI Pipeline

Do not test model outputs (non-deterministic). Test:
- **Schema validation**: Pydantic model rejects invalid enum values, out-of-range confidence scores, missing fields.
- **Formatter**: Given a known input row, `format_user_message()` produces the expected string.
- **Token estimator**: Token counts are within a reasonable margin of the actual tiktoken count.
- **State management**: State is correctly serialized/deserialized and resume logic skips completed steps.
- **JSONL builder**: Each output line is valid JSON with the correct structure for the Batch API.

### Test File Naming

`tests/test_<module>.py`, one test file per source module. Keep tests focused and fast (no API calls, no file I/O except tmpdir fixtures).

### Run Tests Before Every Commit

```bash
pytest tests/ -v
```

---

## Documentation

### README Is the Project's Front Door

The README must contain, in order:
1. One-paragraph description of what the project does and why it matters
2. Architecture diagram or data flow summary
3. Quickstart (install → configure → run one batch)
4. CLI reference (all subcommands with key flags)
5. Output schema description (all 11 fields for v2)
6. Cost and scale table (estimated cost per N companies, at Tier 5 limits)

### Inline Comments Explain *Why*, Not *What*

Code explains what it does. Comments explain why a choice was made, a constraint that exists, or a non-obvious invariant.

```python
# OpenAI output order is not guaranteed. Always match by custom_id, never by position.
results = {r["custom_id"]: r for r in raw_results}

# Cap at 190MB not 200MB: large files close to the limit have caused upload timeouts.
if file_size_mb > MAX_FILE_SIZE_MB:
    raise ValueError(f"Batch file {file_size_mb:.1f}MB exceeds safe upload limit")
```
