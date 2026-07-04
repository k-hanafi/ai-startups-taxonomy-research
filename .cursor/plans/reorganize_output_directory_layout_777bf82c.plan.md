---
name: Reorganize output directory layout
overview: Restructure each arm's output folder so the merged CSV sits at the top level and all batch processing artifacts are consolidated under a single `batchfiles/` subfolder with shortened names.
todos:
  - id: context-paths
    content: Update all 6 path helpers in src/context.py to route through batchfiles/ with shortened names
    status: completed
  - id: broken-import
    content: "Fix classify.py line 274: from src.config -> from src.openai_config"
    status: completed
  - id: output-tokens
    content: Bump MAX_OUTPUT_TOKENS from 450 to 800 in src/openai_config.py
    status: completed
  - id: verify
    content: Run the test suite to confirm nothing breaks
    status: completed
isProject: false
---

# Reorganize Output Directory Layout

## Target structure

```
outputs/
  baseline/
    classified_baseline.csv        # the deliverable
    batchfiles/                    # everything else
      requests/                    # JSONL sent to OpenAI  (was batch_requests/)
      results/                     # raw JSONL from OpenAI (was batch_results/)
      errors/                      # error JSONL           (was batch_errors/)
      parsed/                      # per-batch CSVs        (was batch_outputs/)
      state.json                   # pipeline checkpoint
      run.log                      # audit log
  arm_a/
    classified_arm_a.csv
    batchfiles/
      ...same...
  arm_b/
    classified_arm_b.csv
    batchfiles/
      ...same...
  analysis/                        # unchanged
```

The merged CSV stays where it is (already at arm root). The `state.json` and `run.log` move from the arm root into `batchfiles/`. All four batch subdirectories move under `batchfiles/` and get shortened names.

## Files to change

All changes are in [src/context.py](src/context.py) -- this is the single source of truth for every path. Every other module already calls `batch_requests_dir()`, `batch_results_dir()`, etc. from context, so renaming the return values there propagates everywhere automatically.

### 1. `src/context.py` -- 6 path helpers

Current:

```python
def batch_requests_dir() -> Path:
    return arm_dir() / "batch_requests"

def batch_results_dir() -> Path:
    return arm_dir() / "batch_results"

def batch_errors_dir() -> Path:
    return arm_dir() / "batch_errors"

def batch_outputs_dir() -> Path:
    return arm_dir() / "batch_outputs"

def state_file() -> Path:
    return arm_dir() / "state.json"

def log_file() -> Path:
    return arm_dir() / "run.log"
```

New:

```python
def batchfiles_dir() -> Path:
    return arm_dir() / "batchfiles"

def batch_requests_dir() -> Path:
    return batchfiles_dir() / "requests"

def batch_results_dir() -> Path:
    return batchfiles_dir() / "results"

def batch_errors_dir() -> Path:
    return batchfiles_dir() / "errors"

def batch_outputs_dir() -> Path:
    return batchfiles_dir() / "parsed"

def state_file() -> Path:
    return batchfiles_dir() / "state.json"

def log_file() -> Path:
    return batchfiles_dir() / "run.log"
```

`merged_csv()` is unchanged -- it already returns `arm_dir() / "classified_baseline.csv"` etc.

### 2. `classify.py` line 274 -- fix broken import (bonus bug fix)

```python
# from src.config import MAX_OUTPUT_TOKENS, PROMPT_CACHE_KEY   # broken
from src.openai_config import MAX_OUTPUT_TOKENS, PROMPT_CACHE_KEY
```

### 3. `src/openai_config.py` -- bump MAX_OUTPUT_TOKENS (from previous analysis)

Increase from 450 to 800 to provide safe headroom after the confidence rubric alignment.

No other files need changes -- every module imports path helpers from `src/context.py`, so the reorganization propagates automatically through:
- [src/builder.py](src/builder.py) -- uses `batch_requests_dir()`
- [src/downloader.py](src/downloader.py) -- uses `batch_results_dir()`, `batch_errors_dir()`, `batch_outputs_dir()`
- [src/merger.py](src/merger.py) -- uses `batch_outputs_dir()`, `merged_csv()`
- [src/state.py](src/state.py) -- uses `state_file()`, `arm_dir()`
- [src/logger.py](src/logger.py) -- uses `log_file()`, `arm_dir()`
- [classify.py](classify.py) -- uses `batch_errors_dir()`, `batch_requests_dir()`
