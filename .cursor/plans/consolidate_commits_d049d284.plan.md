---
name: Consolidate commits
overview: Soft-reset the 7 commits back to the pre-change base and re-commit as 4 cohesive commits on main.
todos:
  - id: reset
    content: git reset --mixed to base commit 2bc325e
    status: completed
  - id: commit-a
    content: "Commit A: prompt + schema + tests"
    status: completed
  - id: commit-b
    content: "Commit B: config + builder (pilot calibration)"
    status: completed
  - id: commit-c
    content: "Commit C: submitter + monitor + classify + downloader (production robustness)"
    status: completed
  - id: commit-d
    content: "Commit D: sampling script + pyproject.toml"
    status: completed
isProject: false
---

# Consolidate 7 Commits into 4

## Current state

7 commits on `main` ahead of origin, starting from base `2bc325e`:

```
1268d39 Add reproducible dataset sampling script and openpyxl dependency
05616f1 Download batch error files before output for earlier failure visibility
6327c81 Add --data CLI flag, billing-limit exit code, and fix test command
b069051 Handle OpenAI billing hard limit with graceful resume workflow
458c0fa Fix deprecated max_tokens → max_completion_tokens in batch requests
e9dc298 Calibrate batch sizing and token estimates from 1k pilot run
ce220bc Make conf_rad nullable for RAD-NA and refine prompt punctuation
```

## Proposed 4 commits

**Commit A — Update classification prompt and make conf_rad nullable**

- [prompts/Multiclassification_prompt.txt](prompts/Multiclassification_prompt.txt) — punctuation/delimiter cleanup
- [src/schema.py](src/schema.py) — `conf_rad` becomes `Optional[int]`, split validators
- [tests/test_schema.py](tests/test_schema.py) — tests for nullable conf_rad + sentinel
- *Standalone domain change to the classification model definition*

**Commit B — Calibrate pipeline defaults from 1k pilot and fix API parameter**

- [src/config.py](src/config.py) — tokens per request 2,500 → 7,500, batch size 7,000 → 5,000
- [src/builder.py](src/builder.py) — `max_tokens` → `max_completion_tokens`
- *Both corrections surfaced by running the pilot batch*

**Commit C — Harden production pipeline: billing limits, --data flag, error download ordering**

- [src/submitter.py](src/submitter.py) — `BillingLimitError`, detect `billing_hard_limit_reached`
- [src/monitor.py](src/monitor.py) — catch billing error, save state, print resume panel
- [classify.py](classify.py) — `--data` flag, exit code 2, `max_completion_tokens` in test cmd, retry cleanup
- [src/downloader.py](src/downloader.py) — download error files before output, skip if exists
- *All production robustness improvements*

**Commit D — Add reproducible dataset sampling script**

- [scripts/sample_dataset.py](scripts/sample_dataset.py) — new script
- [pyproject.toml](pyproject.toml) — add `openpyxl` dependency
- *Standalone tooling addition*

## Implementation

1. `git reset --mixed 2bc325e` to put all 7 commits' changes back in the working tree
2. Stage and commit each group (A through D) sequentially

