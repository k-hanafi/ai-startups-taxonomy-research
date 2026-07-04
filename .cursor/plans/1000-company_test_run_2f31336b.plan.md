---
name: 1000-company test run
overview: Run the classification pipeline on a random 1000-company sample and produce an Excel analysis workbook for validating prompt design, subclass distribution, and edge-case handling before scaling to the full 269K dataset.
todos:
  - id: sample-script
    content: Create scripts/sample_dataset.py to draw a reproducible random 1000-row sample from the full CSV
    status: completed
  - id: cli-data-arg
    content: Add --data CLI argument to classify.py (prepare, run, retry) to accept an alternate input CSV path
    status: completed
  - id: add-openpyxl
    content: Add openpyxl to pyproject.toml dependencies
    status: completed
  - id: analysis-script
    content: Create scripts/analyze_test_run.py that produces the 6-sheet Excel analysis workbook
    status: cancelled
  - id: run-sample
    content: Generate the 1000-company sample CSV
    status: completed
  - id: run-pipeline
    content: "Execute the pipeline: python classify.py run --data data/sample_1000.csv"
    status: in_progress
  - id: run-analysis
    content: Run the analysis script and open the Excel workbook for review
    status: pending
isProject: false
---

# 1000-Company Validation Test Run

## Step 1: Pre-sample the dataset

Create a script `scripts/sample_dataset.py` that:

- Reads the full CSV (`data/company_us_short_long_desc_.csv`, ~269K rows)
- Draws a random sample of 1000 rows with a fixed seed for reproducibility
- Writes to `data/sample_1000.csv`
- Prints basic stats (founding year distribution, % with long descriptions, keyword coverage) to confirm the sample is representative

## Step 2: Point the pipeline at the sample and run

Run the existing pipeline against the sample:

```bash
python classify.py run --rows 0:1000 --concurrency 1
```

But since the pipeline hardcodes `DATA_CSV` to the full file, we have two clean options:

- **Option A (chosen):** Add a `--data` CLI argument to `classify.py` so the user can pass `--data data/sample_1000.csv` without touching the default path. This is a small, reusable change -- just a new argument threaded to `_cmd_prepare` and `_cmd_run`.
- No `--rows` needed since the sample CSV is already 1000 rows.

Changes to [classify.py](classify.py):

- Add `--data` argument to `prepare`, `run`, and `retry` subcommands (default: current `DATA_CSV`)
- Thread the resolved path through `_cmd_prepare`, `_cmd_run`, `_cmd_retry`

## Step 3: Build the Excel analysis workbook

Create `scripts/analyze_test_run.py` that reads the merged CSV output and produces an Excel workbook (`outputs/test_run_analysis.xlsx`) with these sheets:

### Sheet 1: "Raw Results"

All 1000 classified rows -- every field from `ClassificationResult` plus the original input fields (short/long description, keywords, year) side by side for easy manual review.

### Sheet 2: "Distribution"

Summary tables:

- `ai_native` counts (0 vs 1) and percentage
- Subclass frequency (all 11 categories) with % of total
- RAD score frequency (RAD-H / RAD-M / RAD-L / RAD-NA)
- Cohort split (PRE-GENAI vs GENAI-ERA)
- Confidence score histograms for both `conf_classification` and `conf_rad`

### Sheet 3: "Cross-Tabs"

Pivot-style cross-tabulations:

- Subclass x RAD score (to validate RAD assignment rules)
- Subclass x Cohort (to see if PRE-GENAI / GENAI-ERA companies cluster differently)
- ai_native x Cohort (key research question)
- Confidence pair distribution (conf_classification x conf_rad)

### Sheet 4: "Rule Validation"

Automated checks against the prompt's RAD assignment rules:

- 0A/0D/0E companies must have RAD-NA -- flag any violations
- 0C-THIN/0C-THICK must have RAD-H or RAD-M -- flag any RAD-L or RAD-NA
- ai_native=1 companies must never have RAD-NA -- flag any violations
- Confidence caps: companies with no long description should have conf <= 4
- Each rule shows pass/fail count and lists violating rows

### Sheet 5: "Edge Cases"

Rows that warrant manual inspection:

- `conf_classification` <= 2 (uncertain AI-native call)
- `conf_rad` <= 2 (uncertain RAD assignment)
- `verification_critique` contains "UNCERTAIN"
- Borderline subclasses (1B vs 0C-THICK, 0A vs 0B)

### Sheet 6: "Spot-Check 50"

50 randomly selected rows with full input + output side-by-side, formatted for manual review. Includes a blank "Reviewer Notes" column for the human analyst.

### Dependency

Add `openpyxl` to `pyproject.toml` dependencies (needed by pandas `to_excel`).

## Step 4: Run the analysis

```bash
python scripts/analyze_test_run.py --input outputs/classified_startups_v2.csv \
    --source data/sample_1000.csv \
    --output outputs/test_run_analysis.xlsx
```

## What to look for in the results

The Excel workbook is designed to surface these potential issues before scaling:


| Signal                        | What it means                         | Action                                     |
| ----------------------------- | ------------------------------------- | ------------------------------------------ |
| Subclass heavily skewed to 0A | Prompt may be too conservative        | Adjust AI-native threshold or examples     |
| Most conf scores are 5        | Model is overconfident                | Tighten confidence cap rules               |
| RAD rule violations           | Structured output or prompt logic gap | Fix prompt constraints or add validators   |
| Many UNCERTAIN flags          | Descriptions may be too sparse        | Consider if this is data quality vs prompt |
| 1B vs 0C-THICK confusion      | Hardest boundary in the taxonomy      | May need more few-shot examples            |


## File changes summary


| File                             | Change                                                 |
| -------------------------------- | ------------------------------------------------------ |
| `scripts/sample_dataset.py`      | **New** -- random sampling script                      |
| `scripts/analyze_test_run.py`    | **New** -- Excel analysis workbook generator           |
| [classify.py](classify.py)       | Add `--data` CLI argument to `prepare`, `run`, `retry` |
| [pyproject.toml](pyproject.toml) | Add `openpyxl` dependency                              |


