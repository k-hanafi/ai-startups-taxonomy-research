# Eval CLI redesign

## STATUS

Last updated: **2026-07-25**

| Field | Value |
|-------|--------|
| **Branch** | `eval/cli-redesign` (stacked on `eval/suite-redesign`) |
| **PR** | #34 (target base: `eval/suite-redesign`) |
| **Goal** | Collapse the paid 9-cell matrix into three beginner commands with a live checklist |
| **Next** | Merge after #33, then run `python -m evals run-evals` for the paid Stage 8 sweep |

## Commands

```bash
python -m evals cost-preview     # per-config + total $ (no API)
python -m evals run-evals        # confirm → Pass A ×3 → cells ×9 → score → dashboard
python -m evals open-dashboard   # opens eval_instances/index.html
```

## Execution graph

1. Cost table + `y/N` confirm (skip with `--yes`).
2. Phase 1: `bank-pass-a` for each of 3 models in parallel.
3. Gate: all Pass A banks green.
4. Phase 2: 9 `run-classification` cells in parallel (Pass A reused from banks).
5. Score each cell with `--confidence-from-raw` as it finishes.
6. Build dashboard + archive instance (blocked if any cell failed).

## Key files

- `evals/cost_preview.py` — offline estimates (shared with `--dry-run`)
- `evals/classification.py` — `bank_pass_a`
- `evals/orchestrate.py` — subprocess supervisor + rich checklist
- `evals/__main__.py` — CLI wiring

## Invariants respected

- Pass A paid once per model (science invariant).
- Same `--limit` on phase 1 and phase 2.
- Resume: completed scored cells skipped; partial banks/cells resume.
- Failed cell does not stop siblings, but blocks the dashboard.
