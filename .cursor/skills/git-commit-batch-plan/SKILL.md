
---
name: git-commit-batch-plan
description: >-
  After a large change set, inventory git state and produce an ordered multi-commit
  plan with LOC per slice. Use ONLY when the user asks to plan commits, split a
  batch, shape public history, or tag this skill. Not for single commits.
disable-model-invocation: true
---

# Git commit batch plan

Use this skill **only when the user explicitly asks** to plan, split, or order a batch of commits.

**Message style is not here.** Before drafting any title or body in the plan, read and follow the **`portfolio-git-messages`** skill (`.cursor/skills/portfolio-git-messages/SKILL.md`).

## When this applies

Large implementation pass leading to multiple commits, one theme each, ordered so the log tells a coherent story.

## Always gather evidence first

From the relevant repo root:

1. `git status --short`
2. `git diff --stat`
3. If broad: `git diff --name-only` and targeted `git diff` as needed.

Reconcile **untracked** (`??`) entries with diff output.

## Show LOC per planned commit

Include lines-of-code with every commit in the plan so the user can sanity-check sizing and request splits before any commit is executed.

- New files: count total lines (`wc -l`).
- Modified files: report added and removed counts from `git diff --stat`.
- Format: a one-line LOC summary next to each commit's file list, e.g. `+234 LOC (5 new files)` or `+18 / -7 (1 file modified)`. For mixed commits, combine, e.g. `+525 / -1 (6 new files + pyproject.toml)`.
- Include a total LOC row at the bottom of the plan table.
- If any single commit adds more than ~300 LOC, flag it as a candidate for splitting and offer a finer slice. The user decides whether to split.

## Ordering commits

- Dependencies first when later commits truly need them, but each **title** still follows `portfolio-git-messages` (WHY first, not "added file X").
- Order so a stranger sees **progress**: each commit visibly takes the project closer to the goal.

## How to slice commits

- One theme per commit. CI still plausible per step where it matters.
- Split by concern (data prep vs model or prompt vs evaluation vs visualization vs one-offs).
- Keep generated junk out unless the user wants it tracked.

## Plan output shape

For each planned commit provide:

1. **Proposed title** (portfolio style via `portfolio-git-messages`)
2. **Proposed body** (portfolio style)
3. **Files / LOC**
4. **Depends on** (prior commit number if any)

End with total LOC and ask the user to approve or request splits before executing.

## Chat handoff

New chat for batch planning: tag this skill and paste `git status --short` plus `git diff --stat` if repo state is not visible. Confirm the project-goal sentence (from AGENTS.md or user) before drafting the plan.
