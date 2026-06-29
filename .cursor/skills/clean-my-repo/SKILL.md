---
name: clean-my-repo
description: >-
  Audit an entire codebase for dead files, stale artifacts, leftover
  experiments, and naming/organization issues, then present a severity-ranked
  cleanup report and execute approved changes on a dedicated branch. Use after
  long bouts of work, before sharing a repo publicly, or when the user asks to
  clean up, declutter, reorganize, or tidy a codebase. This is cleanup only:
  no functionality changes, refactors, or debugging.
disable-model-invocation: true
---

# clean-my-repo

Janitor pass over a whole repo. Goal: a portfolio-grade, production-clean, well-organized codebase. You find clutter, rank it, and only touch files after explicit approval — on a branch, never on `main`.

This skill is language- and project-agnostic. Detect the repo's nature first; do not assume it is a research repo, a web app, or any specific stack.

**Hard rules**

- Phases 1-2 are read-only. Make zero changes until the user approves specific items.
- Cleanup only. Never change behavior, refactor logic, fix bugs, or "improve" code that works.
- Flag from evidence, never vibes. "Not modified recently" is not evidence.
- Never merge or push. You hand off a branch for the user to review.

## Workflow

```
- [ ] Phase 1: Scan (read-only) — gather evidence
- [ ] Phase 2: Ranked report in chat — wait for approval
- [ ] Phase 3: Execute approved items on a branch
- [ ] Phase 4: Hand off the diff
```

## Phase 1 — Scan (read-only)

First, profile the repo so detection generalizes:

- Languages and stack: read manifests (`package.json`, `pyproject.toml`, `go.mod`, `Cargo.toml`, etc.).
- Entry points: CLIs, `main`, server bootstrap, exported package index, build configs, test dirs.
- Tracked vs ignored: `git ls-files` for what's committed; read `.gitignore` for what's intentionally excluded.
- Project conventions: read `README`, `AGENTS.md`/`CLAUDE.md`, and any `docs/` to learn what's intentional before flagging it.

Then hunt for candidates, gathering concrete evidence for each:

- **Dead/orphan files**: not imported, referenced, or reachable from any entry point. Confirm with a repo-wide reference search (ripgrep the basename and module path). Record the search result as evidence.
- **Abandoned work**: one-off dashboards, scratch scripts, spike/experiment files, or research directions no longer wired into anything.
- **Stale artifacts**: generated output committed by mistake (build dirs, caches, logs, `.DS_Store`, coverage, large data dumps) that should be git-ignored instead of tracked.
- **Duplicates**: near-identical files, `copy`/`final`/`v2`/`old`/`backup` variants, superseded versions left behind.
- **Naming issues**: vague or misleading names (`utils`, `helpers`, `tmp`, `test2`, `untitled`, `final_v3`) and names that misrepresent contents. Propose a clearer name.
- **Structure issues**: flat directories that should be grouped, files in the wrong folder, an obvious missing subfolder, inconsistent layout across sibling dirs.

**Guardrails (keep it safe and agnostic):**

- Require a real reference/usage search before calling anything dead. Show the evidence.
- Treat intentional files as off-limits unless clearly broken: configs, lockfiles, CI workflows, `LICENSE`, `.gitignore`, `README`, `AGENTS.md`, dotfiles, license headers.
- Respect `.gitignore`. Large generated/data dirs are usually ignored on purpose — flag them for awareness, never delete blindly.
- Surface untracked/uncommitted files (`git status`) separately as "review these yourself"; never delete them.
- Dynamic references exist (string-based imports, reflection, config-driven loading, glob entry points). When a file *looks* dead but could be loaded indirectly, drop its confidence and mark it verify-first, not safe-to-remove.

## Phase 2 — Ranked report (chat)

Output plain markdown grouped by severity tier. Lead with a one-line summary (e.g. "23 items: 6 safe to remove, 4 likely dead, 9 organization, 4 polish").

Tiers:

- **P0 — Safe to remove**: clearly unreferenced, reversible, zero risk.
- **P1 — Likely dead**: probably removable, but verify one named thing first.
- **P2 — Organization**: rename / move / restructure for clarity.
- **P3 — Polish**: nice-to-have naming and tidiness.

For each item give: path, what it is, why flagged, **evidence** (the search/command and its result), recommended action (delete / rename `X`→`Y` / move `X`→`Y/`), confidence (high/med/low), and risk.

Then stop and ask the user to approve. They may approve a subset, all, or nothing. Do not proceed without an explicit answer.

## Phase 3 — Execute on a branch

Apply **only** approved items, on a dedicated branch so nothing is irreversible:

1. Ensure a clean working tree (`git status`); if there are unrelated uncommitted changes, ask before continuing.
2. Create/switch to a cleanup branch: `git switch -c chore/clean-my-repo` (reuse it if it already exists).
3. Use git-native ops to preserve history and stay reversible:
   - Delete: `git rm <path>`
   - Rename/move: `git mv <old> <new>`
4. **On every rename/move, update all references** (imports, paths, docs, config) so nothing breaks. Search the repo for the old path/name and fix each hit.
5. If structure changed, update references in `README` and `AGENTS.md`/`CLAUDE.md` in the same pass so docs stay accurate.
6. If the repo has tests, run them to confirm nothing broke. If they fail due to your changes, fix or revert that item.

## Phase 4 — Hand off

- Show `git status` and `git diff --stat` so the user sees exactly what changed.
- Summarize what was removed/renamed/moved and what was deferred.
- Remind the user this is on `chore/clean-my-repo` for them to review and merge. **Never merge or push** unless they explicitly ask.
