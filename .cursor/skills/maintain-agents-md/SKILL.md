---
name: maintain-agents-md
description: >-
  Initialize and maintain AGENTS.md at repo root in every Cursor project.
  Use proactively at session start, when AGENTS.md is missing, and whenever
  repo structure, architecture, phases, or entry points change. Applies to
  all projects — not batchkit-specific.
---

# Maintain AGENTS.md (all projects)

`AGENTS.md` at the repo root is the provider-agnostic agent briefing (Cursor, Claude Code, Codex, cloud agents). **Initialize it if missing. Keep it current as you work — do not wait to be asked.**

Canonical global policy also lives in `~/.cursor/user-rules/agents-md-maintenance.md`.

## If AGENTS.md is missing

Create it at the workspace root before substantive work (unless the user says skip or the project is a one-file throwaway).

Initial sections:

1. Project overview + current status
2. Tech stack
3. Repository layout (exists vs planned)
4. Development commands
5. Where to work (task → file)
6. Maintaining this file (checklist below)

Target ~100–200 lines. Compress repo knowledge — do not require the agent to explore blindly.

## When to update

Same session / same PR as the code change:

- Major milestone or phase complete/started
- Top-level module added, removed, or renamed
- Architecture, domain model, or data flow changed
- Dev commands, dependencies, or CI changed
- Entry points moved or added

## What to update

1. Build status / roadmap
2. File map
3. Planned-but-not-created list
4. Where-to-work table
5. Tech stack (major deps)
6. Domain model (new types)

## Rules

- Surgical edits only; preserve structure and tone
- No session chatter or trivial bugfix noise
- Commit with the repo when version-controlled
- Nested `backend/AGENTS.md` etc. OK in monorepos; root file stays project-wide

## Minimal template (new repos)

```markdown
# AGENTS.md

Instructions for AI coding agents. Read before making changes.

## Project overview
[What it does, current status]

## Tech stack
[Languages, frameworks, tools]

## Repository layout
[What exists vs planned]

## Development commands
[Install, test, lint, run]

## Where to work
| Task | Start here |
|------|-----------|

## Maintaining this file
Update this file when structure, architecture, or milestones change.
See ~/.cursor/user-rules/agents-md-maintenance.md for full checklist.
```
