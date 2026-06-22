---
name: multi-agent-chrome
description: >
  Orchestrate parallel browser work using multiple isolated Chrome instances or Cursor Task subagents. Use for the same browser action across many targets (forms, JS-rendered pages). Triggers on "multi chrome", "parallel browser agents", "multi-agent chrome", or /multi-agent-chrome.
disable-model-invocation: true
---

# Multi-Agent Chrome Orchestrator

Run 1–5 browser workers in parallel for repetitive web tasks (forms, scraping JS pages, bulk navigation).

## Cursor environment

- **Default:** spawn **Task** subagents (`subagent_type`: `generalPurpose` or `explore`). Each subagent uses **cursor-ide-browser** MCP for one browser session. Coordinate via a shared `chat.md` file.
- **Advanced:** launch isolated Chrome instances on ports 9223–9227 with the bundled scripts (macOS + Google Chrome required).

Workspace root: `~/.cursor/agent-workspace/multi-chrome/`

## When to use

- Submitting contact forms across many sites
- Filling applications on multiple platforms
- Scraping JS-rendered pages that need a real browser
- Any repetitive browser task where sequential work is too slow

## Architecture (Task-based, recommended)

```
~/.cursor/agent-workspace/multi-chrome/
├── chat.md                 # Shared coordination file
├── scripts/
│   ├── launch_chrome.sh    # Optional: isolated Chrome on 9223+
│   └── kill_chrome.sh
```

## Step-by-step (Task + cursor-ide-browser)

### 1. Size the pool

Match agent count to workload (e.g. 10 forms → 3–5 agents). Do not over-provision.

### 2. Initialize chat.md

Create or reset `~/.cursor/agent-workspace/multi-chrome/chat.md`:

```markdown
# Multi-Chrome Agent Chat

## Orchestrator
[{timestamp}] Launching N agents for: {task summary}

### Agent 1 Tasks
1. {url and actions}

### Agent 2 Tasks
1. {url and actions}

## Agent 1

## Agent 2
```

### 3. Spawn Task subagents in parallel

For each agent N, use the **Task** tool with a prompt like:

```
You are Agent N. Read ~/.cursor/agent-workspace/multi-chrome/chat.md.
Find tasks under "### Agent N Tasks". Execute them using cursor-ide-browser
(navigate, snapshot, click, fill). Append progress under "## Agent N" with
timestamps. Use [WORKING], [DONE], or [ERROR] tags. Do not edit other agents' sections.
```

Spawn all agents in one turn (parallel Task calls).

### 4. Monitor

Read `chat.md` until all agents report `[DONE]` or `[ERROR]`.

### 5. Tear down

If you launched isolated Chrome instances, run:

```bash
bash ~/.cursor/skills/multi-agent-chrome/scripts/kill_chrome.sh
```

## Optional: isolated Chrome instances (macOS)

When Task subagents need separate browser profiles (not the IDE browser):

```bash
bash ~/.cursor/skills/multi-agent-chrome/scripts/launch_chrome.sh [COUNT]
```

Ports 9223–9227 (max 5). Port 9222 is often reserved for a primary DevTools session.

Then configure each Task subagent to use Chrome DevTools MCP pointed at its port (if you have that MCP server installed in Cursor).

## Chat protocol

| Tag | Meaning |
|-----|---------|
| `[WORKING]` | Agent is processing |
| `[DONE]` | All assigned tasks complete |
| `[ERROR]` | Blocker (CAPTCHA, timeout, etc.) |
| `[WAITING]` | Idle, awaiting tasks |
| `[ABORT]` | Orchestrator requests stop |

## Edge cases

- **CAPTCHA:** Report `[ERROR] CAPTCHA on {url}`; reassign or skip.
- **Rate limiting:** Spread URLs across agents.
- **Login required:** Include login steps in the task list or pre-authenticate.
- **Sensitive data:** Never put real passwords or PII in chat.md; use env vars or ask the user to fill secrets manually.

## Security

Do not use `--dangerously-skip-permissions` or hardcoded third-party identities from demo materials. Use the user's own credentials and approved test data only.
