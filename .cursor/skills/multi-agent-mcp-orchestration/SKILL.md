---
name: multi-agent-mcp-orchestration
description: >
  Register other AI tools (Codex, Gemini, secondary Claude instances) as MCP servers so the primary agent plans and delegates implementation. Use for complex multi-model dev loops with role separation. Triggers on "MCP orchestration", "register codex as MCP", "multi-model workers", or /multi-agent-mcp-orchestration.
disable-model-invocation: true
---

# Multi-Agent MCP Orchestration

Wire specialized tools as MCP servers. The Cursor agent becomes the manager: plan, delegate, collect results, validate.

## Cursor environment

- Add MCP servers in **Cursor Settings → MCP** (or project `.cursor/mcp.json` when supported).
- The orchestrator (you) calls MCP tools via **CallMcpTool** after reading each tool's schema in the project's `mcps/` descriptors.
- Do not assume Codex, Gemini, or other workers exist until the user has installed and authenticated them.

## Why it works

Different models have different strengths. A strong planner plus sandboxed implementers (Codex) or video/long-context tools (Gemini) can outperform a single model if roles are explicit and outputs are validated.

## Setup pattern

```bash
# Example: register a worker (exact command depends on the tool the user installed)
claude mcp add codex-review -- codex mcp-server
```

In Cursor, equivalent configuration lives in MCP settings JSON. Name each server by role (`codex-frontend`, `codex-tests`, etc.).

## Orchestration workflow

1. **Receive** a feature or fix request from the user.
2. **Decompose** into subtasks with clear inputs and expected outputs per worker.
3. **Assign roles** (frontend, backend, tests, docs) and call the matching MCP tool.
4. **Validate** worker output against the plan (types, tests, consistency).
5. **Integrate** into the repo; run project tests before declaring done.

## Role definition template

For each MCP worker, document in the prompt:

- **Scope** — what it may touch
- **Output format** — files, patches, or structured JSON
- **Failure handling** — retry once, then escalate to the user
- **Forbidden** — secrets in logs, force-push, CI workflow edits to greenwash failures

## When to use

- Large features that parallelize across layers
- When a dedicated worker is already configured and trusted

## When not to use

- Small or single-file changes
- When no secondary MCP servers are configured

## References

- MCP specification: https://modelcontextprotocol.io
- MCP server registry: https://github.com/modelcontextprotocol/servers
