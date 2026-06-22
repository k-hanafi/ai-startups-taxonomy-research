---
name: model-chat
description: >
  Spawn 5+ Claude API instances into a shared debate room with round-robin turns and a final synthesizer. Uses the model-chat Python script. Triggers on "model chat", "multi-model debate", "agent debate", "spawn a chat room", or /model-chat. Pass a topic as the argument. Alternative without API key: use the agent-chatrooms skill (Task tool).
disable-model-invocation: true
---

# Model Chat

Spawn multiple Claude instances into a shared conversation room. They debate across several rounds in parallel within each round. A synthesizer merges the debate into a final answer.

**Why this works:** Same model with slight framing variations surfaces different failure modes. Consensus filters hallucinations; divergences reveal genuine judgment calls.

## Cursor environment

- Prefer **agent-chatrooms** (Task subagents, no extra API key) unless the user explicitly wants this script or already has `ANTHROPIC_API_KEY` configured.
- Run the orchestrator via Bash from any project directory.

## Prerequisites

```bash
pip install anthropic python-dotenv
```

Set `ANTHROPIC_API_KEY` in `~/.env` or `~/.cursor/skills/model-chat/.env`.

## Execution

### 1. Parse the request

Extract:

- **Topic/problem** to debate
- **Mode**: auto (default) or interactive (`--interactive`)
- **Agent count**: default 5
- **Round count**: default 5

### 2. Run the orchestration script

```bash
python3 ~/.cursor/skills/model-chat/model_chat.py "<topic>"
```

Interactive mode:

```bash
python3 ~/.cursor/skills/model-chat/model_chat.py "<topic>" --interactive
```

Optional flags:

- `--agents N` — number of agents (default 5)
- `--rounds N` — debate rounds (default 5)

### 3. Deliver results

The script prints the debate live and saves:

- `~/.cursor/agent-workspace/model-chat/<YYYYMMDD-HHMMSS>/conversation.json`
- `~/.cursor/agent-workspace/model-chat/<YYYYMMDD-HHMMSS>/synthesis.md`
- `~/.cursor/agent-workspace/model-chat/latest` — symlink to the most recent run

Present to the user:

- Brief summary of agreements and disagreements
- The synthesis (or path to `synthesis.md`)
- Notable debate moments

## How it works internally

1. **Agents** with framing variations (systems thinker, pragmatist, edge-case finder, and others)
2. **Round-robin**: each round, all agents see full history and respond in parallel
3. **Default 5 rounds**
4. **Synthesizer**: reads the transcript and produces a structured merged answer
5. **Interactive mode**: optional user input between rounds

## Output files

| File | Description |
|------|-------------|
| `conversation.json` | Full structured conversation log |
| `synthesis.md` | Final synthesized answer |
