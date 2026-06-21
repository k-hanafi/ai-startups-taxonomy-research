---
name: agent-chatrooms
description: >
  Spawn multiple agents into a shared conversation where they debate, disagree, and converge on a solution via a shared chat.json file. Use for architecture decisions, code review debates, design trade-offs, or any problem that benefits from adversarial reasoning. Triggers on "chatroom", "agent debate", "have agents discuss", "agent chatroom", "debate this", or /agent-chatrooms. Also triggers on phrases like "have 3 agents argue about", "multi-agent discussion", or "let them debate".
disable-model-invocation: true
---

# Agent Chatrooms

## Cursor environment

- Spawn subagents with the **Task** tool (`subagent_type`: `generalPurpose`, `explore`, or `shell` as appropriate).
- Optional `model` on Task only when the user requests a specific model.
- Ephemeral artifacts: `~/.cursor/agent-workspace/<subdir>/` (create as needed). For repo-scoped runs, use `.cursor/agent-workspace/<subdir>/` in the project.
- Use **AskQuestion** for structured user questions.


Spawn N agents (default 3) into a simulated shared conversation. Each agent reads the full chat history before responding, building on, challenging, or refining previous contributions. You (the orchestrator) act as PM — moderating rounds, reading the debate, and extracting the final output.

**Why this works:** Sequential handoffs lose context — the second model doesn't know WHY the first made its decisions. A shared conversation preserves reasoning chains and enables genuine debate. When Agent A says "this needs a queue" and Agent B says "a simple loop is fine," that disagreement is more valuable than either agent's confident solo answer. Agents challenge assumptions, catch errors, and synthesize diverse reasoning paths.

## Execution

### 1. Parse the request

Extract from the user's message:
- **Problem/question** to debate
- **Agent count N** — default 3. User can override ("have 5 agents debate")
- **Round count R** — default 3 rounds. User can override ("debate for 5 rounds")
- **Agent roles** (optional) — user may specify roles ("have an architect, a security engineer, and a frontend dev debate this"). If not specified, assign diverse default roles.

If the problem is vague, ask the user to sharpen it before spawning.

### 2. Assign agent roles

Each agent gets a distinct perspective to maximize productive disagreement. If the user didn't specify roles, choose from these defaults based on the problem domain:

**Software engineering:**
1. **Architect** — thinks in systems, interfaces, scalability, and long-term maintainability
2. **Pragmatist** — optimizes for shipping fast, minimal complexity, "good enough" solutions
3. **Critic** — finds edge cases, failure modes, security holes, and unstated assumptions

**Product/design:**
1. **User advocate** — optimizes for UX, simplicity, user delight
2. **Business strategist** — optimizes for revenue, growth, competitive advantage
3. **Engineer** — grounds the discussion in technical feasibility and cost

**Strategy/decisions:**
1. **Optimist** — sees opportunity, upside potential, reasons to act
2. **Skeptic** — sees risk, downside, reasons to wait
3. **Synthesizer** — finds the middle path, integrates both perspectives

For N > 3, add more roles that create productive tension with the existing ones.

### 3. Initialize the chat file

Create `~/.cursor/agent-workspace/chatroom/chat.json` with the initial structure:

```json
{
  "problem": "{problem statement}",
  "context": "{any relevant context, code, constraints}",
  "agents": [
    {"name": "Agent A", "role": "{role}", "framing": "{role description}"},
    {"name": "Agent B", "role": "{role}", "framing": "{role description}"},
    {"name": "Agent C", "role": "{role}", "framing": "{role description}"}
  ],
  "rounds": [],
  "final_output": null
}
```

### 4. Run debate rounds

For each round (1 through R), spawn all N agents in parallel. Each agent reads the full chat history and contributes.

#### Agent spawn config:
- `subagent_type: "general-purpose"`
- `model: "sonnet"` (default — user can override to opus for deeper reasoning)

#### Round procedure:

**Round 1 — Opening positions:**
Each agent states their initial take on the problem.

Agent prompt:
```
You are {role}: {role_description}

PROBLEM:
{problem}

CONTEXT:
{context}

This is Round 1 of a multi-agent debate. State your initial position on this problem. Be specific and concrete — propose actual solutions, not vague principles. Take a clear stance.

Other agents will challenge your position in subsequent rounds, so make your reasoning explicit.

Respond in this format:
POSITION: [Your one-sentence stance]
REASONING: [Your detailed argument — 3-5 key points]
PROPOSAL: [Your concrete recommendation]
CONCERNS: [What could go wrong with your approach]

Write your response directly — do not write to any files.
```

**Rounds 2+ — Debate:**
Each agent reads all previous responses and responds.

Agent prompt:
```
You are {role}: {role_description}

PROBLEM:
{problem}

PREVIOUS DISCUSSION:
{all previous round entries, formatted as "Agent X (Role): response"}

This is Round {N} of a multi-agent debate. Read the previous discussion carefully.

Your job:
1. Respond to the strongest counterargument against your position
2. Identify where you AGREE with other agents (concede good points)
3. Identify where you still DISAGREE and why
4. Refine your proposal based on the discussion so far

Do NOT just repeat your previous position. Engage with what others said. Change your mind if they made a better argument.

Respond in this format:
AGREEMENTS: [What other agents got right]
DISAGREEMENTS: [Where you still differ and why]
REFINED PROPOSAL: [Your updated recommendation]
CONFIDENCE: [1-10 how confident you are in your refined position]

Write your response directly — do not write to any files.
```

#### After each round:
1. Collect all agent responses
2. Append to the `rounds` array in `chat.json`:
```json
{
  "round": 1,
  "entries": [
    {"agent": "Agent A", "role": "Architect", "response": "..."},
    {"agent": "Agent B", "role": "Pragmatist", "response": "..."},
    {"agent": "Agent C", "role": "Critic", "response": "..."}
  ]
}
```
3. Check for convergence: if all agents agree (confidence 8+, proposals aligned), stop early — no need for more rounds.

### 5. Synthesize the final output

After the last round, you (the orchestrator) read the full debate and produce a synthesis. Do NOT spawn another agent for this — you do it yourself.

Analyze:
- **Where did agents converge?** — these are high-confidence conclusions
- **Where did they remain split?** — these are genuine trade-offs the user needs to decide
- **What concerns were raised but unresolved?** — these are risks to monitor
- **Did any agent change their mind?** — mind-changes are strong signals

### 6. Write the final report

Update `~/.cursor/agent-workspace/chatroom/chat.json` with the `final_output`, and write a human-readable summary to `~/.cursor/agent-workspace/chatroom/chatroom_report.md`:

```markdown
# Agent Chatroom Report

**Problem**: {problem}
**Agents**: {N} | **Rounds**: {R}
**Date**: {date}

## Participants
| Agent | Role | Final Confidence |
|-------|------|-----------------|
| Agent A | {role} | {confidence}/10 |
| Agent B | {role} | {confidence}/10 |
| Agent C | {role} | {confidence}/10 |

## Consensus
{What all agents agreed on by the final round}

## Key Disagreements
{Where agents remained split — present both sides fairly}

## Recommended Action
{Your synthesis as orchestrator — the best path forward considering all perspectives}

## Unresolved Risks
{Concerns raised during debate that weren't fully addressed}

## Debate Highlights
{The most interesting exchanges — where minds changed or strong counterarguments emerged}

## Full Transcript
{Link to chat.json or inline the key exchanges}
```

### 7. Deliver results

Present to the user:
- **One-paragraph synthesis** of the debate outcome
- **The recommended action** (your call as orchestrator, informed by the debate)
- **The sharpest disagreement** (where the user's judgment is needed)
- File paths to report and chat.json

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| N | 3 | Number of agents in the chatroom |
| R | 3 | Number of debate rounds |
| model | sonnet | Model for each agent |
| roles | auto | Agent roles (auto-assigned or user-specified) |

User can override: "have 5 opus agents debate for 4 rounds" or "debate between an architect and a junior dev".

## Cost considerations

- 3 sonnet agents x 3 rounds = 9 agent calls (~$0.30-0.50)
- 3 opus agents x 3 rounds = 9 agent calls (~$3-5)
- 5 agents x 5 rounds = 25 calls — gets expensive with opus
- Default to sonnet. Only use opus if user explicitly requests it or the problem requires deep technical reasoning.
- Early convergence saves cost — stop if agents agree before all rounds complete.

## Edge cases

- **N < 2**: Warn the user — a chatroom needs at least 2 agents. Minimum 2.
- **Agents all agree immediately**: Stop after round 1. Report unanimous consensus. This is a valid (and cheap) outcome.
- **Agents deadlock**: After R rounds with no convergence, report the deadlock honestly. The finding is that this is a genuine judgment call with no dominant answer.
- **Agent goes off-topic**: Exclude that response from synthesis and note the effective agent count.
- **Existing chatroom files**: Check `~/.cursor/agent-workspace/chatroom/` for existing files. Overwrite without asking (these are ephemeral debate artifacts, not persistent data).
- **User specifies custom roles**: Use exactly what they specify. Don't add extra roles unless asked.

## Output files

| File | Description |
|------|-------------|
| `~/.cursor/agent-workspace/chatroom/chat.json` | Full structured debate transcript |
| `~/.cursor/agent-workspace/chatroom/chatroom_report.md` | Human-readable synthesis report |

Previous reports are overwritten — these are ephemeral debate tools, not archives.
