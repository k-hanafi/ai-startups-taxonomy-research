
---
name: git-commit-batch-plan
description: >-
  After a large change set, inventory git state and produce an ordered plan of
  commits for a portfolio audience (recruiters, research advisors). WHY-first
  titles tied to the project goal, plain-English bodies, no insider jargon.
  Bugbot fix batches use the prefix `Bugbot fix: ` on titles, same style otherwise.
  Use when planning commits, splitting work, or shaping public history.
---

# Git: portfolio-grade commit plan after a long work session

## Audience and goal

This is **not** casual pair-programming notes. The log is a **public portfolio**. The reader is usually a **recruiter or research professor** who do not review code. They scan titles. They sometimes open one commit and read the body for 10 to 20 seconds.

They should be able to answer, just from the message:

1. What did this commit add to the project?
2. How does it move the **overarching project goal** forward?
3. Was the engineer thoughtful about it (tradeoffs, judgment, what was left out)?

If a bullet only makes sense to someone already in the codebase, it is wrong for this audience.

## The project-goal anchor

Before drafting commit messages, write down (or have the user confirm) **one sentence** describing the **overarching goal of the project**. Examples:

- "Classify 265k startups into a research-grade AI-native taxonomy with high-confidence evidence."
- "Run reproducible LLM batch jobs at scale without leaking prompts, cost, or output schema."

Every commit body should **link back to this goal in plain language**. Recruiters do not know what `website_alive` is. They do know what "skip dead websites so we do not waste money" means.

If the user has not stated the goal, ask once before generating the plan.

## Strict punctuation (titles and bodies)

**Never** use an em dash or a semicolon to separate clauses in commit titles or commit bodies. Use commas, periods, parentheses, or new sentences. Apply the same rule to example titles and bodies in chat when you draft commits.

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

- Dependencies first when later commits truly need them, but the **title** still states **why** that foundation matters for the project goal, not "added file X".
- Order so a stranger sees **progress**: each commit visibly takes the project closer to the goal.


## How to slice commits

- One theme per commit. CI still plausible per step where it matters.
- Split by concern (data prep vs model or prompt vs evaluation vs visualization vs one-offs).
- Keep generated junk out unless the user wants it tracked.

## Commit titles: WHY first, plain English, one line

The title is the **main scan line**. A recruiter must understand the value from the title alone, with **no project context**.

Rules:

- 50 to 75 characters. Go longer only if shortening hides the *why*.
- Imperative mood is fine. Lead with **value or problem solved**, mechanism second if at all.
- Prefer concepts a non-engineer can picture (cost, accuracy, reliability, clarity, speed, reproducibility) over module or file names.
- No insider acronyms or filenames in the title unless they are common English (CSV, API, README, CLI are fine).

Examples:

**Weak (insider only):** `Add tavily_crawl.py`  
**Stronger (value visible):** `Pull homepage evidence from each company website to lift classifier confidence`

**Weak:** `Add paths.py`  
**Stronger:** `Standardize where pipeline outputs are saved so research artifacts are reproducible`

**Weak:** `Gate Tavily on website_alive`  
**Stronger:** `Skip dead websites before paid crawls so the research budget is not wasted`

If the why does not fit cleanly, put the extra nuance in the **first line of the body**, still short and still plain English.

## Bugbot fix commits

When committing changes that address findings from a Bugbot review (or follow-up fixes for the same class of issues), **prefix the title** with `Bugbot fix: ` and keep everything after the prefix in the **same WHY-first, plain-English style** as every other commit in this skill.

**Title shape:**

```
Bugbot fix: <same style as a normal commit title>
```

The prefix is the only difference. Do not switch to insider language, file names, or "fix bug in extract.py" style titles. The reader scanning the log should still see the value in the words after the prefix.

**Examples:**

**Normal commit title:** `Rebuild scrape counters from the log so interrupted runs resume correctly`

**Bugbot fix title:** `Bugbot fix: Rebuild scrape counters from the log so interrupted runs resume correctly`

**Normal commit title:** `Reject coverage hits when archive timestamps cannot be parsed`

**Bugbot fix title:** `Bugbot fix: Reject coverage hits when archive timestamps cannot be parsed`

**Rules for Bugbot fix commits:**

- Use the prefix on **every** commit in a Bugbot fix batch, not only the first.
- **Bodies** follow the same rules as other commits (goal anchor, bullets, verification line). Optionally mention that the change closes a review finding in one plain sentence, without pasteing Bugbot severity labels or internal line numbers in the title.
- Still one theme per commit. Split large fix sets the same way you would split feature work.
- The 50 to 75 character title guideline applies to the **whole** line including the prefix. Go longer only if shortening hides the why.

## Commit bodies: high-level value, then judgment

Bodies must be **readable by an outsider**. The first sentence (or first bullet) **must connect this commit to the project goal in plain English**. Then add 2 to 5 short, concrete points that show why the change was thoughtful, not mechanical.

**Shape (pick one):**

- **Option A:** 1 to 2 short opening sentences in plain English that tie the change to the project goal, then 2 to 5 bullets.
- **Option B:** 3 to 6 bullets only, where the **first bullet** restates the value to the project in one line.

What each body should usually include:

1. **Why this matters for the project**, in language a non-engineer can follow.
2. **What changed**, at a high level (what the system can now do, or stops doing).
3. **One sign of judgment**: a tradeoff, a thing deliberately not done, a constraint you respected (cost, time, scope, reproducibility).
4. **One concrete detail** when it adds credibility (a number, a default, a path, a behavior). Skip if it would only mean something to someone already in the code.
5. **Verification** in one honest line. Examples: `Tests: pytest`, `Verified manually on a 100-row sample`, `Not run: paid API call, dry-run only`.

Length target: most bodies are **3 to 6 lines total**. If it is getting longer, the commit is probably two commits.

**Do not:**

- Use LLM or corporate filler such as *leverage, robust, seamless, comprehensive, holistic, empower, delve, landscape, synergy, cutting-edge, world-class*.
- Claim "strong design" or "intentional engineering" in the abstract. Show it with one specific choice.
- Drop function names, internal variable names, or column names without translating them. If you must mention `website_alive`, gloss it as "the dead-website flag" first.
- Write narration of which files were touched. The reader can see the diff.

**Do:**

- Write like the engineer who owns the decision is explaining it briefly to a smart non-specialist.
- Translate jargon. Replace "MECE taxonomy boundaries" with "non-overlapping category definitions". Replace "RAD signal" with "researcher-assigned difficulty signal" or just "difficulty rating", whichever fits.
- Keep verbs concrete: cut, replace, narrow, default, gate, retry, log, document.

### Worked example

**Bad (insider, mechanical):**

```
Gate Tavily spend with website_alive on a single lean classifier CSV

- Default artifact is outputs/tavilycrawl/classifier_input.csv.
- Crawl queue skips rows with website_alive=false.
- Master join trimmed to fields we still inject into prompts.
```

**Good (recruiter-readable, goal-anchored):**

```
Skip dead company websites before paid web crawls to protect the research budget

The project classifies 265k startups using web evidence, so every wasted crawl
is real money and noisy data. This commit checks each homepage once and
remembers whether it is reachable, then only paid crawls run for live sites.
Companies with dead sites still get classified using the data we already have.

- One canonical input file replaces a fan-out of intermediate CSVs, so reruns are predictable.
- Trimmed the company profile down to the fields that actually feed the model, on purpose.
- Verified end-to-end on a small sample. Full crawl is gated until budget review.
```

### Worked example (Bugbot fix)

**Title:**

```
Bugbot fix: Backfill classifier CSV from snapshots so crashes cannot drop companies
```

**Body (same shape as a normal commit):**

```
Stage D reads the processed scrape file, but a crash could mark a company done in
the log without a matching row. This commit closes that gap so the historical
comparison panel stays complete after an interrupted run.

- Startup reconciles counters and backfills any missing success rows from the log.
- Processed CSV is saved before the log line on each success.
- Tests: pytest wayback_machine/tests/test_state.py.
```

## Chat handoff

New chat for commits: tag this skill and paste `git status --short` plus `git diff --stat` if the repo state is not visible. Confirm the project-goal sentence before drafting messages.
