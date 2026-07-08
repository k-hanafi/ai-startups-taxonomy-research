
---
name: portfolio-git-messages
description: >-
  Portfolio-grade git commit and PR message style for a public research log.
  WHY-first plain-English titles, goal-anchored bodies, Bugbot fix prefix rules.
  Read and follow before EVERY git commit, amend, squash, PR title, or PR body.
  Use whenever committing, merging, drafting public history, or writing PR text.
disable-model-invocation: false
---

# Portfolio git messages

**Mandatory:** Read this skill before writing any commit message or PR title/body.

The git log is a **public portfolio**. Readers are recruiters and research advisors who scan titles and spend 10 to 20 seconds on a body. They should answer from the message alone:

1. What did this add to the project?
2. How does it move the **project goal** forward?
3. Was the engineer thoughtful (tradeoffs, scope, what was left out)?

If a bullet only makes sense to someone already in the codebase, rewrite it.

## Project goal anchor

Before drafting, get **one sentence** for the project goal:

- Read `AGENTS.md` or `README.md` in the repo when present.
- If unclear, ask once. Do not block a tiny fix, but do anchor larger commits.

Every body ties back to that goal in plain language. Translate jargon (e.g. "dead-website flag" not `website_alive` alone).

## Punctuation

**Never** use an em dash or semicolon to separate clauses in titles or bodies. Use commas, periods, parentheses, or new sentences.

## Titles: WHY first, one line

- 50 to 75 characters. Longer only if shortening hides the why.
- Imperative mood is fine. Lead with **value or problem solved**, mechanism second.
- Prefer cost, accuracy, reliability, clarity, speed, reproducibility over file or module names.
- No insider acronyms unless common English (CSV, API, README, CLI).

**Weak:** `Add tavily_crawl.py`  
**Strong:** `Pull homepage evidence from each company website to lift classifier confidence`

If the why does not fit in the title, put nuance in the first body sentence.

## Bugbot fix commits

Prefix: `Bugbot fix: ` then the **same WHY-first title style**. No insider titles like "fix extract.py".

```
Bugbot fix: Backfill classifier CSV from snapshots so crashes cannot drop companies
```

- Prefix on **every** commit in a Bugbot fix batch.
- Bodies follow the same rules below. One plain sentence may note it closes a review finding. No severity labels or line numbers in titles.

## Bottom-up layout (light steer)

**Prefer** skimmable messages: the first sentence or two should answer *what this does and why it matters* in plain English, before file paths, module names, or implementation detail. Put technical bullets underneath for readers who want depth. Adapt when a different shape is clearer.

## Bodies: value, then judgment

Pick one shape:

- **Option A:** 1 to 2 opening sentences tying to the project goal, then 2 to 5 bullets.
- **Option B:** 3 to 6 bullets where the **first** restates project value.

Include when relevant:

1. Why this matters (non-engineer language).
2. What the system can now do or stop doing.
3. One sign of judgment (tradeoff, deliberate omission, constraint respected).
4. One concrete detail if it adds credibility (number, default, behavior).
5. **Verification** in one honest line: `Tests: pytest`, `Not run: paid API call`, etc.

Target **3 to 6 lines total**. Longer probably means split the commit.

**Do not:** corporate filler (*leverage, robust, seamless, comprehensive, holistic, empower, delve, landscape, synergy, cutting-edge, world-class*), file-change narration, raw function or column names without gloss.

**Do:** concrete verbs (cut, replace, narrow, default, gate, retry, log, document).

## No baby commits on main

The public log is judged on substance per commit, not commit count. Micro-commits of plan or STATUS files read as contribution-graph padding.

- **Never** commit plan/STATUS/markdown housekeeping edits to main as standalone micro-commits. Batch them.
- Plan/STATUS updates accumulate in the working tree and land as **one consolidated commit per real milestone** (a PR merge, a locked pivot batch, a session end) with a substantive WHY-first message.
- Prefer riding doc/plan updates on the relevant PR branch instead of direct-to-main whenever a related PR exists.
- Rule of thumb: if a commit touches only `.md` files and takes under a minute to explain, it probably belongs batched with the next substantive commit.
- Direct-to-main commits are reserved for consolidated milestone records, never per-thought updates.

## PR titles and bodies

PR titles follow the same WHY-first title rules as commits.

PR bodies: **Summary** first, then **Test plan** checklist.

- **Summary headline (1 to 2 sentences):** what the PR does and why, readable without opening the diff. This is the only part many readers need.
- **Summary detail (optional bullets below):** tradeoffs, scope, concrete behavior. Technical specifics belong here, not in the headline.
- Same plain-English tone throughout. No em dashes or semicolons as clause separators.

## More examples

See [examples.md](examples.md) for full good/bad commit and Bugbot fix samples.
