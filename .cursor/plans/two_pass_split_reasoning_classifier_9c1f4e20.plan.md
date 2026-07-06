---
name: Two-pass split-reasoning classifier
overview: Split the single-shot classifier into two sequential API calls with different reasoning configs, because reasoning and logprobs are mutually exclusive per request. Pass A (reasoning off, logprobs on) makes the binary ai_native decision with calibrated logprob confidence; Pass B (reasoning high, family-constrained) makes the subclass + RAD decision. Validate in the eval harness first, then promote to production.
todos:
  - id: s1-prompts
    content: "STAGE 1 (branch two-pass/stage-1-prompts, own PR): decompose system_classifier_prompt.txt per the mapping table into binary_gate_prompt.txt (Pass A, binary-only few-shots, collapsed decision procedure) and family-parameterized subclass_rad_prompt.txt (Pass B, {family_block} for 1A-1G vs 0A-0C + sorted few-shots); keep the AI-native definition single-source; user reviews prompt text before merge"
    status: pending
  - id: s1-smoke
    content: "STAGE 1: cheap live smoke of each prompt on 3-5 golden rows (nano) to catch format/instruction breakage before the PR merges - not an accuracy eval"
    status: pending
  - id: s2-schema
    content: "STAGE 2 (branch two-pass/stage-2-implementation, own PR): BinaryResult schema (ai_native only) and family-parameterized SubclassResult schema (Literal 1A-1G OR 0A-0C + boundary_disagreement flag); cohort computed in code"
    status: pending
  - id: s2-runner
    content: "STAGE 2: two-pass runner in evals/ - Pass A at effort=none+logprobs, cohort in code, Pass B at effort=high with the family schema chosen from Pass A; assemble the full record; offline tests"
    status: pending
  - id: s3-eval
    content: "STAGE 3 (paid, after both PRs merge): run two-pass over the golden set; measure binary calibration (Pass A logprobs) + conditioned subclass accuracy vs the banked single-pass baselines (none 41% / high 66% vs Fable)"
    status: pending
  - id: s3-decide
    content: "STAGE 3: go/no-go vs single-pass high baseline; if go, write the production promotion plan and update the logprob plan's implementation half"
    status: pending
isProject: false
---

# Two-Pass Split-Reasoning Classifier

Realizes the confidence half of
[.cursor/plans/logprob_confidence_classifier_17f55781.plan.md](logprob_confidence_classifier_17f55781.plan.md)
with a concrete architecture forced by the eval-harness findings.

## Empirical basis (verified live, 2026-07-05/06, gpt-5.4-nano, n=100)

- Reasoning models reject `temperature` (400).
- Logprobs and reasoning are mutually exclusive per request: logprobs return
  only at `reasoning.effort=none`; `minimal/low/medium/high` all 400 on logprobs.
- Binary `ai_native`: `none` and `high` both agree with Fable 93%, and agree
  with each other 96%. Reasoning adds ~nothing to the binary axis.
- Subclass (10-way): `high` 66% vs `none` 41% (kappa 0.57 vs 0.32). Reasoning
  is load-bearing. 88% of `none`'s subclass misses stay in the correct family,
  and the damage is `none` collapsing 0A/0B/0C into 0A (53 vs Fable's 13).
- RAD weak in both regimes (~46-51%) — a separate signal-quality problem.

Conclusion: harvest logprob confidence where reasoning is free (binary), and
pay for reasoning where it matters (subclass), which is impossible in one call.

## Architecture

```
evidence ─▶ Pass A: binary gate      (effort=none, logprobs on, schema={ai_native})
                │  └─ logprob/entropy on the single decision token = binary confidence
                ▼
           cohort computed in code   (founded_date vs 2023-03-14; no LLM)
                │
                ▼
           Pass B: subclass + RAD     (effort=high, logprobs off)
                │  ├─ told "prior analysis found ai_native=X; choose within [family]"
                │  ├─ schema Literal constrained to 1A-1G OR 0A-0C (API-enforced)
                │  ├─ RAD rules shown only when X=1 (else RAD-NA by construction)
                │  └─ boundary_disagreement flag when the smart pass distrusts the gate
                ▼
           assemble 11-field ClassificationResult
```

## Key decisions (locked 2026-07-06)

- **Sequential, conditioned.** Pass B receives Pass A's verdict and is
  hard-constrained to that family; halves the label space and attacks the 0A
  collapse. A `boundary_disagreement` boolean records (but does not act on)
  cases where high-reasoning Pass B distrusts the cheap gate — a calibration
  check on the binary confidence, at zero control-flow cost.
- **Separate prompts.** Pass A is a lean, directive binary prompt (union of
  1A-1G as *mechanisms* + the two guardrails + tie-break-to-0 + binary-boundary
  few-shots). Lean matters more here because reasoning is off. Pass B carries
  one family's detailed definitions + reasoning steps + intra-family few-shots.
- **Shared AI-native definition (invariant).** The binary boundary is *defined
  as* "falls into one of 1A-1G." Prompt A's condensed definition and Prompt B's
  detailed family must stay consistent; changing the taxonomy updates both in
  the same change (add to AGENTS.md, like the evidence.py re-vendor rule).
- **Minimal Pass A output** so the decision-token logprob is clean (not buried
  in a 195-token JSON blob).
- **cohort is deterministic**, computed in code, removed from both LLM passes.
- **Confidence sources by axis:** binary = logprob/entropy (Pass A); subclass +
  RAD = self-reported conf_* fields, optionally repeat self-consistency.

## Prompt decomposition (from the current monolith)

`prompts/system_classifier_prompt.txt` (744 lines) splits by section into the
two new prompts. Mapping:

| Current section (lines) | Goes to | Notes |
|---|---|---|
| Role header (1-4) | both (reworded) | A: "decide AI-native or not"; B: "assign subclass + RAD within a known family" |
| INPUT FORMAT + evidence hierarchy (6-36) | both (shared, byte-identical) | Same evidence feeds both passes |
| AI-native DEFINITION + wrapper note (42-52) | A gets the binary core ("wrappers ARE AI-native"); B gets the 1C-vs-1D depth half | This is the **shared-definition invariant** anchor |
| CRITICAL GUARDRAIL marketing-vs-mechanism (54-62) | **A** | The binary trap; keep verbatim |
| DECISION PROCEDURE (64-79) | **split** | A keeps the counterfactual (Step 3-4) collapsed to 1-vs-0 + coin-flip-to-0. Steps 1-2 (0C sorting) and the 0A/0B cut move to **B's zero-family branch** |
| Subclasses 1A-1G (81-173) | **B** (one-family branch) | Shown only when Pass A = 1 |
| Subclasses 0A/0B/0C (175-212) | **B** (zero-family branch) | Shown only when Pass A = 0 |
| DIMENSION 2 RAD, all (214-300) | **B**, only when family = 1 | When family = 0, RAD is forced RAD-NA in code, rules not shown |
| TEMPORAL COHORT (302-317) | **code** computes value; condensed text to **B** | B needs cohort as an input for RAD defaults; the label itself is deterministic |
| conf_classification rubric (323-341) | **B** | Binary confidence comes from Pass A logprobs, so this rubric now scores subclass only |
| conf_rad / reasons / sources / critique (343-370) | **B** | B produces the narrative fields |
| EDGE CASES (372-397) | **split** | Binary/pivot/services-to-0C/coin-flip to A; 1C-vs-1D, 1E, MLOps-to-1B, labeling to B |
| FEW-SHOT EXAMPLES (399-744) | **split + reformatted** | See below |

**Few-shot reformatting is required, not just copying.** Pass A examples must
have binary-only outputs (`ai_native: 0|1`, nothing else), curated for the 1-vs-0
boundary and the marketing traps (Stripe 0A, Notion 0B, Futuremood 0C, plus 2-3
clear 1s). Pass B examples keep their full subclass+RAD outputs but are sorted
into the family branch they belong to (1A-1G examples for the one-family prompt,
0A/0B/0C for the zero-family prompt).

**Two prompt files, but Pass B is family-parameterized.** Options: one
`subclass_rad_prompt.txt` with a `{family_block}` placeholder filled at request
build time (one-family vs zero-family definitions + few-shots), or two physical
files. Prefer the placeholder so the shared framing/rubrics stay single-source.

## Costs / tradeoffs

- 2x calls per company (88k for the full set). Acceptable: cost is not a
  constraint, Batch API handles it, caching works per-pass.
- Latency doubles per company — irrelevant for batch.
- Reference for all agreement numbers is Fable drafts, not human-verified gold;
  the verdict pass will firm absolute numbers. The none-vs-high comparison is
  reference-independent.

## Execution workflow: two stages, two branches, two PRs (locked 2026-07-06)

The prompt content and the plumbing are separable risks, so they ship
separately (same per-stage loop as the eval harness: local Bugbot before push,
GitHub Bugbot on the PR, squash-merge):

- **Stage 1 — Prompts** (`two-pass/stage-1-prompts`). Pure prompt-engineering
  PR: the two new prompt files decomposed per the mapping table, plus a cheap
  3-5 row live smoke per prompt (format sanity, not accuracy). No Python
  behavior changes. The diff is prose, so the user reviews the taxonomy text
  itself before it becomes the new source of truth.
- **Stage 2 — Implementation** (`two-pass/stage-2-implementation`). Schemas,
  request builders, the two-pass runner in `evals/`, cohort-in-code, offline
  tests. Depends on Stage 1's merged prompt files but on nothing else new.
- **Stage 3 — Paid validation run + go/no-go.** Not a PR of its own; produces
  run artifacts + the decision, recorded back into this plan.

Sequencing with the eval harness ([golden_set_eval_harness_f23e64d5.plan.md](golden_set_eval_harness_f23e64d5.plan.md)):
harness PR 3 (runner, open) merges first since Stage 2 builds on its run-dir
conventions. Harness Stage 4 (logprob extraction) now targets Pass A's
binary-only output (a near-single-token JSON), which simplifies it. Harness
stages 5-9 (batch parity, scorer, experiments, dashboard, gate report) proceed
against two-pass runs.

## Superseded assumptions in the parent logprob plan

[logprob_confidence_classifier_17f55781.plan.md](logprob_confidence_classifier_17f55781.plan.md)
"Locked config" was based on user research that the eval harness has now
disproven live: `temperature=0` is rejected by reasoning models, and logprobs
are NOT exposed with reasoning on. Its single-call design (reasoning=medium +
temperature=0 + top_logprobs on one request) is impossible; this plan is the
replacement architecture. Its extraction algorithm (composed-joint subclass
probability, autoregressive conditioning rule) mostly dissolves: Pass A gives
a clean binary token, and Pass B carries no logprobs. Keep its schema naming
work (evidence_sufficiency rename) for the production promotion.

## Validation gate (before production)

Two-pass conditioned subclass accuracy must beat the single-pass high baseline
(66% vs Fable) by enough to justify the second call; binary logprob confidence
must show usable calibration (reliability + selective prediction). Both measured
in evals/ before any src/ change.
