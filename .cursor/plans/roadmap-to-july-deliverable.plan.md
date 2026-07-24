# Roadmap to July Deliverable

Master plan / PRD for the professor meeting (late July 2026). Owned by the
orchestrator chat; workers execute scoped units and report back. Update the
STATUS block after every milestone, decision, or pivot.

---

## STATUS

- **State (2026-07-23 eve):** WU-0/1p/4a/5a all DONE. Deliverable 3 fallback
  is on main. Next: eval suite redesign plan, then the user-run Stage 8
  sweep, then WU-2/WU-3.
- **WU-4a DONE:** dead-cohort V1 classification complete after two recovery
  fixes (download crash resume; batch 1 split for the 200 MiB Batch API
  file cap). 15,714 verdicts; survivorship_corrected.csv written (44,386
  rows; 15,682 dead verdicts overlaid, 32 IDs unmapped, parked). Headline:
  dead AI-native rate 5.0% vs 14.1% alive-evidence; corrected rate 11.2%
  vs 15.6% biased.
- **WU-5a DONE (PR #31 MERGED):** alive-vs-dead dashboard on real data,
  evidence-only universe (21,990 alive + 15,682 dead), 4-act survivorship
  section, insights dashboard retired. Median-confidence card replaced with
  mean 4.2/5 (median legitimately pins at 5; 52.8% of rows self-report max
  confidence, itself a motivating stat for V2 logprob confidence).
- **Eval dashboard:** mock instance passed visual inspection (GO for real
  runs). PR #32 (cost-breakdown popover) CLOSED by decision, to be folded
  into the redesign.
- **NEW SCOPE, eval suite redesign:** 3 sections only: (1) Pipeline
  robustness (logprob extraction + batch parity, will it survive
  production), (2) Model benchmarks (accuracy/confidence quality/cost/
  latency across the GPT family), (3) Confidence correctness correlation
  (reliability bins, ECE, selective prediction; early signal ECE 0.077,
  top-half confidence -> 100% accuracy). Everything else out of scope by
  decision. Professional product voice (no design-inspiration references
  in user-facing copy), de-slop the styling. Sequence: plan -> one-shot
  implement -> debug, on a separate branch.
- **In-flight workers:** none.
- **User actions pending:** Stage 8 sweep (9 cells, from main) whenever
  ready; it is independent of the redesign.

---

## Decisions (locked 2026-07-22)

- **D1:** Eval matrix = gpt-5.4-nano / gpt-5.4-mini / gpt-5.6-luna x Pass B
  low/medium/high. 9 runs total, 100 golden rows each. Luna must be added to
  `evals/config.py` models + pricing first.
- **D2:** The user personally executes ALL paid eval and classifier runs in
  their own terminal. Orchestrator/workers only prepare code and hand over
  exact commands. Dry-run cost estimates precede every paid run.
- **D3:** Deliver results for both cohorts. Alive = the ~22k evidence rows
  (V1 results already exist for all 44,387). Dead = the 15,714 with archive
  evidence.
- **D4:** Fallback-first sequencing. Run the EXISTING V1 classifier on the
  dead cohort now (before the eval sweep) so an alive-vs-dead comparison
  dashboard exists no matter what. Only after that dashboard exists do we
  build V2.
- **D4b (architecture):** Production V2 uses the SYNC Responses API, not the
  Batch API. Rationale: two-pass over Batch is a two-stage DAG with
  resume/double-billing risk (the plan's riskiest item); sync reuses the
  already-working eval runner pattern. Cost: forfeits the 50% batch
  discount (~2x model spend on ~37k rows, acceptable at nano-class pricing).
- **D4c (structure):** V1 and V2 are BOTH maintained, as standalone sibling
  packages: `src/single_pass_classifier/` (today's `src/` modules moved) and
  `src/two_pass_classifier/` (new, promoted from `evals/`). Underscores, not
  hyphens (hyphens are illegal in Python package names). The restructure
  happens only AFTER the V1 fallback run completes, so `classify_dead.py`
  is never destabilized mid-run.

---

## The three deliverables

| # | Deliverable | What "done" looks like |
|---|------------|------------------------|
| 1 | **V2 progress presentation dashboard** | Narrative HTML telling the V1 -> V2 story: single-pass + self-reported 1-5 confidence -> two-pass split-reasoning + logprob-calibrated confidence + golden-set eval harness. |
| 2 | **Production eval dashboard instance** | LangSmith-style eval dashboard rendering REAL Stage 8 results (9 configs, 100-row golden set), ready for config discussion. |
| 3 | **Production dashboard, alive + dead** | Classification dashboard over ~22k alive + ~15.7k dead with an alive-vs-dead section isolating survivorship bias. V1 version first (fallback), V2 version after migration. |

---

## Verified current state (2026-07-22)

### Eval harness (`evals/`) — feeds deliverables 1 + 2
- DONE: stages 0-7 + cost-extrapolation pivot merged; Stage 9 mock dashboard.
  Pass A batch parity validated (moot for production now that V2 is sync,
  still evidence of correctness).
- Golden set: `evals/golden/golden_set.csv`, 100 stratified rows with
  `draft_*` provisional labels.
- NOT DONE: Stage 8 paid sweep. Blocker: luna missing from
  `evals/config.py` `EVAL_MODELS` + `EVAL_MODEL_PRICING` (worker fixing).
- Per-cell commands (user-run): `python -m evals run-two-pass --model M
  --effort-b E` -> `python -m evals score <run_id> --confidence-from-raw` ->
  `python -m evals dashboard --runs <ids...>`.

### Survivorship pipeline (`wayback_machine/`) — feeds deliverable 3
- DONE through paid extract: 19,044 targets, 15,714 with evidence in
  `wayback_machine/outputs/processed/scrape_processed_dead.csv`.
- REMAINING (user-run, V1 fallback = WU-4a):
  1. `python3 wayback_machine/scripts/build_classifier_input_dead.py` (free)
  2. `python3 wayback_machine/scripts/classify_dead.py prepare --dry-run`
  3. `caffeinate -ims python3 wayback_machine/scripts/classify_dead.py run`
     (paid, Batch API, `CLASSIFY_NS=wayback_dead`)
  4. `python3 wayback_machine/scripts/merge_survivorship.py` (free) ->
     `outputs/wayback_dead/survivorship_corrected.csv`

### V2 production migration — now sync-API, standalone package
- V2 lives in `evals/` (sync Responses API, 100-row scale). Production
  promotion = scale-up of the sync runner, NOT a batch DAG.
- To port: Pass A binary gate (effort none, top_logprobs=15,
  `prompts/binary_gate_prompt.txt`) -> Pass B family-constrained
  subclass+RAD (`prompts/subclass_rad_prompt.txt` + family blocks); logprob
  confidence via `evals/logprob_extract.py`; cohort computed in code.
- Scale-up needs: concurrency + rate limiting, resume checkpointing (never
  re-pay a completed Pass A), per-pass error files, cost accounting, output
  CSV contract with logprob columns.

### Dashboards (`data visualization/`)
- `build_classification_dashboard.py`: 5-section template (reads legacy V1
  CSV). Fork target for deliverable 3.
- `survivorship_analysis.py` + `build_survivorship_insights_dashboard.py`:
  alive-vs-dead comparisons + 2 logistic regressions built; PREVIEW mode
  until `survivorship_corrected.csv` exists. Becomes real after WU-4a.
- Eval dashboard: mock by default; real via `--runs`/`--scored`.
- No V2 narrative dashboard exists (deliverable 1 gap).

---

## Sequencing (decided)

```
NOW (parallel):
  [user, iTerm]  WU-4a  V1 fallback: classifier_input_dead -> dry-run ->
                        classify_dead run -> merge_survivorship
  [worker]       WU-0   rebase/land PR #29 + AGENTS.md staleness fix
  [worker]       WU-1p  luna in evals/config.py (models + pricing) -> PR

THEN:
  [worker]       WU-5a  V1 alive-vs-dead dashboard (deliverable 3, fallback
                        edition) from survivorship_corrected.csv
  [user, iTerm]  WU-1   Stage 8 sweep: 9 x (run-two-pass -> score) -> dashboard
  [worker]       WU-2   snapshot real eval dashboard into presentation dir

THEN (V2 build, only after WU-4a results are banked):
  [worker]       WU-3   src restructure: src/single_pass_classifier +
                        src/two_pass_classifier (sync API, scaled eval runner)
  [user, iTerm]  WU-4b  V2 sync runs: alive evidence cohort + dead cohort
                        (config chosen from Stage 8 results)
  [worker]       WU-5b  V2 production dashboard w/ survivorship section
  [worker]       WU-6   V2 narrative dashboard (can draft early, numbers last)
```

Critical path to a safe meeting: WU-4a -> WU-5a (fallback deliverable 3),
WU-1p -> WU-1 -> WU-2 (deliverable 2). Everything V2 is upside on top.

---

## Work units

### WU-0 — Housekeeping (worker, in flight)
Rebase `cursor/eval-dashboard-langsmith-ux-0263` onto main, resolve
conflicts, make PR #29 mergeable. Fix AGENTS.md staleness (extract done,
roadmap plan exists). Verify: PR mergeable, `pytest` green.

### WU-1p — Eval config prep (worker, in flight)
Add `gpt-5.6-luna` to `EVAL_MODELS` and `EVAL_MODEL_PRICING` (pricing from
OpenAI's published rates; if unpublishable, flag for user). Align matrix
with D1. PR to main. Verify: `pytest evals/tests` green, dry-run works for
all 3 models.

### WU-1 — Stage 8 sweep (USER-RUN, paid)
Dry-run each of 9 cells first, then execute. Commands handed over after
WU-1p lands. Verify: 9 scored.json files with accuracy/cost/latency.

### WU-2 — Real eval dashboard (worker, free)
`python -m evals dashboard --runs <9 ids>`; snapshot HTML into
`01_Presentation_Materials/`. Verify: no SYNTHETIC banner, 9 configs render.

### WU-4a — V1 fallback dead-cohort classification (USER-RUN, paid)
Commands in "Survivorship pipeline" above. Produces
`survivorship_corrected.csv`. Verify: merge prints before/after AI-native
rate; row counts match 15,714 dead verdicts overlaid.

### WU-5a — V1 alive-vs-dead dashboard (worker, free)
Fork/extend `build_classification_dashboard.py` +
`survivorship_analysis.py` into the fallback edition of deliverable 3:
V1 classifications, alive evidence cohort vs dead cohort, survivorship-bias
section (distribution deltas, biased vs corrected AI-native rate, logistic
forests). Verify: HTML renders with real (non-PREVIEW) dead verdicts.

### WU-3 — V2 standalone package + sync scale-up (worker, largest unit)
1. Move existing `src/` modules into `src/single_pass_classifier/`; fix
   imports (`classify.py`, wayback scripts, tests). No behavior change.
2. Create `src/two_pass_classifier/`: promote two-pass runner, logprob
   extraction, schemas, prompts wiring from `evals/`; add concurrency,
   resume checkpoints, rate limiting, cost accounting for ~37k rows.
3. Smoke: 10-row run matches eval-path results on the same rows.
Config (model + effort B) injected from Stage 8 results. Verify: full test
suite green, smoke parity, V1 path still runs (classify_dead regression).

### WU-4b — V2 production runs (USER-RUN, paid, sync)
Alive evidence cohort + dead cohort through `src/two_pass_classifier` with
the chosen config. Dry-run cost gate first. Verify: output CSVs complete,
resume tested, cost within estimate.

### WU-5b — V2 production dashboard (worker, free)
WU-5a dashboard re-pointed at V2 outputs; confidence audit section becomes
logprob calibration. Verify: HTML with V2 numbers, alive-vs-dead section.

### WU-6 — V2 narrative dashboard (worker, free, parallelizable)
The V1 -> V2 story: why self-reported confidence was weak -> two-pass
design -> logprob calibration -> golden set + eval harness -> Stage 8
results -> production V2. Draft early against known material; inject final
numbers after WU-1/WU-4b. Verify: standalone professor-readable HTML.

---

## Risks

- Meeting is days away: the fallback path (WU-4a + WU-5a) is the safety
  net and runs first by design.
- Sync V2 forfeits the 50% batch discount (~2x model spend, accepted D4b).
- Sync at 37k-row scale needs disciplined rate limiting + resume; smoke
  test before full runs.
- Stage 8 gold labels are provisional drafts; present eval numbers as
  provisional.
- Dead cohort is 15,714 of 19,044 targets (~18% archive empties); the
  dashboard must state this.
- Fair comparison: alive vs dead must use the SAME classifier version per
  dashboard edition (V1-vs-V1 fallback, V2-vs-V2 final). Never mix.
- `src/` restructure (WU-3.1) must wait until WU-4a finishes so the V1
  batch run is never disturbed mid-flight.
