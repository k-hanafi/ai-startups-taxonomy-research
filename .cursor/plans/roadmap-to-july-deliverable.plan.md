# Roadmap to July Deliverable

Master plan / PRD for the professor meeting (late July 2026). Owned by the
orchestrator chat; workers execute scoped units and report back. Update the
STATUS block after every milestone, decision, or pivot.

---

## STATUS

- **State:** Plan created 2026-07-22 from four repo audits (eval harness,
  survivorship pipeline, V2 migration scope, dashboard inventory).
- **Current branch:** `cursor/eval-dashboard-langsmith-ux-0263` (draft PR #29,
  CONFLICTING with main, 19 commits behind).
- **Key discovery:** AGENTS.md is stale. The paid Stage C dead-cohort extract
  is ALREADY DONE on disk: 19,044 targets covered, 15,714 with usable evidence
  (`wayback_machine/outputs/processed/scrape_processed_dead.csv`). Remaining
  survivorship stages: D (free) -> E (paid classify) -> F (free merge).
- **Decisions pending (user):** D1 model matrix, D2 budgets, D3 V2 scope for
  the alive cohort, D4 fallback if V2 batch migration slips. See "Open
  decisions".
- **In-flight workers:** none.
- **Next steps:** resolve D1-D4, then kick off WU-1 (Stage 8 sweep prep) and
  WU-3 (V2 batch migration scaffold) in parallel.

---

## The three deliverables

| # | Deliverable | What "done" looks like |
|---|------------|------------------------|
| 1 | **V2 progress presentation dashboard** | A narrative HTML dashboard telling the V1 -> V2 story: single-pass + self-reported 1-5 confidence -> two-pass split-reasoning classifier + logprob-calibrated confidence + golden-set eval harness. Presentable to a professor in one sitting. |
| 2 | **Production eval dashboard instance** | The existing LangSmith-style eval dashboard rendering REAL Stage 8 sweep results (not the mock fixture) over the 100-row golden set, ready to discuss config choice. |
| 3 | **Unified production dashboard: alive + dead** | `full_baseline_cohort.html`-class dashboard over V2 classifications of ~20k alive (evidence rows) + ~15.7k dead companies, with a new alive-vs-dead section isolating survivorship bias. Requires migrating the `src/` batch classifier to the V2 two-pass design first. |

---

## Verified current state (2026-07-22)

### Eval harness (`evals/`) — feeds deliverables 1 + 2
- DONE: stages 0-7 + cost-extrapolation pivot merged; Stage 9 mock dashboard
  built. Batch parity validated for Pass A (logprobs survive Batch API).
- Golden set: `evals/golden/golden_set.csv`, 100 stratified rows, all with
  `draft_*` provisional labels (human `gold_*` review waived).
- NOT DONE: Stage 8 paid sweep (full 100-row two-pass runs per config).
  Banked so far: 3 single-pass nano reference runs + 2-row two-pass smokes.
- Config drift to fix before sweep: `evals/config.py` `EVAL_MODELS` lists
  nano/mini/gpt-5.4/gpt-5.5 but the locked dashboard matrix is
  nano/mini/**gpt-5.6-luna** x Pass B low/medium/high; luna missing from
  models list and pricing table.
- Run commands per cell (paid, sync, outside sandbox):
  `python -m evals run-two-pass --model M --effort-b E` then
  `python -m evals score <run_id> --confidence-from-raw` then
  `python -m evals dashboard --runs <ids...>`. Dry-run flag exists.

### Survivorship pipeline (`wayback_machine/`) — feeds deliverable 3
- DONE: cohort (22,002) -> death probe (19,044 ok) -> targets (19,044) ->
  **paid extract complete** (15,714 with evidence; ~3.2k permanent archive
  empties; 0 retryable failures left).
- REMAINING:
  1. `python3 wayback_machine/scripts/build_classifier_input_dead.py` (free)
  2. `python3 wayback_machine/scripts/classify_dead.py prepare --dry-run`
     then `classify_dead.py run` (PAID; isolated under `CLASSIFY_NS=wayback_dead`)
  3. `python3 wayback_machine/scripts/merge_survivorship.py` (free) ->
     `outputs/wayback_dead/survivorship_corrected.csv`
- Note: classify_dead currently runs the V1 single-pass classifier. Whether
  it should wait for V2 is decision D3/D4.

### V2 production migration (`src/` + `classify.py`) — blocker for deliverable 3
- V2 exists only in `evals/` and only via the synchronous Responses API.
  Production `src/` is still single-pass Batch.
- Architecture to port: Pass A = binary gate, effort none, top_logprobs=15,
  `prompts/binary_gate_prompt.txt`; Pass B = family-constrained subclass+RAD,
  configurable effort, `prompts/subclass_rad_prompt.txt` + family blocks.
  Cohort computed in code. Confidence from logprob extraction
  (`evals/logprob_extract.py`).
- Hardest piece: two-stage Batch orchestration (Pass A batch -> download ->
  join by custom_id -> build family-split Pass B batches -> download ->
  assemble CSV), with resume that never re-pays a completed Pass A.
- Validated: Pass A batch parity (byte-identical bodies, logprobs intact).
  NOT validated: Pass B on Batch, end-to-end A->B join at scale.
- Final model/effort config comes from Stage 8 results, but scaffolding can
  start now with config as a parameter.

### Dashboards (`data visualization/`) — deliverables 1 + 3
- `build_classification_dashboard.py` (full_baseline_cohort.html): 5 sections
  (Overview, Landscape, RAD, Cohort Dynamics, Confidence Audit); reads the
  legacy V1 CSV. Template for deliverable 3, needs new input + new section.
- `survivorship_analysis.py` + `build_survivorship_insights_dashboard.py`:
  alive-vs-dead comparisons + 2 logistic regressions already built; PREVIEW
  mode until `survivorship_corrected.csv` exists. Reuse for the new
  alive-vs-dead section.
- Eval dashboard: mock by default; real data via `--runs`/`--scored`.
- Gap: no V2 narrative/methodology dashboard exists at all (deliverable 1).

---

## Dependency graph

```
D1+D2 decisions
   |
   v
WU-1 Stage 8 sweep (paid) ----> WU-2 real eval dashboard  [deliverable 2]
   |                                   |
   | (config pick)                     v
   v                            WU-6 V2 narrative dashboard [deliverable 1]
WU-3 V2 batch migration (can scaffold in parallel, config injected late)
   |
   v
WU-4 production V2 runs (paid): alive evidence cohort + dead cohort
   |
   v
WU-5 merge_survivorship + unified alive-vs-dead dashboard [deliverable 3]

Housekeeping (parallel, cheap): WU-0 rebase/land PR #29; sync AGENTS.md.
```

Critical path: D1 -> WU-1 -> config pick -> WU-4 -> WU-5. WU-3 scaffolding
and WU-6 drafting are parallelizable and should start immediately.

---

## Work units (kickoff-ready)

### WU-0 — Housekeeping (free, ~1h agent time)
Rebase `cursor/eval-dashboard-langsmith-ux-0263` onto main, resolve
conflicts, land PR #29 (the real eval dashboard build depends on this UX).
Update AGENTS.md: survivorship status (extract DONE), this plan's existence.
- Owner: implementer worker. Verify: PR mergeable, `pytest` green.

### WU-1 — Stage 8 eval sweep prep + execution (PAID)
1. Fix `evals/config.py`: add `gpt-5.6-luna` to `EVAL_MODELS` + pricing;
   align matrix with D1 decision.
2. Dry-run every cell; report total estimated input cost for approval.
3. Execute sweep (user runs, or worker runs outside sandbox with keys):
   N configs x (run-two-pass -> score --confidence-from-raw -> report).
- Owner: implementer for (1), user/worker for (3). Verify: scored.json per
  cell, accuracy/cost/latency populated.

### WU-2 — Real eval dashboard instance (deliverable 2, free)
`python -m evals dashboard --runs <stage-8 run ids>`; confirm SYNTHETIC
banner gone, all configs render, Pareto/confidence/latency tabs correct.
Snapshot HTML into `01_Presentation_Materials/`.
- Depends: WU-1. Verify: dashboard opens with real runs only.

### WU-3 — V2 two-pass Batch migration (biggest engineering item)
Port the two-pass design from `evals/` into the production Batch pipeline:
- `src/schema.py`: BinaryResult + family-constrained Pass B schemas; keep
  final CSV contract + new logprob-confidence columns + boundary_disagreement.
- `src/builder.py`: pass_a/pass_b request builders (cache-stable prefixes,
  logprobs only on A, family-split B prefixes).
- `classify.py` + `src/state.py`: two-stage state machine
  (A prepare/submit/download -> join -> B prepare/submit/download ->
  assemble); resume must never re-pay completed A batches.
- Promote `evals/logprob_extract.py` -> `src/logprobs.py`.
- Tokens/dry-run cost for 2x calls; keep `CLASSIFY_NS` isolation.
- Smoke test: tiny end-to-end two-pass Batch run (~10 rows) before scale.
Model/effort left as config; injected after WU-1.
- Owner: dedicated implementer worker (large unit; own branch/PR). Verify:
  smoke run produces assembled CSV matching eval-path results on same rows.

### WU-4 — Production V2 classification runs (PAID)
1. `build_classifier_input_dead.py` (free) -> classifier_input_dead.csv.
2. Dry-run costs for both cohorts with chosen config; get approval.
3. Run V2 batch classify: alive evidence cohort (classifier_input.csv) under
   default NS; dead cohort under `CLASSIFY_NS=wayback_dead`.
- Depends: WU-3 + config from WU-1. Verify: output CSVs complete, custom_id
  match counts, cost within estimate.

### WU-5 — Merge + unified alive-vs-dead dashboard (deliverable 3)
1. `merge_survivorship.py` -> survivorship_corrected.csv (adapt to V2
   columns if needed).
2. New builder (fork of `build_classification_dashboard.py`): V2 production
   input, existing 5 sections adapted (confidence audit becomes logprob
   calibration), plus new "Survivorship bias isolated" section reusing
   `survivorship_analysis.py` cohort comparisons + logistic forests.
- Depends: WU-4. Verify: HTML renders alive+dead KPIs, distribution deltas,
  corrected vs biased AI-native rate.

### WU-6 — V2 progress narrative dashboard (deliverable 1)
New presentation HTML: the story of V1 -> V2. Sections: why V1 confidence
was weak (post-hoc self-report) -> two-pass design (gate + family-constrained
subclass) -> logprob confidence/calibration -> golden set + eval harness ->
Stage 8 results summary -> what production V2 changes. Reuse house STYLE.
Content can be drafted NOW against mock/known material; final numbers
injected after WU-1.
- Owner: implementer worker, parallel from day 1. Verify: professor-readable
  standalone HTML in `01_Presentation_Materials/`.

---

## Open decisions (user)

- **D1 — Stage 8 matrix:** confirm nano / mini / gpt-5.6-luna x Pass B
  low/medium/high (9 cells) as locked in the dashboard, or trim. Requires
  adding luna pricing to config.
- **D2 — Budgets:** approve (a) Stage 8 sweep spend (dry-run numbers first),
  (b) production batch spend for alive + dead V2 runs (dry-run first).
- **D3 — Alive cohort scope for V2:** re-classify the full 44,387 live rows,
  or only the ~22k with real website evidence (the "~20k alive" framing)?
  Recommendation: evidence rows only for the meeting; the no-evidence
  remainder is exactly what the dead cohort replaces.
- **D4 — Fallback if WU-3 slips:** the meeting is days away and WU-3 is the
  riskiest unit. Fallback: run the dead cohort through the EXISTING V1
  classifier (classify_dead.py works today), build deliverable 3's
  alive-vs-dead comparison on V1-vs-V1 (methodologically consistent), and
  present V2 as "validated in eval, production migration in progress".
  Decide now whether fallback is acceptable so we can trigger it early.

## Risks

- WU-3 (two-stage Batch DAG) is high complexity; resume/double-billing bugs
  are the main hazard. Mitigate with the 10-row smoke and Pass-A-bank reuse.
- Stage 8 gold labels are provisional drafts (human review waived); frame
  eval numbers as provisional to the professor.
- Only 15,714 of 19,044 dead targets have evidence (~18% archive empties);
  the comparison cohort is ~15.7k, not 20k. State this in the dashboard.
- Fair-comparison invariant: alive and dead MUST be classified by the same
  classifier version for the survivorship section to be publishable.
