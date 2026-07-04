---
name: Logprob confidence classifier
overview: "Replace the model's self-reported 1-5 \"confidence\" (which really measured input-signal richness) with two honest, separately-named signals: a reframed verbalized `evidence_sufficiency` and a new token-logprob-derived confidence block. Config: reasoning=medium + temperature=0 + top_logprobs=15. Schema audited and locked 2026-07-04. GATE: before any pipeline changes, build a 100-company golden-dataset evaluation harness (fresh chat, own branch) that validates the schema, extraction, and research approach end-to-end."
todos:
  - id: eval-harness
    content: "GATE (fresh chat, own branch): build the 100-company golden-dataset eval harness. Must resolve: reasoning none-vs-medium A/B (does deliberation collapse logprob spread?), real tokenization of subclass/rad values, pre- vs post-mask logprobs, batch include+top_logprobs support, end-to-end extraction correctness, run-level QA report. Only after this passes does the pipeline work below start."
    status: in_progress
  - id: confidence-schema
    content: "Finalize the model-side confidence field schema (per-axis ranking/dist, top1, margin, entropy, perplexity, diagnostics) with the user before any implementation."
    status: completed
  - id: sanity-smoke
    content: "Superseded by the eval harness (absorbed: sanity call, fixture capture, none-vs-medium A/B)."
    status: cancelled
  - id: config
    content: Add logprob/decoding constants to src/config.py (TOP_LOGPROBS=15, REASONING_EFFORT='medium', DECODING_TEMPERATURE=0, LOGPROB_INCLUDE) and fix stale MODEL_PRICING in src/tokens.py (mini 0.75/4.50, gpt-5.4 output 15).
    status: pending
  - id: builder
    content: Extend build_request_body() in src/builder.py to add include, top_logprobs=15, reasoning=medium, temperature=0 while keeping the cacheable instructions+schema prefix byte-stable.
    status: pending
  - id: schema-prompt
    content: Rename conf_classification->evidence_sufficiency and conf_rad->evidence_sufficiency_rad in src/schema.py (+validators); reword prompts/system_classifier_prompt.txt field name and all OUTPUT examples to measure input-evidence sufficiency.
    status: pending
  - id: logprobs-module
    content: "Create src/logprobs.py: reconstruct token char spans, validate bytes==JSON values, locate ai_native/subclass/rad decision tokens, renormalize over valid enum candidates, compute top1_prob/margin/entropy + headline logprob_confidence."
    status: pending
  - id: downloader
    content: Update src/downloader.py to capture the logprobs array from the response body, call src/logprobs.py, and widen the output CSV with logprob_confidence + per-axis columns + logprob_extraction_ok.
    status: pending
  - id: classify-test
    content: Update classify.py _cmd_test to send the logprob params and print a rich per-axis logprob distribution table for one company.
    status: pending
  - id: smoke-fixtures
    content: Extend scripts/smoke_test_logprobs.py with temperature=0 and a --save-fixture flag that writes an anonymized raw response for golden tests.
    status: pending
  - id: tests
    content: Add tests/test_logprobs.py with 1-2 committed tests/fixtures/*.json (captured under the production config) asserting extraction correctness, byte-match validation, and metric values.
    status: pending
  - id: downstream
    content: "Update renamed-column consumers: merge_survivorship.py and the survivorship/classification dashboard scripts to use evidence_sufficiency (and optionally surface logprob_confidence)."
    status: pending
  - id: agents-md
    content: Update AGENTS.md domain model, data flow, and config invariants for the renamed fields + logprob columns + reasoning-none/temperature-0 decoding.
    status: pending
  - id: rollout
    content: "Paid/manual outside sandbox: smoke-verify mini+logprobs, capture fixtures, validate a 20-row batch, dry-run cost, then run both strands (live + wayback_dead) and regenerate dashboards."
    status: pending
isProject: false
---

# Logprob-based confidence for the classifier

## GATE (decided 2026-07-04): 100-company golden eval harness first

Too many unvalidated assumptions remain to touch the production pipeline safely. Before ANY implementation below, a **100-company golden-dataset evaluation harness** is built in a fresh chat on its own branch, testing schema + extraction + research approach end-to-end. It must answer:

1. **Reasoning none-vs-medium A/B** — does deliberation collapse the logprob spread (top1 ~0.99 everywhere) and gut margin/entropy? This is the only open question that could invalidate the methodology.
2. **Real tokenization** of `subclass` (`"1A"` single vs `1`+`A` split) and `rad_score` suffix — pins the extractor.
3. **Pre- vs post-mask logprobs** — decides whether `valid_mass` has epistemic meaning or is extraction-diagnostic only.
4. **Batch API** accepts `include` + `top_logprobs` and returns the sync-shaped logprob arrays.
5. **Extraction correctness** — byte-reconstruction, span alignment, golden fixtures, emitted-vs-argmax agreement (~100% expected at temp 0).
6. **Truncation risk** — measure real reasoning-token usage to size `MAX_OUTPUT_TOKENS` (current 1000 will truncate under reasoning=medium; reasoning tokens count against the cap) and the cost model.

The rest of this plan is the post-gate implementation reference.

## Locked config (user-confirmed by research, 2026-07-03)

User research confirmed `temperature=0` does NOT require reasoning off, and logprobs are exposed with reasoning on for the 5.4 family.

- Model: **`gpt-5.4-mini`** default (better judgment/calibration than nano). One-line change in [src/config.py](src/config.py); revisit freely.
- **`reasoning = {"effort": "medium"}`** — keep deliberation for the nuanced 10-subclass taxonomy.
- **`temperature = 0`** — greedy: the emitted class IS the argmax of the base distribution, so label and confidence agree by construction.
- **`top_logprobs = 15`**, `include = ["message.output_text.logprobs"]`.
- Stored label = argmax over valid enum candidates from the logprob table (belt-and-suspenders: holds the "highest logprob = chosen class" invariant even if a rare non-greedy quirk appears).
- Scope: **re-run both strands** (live 44.4k + dead ~19k) so every row has the new fields and the survivorship comparison stays apples-to-apples.

Facts we rely on:

- **Logprobs are temperature-independent.** OpenAI returns `log(softmax(raw_logits))`, computed *before* temperature scaling — the confidence signal is the honest base distribution, never a degenerate temp-0 one-hot.
- **`logprob_confidence` is a model-internal probability, not calibrated P(correct).** Without a gold set (out of scope for v1) we use it for *relative* ranking and ambiguity flags (margin/entropy), and we say so plainly in the paper.

## Model-side confidence field schema (LOCKED 2026-07-04, post-audit)

Per-axis candidate sets: `ai_native` = {0,1}, `subclass` = 10 values, `rad` = {H,M,L} (NA excluded; all `rad_*` fields null when ai_native=0, and `rad_valid_mass` incidentally records NA leakage).

Per axis (`ainat_`, `subclass_`, `rad_`):
- `{axis}_ranking` — JSON string, ordered `[{label, prob}, ...]` renormalized over valid candidates. THE primitive; everything else derivable.
- `{axis}_top1_prob` — probability of chosen class (headline scalar).
- `{axis}_margin` — p1 − p2 (interpret only when top1 < ~0.9).
- `{axis}_entropy` — normalized entropy 0–1 (per-axis N; never pool raw across axes).
- `{axis}_perplexity` — exp(raw entropy) = effective class count (prose/figures; never alongside entropy in a regression).
- `{axis}_top2_label` — runner-up class (confusion-graph analysis; filter to ambiguous rows first).
- `{axis}_valid_mass` — un-renormalized mass on valid candidates. **Extraction diagnostic only** (returned logprobs appear post-grammar-mask, so it cannot detect "model wanted to refuse").

Row-level: `logprob_confidence` (headline), `logprob_extraction_ok` (False -> all confidence fields nulled), `reasoning_tokens` (from usage; **endogenous** — descriptive covariate only, never a causal regressor). `output_perplexity` was CUT (confounded by prose length/wording; recomputable from raw JSONLs).

**Critical definitional rule (autoregressive conditioning):** the output emits `ai_native` before `subclass`, so the raw subclass-position distribution is conditional on the already-committed branch — cross-branch probs are ~0 by construction, not by evidence. Therefore the subclass distribution is the **composed joint**: `P(1D) = P(ai_native=1 at its own token) x P(D | branch 1 at the subclass letter)`. Opposite-branch entries appear as an honest aggregate (e.g. `"0*"`) unless single-token subclass values make the full 10-way directly observable. `logprob_confidence` = this composed joint top-1. Must be documented in the paper's methods.

## Confidence field redesign

Split the one overloaded signal into two orthogonal, honestly-named ones:

- **Verbalized, input-side** (model still emits in JSON): rename `conf_classification` -> `evidence_sufficiency`, `conf_rad` -> `evidence_sufficiency_rad`, and reword the prompt to ask "how much usable signal did the input contain" rather than "how confident are you."
- **Model-side, derived by us** (NOT in the request schema; computed post-hoc and appended as CSV columns): `logprob_confidence` (headline = joint top-1 prob of the chosen `subclass`), plus per-axis `*_top1_prob`, `*_margin`, `*_entropy` for `ai_native` / `subclass` / `rad`, an optional `logprob_dist_json`, and `logprob_extraction_ok`.

Keeping derived metrics out of `ClassificationResult` is deliberate: the request schema must stay byte-stable for prompt caching, and the model shouldn't be asked to emit numbers we can compute exactly from its own logits.

## Extraction design (`src/logprobs.py`, new)

The output (temp 0; reasoning tokens are hidden and precede the message) is a single strict-JSON object streamed as `output_text` tokens, each carrying `{token, bytes, logprob, top_logprobs[]}`. Locate field values by structural JSON parse with char offsets — never string search (`"1D"` can also appear inside `reasons_3_points`). Merge the emitted `entry.{token,logprob}` into the candidate pool before renormalizing (chosen token may be absent from its own top-k list).

Algorithm:
1. Concatenate token `bytes` while tracking each token's char span; assert the reconstruction equals the response text and that the parsed JSON round-trips (**fail loudly on mismatch** -> `logprob_extraction_ok=False`).
2. Locate the value span for each axis by JSON position:
   - `ai_native`: bare `0`/`1` token -> renormalize top_logprobs over `{"0","1"}`.
   - `rad_score`: shared prefix `RAD-`; read the *suffix* token (`H`/`M`/`L`/`NA`) -> renormalize over the 4 RAD values.
   - `subclass`: read the digit token (1 vs 0) and the letter token; `subclass` prob = P(digit) x P(letter | digit). This handles both single-token (`"1A"`) and split (`"1`,`A"`) tokenizations; the fixture pins the exact case.
3. Per axis compute `top1_prob`, `margin = p1 - p2`, `entropy = -sum p log p / ln(N)` over valid candidates only.

Golden tests are non-negotiable here: a wrong token index yields silently-wrong confidence.

## Files to change

- [src/config.py](src/config.py): add `TOP_LOGPROBS=15`, `REASONING_EFFORT="medium"`, `DECODING_TEMPERATURE=0`, `LOGPROB_INCLUDE`; raise `MAX_OUTPUT_TOKENS` (~8000 — reasoning tokens count against the cap; size from harness data); fix `MODEL_PRICING` (mini $0.75/$4.50, gpt-5.4 output $15) and add reasoning-token cost estimation in [src/tokens.py](src/tokens.py).
- [src/builder.py](src/builder.py) `build_request_body()`: add `include`, `top_logprobs`, `reasoning`, `temperature` (prefix stays byte-stable -> caching intact).
- [src/schema.py](src/schema.py): rename the two conf fields + validators; update [prompts/system_classifier_prompt.txt](prompts/system_classifier_prompt.txt) field name + wording in all OUTPUT examples.
- `src/logprobs.py` (new): extraction + metrics above.
- [src/downloader.py](src/downloader.py): capture the logprobs array (extend `_assistant_json_from_batch_body`), call `src/logprobs.py`, and widen `_append_to_output_csv` fieldnames to `ClassificationResult` fields + logprob columns.
- [classify.py](classify.py) `_cmd_test`: send the same logprob params and print a rich per-axis distribution table.
- [scripts/smoke_test_logprobs.py](scripts/smoke_test_logprobs.py): add `--save-fixture` (anonymized raw response) + `temperature=0`.
- `tests/test_logprobs.py` + `tests/fixtures/*.json` (new).
- Downstream renamed-column consumers: [wayback_machine/scripts/merge_survivorship.py](wayback_machine/scripts/merge_survivorship.py), [data visualization/02_Analysis_Code/survivorship_analysis.py](data visualization/02_Analysis_Code/survivorship_analysis.py), `build_survivorship_insights_dashboard.py`, `build_classification_dashboard.py`.
- [AGENTS.md](AGENTS.md): domain model (renamed fields + logprob columns), data-flow note, config invariants.

## Rollout (paid stages run OUTSIDE the sandbox, `keys/openai.env`)

0. **Eval harness gate passes** (see top section; fresh chat, own branch).
1. Land pipeline code + offline tests (no API): `pytest` green on golden fixtures captured by the harness.
2. Verify batch logprob shape on a tiny `prepare`+`submit` (e.g. `--rows 0:20`) if the harness didn't already cover Batch API.
3. Dry-run cost (now including reasoning tokens), then `run` both strands (live via default input; dead via `classify_dead.py run`, `CLASSIFY_NS=wayback_dead`).
4. Post-run QA report: % extraction-ok, valid_mass distribution, top1 histograms per axis, emitted-vs-argmax agreement rate.
5. Update dashboards/merge, regenerate.

## Out of scope (v1)

Gold-set calibration (ECE / temperature-scaling / Platt), self-consistency sampling, and any claim that `logprob_confidence` equals P(correct).

## Open validation gates

All absorbed into the eval-harness gate (top section). Schema is confirmed; implementation starts only after the harness passes.