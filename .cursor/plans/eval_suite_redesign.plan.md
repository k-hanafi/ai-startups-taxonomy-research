# Classifier Eval Suite redesign

Product-grade redesign of the eval dashboard into a three-tab "Classifier
Eval Suite". Implemented on branch `eval/suite-redesign`; folds in the
cost-breakdown popover ladder from closed PR #32 (commit `1a40f82`,
cherry-picked as the base of this branch).

---

## STATUS

- **State (2026-07-23):** IMPLEMENTED on `eval/suite-redesign`. Metrics
  layer + fixture extended, HTML/CSS/JS rewritten, tests updated,
  headless-verified, PR open. Real Stage 8 runs plug in via
  `--runs` / `--scored` unchanged.
- Base: `origin/main` at d97d8bc + cherry-pick of 1a40f82 (cost popover).
- No paid API calls anywhere in this work; `OPENAI_API_KEY=placeholder`
  covers all tests and builds.

---

## Locked design contract (user-approved, do not deviate)

1. **Product name:** "Classifier Eval Suite" in the header. Purge every
   user-facing mention of LangSmith or any design-inspiration/meta-prompt
   references from HTML copy, titles, banners, and builder docstrings that
   leak into the page. The page must read like a polished product an
   enterprise eval vendor would ship.
2. **Layout:** tabbed app-like SPA feel (not a scrolly report): top app bar
   with product name, tab navigation, exactly THREE tabs in this order:
   - **Pipeline robustness**: answers "do logprob extraction and Batch API
     parity behave as intended, will this classifier survive production?"
     Content: tokenization pinned (decision-token extraction stable),
     probability mass accounted (valid_mass distribution/threshold),
     sync-batch parity verified (request-body identity, logprob shape).
     Render as a checks panel: each check gets a pass/fail/pending badge
     plus the supporting numbers and a one-line plain-English meaning.
   - **Model benchmarks**: answers "which GPT-family model should
     production use?" Compared on accuracy, confidence quality, cost,
     latency (latency labeled as a production-practice metric). Carry
     over: the experiments leaderboard table, model-group filter chips +
     search, Pareto chart, latency p50/p95 chart, and the cost-breakdown
     popover ladder (per-pass tokens x prices -> cache -> batch -> scale
     to production N) on every cost value.
   - **Confidence correctness correlation**: answers "how correlated is
     logprob confidence with actual correctness?" Content: reliability
     diagram (binned confidence vs accuracy), ECE per config,
     selective-prediction curve (coverage vs accuracy when abstaining
     below a confidence cutoff). The mock may seed the documented early
     signal (ECE 0.077, top-half confidence -> 100% accuracy) as its
     synthetic anchor.
   Anything beyond these three questions is out of scope BY DECISION:
   delete the confusion tab and any other leftover views.
3. **Data:** all SYNTHETIC until the Stage 8 sweep. Extend the mock fixture
   with whatever new blocks the three tabs need (robustness checks,
   reliability bins, selective curves), internally consistent with each
   config's numbers. Keep one honest, professionally worded synthetic-data
   notice (flat styling, not shouty). The real-data path must keep working:
   `--runs`/`--scored` populates benchmarks + confidence from scored.json
   fields where they exist, robustness renders "pending" for anything not
   yet recorded, nothing fabricated at render time.
4. **Visual language, de-slopped flat enterprise:** square corners
   (border-radius 0 or near-0), hairline 1px borders, NO shadows, NO
   rounded pill cards, dense data tables, monospace numerals, generous
   whitespace, muted single accent color, restrained type scale. Kill the
   current rounded-rectangle card look entirely.

---

## Per-tab content spec

### Tab 1: Pipeline robustness

Question: do logprob extraction and Batch API parity behave as intended,
will this classifier survive production?

Rendered as a checks panel of exactly three checks. Each check row shows a
PASS / FAIL / PENDING badge, a one-line plain-English meaning, and a
compact numbers table.

| Check id | Title | Real-data source | Pass condition |
|----------|-------|------------------|----------------|
| `tokenization_pinned` | Decision-token extraction stable | Per-run `calibration.n` vs `calibration.n_eligible` (recorded by scoring; extraction only yields a value when token-byte reconstruction and structural key location both succeed for a row) | Every run that carries calibration covers all eligible rows; PENDING when no run records calibration; FAIL when any run's coverage is incomplete |
| `probability_mass` | Probability mass accounted | Optional per-run `robustness.valid_mass` block (n, min, mean, p50, share_below_threshold, threshold). Not yet emitted by scoring, so real runs render PENDING until it is recorded | All recording runs show zero rows below the threshold; FAIL when any recording run has rows below it |
| `batch_parity` | Sync-batch parity verified | Optional per-run `robustness.batch_parity` block (verdict, n_rows, n_checks, n_failed, model), mirroring the gate-Q4 parity report. Not yet embedded in scored.json, so real runs render PENDING | All recorded verdicts are PASS; FAIL on any recorded FAIL |

Aggregation lives in `evals/dashboard_metrics.py::build_robustness` and is
exposed as `metrics["robustness"]["checks"]` (list of dicts with `id`,
`title`, `status` in {pass, fail, pending}, `meaning`, `stats` list of
label/value pairs, `per_model` rows). The renderer never invents numbers:
missing blocks surface as "pending" with a note naming the artifact that
will populate them.

### Tab 2: Model benchmarks

Question: which GPT-family model should production use?

- Toolbar: model-group chips (nano / mini / luna), per-config chips,
  search box, visible count, show all / clear. Filter state is global and
  also drives the confidence tab.
- Leaderboard table (dense, ranked by subclass accuracy): rank, config
  (label + model, effort, n scored, partial badge), subclass accuracy +
  inline bar, AI-native accuracy, RAD accuracy, mean confidence, projected
  production cost with the cost-breakdown popover (per-pass tokens x
  prices -> cache -> batch -> scale to production N) on every cost value,
  latency p50 (labeled as a production-practice metric, not a quality
  metric).
- Pareto chart: projected production $ (log x) vs subclass accuracy, CI
  whiskers, partial-run markers.
- Latency chart: p50/p95 grouped bars per config, production-practice
  framing in the caption.

### Tab 3: Confidence correctness correlation

Question: how correlated is logprob confidence with actual correctness?

- Reliability diagram: per-model binned mean confidence vs accuracy with a
  diagonal reference; one trace per visible model group (Pass A is banked
  once per model, so calibration is a model property, not an effort
  property). Marker size encodes bin count.
- ECE per config: bar chart, lower is better.
- Selective-prediction curve: coverage (fraction answered, most-confident
  first) vs accuracy, one trace per visible model group, with the
  full-coverage accuracy as the right edge.
- Data fields per config row: `reliability_bins`, `ece`,
  `selective_curve` (full coverage grid), `selective_acc_50`,
  `mean_confidence`, `share_above_90`.

## Fixture extension (synthetic, internally consistent)

Per model (nano / mini / luna), 100 synthetic per-row (confidence, correct)
pairs are generated once and all derived statistics are computed from the
same rows, so every displayed number recomputes from every other:

- `calibration.reliability.bins` (10 equal-width bins) and `.ece` computed
  from the rows;
- `calibration.selective_prediction` on the full 0.1..1.0 coverage grid
  computed from the rows; accuracy at coverage 1.0 equals the model's
  `axes.ai_native.accuracy`;
- `screen.mean_confidence`, `screen.share_above_90`,
  `screen.selective_acc_50` all recompute from the same rows;
- the three runs of one model share identical calibration blocks
  (bank-once Pass A design);
- nano anchors the documented early signal: ECE 0.077 and 100% accuracy on
  the top-confidence half.

Each run also carries a synthetic `robustness` block (tokenization /
valid_mass / batch_parity) so the mock panel shows all three checks green
with plausible supporting numbers.

## Out of scope by decision

- Confusion tab (deleted).
- Summary line chart and vs-baseline table from the old Experiments /
  Charts layout (not in the locked carry-over list; the `vs_baseline`
  metrics field remains in the data layer for future use).
- Any new scoring-side instrumentation (emitting `robustness` blocks from
  real runs is future work; the dashboard renders PENDING until then).

## Verification checklist

- `OPENAI_API_KEY=placeholder python3 -m pytest evals/tests -q` green.
- `python3 -m evals dashboard` rebuilds `eval_dashboard.html` from the mock.
- Headless browser: all three tabs render, chips + search filter, cost
  popover opens/closes, zero console errors; viewport screenshot per tab.
- No "LangSmith" (or other design-inspiration references) anywhere in the
  builder or the generated page.
