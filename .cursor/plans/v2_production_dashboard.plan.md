# V2 production dashboard (alive + dead)

## STATUS

- **State:** DRAFTING. Inherits WU-5a layout (`v1_alive_dead_dashboard.plan.md`).
  Spec work starts with section 5 (Confidence). Other sections stay "same as WU-5a
  unless this plan overrides them."
- **Roadmap unit:** WU-5b in `roadmap-to-july-deliverable.plan.md`.
- **Hard dependency:** V2 professor CSV from the production runner
  (`outputs/two_pass_classifier/runs/<run>/classifications.csv`), evidence-only
  alive + dead under one classifier version. Never mix V1 and V2 verdicts.
- **Locked so far (2026-07-28):** Confidence is three subsections, one per
  professor confidence field, each opening with a measurement explainer before
  any charts.

## Deliverable (unchanged intent)

- **Output:** presentation HTML under
  `data visualization/01_Presentation_Materials/` (exact filename TBD when
  implementation starts; working name `v2_alive_dead_cohort.html`).
- **Builder:** fork/adapt `build_v1_alive_dead_dashboard.py` against the V2
  18-column professor contract.
- **Universe:** evidence-only, corrected-only base sections (same locked
  decisions as WU-5a).

## Input contract (V2)

Primary: `outputs/two_pass_classifier/runs/<run>/classifications.csv`

Relevant confidence columns:

| Column | Type | How it is produced |
|---|---|---|
| `ai_native_confidence` | float in (0, 1], or blank if extraction failed | Pass A logprob extraction (`two_pass_classifier/confidence.py`), not model-written |
| `subclass_confidence` | integer 1..5 | Pass B self-report from the prompt confidence scale |
| `rad_confidence` | integer 1..5, or blank when `rad_score` is `RAD-NA` | Pass B self-report; omitted for not-AI-native family |

## Dashboard structure

Nav stays: Overview, Landscape, RAD, Cohorts, **Confidence**, Survivorship
(four acts). Only **Confidence** is redesigned below. Base sections and the
Survivorship acts remain the WU-5a plan until a later edit names changes.

---

## Section 5: Confidence (V2 redesign)

**Job of this section:** teach the reader how each of the three confidence
numbers is measured, then show the production distributions. This is a methods
plus audit section, not a golden-set calibration page (ECE / reliability bins
live on the eval suite dashboard).

**Nav / layout:** one top-level Confidence section with three sub-anchors:

1. AI-native confidence (`ai_native_confidence`)
2. Subclass confidence (`subclass_confidence`)
3. RAD confidence (`rad_confidence`)

Each subsection is one composition: short measurement explainer, then the
charts for that field. Do not dump all three chart banks before the explainers.

### 5.1 AI-native confidence (logprob)

**Measurement explainer (required, before charts).** Plain-English primer,
professor-readable, no API jargon wall. Cover:

1. Pass A only answers `ai_native` as `0` or `1`. The model does not write a
   confidence score for this field.
2. The API still returns **log-probabilities** on the sampled tokens: for each
   generated token, how likely the model thought that token was, plus the next
   few alternatives (`top_logprobs`).
3. We locate the decision digit token for `ai_native`, convert those
   log-probabilities into ordinary probabilities on `{0, 1}`, and take the
   probability mass on the digit the model actually sampled. That probability
   is `ai_native_confidence` (higher means the model was more committed to the
   bit it output).
4. Optional one-sentence honesty note: when the opposing digit is missing from
   the reported alternatives (censored case), production uses the validated
   midpoint bound from `confidence.py` rather than pretending the opponent had
   zero mass. Blank cells mean extraction was unavailable for that row.

Keep the explainer concrete (token probabilities on a 0/1 decision), not a
full information-theory digression. Point to the production module only in a
methods footnote if needed.

**Visualizations (after the explainer).** Adapt the audit charts to a
continuous `[0, 1]` field (histogram / density-friendly summary), not a 1-5
bar. Exact chart list can be tightened in a later plan pass; minimum bar:

- Distribution of `ai_native_confidence` over the evidence-only corrected frame
- Split or filter affordance consistent with the rest of the dashboard
  (e.g. All / Live / Dead), once the surrounding filter shell is decided

### 5.2 Subclass confidence (prompt 1-5)

**Measurement explainer (required, before charts).** State clearly that this
score is **model-written**, not extracted from token probabilities. Summarize
how Pass B was instructed, using the production scale from
`two_pass_classifier/prompts/subclass_rad_prompt.txt` (paraphrase for readers,
keep the anchors):

- **5:** Substantial, internally consistent evidence makes the decision
  unambiguous.
- **4:** Strong evidence supports the decision, with one plausible adjacent
  choice.
- **3:** Informative but incomplete evidence requires judgment between
  alternatives.
- **2:** Thin, generic, stale, or contradictory evidence supports only a weak
  choice.
- **1:** Evidence is insufficient and a stated family fallback rule is required.

Also note the prompt rule that critiques must explicitly explain any confidence
below 3. Schema bounds are integers 1 through 5 (not 0).

**Visualizations (after the explainer).** Discrete 1-5 distribution for
`subclass_confidence`, plus any by-subclass mean/median breakdown carried over
from the V1 confidence audit (re-pointed at this field).

### 5.3 RAD confidence (prompt 1-5)

**Measurement explainer (required, before charts).** Same self-report mechanism
and the same 1-5 scale as subclass confidence (Pass B prompt). Call out the
family difference:

- AI-native rows: model emits `rad_confidence` with `rad_score`.
- Not-AI-native rows: `rad_score` is `RAD-NA` and `rad_confidence` is blank.
  Charts must exclude blanks from the denominator and say so in the caption.

**Visualizations (after the explainer).** Discrete 1-5 distribution for
non-blank `rad_confidence`, analogous to the V1 `conf_rad` chart.

### Deliberate non-goals for section 5

- No golden-set ECE / reliability diagrams here (that is the eval suite).
- No mixing V1 `conf_classification` / `conf_rad` into these panels.
- No collapsing the three fields into one "overall confidence" score.

---

## Still inherited from WU-5a (until overridden)

- Overview, Landscape, RAD, Cohorts base sections (column renames only as needed
  for the V2 CSV).
- Survivorship Acts 1-4, including Act 4.1 alive-vs-dead confidence (will need a
  follow-up edit to say which of the three V2 fields that act uses).
- Evidence-only universe and corrected-only base sections.

## Open follow-ups (not locked this pass)

1. Exact chart inventory and KPI cards inside 5.1 / 5.2 / 5.3.
2. Whether Overview still surfaces a single confidence headline card, and if so
   which field it uses.
3. How Survivorship Act 4.1 compares alive vs dead once confidence is three
   fields (likely three small charts or a field selector).
4. Output HTML filename and builder module name.
