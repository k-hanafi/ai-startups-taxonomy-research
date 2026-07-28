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
- Survivorship Acts 1-4, subject to the consolidation pass below.
- Evidence-only universe and corrected-only base sections.

---

## Chart consolidation (redundancy cuts)

**Status: PROPOSED, awaiting confirmation.** Counted from the shipped
`build_v1_alive_dead_dashboard.py`, the inherited design renders **31 chart
boxes** (2 Landscape, 2 RAD, 2 Cohorts, 3 Confidence, 3 Act 1, 9 Act 2, 5 Act 3,
5 Act 4), not the ~18 the plan prose implies. Several state the same fact on a
different scale, which makes the page longer without making the argument
stronger. Each cut below names the chart it duplicates and what is kept instead.

Guiding rule: one fact, one chart, on the most interpretable scale. Prefer
**mortality rate** (dead / dead + survivor) over ratios of shares, because a
percentage of a named group is the scale a reader can restate correctly.

### Safe cuts (pure duplicates)

| Cut | Duplicates | Keep instead |
|---|---|---|
| **1.2 Composition before and after** (two 100 percent stacked bars, 10 segments each) | Same comparison as 1.3, which shows the difference directly | **1.3 Share shift per subclass** (diverging bar). Ten-segment stacked bars cannot be compared segment to segment by eye; the delta chart is the finding. |
| **2.2 Subclass lift** (dead share / survivor share) | Ratio of the two series already drawn in 2.1, and answers the same question as 2.2b on a harder scale | **2.1** (levels, with the significance stars) plus **2.2b** (mortality). Lift reads as "1.4x" with no denominator a reader can name. |
| **2.6 companion bar** (mortality percent per funding bucket) | Funding appears as the log-funding term in 3.1 and as heatmap rows in 3.3 | **2.6 main chart only** (bucket shares, survivor vs dead). |

### Merges (two charts, one story)

| Merge | Why |
|---|---|
| **2.2b** currently renders mortality by subclass AND mortality by defensibility group as separate charts. Render **one** ranked mortality-by-subclass bar chart, with bars colored by defensibility group and a legend. | The group chart is an aggregation of the same bars. Coloring shows both the ranking and the grouping in one read, and makes the "commoditizable genres die more" claim visible rather than asserted across two charts. |
| **4.2** currently renders two sensitivity bar charts (thin history, snapshot age). Render as **one compact table** of AI-native rate plus n per cut. | Robustness cuts are null results. A table saying the rate barely moves communicates "nothing hinges on this" faster than two bar charts that look like findings. |

### Cut on grounds of self-admitted weakness

- **2.4 Mortality by founding era.** The inherited plan already prints a caption
  explaining that raw era mortality mostly measures exposure time and that the
  regression handles it properly. A chart whose own caption argues against
  reading it costs attention and buys nothing. The era term stays in both forest
  plots (3.1, 3.2), which is where it is interpretable. Keep the fact as one
  sentence in the Act 2 intro or the methods box if wanted.

### V2-specific de-duplication: confidence

Three confidence fields multiply any alive-vs-dead confidence comparison by
three. Locked split of responsibility:

- **Section 5** owns the measurement explainers and the overall distributions for
  all three fields (see section 5 above).
- **Act 4.1** shows exactly **one** alive-vs-dead confidence comparison, on
  `ai_native_confidence`, because it is the measured (logprob) quantity and
  `ai_native` is the headline axis of the correction. The evidence-thinness
  caveat stays printed under it.
- Alive-vs-dead splits of `subclass_confidence` and `rad_confidence` are NOT
  charted. If the gap is worth noting, it goes in one sentence of the Act 4
  methods box.

This resolves former open follow-up 3.

### Flagged, not cut (decide explicitly)

- **Landscape evidence filter pill group** (`All / Live only / Dead only`). It
  lets a reader hand-build an untested version of Act 2.1 in a section that is
  supposed to be the corrected baseline, and it costs the 45-key precomputed
  filter grid. Options: drop the evidence dimension and keep the section purely
  corrected (simpler, one story per section), or keep it as an exploration
  affordance. Recommendation: drop it, since the flagship section owns the
  comparison with statistics attached.
- **3.4 Deaths over time.** Scientifically the weakest chart (death anchor is the
  last archive capture, coverage-bounded, marked exploratory), but it is the only
  time-axis chart on the page and carries the frontier-release narrative. Keep,
  with the exploratory caption.

### Additional redundancy found in the shipped build (not in the plan prose)

- **Act 3 Model 3 forest vs 3.3 dependency-trap heatmap.** Model 3 (the
  `log_funding x rad_score` interaction) shipped, so the same interaction is now
  drawn twice: once as odds ratios, once as a mortality heatmap. Recommendation:
  keep the heatmap (a reader can name what each cell means) and drop the Model 3
  forest, or keep Model 3 and demote the heatmap to a caption number. Do not ship
  both.
- **Act 4 "Strict dead definition" chart.** Present in the build but absent from
  the plan's Act 4 list. It belongs with 4.2 robustness; fold it into the same
  compact robustness table rather than giving it its own chart box.
- **Landscape "Subclass Distribution" vs Cohorts "Subclass Distribution by
  Cohort".** The second is the first split by cohort, so the totals are
  recoverable from it. Candidate for dropping the unsplit version, though the
  unsplit bar is the natural establishing shot for the page. Flagged, not cut.

### Net effect

31 chart boxes down to roughly 18 to 20, with no fact removed from the argument:
duplicates cut, pairs merged, self-disclaimed charts dropped, and the confidence
comparison run once on the measured field instead of three times.

## Open follow-ups (not locked this pass)

1. Exact chart inventory and KPI cards inside 5.1 / 5.2 / 5.3.
2. Whether Overview still surfaces a single confidence headline card, and if so
   which field it uses.
3. Confirm the consolidation cuts above, and decide the flagged Landscape filter.
4. Output HTML filename and builder module name.
