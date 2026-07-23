# V1 alive-vs-dead presentation dashboard (survivorship flagship)

## STATUS

- **State:** APPROVED and IN IMPLEMENTATION on branch `dashboards/v1-alive-dead`.
- **Locked user decisions (2026-07-23):**
  1. **Evidence-only universe, entire dashboard.** Every section, base sections
     included, covers ONLY companies with real website evidence. Alive side =
     companies Tavily successfully scraped (non-empty `website_evidence` in
     `outputs/tavilycrawl/processed/classifier_input.csv`, roughly 22k). Dead
     side = the ~15.7k with recovered archive evidence
     (`evidence_source == "wayback_dead"` after the merge). NO metadata-only
     classifications anywhere, neither alive nor dead. Act 1's biased vs
     corrected headline is computed on this universe: biased = evidence-based
     alive only; corrected = evidence-based alive + evidence-based dead.
     (This resolves former open question 1: stricter than any prior option.)
  2. **Base sections are corrected-only.** No biased/corrected toggle.
  3. **`survivorship_insights.html` is retired.** Its builder
     `build_survivorship_insights_dashboard.py` is deleted in this branch;
     `survivorship_analysis.py` stays (its compute lives on inside the new
     dashboard). AGENTS.md references updated. Nothing else imports the
     deleted builder (verified: only docs/plans mention it).
- **PREVIEW nuance under decision 1:** before the merge lands there are zero
  `wayback_dead` rows, so PREVIEW mode uses the full dead work list
  (metadata-only labels) as the stand-in dead cohort, behind a loud banner.
  The moment `survivorship_corrected.csv` exists, re-running the builder
  switches every dead-side number to the evidence-based ~15.7k automatically.
- **Hard dependency:** `outputs/wayback_dead/survivorship_corrected.csv` does not exist yet.
  The classify-dead batch recovery is still running (`outputs/wayback_dead/batch_data/state.json`
  present, no `wayback_dead_classifications.csv` yet), so `merge_survivorship.py` has not run.
  The builder must therefore support the same PREVIEW fallback `survivorship_analysis.py`
  already implements (production CSV + `scrape_targets_dead.csv` tagging), and print a
  PREVIEW banner until the corrected CSV lands.
- **Next steps:** (1) user answers the open questions at the bottom, (2) implement work
  items in order, (3) once the dead classifications land, run `merge_survivorship.py`
  then rebuild; numbers refresh with zero code changes.

## Deliverable

- **Output:** `data visualization/01_Presentation_Materials/v1_alive_dead_cohort.html`
  (single-file HTML, Plotly via CDN, no build step, opens from disk).
- **Builder:** `data visualization/02_Analysis_Code/build_v1_alive_dead_dashboard.py`.
- **Compute:** extend `data visualization/02_Analysis_Code/survivorship_analysis.py`
  (the existing pure-metrics module) with a small number of additive helpers. No new
  compute module.

## Architecture decision

Keep the compute/render split the repo already uses (`survivorship_analysis.py` returns
one JSON-able metrics dict; a builder renders it). Rationale: the split is already proven
by `build_survivorship_insights_dashboard.py`, it keeps statistics testable without
HTML in the way, and the flagship section needs roughly 80 percent of metrics that
`survivorship_analysis.py` computes today (`_correction`, `_ai_vs_survival`,
`_rad_survival`, `_dependency_trap`, `_vertical`, `_temporal`, `_confidence`, `_flips`,
`_regression`).

Concretely:

1. **Extend `survivorship_analysis.py`** with four additive helpers (details in the
   statistics section): per-subclass proportion tests, the coverage funnel, the
   thin-history/snapshot-age sensitivity cut, and funding-bucket distributions by
   survival status. All return plain dicts merged into `compute_metrics()` output.
   Existing keys and the PREVIEW behavior are untouched, so
   `build_survivorship_insights_dashboard.py` keeps working.
2. **New builder forks the HTML shell** of `build_classification_dashboard.py`
   (nav, STYLE block, section/chart-box/insight markup, IntersectionObserver wiring)
   rather than importing it: the shell there is one f-string, not a reusable function,
   and this is a one-off presentation artifact. Copy-and-adapt is simpler and safer
   than refactoring a working dashboard into shared templates for a single consumer.
3. **Reuse `compute_metrics()` from `build_classification_dashboard.py` by file-path
   import** (same `importlib` pattern the insights builder uses) for the adapted base
   sections: it is already a pure function of a DataFrame with the exact columns
   `survivorship_corrected.csv` carries, so pointing it at the corrected frame gives
   the Overview/Landscape/RAD/Cohort/Confidence numbers for free.
4. **Do not modify or retire** `build_classification_dashboard.py` or
   `build_survivorship_insights_dashboard.py` in this change. Whether the insights
   dashboard is retired afterward is an open question below.

What is deliberately NOT built: no framework, no shared template module, no config
system, no CLI beyond input-path overrides mirroring the insights builder.

## Input-data contract

All reads are `dtype=str, keep_default_na=False` (house pattern in
`survivorship_analysis._read`), converted where needed.

### 1. `outputs/wayback_dead/survivorship_corrected.csv` (primary; DEPENDENCY)

Written by `wayback_machine/scripts/merge_survivorship.py`. Exact expected schema
(production columns plus three merge columns):

| Column | Type | Use |
|---|---|---|
| `CompanyID` | str (org uuid) | join key everywhere |
| `CompanyName` | str | hover labels only |
| `ai_native` | "0"/"1" | both axes |
| `subclass` | 1A..1G, 0A..0C | composition, mortality |
| `rad_score` | RAD-H/M/L/NA | dependency analysis |
| `cohort` | PRE-GENAI / GENAI-ERA | era cuts (classifier output, see caveats) |
| `conf_classification` | 1..5 | confidence audit |
| `conf_rad` | 1..5 or empty | confidence audit |
| `reasons_3_points`, `sources_used`, `verification_critique` | str | not charted |
| `evidence_source` | live / wayback_dead / dead_metadata | THE cohort definer |
| `snapshot_ts` | YYYYMMDDHHMMSS or empty | snapshot-age sensitivity |
| `thin_history` | "True"/"False"/empty | robustness cut |

Fallback while absent: `outputs/production_csvs/production_classifications.csv`
(same schema minus the last three columns) with `evidence_source` synthesized from
`scrape_targets_dead.csv` membership, exactly as `survivorship_analysis.load_frame`
does today (PREVIEW mode).

### 2. `data/master_csv.csv` (covariates, joined on `org_uuid` = `CompanyID`)

Verified header: `org_uuid, name, homepage_url, short_description, Long description,
category_list, category_groups_list, founded_date, employee_count, total_funding_usd,
website_alive`. Used: `total_funding_usd` (funding buckets, log funding),
`category_groups_list` (primary vertical), `founded_date` (founded-year check),
`website_alive` (strict-dead definition). `employee_count` is available but excluded
from the main design (stale and coarse for dead firms; see cut list).

### 3. `wayback_machine/data/scrape_targets_dead.csv` (dead-cohort provenance)

Verified header: `org_uuid, name, homepage_url, founded_date, closest_ts, snapshot_url,
select_paths, website_alive, thin_history, death_ts, days_before_death`. Used:
`death_ts` (deaths-over-time), `days_before_death` (snapshot-age sensitivity),
`thin_history`, plus membership itself (defines the 19,044 with-snapshot set).

### 4. `wayback_machine/data/not_found_cohort.csv` (funnel top only)

Only its row count (~22,002) feeds the coverage funnel. If reading it is awkward,
hardcode the probe-verified count with a comment pointing at the source; it is frozen.

### 5. `outputs/tavilycrawl/processed/classifier_input.csv`

Only `org_uuid` + `website_evidence` non-emptiness, to define true survivors
(`has_live_evidence`), as `survivorship_analysis` already does.

### Cohort definitions (evidence-only universe, per locked decision 1)

- **survivor:** `evidence_source == "live"` and non-empty live `website_evidence`
  (~22k). The ONLY alive companies on this dashboard.
- **dead:** `evidence_source == "wayback_dead"` (~15.7k evidence-based verdicts).
  The ONLY dead companies on this dashboard. In PREVIEW mode (corrected CSV
  absent) the full dead work list stands in, behind the banner.
- **dead_strict:** dead and `website_alive == False` and not thin_history
  (robustness subset only, Act 4).
- **excluded from the dashboard entirely:** live rows with empty evidence, and
  `dead_metadata` rows (not-found companies whose archive evidence was never
  recovered). They appear only as counts in the Act 4 coverage funnel.

Dashboard universe = survivor + dead, roughly 38k of the 44,387.

## Dashboard structure

Nav sections: Overview, Landscape, RAD, Cohorts, Confidence, then the flagship
**Survivorship** section (with sub-anchors for its four acts). Footer carries model,
method, generation date, and the PREVIEW/evidence-based flag.

### Base sections (adapted from `full_baseline_cohort.html`)

All five run on the **corrected** frame (survivors + recovered dead), so every number
is the survivorship-corrected view of the ecosystem. Adaptations only; chart types and
layout stay as in `build_classification_dashboard.py`.

1. **Overview.** Same 5 KPI cards plus one new card: `evidence_source` split
   (n live / n wayback_dead / n dead_metadata). Headline title becomes
   "Survivorship-corrected classification of 44,387 US startups".
2. **Landscape.** Same subclass bar + AI-rate-by-cohort. Add one filter pill group:
   `Evidence: All / Live only / Dead only` (client-side, precomputed like the existing
   `FILTER_DATA` cohort x min-confidence grid, extended with the evidence dimension;
   3 x 3 x 5 = 45 keys, still tiny).
3. **RAD.** Unchanged charts (RAD distribution, subclass x RAD heatmap) on the
   corrected frame.
4. **Cohort dynamics.** Unchanged charts on the corrected frame.
5. **Confidence audit.** Unchanged charts on the corrected frame; the survivor-vs-dead
   confidence comparison lives in the flagship section instead, to avoid duplication.

### Flagship section: Survivorship (four acts)

Every chart lists: title, type, axes, filters, and the one-sentence insight a reader
should be able to state. Charts marked REUSE have compute already in
`survivorship_analysis.py`; NEW means a new helper is needed.

#### Act 1: The bias exists

- **1.1 Headline correction** (REUSE `_correction`, recomputed on the
  evidence-only universe). Grouped bar. x: two views (biased = evidence-based
  alive only; corrected = evidence-based alive + evidence-based dead); y:
  AI-native rate % with Wilson 95 percent CIs as error bars. No filters.
  Insight: "Counting only companies alive today misstates the AI-native rate by X points."
- **1.2 Composition before and after** (REUSE `_correction` subclass dists). Two 100
  percent stacked horizontal bars (biased vs corrected), segments = 10 subclasses in
  house colors. No filters.
  Insight: "The survivor lens systematically over-counts some subclasses and
  under-counts others."
- **1.3 Share shift per subclass** (NEW, trivial: corrected share minus biased share).
  Diverging horizontal bar. y: subclass; x: share change in percentage points, red for
  losses, green for gains. No filters.
  Insight: "Subclass S gains/loses D points of ecosystem share once the dead are counted."
- **1.4 Metadata vs evidence flips** (REUSE `_flips`; only meaningful post-merge).
  Small bar or KPI trio: percent of the recovered dead whose `ai_native`, `subclass`,
  `rad_score` changed vs their metadata-only label. No filters.
  Insight: "Recovered evidence overturned the metadata-only verdict for X percent of
  dead companies, so the correction is real, not relabeling."

#### Act 2: Who dies

- **2.1 Subclass distribution, alive vs dead** (REUSE `_ai_vs_survival` + NEW tests).
  Grouped bar, normalized. x: 10 subclasses; y: share of cohort %; two series
  (survivor, dead). Significance stars on subclasses where the BH-adjusted
  two-proportion z-test rejects at alpha 0.05; caption states the correction.
  No dead-definition filter (locked decision 1 fixes the dead cohort to the
  evidence-based set; the strict subset lives in Act 4 as robustness).
  Insight: "Subclasses 1C and 1G are significantly over-represented among the dead."
- **2.2 Subclass lift** (REUSE `subclass_lift`). Vertical bar. x: subclass; y: dead
  share / survivor share, dotted line at 1.0.
  Insight: "A lift above 1 means the genre dies more than it lives."
- **2.2b Mortality by subclass and by defensibility group** (REUSE
  `_rad_survival.by_subclass` / `.by_subclass_group`, carried over from the
  retired insights dashboard). Horizontal bars ranked by mortality %, n printed.
  Insight: "Commoditizable genres (1C, 1G) die at a higher rate than defensible
  ones (1A, 1B, 1E)."
- **2.3 Mortality by RAD** (REUSE `_rad_survival.by_rad`). Horizontal bar, AI-native
  firms only. y: RAD-H/M/L; x: mortality % (dead / dead+survivor) with n labels.
  Insight: "High foundation-model dependency tracks higher mortality: the
  dependency-trap hypothesis in one chart."
- **2.4 Mortality by founding era** (NEW, one `_mortality_by` call on `founding_era`).
  Two bars, PRE-GENAI vs GENAI-ERA mortality %.
  Insight: "GENAI-ERA firms are younger, so raw era mortality mostly measures exposure
  time; the regression handles this properly." (Printed as the chart caption.)
- **2.5 Vertical mortality** (REUSE `_vertical`). Horizontal bar, AI-native only,
  top 8 primary category groups by n. y: vertical; x: mortality % with n.
  Descriptive only, no per-vertical tests (small cells).
  Insight: "Mortality is not uniform across markets; vertical V is the deadliest
  large vertical."
- **2.6 Funding by survival status** (NEW helper). Grouped bar. x: funding buckets
  (unknown, <$1M, $1-10M, $10-100M, $100M+); y: share of cohort %; series survivor
  vs dead. Companion single bar: mortality % per bucket.
  Insight: "Dead companies cluster in the unfunded and sub-$1M buckets; funding buys
  survival time." Caption flags the unknown-heavy Crunchbase funding field.

#### Act 3: Why they die

- **3.1 Model 1 forest plot** (REUSE `_regression.model1`). Odds-ratio dot plot with
  95 percent CI bars, log x-axis, line at OR 1.0. Terms: AI-native, log funding,
  founding era, top-8 vertical dummies. Full outcome frame.
  Insight: "Net of funding, era, and vertical, AI-nativeness multiplies death odds
  by X (or divides, if protective)."
- **3.2 Model 2 forest plot** (REUSE `_regression.model2`). Same style, AI-native
  firms only. Terms: RAD-H and RAD-M vs RAD-L, defensibility group, log funding, era.
  Insight: "Among AI firms, dependency itself predicts death after controlling for
  resources, validating RAD as more than a label."
- **3.3 Funding x RAD mortality heatmap** (REUSE `_dependency_trap`). Heatmap. rows:
  funding buckets; cols: RAD-H/M/L; cell: mortality % + n.
  Insight: "The deadliest cell is high dependency with thin funding."
  Implementation note: attempt a Model 3 adding a `log_funding : C(rad_score)`
  interaction; ship it only if it converges with stable CIs, otherwise the heatmap
  stands alone as the descriptive interaction view (stated in the plan as a
  try-then-fall-back work item, not a promise).
- **3.4 Deaths over time** (REUSE `_temporal`). Bars of last-seen month counts since
  2022-01 with lines for commoditizable (1C, 1G) vs defensible (1A, 1B, 1E) genres and
  dotted reference lines at ChatGPT / GPT-4 / GPT-4 Turbo / GPT-4o releases. Marked
  exploratory in the caption (death anchor = last Wayback capture, coverage-bounded).
  Insight: "Do commoditizable genres die in waves after frontier releases?"

#### Act 4: Honesty and robustness

- **4.1 Confidence, survivor vs dead** (REUSE `_confidence`). Grouped bar. x:
  confidence 1..5; y: share of cohort %; series survivor vs dead_recovered. Printed
  caveat directly under the chart: archive evidence is a single pre-death homepage vs
  a multi-page live crawl, so a confidence gap partly reflects evidence quality, not
  company quality.
  Insight: "Dead-cohort verdicts are (or are not) materially less confident than live
  ones, and here is why that comparison is not apples-to-apples."
- **4.2 Thin-history and snapshot-age sensitivity** (NEW helper). Two small charts:
  (a) AI-native rate among the dead, thin_history true vs false, with counts;
  (b) AI-native rate among the dead by `days_before_death` bucket (0-30, 31-90,
  91-365, 365+ days between the used snapshot and the death anchor).
  Insight: "The headline dead-cohort AI rate does not hinge on companies with thin
  archives or stale snapshots."
- **4.3 Coverage funnel** (NEW helper). Funnel or stepped horizontal bar:
  22,002 not-found -> 19,044 with a usable pre-death snapshot -> ~15.7k classified
  with recovered evidence, plus the residual ~2,958 with no archive and the extract
  shortfall. Caption names who is still missing (companies too small or short-lived
  to be archived) and the direction of remaining bias (the corrected dataset still
  under-represents the smallest, fastest-dying firms, so the correction is a floor).
  Insight: "Even the corrected view misses the least-archived companies; the true
  bias is at least as large as measured."
- **4.4 Methods box** (no chart). Printed text block listing the caveats below.

## Statistical choices

1. **Two-proportion z-tests per subclass** (alive vs dead shares), via
   `statsmodels.stats.proportion.proportions_ztest` (statsmodels is already an
   `analysis` extra dependency). 10 tests, so control the false discovery rate with
   Benjamini-Hochberg at alpha 0.05; BH over Bonferroni because we care about ranking
   the story, not strict familywise error, and cells are large enough that power is
   not the issue. Also report one global chi-square on the 10x2 table as the omnibus
   check. Annotate stars only where BH-adjusted p < 0.05; print the adjustment method
   on the chart.
2. **Wilson score intervals** for the headline AI-native rates (better than normal
   approximation near the observed 3-10 percent rates; one-line formula, no new deps).
3. **Logistic regressions** reused unchanged from `survivorship_analysis._regression`
   (Model 1: death ~ AI-native + log funding + era + vertical; Model 2, AI-native
   only: death ~ RAD (vs RAD-L) + defensibility group + log funding + era). Odds
   ratios with 95 percent CIs on forest plots. Optional Model 3 with a
   funding x RAD interaction, shipped only if it converges cleanly.
4. **Mortality definition** everywhere: dead / (dead + survivor) within the outcome
   frame, excluding the `excluded` rows (live but no evidence, not in the dead list),
   exactly as `_mortality_by` does now.
5. **No survival-time analysis** (Kaplan-Meier etc.): survivors have no time-at-risk
   origin comparable to `death_ts`, so any hazard estimate would be fake precision.

## Caveats to print on the dashboard itself

1. "Dead" means Tavily could not extract the live site: a strong proxy for failure,
   not a death certificate. A strict subset (site confirmed offline, non-thin archive
   history) is shown alongside so findings do not hinge on the definition.
2. Dead-company evidence is a single archived homepage; live evidence is a multi-page
   crawl. Confidence and verdict-richness gaps partly reflect evidence quality.
3. Dead companies are judged on their messaging as of the pre-death snapshot date;
   survivors on today's site. The comparison cannot fully separate death from
   evidence recency; founding era is controlled in the regression.
4. Coverage is incomplete: ~2,958 of the 22,002 not-found companies have no usable
   archive, and they skew small and short-lived, so residual survivorship bias
   remains in the corrected view (the measured correction is a lower bound).
5. `total_funding_usd` is Crunchbase-reported and unknown-heavy; funding charts keep
   an explicit "unknown" bucket rather than dropping rows.
6. `cohort` (PRE-GENAI vs GENAI-ERA) is a classifier output derived from founding
   date, not an independent measurement.
7. Statistical annotations use BH-adjusted two-proportion z-tests; adjusted alpha
   and test counts are printed with the chart.
8. Until `survivorship_corrected.csv` exists, a PREVIEW banner states dead verdicts
   are metadata-only and all dead-side numbers are provisional.

## Coverage checklist: retired survivorship_insights.html sections

Per the user's requirement, every section of the retired
`build_survivorship_insights_dashboard.py` is either (a) present in the new
dashboard or (b) consciously dropped with a reason.

| Retired section / chart | Verdict | Where / why |
|---|---|---|
| Overview hero metrics + dead-definition insight | (a) present | Flagship intro block + Act 4 methods box; cohort counts in nav meta |
| 01 AI-native rate: survivor vs dead (all) vs dead (strict) | (a) present, adapted | Act 4 robustness chart as survivor vs dead vs dead_strict; the "dead (all not-found incl. metadata-only)" bar is gone per locked decision 1 (evidence-only universe) |
| 01 Survivorship correction (biased vs corrected) | (a) present, improved | Act 1.1 with Wilson CIs, on the evidence-only universe |
| 01 Subclass lift | (a) present | Act 2.2 |
| 02 Mortality by RAD | (a) present | Act 2.3 |
| 02 Mortality by defensibility group | (a) present | Act 2.2b |
| 02 Mortality by subclass (ranked) | (a) present | Act 2.2b |
| 03 Dependency-trap heatmap (funding x RAD) | (a) present | Act 3.3 |
| 04 Vertical mortality | (a) present | Act 2.5 |
| 05 Deaths over time + model-release markers | (a) present | Act 3.4 |
| 06 Regression forest, Model 1 | (a) present | Act 3.1 |
| 06 Regression forest, Model 2 | (a) present | Act 3.2 |
| 07 Confidence survivor vs dead | (a) present, improved | Act 4.1 with the evidence-thinness caveat printed |
| 07 Flips insight (metadata vs evidence) | (a) present | Act 1.4 |
| 07 Strict-subset + limits insights | (a) present | Act 4 robustness chart + methods box |

Nothing is dropped outright; the only conscious change is that all "dead (full
proxy incl. metadata-only)" framings collapse to the evidence-based dead cohort,
per locked decision 1.

## Cut or flagged from the brief (with reasons)

- **Employee-size cuts:** `employee_count` exists in master but is a coarse Crunchbase
  bucket, frequently stale for dead firms (last-known headcount, not at-death).
  Excluded from the main design; can be added as a robustness appendix if wanted.
- **Kaplan-Meier / time-to-death curves:** no comparable time-at-risk for survivors.
  Cut (deaths-over-time bar chart 3.4 is the honest exploratory alternative).
- **Per-vertical significance tests:** AI-native x vertical x survival cells get
  small; vertical mortality stays descriptive with n printed on each bar.
- **Geography, investor, or founder covariates:** columns do not exist in
  `master_csv.csv`. Cut.
- **Funding x RAD as a formal interaction term:** kept as an attempt with an explicit
  fallback to the descriptive heatmap, since sparse RAD-L cells may not support it.

## Implementation work items (in order)

1. Extend `survivorship_analysis.py` with four helpers, merged into
   `compute_metrics()`: `_subclass_tests` (z-tests + BH + chi-square),
   `_funding_by_survival` (bucket shares + per-bucket mortality),
   `_sensitivity` (thin_history and days_before_death cuts),
   `_funnel` (coverage funnel counts). Keep all existing keys stable.
   Verify: run the module directly in PREVIEW mode; JSON contains the new keys and
   `pytest` (if any dashboard tests exist) still passes.
2. Add Wilson CIs to `_correction` and `_ai_rate` outputs (additive keys).
   Verify: rates unchanged, CI bounds sane on PREVIEW data.
3. Create `build_v1_alive_dead_dashboard.py`: fork the HTML shell and STYLE from
   `build_classification_dashboard.py`, import `compute_metrics` from it by file path
   for the base sections, import `survivorship_analysis` for the flagship section,
   add the evidence-source filter grid, render all Act 1-4 charts, PREVIEW banner,
   caveats box, footer. Output `v1_alive_dead_cohort.html`.
   Verify: builds in PREVIEW mode, opens locally, every chart renders, no console
   errors, nav anchors work.
4. Try Model 3 (funding x RAD interaction); keep only if it converges with stable CIs.
   Verify: convergence flag + CI widths inspected by hand.
5. After the batch recovery completes and `merge_survivorship.py` runs: rebuild,
   confirm the PREVIEW banner disappears, spot-check headline numbers against the
   merge script's printed before/after table, and confirm flips (1.4) populates.
6. Update `AGENTS.md` (repository layout + status tables) in the same change.

## Open questions (resolved 2026-07-23, see STATUS)

1. **Dead cohort scope:** RESOLVED, stricter than proposed: evidence-only
   universe everywhere (alive = Tavily-scraped only, dead = wayback_dead only).
2. **Base sections:** RESOLVED: corrected-only, no toggle.
3. **survivorship_insights.html:** RESOLVED: retired; builder deleted in this
   branch, compute module kept, content absorbed (see coverage checklist).
4. **Exact classified count for the funnel** (plan assumes ~15,714): still to
   confirm once the batch recovery finishes; the funnel reads it from data, so
   no code change is needed.
