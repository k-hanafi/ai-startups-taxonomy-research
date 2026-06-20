# Historical AI-Messaging Study — Master Plan

> Living document. This is the single source of truth for the Wayback Machine
> ("how did startup AI messaging evolve since GPT-4?") workstream. Update it as
> decisions change. The agent should read this first when resuming work.

Last updated: 2026-06-08

---

## 1. Research question

How did the AI messaging on startup company websites evolve from the release of
GPT-4 (March 14, 2023) to today?

We already classify companies' **current** websites with our taxonomy
(AI-native subclass + RAD score). The new strand re-runs that *same* classifier
on the **March-2023 version** of the same companies' websites, captured by the
Internet Archive's Wayback Machine, to measure the before/after shift.

## 2. The core design idea

`classify.py` does not care where `website_evidence` comes from — it only reads
the columns in `CLASSIFIER_INPUT_COLUMNS`. So the entire job is to produce a
second `classifier_input_2023.csv` with identical columns, where
`website_evidence` is the 2023 snapshot instead of today's site. Then we run the
*existing, unchanged* classifier on it and diff the two outputs.

We are NOT building a new classifier or new taxonomy. We are building a second
**evidence source** that plugs into the same socket.

## 3. Phases

| Phase | Goal | Status |
|-------|------|--------|
| 0. Discovery / feasibility | Measure how much of our cohort is even retrievable at March 2023 before building anything. Produce a dashboard for the research team. | DONE — GO (see Findings §4a) |
| 1. Cohort freeze | Lock the ~20k companies (Tavily-returned set) we will study longitudinally. | started (cohort extracted) |
| 2. Historical scrape infra | CDX discovery + Tavily extract over snapshots -> `classifier_input_2023.csv`. | BUILT (see §5c); awaiting recovery-probe finish to freeze targets |
| 3. Classify 2023 | Run existing classifier on the 2023 input. | not started |
| 4. Longitudinal analysis | Diff 2023 vs today; dashboards for the paper. | not started |

**We do not advance past Phase 0 until the coverage numbers justify it.** If only
~half the cohort is retrievable at March 2023, we revisit the cohort or the date.

## 4a. Phase 0 coverage results (CDX probe, n=300 sample, 2026-06-08)

Verdict: **GO.** Coverage is well above the 50% go/no-go threshold.

- Cohort = **22,032** companies (the Tavily-returned set; 49.6% of 44,387 scanned).
- Sample 300, **265 resolved** cleanly, 35 dropped on Archive throttling.
- **76.6% ± 5.1%** of resolved companies have a March-2023 homepage capture.
- **81.9% ± 5.0%** for companies founded ≤2022 (the fair comparison denominator).
- **0 "never archived"** — every miss is a domain in the Archive lacking a capture in
  the Dec 2022 – Jun 2023 window. Coverage falls with youth: 2021 90% -> 2022 69% ->
  2023 54% -> 2024+ 0%. So misses are young companies, not Archive gaps.
- Temporal fidelity is strong: median drift **10 days** from March 14; 173/203 captures
  within 30 days.
- Operational reality: the Archive throttles hard (~12% of CDX calls dropped even at
  concurrency 4). The full scrape will be **rate-limit-bound and slow**, not credit-bound.
- Dashboard: `canvases/march-2023-historical-coverage.canvas.tsx` (Cursor canvas).
- Repro: `wayback_machine/scripts/probe_coverage.py` then `summarize_coverage.py`.

## 4. Findings (from spikes — these are facts, not assumptions)

- **Tavily `/extract` works on archived URLs.** Pointed at
  `web.archive.org/web/20230301000225id_/https://stripe.com/`, it returned the
  real March-2023 Stripe homepage as markdown (13.7k chars), no Archive chrome.
  The `id_` suffix on the timestamp is what strips the Wayback toolbar.
- **Extract response shape == crawl response shape** (`results:[{url,raw_content}]`),
  so our existing `compact_tavily_response()` works on it unchanged. Same cleaning,
  same `website_evidence` format -> clean like-for-like comparison.
- **Extract does NOT return a `usage`/credits field** (only `response_time`), unlike
  crawl. Historical budget control will be call-count-based, not response-parsed.
- **Use CDX, not the availability API, for discovery.** The `availability` endpoint
  returned empty for stripe.com twice (flaky under load); the CDX Server API
  returned exact snapshots reliably.

## 5b. Phase 2 scrape — decisions locked (2026-06-10)

- **Engine: Tavily `/extract`** on the archive snapshot URLs, mirroring the live crawl so
  the 2023 classifications are directly comparable to today's. (User confirmed; doesn't mind paying.)
- **We already have all snapshot URLs** for free: `web/{closest_ts}id_/{homepage_url}` derived
  from `coverage_full.csv`. No HTML pre-download needed — Tavily fetches the URLs itself.
  (Tavily `/extract` takes URLs only; it cannot read local HTML — verified from its tool schema.)
- **Scope: retrievable companies only** (status ok & has_2023). The true before/after panel.
- **Stage C structure:** modular fetch+extract; first run is Tavily-direct on the URL list.
- **Recovery pass running:** re-probing the ~9,499 throttle-failed companies (resumable, skips the
  12,533 already resolved) to find the TRUE retrievable count before freezing the scrape list.
- **Research goal (why):** run the existing LLM taxonomy classifier on the 2023 Tavily evidence,
  then diff against today's classifications to measure how AI-nativeness/RAD shifted since GPT-4.

## 5c. Phase 2 infrastructure — BUILT (2026-06-09)

`wayback_machine/` refactored into a self-contained sub-project (zero `src`
imports; the cleaner is vendored + golden-tested so the folder is liftable).
Design sign-off: standalone isolation, moderate hygiene (README + tests +
.gitignore, shared venv), spike script included.

Modules: `config.py` (tunables), `paths.py`, `cohort.py` (vendored column
contracts + snapshot-URL builder + retrievable filter), `evidence.py` (frozen
copy of `src/website_evidence.py`), `state.py` (atomic resume + JSONL heal +
completed-ids), `extract.py` (resumable Tavily `/extract` engine: sliding-window
limiter, per-call retries, outage loop, SIGINT graceful stop, call-count budget,
heartbeat, run manifest), `targets.py` (Stage B), `classifier_input.py` (Stage D).
CLIs in `scripts/`: `build_targets.py`, `spike_extract.py`, `run_extract.py`,
`build_classifier_input_2023.py`. Tests: golden cleaner + cohort helpers.

Key design choices:
- **Existence filter at freeze time**: Stage B keeps only retrievable companies
  founded `<= 2023-03` (`COHORT_FOUNDED_CUTOFF`). Catches companies that slip
  through retrievability via a post-launch snapshot.
- **Final CSV metadata base = `data/master_csv.csv`** (the same static file the
  live crawl joined against), with only the two evidence columns swapped to 2023
  values. Guarantees the ONLY difference fed to the classifier is the evidence.
- **Archive URL rewritten back to the homepage** before cleaning, so the 2023
  evidence header matches the live format exactly.
- **Budget is call-count based** (extract returns no usage field): basic billing
  is 1 credit / 5 successful extractions.
- Empty/thin extracts are terminal + skipped on resume (delete JSONL lines to retry).

Run order (network stages outside the sandbox):
`build_targets.py` → `spike_extract.py --n 50` → `run_extract.py` →
`build_classifier_input_2023.py` → existing `classify.py`.

## 5. Decisions locked

- **Folder layout:** one top-level `wayback_machine/` package (underscore so it is
  importable; a space breaks Python imports). It will hold data, the discovery
  dashboard, and later the scrape code. Reuses `src/website_evidence.py` etc.
- **Extraction method:** Tavily `/extract` on the snapshot URL (identical pipeline
  to live crawl -> cleanest comparison).
- **Pages per company:** homepage only to start (carries most AI messaging, most
  robust against partial archive coverage).
- **Discovery primitive:** CDX Server API.
- **Target date:** 2023-03-14 (GPT-4 launch). Window: 2022-12-01 to 2023-06-30,
  pick the capture closest to the target. Record the actual snapshot date per company.

## 5a. Overnight full-census run (started 2026-06-08 ~06:40 UTC)

Running the coverage probe across the FULL 22,032 cohort, shuffled (seed 42) so any
partial result is an unbiased random sample. Writing to `wayback_machine/data/coverage_full.csv`.
Config: concurrency 6, `--skip-ever-call`, wrapped in `caffeinate -ims`.

### Final results (completed 2026-06-08)

- **22,032** probed (full census, shuffled seed 42)
- **12,533** resolved cleanly; **9,499** Archive throttle errors (retriable)
- **9,089** have a March-2023 homepage capture → **72.5% ± 0.8** of resolved
- **79.9%** for companies founded ≤2022
- **~16,000** projected retrievable across full cohort
- Output: `wayback_machine/data/coverage_full.csv`

### Tomorrow's steps (copy/paste)
```bash
cd /Users/k/Desktop/ai-native-startup-classification
# 1. Check progress (rows written so far)
wc -l wayback_machine/data/coverage_full.csv
# 2. If you want to keep filling it in, just re-run the same command (it resumes):
caffeinate -ims python3 wayback_machine/scripts/probe_coverage.py \
  --sample-size 0 --concurrency 6 --skip-ever-call \
  --output wayback_machine/data/coverage_full.csv
# 3. Recompute dashboard aggregates from whatever is done:
python3 wayback_machine/scripts/summarize_coverage.py \
  --input wayback_machine/data/coverage_full.csv
# 4. Then update the canvas data block with those numbers.
```
Canvas to refresh: `canvases/march-2023-historical-coverage.canvas.tsx`.

## 6. Open questions

- ~~Final cohort definition~~ RESOLVED (2026-06-09): scrape only companies that
  existed at GPT-4 launch — `founded_date <= 2023-03` AND retrievable. Stage B
  enforces this (`build_targets`, `--founded-cutoff`). On the full cohort, 1,321
  of 22,032 are founded after March 2023 (557 in 2023 Apr–Dec, 764 in 2024) and
  would be dropped; the rest of the drop happens via retrievability.
- Acceptable date drift: how far from March 14 is still "March 2023" for the paper?
- If coverage is low, do we widen the window, change the cohort, or pick a
  different inflection date?

## 7. Artifacts

- `wayback_machine/data/wayback_cohort.csv` — frozen ~20k cohort metadata (no
  evidence text; keeps `live_evidence_chars` as a richness signal).
- `wayback_machine/data/coverage_sample.csv` — CDX coverage probe results on a sample.
- Canvas dashboard: March-2023 retrievability of the cohort.
