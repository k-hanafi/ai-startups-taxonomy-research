# AGENTS.md

Briefing for AI coding agents working in this repo. **Read this first** — it
replaces an exhaustive codebase search. It is auto-injected into every chat.

If you change the repo's structure, architecture, data flow, commands, or
status, **update this file in the same change**. See [Maintaining this file](#maintaining-this-file).

Last updated: 2026-07-22 · Active branch: `eval/luna-config` (luna pricing re-verified + survivorship status corrected; next = paid 9-cell matrix)

---

## Project overview

Research codebase for a two-axis taxonomy of AI-native startups (UBC student
research; companion SSRN paper "Prompted to Start"). Every company gets:

- an **AI-native axis** — `ai_native` 0/1 plus a `subclass` (1A–1G AI-native,
  0A/0B/0C not), and
- a **RAD score** (Resource-Adjusted AI Dependency: how dependent/defensible the
  company is vs. foundation-model providers).

The pipeline enriches Crunchbase rows with live website evidence, then classifies
them with an LLM via the OpenAI Batch API. There are **three strands**, all
feeding the *same* classifier:

1. **Live** (built, run): classify companies on today's websites.
2. **Historical / wayback** (built; paused awaiting recovery probe): re-run the
   *unchanged* classifier on each company's **March-2023 (GPT-4 launch)** homepage
   from the Internet Archive, to measure how AI messaging shifted.
3. **Survivorship-bias** (merged to `main`): recover **pre-death** snapshots for
   the ~22k companies Tavily couldn't extract, classify them, and merge back so the
   dataset isn't biased toward survivors.

**Core invariant:** `classify.py` only reads `CLASSIFIER_INPUT_COLUMNS`. Each
strand is just a different way to produce `website_evidence`; the classifier and
taxonomy never change. The only thing that differs across strands is the evidence.

## Status / roadmap

| Strand | Stage | Status |
|--------|-------|--------|
| Live | crawl → classify → merge | DONE — 44,387 companies classified (`production_classifications.csv`) |
| Historical (wayback) | coverage probe done; infra built | PAUSED — GO verdict (~16k retrievable at Mar-2023); awaiting recovery probe before paid extract |
| Survivorship-bias | probe done → extract DONE → classify → merge | IN PROGRESS — paid Stage C extract complete (19,044 targets covered, 15,714 with evidence in `scrape_processed_dead.csv`); next: `build_classifier_input_dead.py` → `classify_dead.py` (paid) → `merge_survivorship.py` |

Authoritative plans (read when resuming a strand; committed under **`.cursor/plans/`**):
- `.cursor/plans/roadmap-to-july-deliverable.plan.md` — master roadmap for the July professor-meeting deliverables (read first).
- `.cursor/plans/PLAN.md` — historical/wayback master plan.
- `.cursor/plans/survivorship_bias_wayback_*.plan.md` — death-anchored CDX probe (active survivorship strand).
- `.cursor/plans/survivorship_tavily_pipeline_*.plan.md` — post-probe Tavily extract + classify pipeline.
- `.cursor/plans/logprob_confidence_classifier_*.plan.md` — logprob-based confidence methodology (active).
- `.cursor/plans/golden_set_eval_harness_*.plan.md` — golden-set eval harness (active; classification committed; #22 dashboard + #24/#25 science on `main`; provisional `draft_*` gold accepted for paid sweep; next = paid 9-cell matrix with Pass A auto-banked once per model).

Cursor writes new plans to `~/.cursor/plans/` by default; copy or sync them into **`.cursor/plans/`** in this repo so they are version-controlled. Legacy copies may still exist in **`plans/`** at repo root. Repo agent skills (committed): **`portfolio-git-messages`**, **`git-commit-batch-plan`**, **`code-structure`**, **`clean-my-repo`** under **`.cursor/skills/`**. **`.cursor/rules/`** stays local.

## Tech stack

Python ≥3.11 · `openai` (Responses + Batch API) · `pandas` · `pydantic` (structured
output) · `tiktoken` (pre-flight cost) · `tenacity` (retries) · `rich` (terminal
UI) · `python-dotenv`. Tavily HTTP API for web crawl/extract (stdlib `urllib`).
Internet Archive CDX API for snapshot discovery. Tests: `pytest`. The
survivorship-insights dashboard adds `statsmodels` (logistic regression),
installed via the `analysis` extra.

## Architecture & data flow

```
LIVE strand
data/master_csv.csv ──update_website_liveness.py──▶ website_alive set in place
        └──run_tavily_crawl.py (Tavily /crawl)──▶ outputs/tavilycrawl/processed/classifier_input.csv
                └──classify.py prepare|submit|download──▶ outputs/production_csvs/production_classifications.csv

HISTORICAL strand (wayback_machine/, self-contained)
coverage_full.csv ──build_targets.py──▶ scrape_targets.csv
        └──run_extract.py (Tavily /extract on archive URLs)──▶ outputs/raw/snapshots.jsonl
                └──build_classifier_input_2023.py──▶ classifier_input_2023.csv ──▶ classify.py

SURVIVORSHIP strand (active; GO = archive crawl matching the live cohort)
classifier_input.csv (empty-evidence rows) ──build_not_found_cohort.py──▶ not_found_cohort.csv
 └──probe_death_coverage.py (death-anchored CDX)──▶ death_coverage.csv
 └──build_targets_dead.py──▶ scrape_targets_dead.csv (if_ snapshot URL + per-company scope)
 └──run_extract_dead.py (Tavily /extract on pre-death snapshot)──▶ scrape_processed_dead.csv
 └──build_classifier_input_dead.py──▶ classifier_input_dead.csv
 └──classify_dead.py run (classify.py under CLASSIFY_NS=wayback_dead)──▶ outputs/wayback_dead/wayback_dead_classifications.csv
 └──merge_survivorship.py──▶ outputs/wayback_dead/survivorship_corrected.csv
 └──build_survivorship_insights_dashboard.py (survivor-vs-dead + logistic regression)──▶ data visualization/01_Presentation_Materials/survivorship_insights.html
```

`classify.py` itself is a state machine: `prepare → submit → download` (or `run`
for all three), with `status`, `retry`, `merge`, and `test`. Every stage reads a
checkpoint and skips finished work, so a 44k-row run is fully resumable.

## Repository layout

### Root
| Path | Purpose |
|------|---------|
| `classify.py` | CLI entry for the classifier (`prepare/submit/status/download/retry/merge/test/run`) |
| `README.md` | Public-facing writeup (taxonomy + pipeline narrative + mermaid diagrams) |
| `pyproject.toml` | Dependencies + pytest config |
| `AGENTS.md` | This file |
| `.cursor/plans/` | Committed Cursor plans (sync from `~/.cursor/plans/` after planning sessions) |
| `.cursor/skills/` | Four committed repo skills: `portfolio-git-messages`, `git-commit-batch-plan`, `code-structure`, `clean-my-repo` |
| `plans/` | Legacy plan copies (prefer `.cursor/plans/` for new work) |

### `src/` — live classification pipeline
| File | Responsibility |
|------|----------------|
| `config.py` | **Single source of truth** for tunables: `DEFAULT_MODEL` (`gpt-5.4-nano`), Tier-5 rate limits, batch sizing, token/cost constants. No magic numbers elsewhere. |
| `paths.py` | All filesystem paths for generated artifacts. `CLASSIFY_NS` env (set before import) reroutes batch state + output CSV under `outputs/<ns>/` for isolated runs (e.g. survivorship) |
| `master_csv.py` | Column contracts (`MASTER_CSV_COLUMNS`, `CLASSIFIER_INPUT_COLUMNS`); URL-validity + tavily-eligible row mask |
| `schema.py` | `ClassificationResult` Pydantic model (11 fields); auto-generates the JSON schema injected into every request |
| `formatter.py` | Maps a CSV row → user message; builds `custom_id` |
| `builder.py` | Writes JSONL batch files (identical cacheable prefix + 1 user msg/line); loads system prompt |
| `tokens.py` | tiktoken token counting + `MODEL_PRICING`; powers `--dry-run` cost reports |
| `submitter.py` | Fault-tolerant file upload + batch create (tenacity backoff); `BillingLimitError` |
| `monitor.py` | Async concurrent batch monitor; sliding-window queue-pressure control (stays under 15B token queue) |
| `downloader.py` | Downloads results, matches to inputs by `custom_id` (never positional), appends to production CSV, tracks cache hits |
| `merger.py` | Distribution + cost report (rich tables); no separate merge needed |
| `state.py` | `state.json` checkpoint (`BatchRecord` lifecycle); atomic writes; resume |
| `logger.py` | Logging setup |
| `website_evidence.py` | Cleans/compacts raw Tavily markdown into evidence text (strips chrome, packs signal-first) |
| `tavily_crawl.py` | Cost-controlled Tavily `/crawl` runner for live homepage enrichment (resumable, rate-limited, budget-capped) |

### `scripts/` — live, network-touching (run outside the sandbox)
| File | Purpose |
|------|---------|
| `update_website_liveness.py` | Probes each homepage, writes `website_alive` true/false into `master_csv.csv` (filters dead/parked before paid crawl) |
| `run_tavily_crawl.py` | Thin CLI wrapper around `src/tavily_crawl.py` |

### `prompts/`
| File | Purpose |
|------|---------|
| `system_classifier_prompt.txt` | **Active** system prompt (loaded by `builder.load_system_prompt`): taxonomy, evidence hierarchy, RAD rules, few-shots |
| `system_prompt.txt` | Earlier prompt version, kept for reference (not loaded by the current pipeline) |

### `wayback_machine/` — historical + survivorship strands (self-contained; zero `src` imports)
| File | Responsibility |
|------|----------------|
| `README.md` | Sub-project guide + stage-by-stage run order |
| `config.py` | Historical tunables: target date, CDX rate limits, `ExtractConfig`, budget, death-anchor lookback (`DEATH_LOOKBACK_DAYS`) |
| `paths.py` | All wayback paths |
| `cohort.py` | Vendored column contracts + snapshot-URL builder + retrievable/existence filters |
| `evidence.py` | **VENDORED** frozen copy of `src/website_evidence.py` (golden-tested; must stay behavior-identical) |
| `cdx.py` | Minimal IA CDX client (`to_host` + rate-limited `cdx_get`, freezes all workers on 429); used by the death probe |
| `state.py` | `ExtractState` resume + JSONL tail-healing + completed-ids reconciliation |
| `extract.py` | Resumable, budget-capped Tavily `/extract` engine (historical analogue of `tavily_crawl.py`) |
| `targets.py` | Stage B: `coverage_full.csv` → `scrape_targets.csv` |
| `targets_dead.py` | **(survivorship)** Stage B: `death_coverage.csv` → `scrape_targets_dead.csv` (emits `if_` crawl URL + per-company `select_paths` scope; no founded cutoff) |
| `extract_dead.py` | **(survivorship)** Stage C: resumable, budget-capped Tavily `/extract` over pre-death `if_`/`id_` snapshots; reuses `extract.py`'s reliability harness + failure-reason instrumentation (rate_limited vs no_archive_content); writes to the crawl-era artifact names to preserve resume state |
| `classifier_input.py` | Stage D: master metadata + 2023 evidence → `classifier_input_2023.csv` (reused by the dead strand) |

### `wayback_machine/scripts/` — thin CLIs
| File | Purpose |
|------|---------|
| `extract_cohort.py` | Build the frozen wayback cohort from live data |
| `probe_coverage.py` | Stage A: CDX coverage probe at the global Mar-2023 anchor |
| `summarize_coverage.py` | Aggregate `coverage_full.csv` for the dashboard |
| `build_targets.py` | CLI for `targets.py` |
| `spike_extract.py` | Small de-risk extract (~50 companies) before the full run |
| `run_extract.py` | CLI for the paid extract engine |
| `build_classifier_input_2023.py` | CLI for `classifier_input.py` |
| `build_not_found_cohort.py` | **(survivorship)** Build `not_found_cohort.csv` from empty-evidence rows |
| `probe_death_coverage.py` | **(survivorship, active)** Death-anchored CDX probe → `death_coverage.csv` |
| `run_probe_recovery.sh` | Shell helper to resume the recovery probe |
| `summarize_death_coverage.py` | **(survivorship)** Aggregate `death_coverage.csv` → compact JSON shared by the findings canvas + `build_survivorship_dashboard.py` |
| `build_targets_dead.py` | **(survivorship)** CLI for `targets_dead.py` |
| `run_extract_dead.py` | **(survivorship, paid)** CLI for the dead-cohort extract engine (`extract_dead.run_extract_dead`); wrap in `caffeinate -ims` outside the sandbox |
| `build_classifier_input_dead.py` | **(survivorship)** CLI: dead evidence → `classifier_input_dead.csv` |
| `classify_dead.py` | **(survivorship)** Sets `CLASSIFY_NS=wayback_dead` then delegates to `classify.main()` — runs the unchanged classifier in an isolated workspace |
| `merge_survivorship.py` | **(survivorship)** Stage F: overlay dead verdicts onto `production_classifications.csv`, tag `evidence_source`, write `survivorship_corrected.csv` + before/after summary |
| `summarize_crawl_failures.py` | **(survivorship)** Offline (stdlib-only, no keys) breakdown of `crawl_dead.jsonl` by `failure_reason` (rate_limited / no_archive_content / transient / network / legacy_empty) |

### `evals/` — golden-set eval harness
| Path | Purpose |
|------|---------|
| `dashboard_metrics.py` | Eval dashboard metrics: scored.json/fixture → chart metrics (ECE, selective@50, vs_baseline, Pass B isolating fields, finalist mean±range aggregates). No OpenAI import. |
| `tests/fixtures/dashboard/dashboard_mock_runs.json` | Synthetic locked matrix; Pass A metrics identical across efforts within each model (bank-once design) |
| `config.py` | Locked matrix `EVAL_MODELS` + `MATRIX_PASS_B_EFFORTS`; `PASS_A_TOP_LOGPROBS=2` (binary); legacy `TOP_LOGPROBS` for old single-pass only |
| `classification.py` | Pass A/B classification runner; Pass A auto-banks under `evals/runs/pass_a_banks/<model>/` (reuse by default; `--rerun-pass-a` / `--pass-a-from` escapes) |
| `logprob_extract.py` | Pass A confidence; requires both `{0,1}` candidates or marks unavailable |
| `scoring.py` | End-to-end axes + `pass_b_metrics` (family-conditional subclass, AI-native-only RAD, boundary_disagreement); `--baseline` paired deltas; refuses partial confidence unless `--allow-partial-confidence` |
| `__main__.py` | CLI: `run-classification` (paid matrix), `matrix` (list cells), `score`, `dashboard`; legacy `run` warns and is not the matrix path |

### Other
| Path | Purpose |
|------|---------|
| `data visualization/01_Presentation_Materials/*.html` | Generated dashboards |
| `data visualization/02_Analysis_Code/*.py` | Scripts that build those dashboards |
| `data visualization/02_Analysis_Code/survivorship_analysis.py` | Survivorship findings compute: survivor-vs-dead cohorts + 2 logistic models (pure metrics dict; PREVIEW from production if `survivorship_corrected.csv` absent) |
| `data visualization/02_Analysis_Code/build_survivorship_insights_dashboard.py` | Renders `survivorship_insights.html` from that compute module (reuses house STYLE) |
| `data visualization/02_Analysis_Code/build_eval_dashboard.py` | Eval viewer: Pareto / leaderboard / ECE+selective / reliability / vs_baseline / latency + config filter. Defaults to mock fixture; `--runs`/`--scored` for real runs. |
| `tests/` | pytest for the live pipeline (schema, formatter, tokens, enrichment, tavily runner) |
| `wayback_machine/tests/` | pytest for wayback (golden cleaner, cohort, state, config, budget, probe) |
| `keys/` | API key env files, e.g. `keys/openai.env` (`OPENAI_API_KEY`). Git-ignored + cursor-ignored. **Never commit.** |
| `data/`, `outputs/`, `wayback_machine/data/`, `wayback_machine/outputs/` | Generated/large data. Git-ignored **and not indexed** — read via terminal/Read, not semantic search. |

## Key data artifacts

| Artifact | What it is |
|----------|-----------|
| `data/master_csv.csv` | 44,387 companies — static Crunchbase metadata + `website_alive`. The base everything joins against. |
| `outputs/tavilycrawl/processed/classifier_input.csv` | master + live `website_evidence`. **Default input to `classify.py`.** |
| `outputs/production_csvs/production_classifications.csv` | 44,387 classified rows (the live output) |
| `outputs/batch_data/state.json` | classify resume checkpoint |
| `wayback_machine/data/coverage_full.csv` | Mar-2023 coverage probe over the 22,032 survivors |
| `wayback_machine/data/not_found_cohort.csv` | ~22,002 companies Tavily couldn't extract (survivorship target) |
| `wayback_machine/data/death_coverage.csv` | Death-anchored probe output (complete: 22,002 rows, 19,044 `ok`) |
| `wayback_machine/data/scrape_targets_dead.csv` | 19,044 dead-cohort extract targets (`if_` snapshot URL + scope); the frozen Stage-C work list |
| `outputs/wayback_dead/survivorship_corrected.csv` | Stage F output: modern dataset with dead verdicts overlaid (survivorship-corrected) |

## Domain model

`ClassificationResult` (11 fields, `src/schema.py`): `CompanyID`, `CompanyName`,
`ai_native` (0/1), `subclass` (1A–1G / 0A–0C), `rad_score` (RAD-H/M/L/NA),
`cohort` (PRE-GENAI / GENAI-ERA, split at GPT-4 launch 2023-03-14),
`conf_classification` (1–5), `conf_rad` (1–5 or null), `reasons_3_points`,
`sources_used`, `verification_critique`.

## Development commands

**`OPENAI_API_KEY` is required at import time.** `src/config.py` reads
`os.environ["OPENAI_API_KEY"]` when imported; `tests/test_tokens.py` pulls that
in, so **`pytest` fails to collect if the variable is unset**. A placeholder
(e.g. `OPENAI_API_KEY=placeholder`) is enough for the full test suite and offline
stages (`prepare`, `prepare --dry-run`, `status`, `merge`) — no API calls.
Real keys are only needed for paid stages (`submit`, `run`, `download`, `retry`,
`test`) and Tavily enrichment. Keys load from `keys/openai.env` / `keys/tavily.env`
when present; env vars take precedence.

```bash
pip install -e ".[dev]"            # install with dev (pytest) extras
pytest                             # live-pipeline tests
pytest wayback_machine/tests       # wayback tests (incl. golden cleaner)

python classify.py prepare --dry-run          # cost plan, no API calls
python classify.py run                         # prepare → submit → download (full)
python classify.py run --data path/to.csv      # classify a different evidence source
python classify.py test --company-name Stripe  # one company, flex pricing (prompt iteration)

python scripts/update_website_liveness.py      # set website_alive
python scripts/run_tavily_crawl.py             # live homepage crawl
# wayback run order: see wayback_machine/README.md

pytest evals/tests -q                       # full eval harness (use OPENAI_API_KEY=placeholder)
pytest evals/tests/test_dashboard_metrics.py   # dashboard metrics (no OpenAI key)
python -m evals matrix                          # list locked 9-cell matrix commands
python -m evals run-classification --model gpt-5.4-nano --effort-b low --require-matrix-cell
# later efforts for the same model auto-reuse Pass A (bank at evals/runs/pass_a_banks/<model>/)
# escape: --rerun-pass-a  |  advanced pin: --pass-a-from <run_id>
python -m evals dashboard                       # build eval_dashboard.html from mock matrix (default)
python -m evals dashboard --runs <run_id>...    # real scored.json only (no auto-discovery)
python -m evals score <run_id> --confidence-from-raw [--baseline <run_id>]
python -m evals score <run_id> --allow-partial                 # incomplete n_scored only
python -m evals score <run_id> --allow-partial-confidence      # incomplete raw confidence only
# legacy: python -m evals run  (single-pass; retired for matrix path; warns)```

## Conventions & invariants (don't break these)

- **No magic numbers** outside `src/config.py` (live) / `wayback_machine/config.py` (wayback).
- **Identical request prefix** across all requests is what enables prompt caching — keep it byte-stable.
- **Match results by `custom_id`**, never by position (batch order is not guaranteed).
- **`wayback_machine/evidence.py` must stay behavior-identical** to `src/website_evidence.py`. If you change the live cleaner, re-vendor and run `pytest wayback_machine/tests`.
- **Only `website_evidence` may differ** between strands fed to the classifier — that's the whole fair-comparison design.
- **Network/paid stages run OUTSIDE the Cursor sandbox** (Tavily crawl/extract, CDX probes, OpenAI). Wrap long runs in `caffeinate -ims` and/or `tmux`.
- **CDX is hard-capped at 60 req/min per IP**; exceeding it risks a 1-hour IP ban. Pace via `cdx.py`'s shared limiter; never raise rpm above ~58.
- `data/`, `outputs/`, `keys/` are git-ignored; `data/` & `outputs/` are also not indexed.

## Where to work

| Task | Start here |
|------|-----------|
| Change taxonomy / output fields | `src/schema.py` (+ `prompts/system_classifier_prompt.txt`) |
| Change classification instructions | `prompts/system_classifier_prompt.txt` |
| Tune cost / rate limits / batch size | `src/config.py` |
| Change row → prompt mapping | `src/formatter.py` |
| Change evidence cleaning | `src/website_evidence.py` → re-vendor `wayback_machine/evidence.py` → run golden test |
| Add/modify a classify subcommand | `classify.py` |
| Live website scraping behavior | `src/tavily_crawl.py` |
| Historical archive scraping | `wayback_machine/extract.py` + `scripts/run_extract.py` |
| Survivorship death probe | `wayback_machine/scripts/probe_death_coverage.py` + `wayback_machine/cdx.py` |
| Survivorship extract→classify→merge | `wayback_machine/extract_dead.py` + `scripts/{build_targets_dead,run_extract_dead,build_classifier_input_dead,classify_dead,merge_survivorship}.py` |
| Dashboards | `data visualization/02_Analysis_Code/` |
| Eval dashboard / config filter | `evals/dashboard_metrics.py` + `build_eval_dashboard.py` (ECE/selective/vs_baseline; mock fixture until paid matrix runs; `--runs` for real data) |
| Eval matrix / scoring | `evals/config.py` (`EVAL_MODELS` + `MATRIX_PASS_B_EFFORTS`); `run-classification` (auto Pass A bank); `matrix`; `score --confidence-from-raw [--baseline]` |

## Maintaining this file

This file is the project's onboarding memory. Keep it self-healing: when your work
changes the repo, update the relevant section **in the same session/PR** — don't
wait to be asked.

Update triggers → what to edit:
- Strand/milestone started or finished → **Status/roadmap** table + the `Last updated` line.
- Top-level module/script added, removed, or renamed → **Repository layout**.
- Data flow, schema, or domain model changed → **Architecture & data flow** / **Domain model**.
- New dependency, command, or invariant → **Tech stack** / **Development commands** / **Conventions**.
- Active branch changed → the `Active branch` line.

Rules: surgical edits only, preserve structure and tone, keep entries one line,
no session chatter. Global policy: `~/.cursor/user-rules/agents-md-maintenance.md`.
