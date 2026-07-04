# Portfolio git message examples

## Bad vs good (feature commit)

**Bad (insider, mechanical):**

```
Gate Tavily spend with website_alive on a single lean classifier CSV

- Default artifact is outputs/tavilycrawl/classifier_input.csv.
- Crawl queue skips rows with website_alive=false.
- Master join trimmed to fields we still inject into prompts.
```

**Good (recruiter-readable, goal-anchored):**

```
Skip dead company websites before paid web crawls to protect the research budget

The project classifies startups using web evidence, so every wasted crawl
is real money and noisy data. This commit checks each homepage once and
remembers whether it is reachable, then only paid crawls run for live sites.
Companies with dead sites still get classified using the data we already have.

- One canonical input file replaces a fan-out of intermediate CSVs, so reruns are predictable.
- Trimmed the company profile down to the fields that actually feed the model, on purpose.
- Verified end-to-end on a small sample. Full crawl is gated until budget review.
```

## Bugbot fix example

**Title:**

```
Bugbot fix: Backfill classifier CSV from snapshots so crashes cannot drop companies
```

**Body:**

```
Stage D reads the processed scrape file, but a crash could mark a company done in
the log without a matching row. This commit closes that gap so the historical
comparison panel stays complete after an interrupted run.

- Startup reconciles counters and backfills any missing success rows from the log.
- Processed CSV is saved before the log line on each success.
- Tests: pytest wayback_machine/tests/test_state.py.
```

## Title pairs (weak → strong)

| Weak | Strong |
|------|--------|
| `Add paths.py` | `Standardize where pipeline outputs are saved so research artifacts are reproducible` |
| `Gate Tavily on website_alive` | `Skip dead websites before paid crawls so the research budget is not wasted` |
| `Bugbot fix: harden dead-cohort resume` | `Bugbot fix: Keep dead cohort labels honest and crawl billing crash-safe` |
