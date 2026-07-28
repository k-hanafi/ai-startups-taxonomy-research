# Production two-pass classifier

`python -m two_pass_classifier` runs this package as a terminal program. The
workflow uses normal OpenAI Responses calls and keeps all generated state under
`outputs/two_pass_classifier/`.

## Recommended run order

```bash
python -m two_pass_classifier build-manifest
python -m two_pass_classifier cost-preview
python -m two_pass_classifier smoke
python -m two_pass_classifier run
```

1. `build-manifest` validates the default live and archive inputs, then writes
   one immutable content-addressed manifest.
2. `cost-preview` counts production-formatted input tokens and estimates the
   full cost without making API calls.
3. `smoke` runs the exact production request path on 10 deterministic companies.
4. `run` starts a new full run only after it finds a successful matching smoke.

The default model is `gpt-5.6-luna`. Pass A reasoning is always `none`, because
its confidence calculation needs token probabilities. Pass B reasoning defaults
to `low`.

## During and after a run

```bash
python -m two_pass_classifier status RUN_ID
python -m two_pass_classifier resume RUN_ID
python -m two_pass_classifier retry RUN_ID
```

- `status RUN_ID` reads local files only and reports progress, failures, usage,
  cost, throughput, ETA, rate utilization, concurrency, and output paths.
- `resume RUN_ID` reloads the immutable settings from the journal and continues
  only missing or retriable work.
- `retry RUN_ID` appends explicit retry events for retriable failures, preserves
  all history, and prints the exact resume command.

`classifications_in_progress.csv` contains complete rows only.
`classifications.csv` appears only when every manifest row is complete.

## Load-bearing flags

- `--manifest PATH`: use a specific immutable manifest instead of discovering
  the newest valid production manifest.
- `--model MODEL`: override the default with `gpt-5.4-nano`, `gpt-5.4-mini`, or
  `gpt-5.6-luna`.
- `--effort LEVEL`: set Pass B reasoning to `low`, `medium`, or `high`. It never
  changes Pass A.
- `--run-id NAME`: choose a readable new run ID. Existing IDs are never reused
  by `run` or `smoke`.
- `--yes`: skip the interactive confirmation before a paid action.
- `--json`: make `status` print machine-readable JSON.
- `--stage pass_a|pass_b`: limit `retry` to one failed stage.
- `--continue`: make `retry` continue the paid run immediately after appending
  retry events.
- `--max-attempts N`: advanced `resume` or `retry --continue` override for
  physical attempts during that continuation. It does not change model
  semantics.

Run any command with `--help` for its full explanation:

```bash
python -m two_pass_classifier smoke --help
```

## API key behavior

`build-manifest`, `cost-preview`, and `status` do not need an API key.

Paid actions load `OPENAI_API_KEY` only after the cost and configuration are
shown and the confirmation succeeds. Set the key in your environment or in
`keys/openai.env`:

```text
OPENAI_API_KEY=your_real_key
```

Environment variables take precedence over `keys/openai.env`.

## Relationship to the eval harness

`two_pass_classifier` is the only owner of classifier behavior. The
`evals` package imports its prompts, schemas, request builders, formatter and
cohort semantics, confidence extraction, supported models, defaults, output
caps, and normal Responses pricing. Evals separately owns the golden set,
three-model by three-effort matrix, Pass A banks, scoring, calibration,
dashboards, and archives.

Existing eval results created before this alignment, including the local
2026-07-27 sweep, used the prior prompt fingerprint. They remain valid
historical artifacts, but their Pass A banks are rejected for reuse. A new
aligned sweep starts with this offline cost gate:

```bash
OPENAI_API_KEY=placeholder python -m evals cost-preview
```

After reviewing that estimate, run the paid sweep with:

```bash
python -m evals run-evals
```

The first command makes no API calls. The second command rebuilds all three
Pass A banks, runs all nine Pass B cells with normal Responses calls, scores
them, and archives the resulting dashboard.
