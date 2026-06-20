# AGENTS.md

## Cursor Cloud specific instructions

This is a single-product Python 3.11+ CLI/batch pipeline (no servers, ports, or
databases). State is file-based under `outputs/` (gitignored). For commands,
conventions, and pipeline run order, see `README.md` and `coding_instructions.md`
rather than duplicating them here.

### Environment

- Dependencies install to the user site (`~/.local`) via `pip install -e ".[dev]"`
  (run automatically by the startup update script). Console scripts land in
  `~/.local/bin`, which is not on `PATH`, so invoke tools as modules:
  `python3 -m pytest` and `python3 classify.py ...`. Use `python3` (there is no
  `python` symlink).
- No linter is configured. `pyproject.toml` only declares the `pytest`/
  `pytest-asyncio` dev tools (the `ruff` mentioned in docs is not actually a
  dependency). "Lint" here means the test suite.

### Critical gotcha: OPENAI_API_KEY is required at import time

`src/config.py` reads `os.environ["OPENAI_API_KEY"]` at module import and raises
`KeyError` if it is unset. Because `src/tokens.py` imports `src/config.py` and
`tests/test_tokens.py` imports `src/tokens.py`, **the entire `pytest` collection
aborts when `OPENAI_API_KEY` is missing** (not just one test).

- A placeholder value (e.g. `OPENAI_API_KEY=placeholder`) is enough to run the
  full test suite and the offline pipeline stages, because none of them call the
  API: `python3 classify.py prepare [--dry-run]` and `status`/`merge` make no
  network calls. `prepare` writes the OpenAI Batch request JSONL + `state.json`
  locally.
- A real OpenAI key (Tier 5 for the documented rate limits) is only needed for
  the paid stages: `submit`, `run`, `download`, `retry`, and `test`.
- `TAVILY_API_KEY` is only needed for the enrichment stage
  (`scripts/run_tavily_crawl.py`); the test suite mocks Tavily.

Add `OPENAI_API_KEY` (and `TAVILY_API_KEY`) as Cursor secrets so they are
injected into future VMs automatically. Otherwise export a placeholder before
running tests. Keys are also loaded from `keys/openai.env` / `keys/tavily.env`
(gitignored) via `python-dotenv` if present; real environment variables take
precedence.

### Input data

The pipeline expects a Crunchbase-derived CSV. `data/` is gitignored and not in
the repo, so there is no input by default. Point any command at a CSV with the
documented columns via `--data <path.csv>` (see `src/formatter.py` for the
expected fields). Quick offline smoke test:

```bash
OPENAI_API_KEY=placeholder python3 classify.py prepare --data <path.csv> --dry-run
```
