# Quick Reference: Local Setup

## One-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/k-hanafi/ai-startups-taxonomy-research.git
cd ai-startups-taxonomy-research

# 2. Set up Python environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. Configure API keys
mkdir -p keys
echo 'OPENAI_API_KEY=your_key_here' > keys/openai.env
echo 'TAVILY_API_KEY=your_key_here' > keys/tavily.env

# 4. Verify (pytest works on a fresh clone; no data files required)
OPENAI_API_KEY=placeholder pytest
```

## Daily Workflow

```bash
# Start your session
cd ai-startups-taxonomy-research
source .venv/bin/activate

# Sync with remote (updates your current branch)
git fetch origin
git pull origin "$(git branch --show-current)"

# Check status
git status
git log --oneline -5

# Run tests
pytest

# Work on your changes...

# Commit and push
git add .
git commit -m "Your descriptive message"
git push origin "$(git branch --show-current)"
```

## Helper Scripts

```bash
# Sync with remote (interactive)
./scripts/sync_with_remote.sh

# Update website liveness
python scripts/update_website_liveness.py

# Run Tavily crawl
python scripts/run_tavily_crawl.py
```

## Common Tasks

### Run Classification Pipeline
```bash
python classify.py prepare --dry-run  # Cost estimate
python classify.py status             # Check progress
python classify.py run                # Full run
```

### Survivorship Pipeline
```bash
# See wayback_machine/README.md for full details
python wayback_machine/scripts/probe_death_coverage.py
python wayback_machine/scripts/build_targets_dead.py
python wayback_machine/scripts/run_crawl_dead.py
```

### Testing
```bash
pytest                        # Live pipeline tests
pytest wayback_machine/tests  # Wayback tests
pytest -v                     # Verbose output
pytest tests/test_schema.py   # Specific test file
```

## Key Files to Read

1. **AGENTS.md** - Project architecture, data flow, commands
2. **LOCAL_SETUP.md** - Detailed setup guide (this file is the quick reference)
3. **wayback_machine/README.md** - Survivorship pipeline stages
4. **plans/** - Detailed project plans

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not set` | `source keys/openai.env` or use placeholder for tests |
| Module not found | `pip install -e ".[dev]"` |
| Git conflicts | `git fetch origin && git status` |
| Tests fail | Check virtual environment is activated |
| Can't find data files | Check `.gitignore` - they're not in git |

## Repository Structure

```
/
├── src/                    # Live classification pipeline
│   ├── config.py          # Tunables, rate limits, costs
│   ├── schema.py          # ClassificationResult model
│   └── ...
├── wayback_machine/        # Historical + survivorship
│   ├── scripts/           # CDX probe, crawl, merge
│   └── tests/             # Wayback tests
├── scripts/                # Network operations
├── data/                   # Git-ignored: input data
├── outputs/                # Git-ignored: results
├── keys/                   # Git-ignored: API keys
├── tests/                  # Live pipeline tests
└── AGENTS.md              # The authoritative guide
```

## Important Conventions

- **No magic numbers** outside `src/config.py` or `wayback_machine/config.py`
- **Match by `custom_id`**, never by position (batch results)
- **Only `website_evidence` differs** between strands
- **CDX rate limit:** Max 58 req/min (60 risks IP ban)
- **Network operations:** Run outside Cursor sandbox locally

## Getting Help

- Read `AGENTS.md` for architecture
- Check test files for usage examples
- Run commands with `--help`: `python classify.py --help`
- Ask the cloud agent for clarification

---

**Current focus:** `main` — survivorship dead-cohort pipeline is merged; next step is running Stage C crawl (see `plans/tavily_runner_workflow.md`).
