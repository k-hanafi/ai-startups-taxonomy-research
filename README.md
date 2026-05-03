# AI Startups Taxonomy Research - 267,790 crunchbase startups

1. **Large-scale Python classification pipeline of Crunchbase Data** to determine if AI/ML is fundamental to the value proposition of startups.

2. **Conduct a comparative statistical analysis** of GPT-5-mini and GPT-5-nano models, evaluating their reasoning quality, classification characteristics, and cost-effectiveness. Beyond simple accuracy metrics, I analyzed agreement patterns, confidence distributions, correlations between model outputs, and the specific nature of disagreements to understand each model's "personality" and the practical trade-offs for production use. 

### Key Files

- **Statistical Analysis Dashboard**: `data visualization/01_Presentation_Materials/dashboard.html`
- **Main Analysis Script**: `data visualization/02_Analysis_Code/classification_analysis.py`
- **Batch Processing Scripts**: 
  - `GPT-5-mini batch API processing/scripts/MTA_multi_batch_gpt5_mini.py`
  - `GPT-5-nano batch API processing/scripts/MTA_multi_batch_gpt5_nano.py`
- **CSV Results**: 
  - `GPT-5-mini batch API processing/output/concatenated_batches_gpt5_mini.csv`
  - `GPT-5-nano batch API processing/output/classified_startups_gpt5_nano.csv`

### Tavily-Enriched Classification Workflow

Generated artifacts are organized under `outputs/`:

- `outputs/tavilycrawl/`: enriched 44k input CSV, Tavily crawl queue, raw crawl JSONL, crawl state, and classifier input with website evidence.
- `outputs/batch_data/`: OpenAI batch request JSONL, downloaded raw results, per-batch CSVs, errors, and batch state.
- `outputs/production_csvs/`: final research CSVs such as `classified_startups_v2.csv`, `classified_startups_v21_migrated.csv`, and new Tavily-enriched classifier outputs.
- `outputs/logs/`: runtime logs.

Run order:

```bash
python scripts/prepare_tavily_enrichment.py
python scripts/run_tavily_crawl.py --budget-credits 100000
python scripts/build_website_evidence.py
python classify.py prepare
python classify.py submit
python classify.py download
python classify.py merge
```
