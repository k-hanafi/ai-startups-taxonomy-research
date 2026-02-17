# AI-Native Startup Classification - 267,790 crunchbase startups

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
