# AI-Native Startup Classification - 267,790 crunchbase startups

1. **Large-scale Python classification pipeline of Crunchbase Data** to determine if AI/ML is fundamental to the value proposition of startups.

2. **Results dashboard**: Standalone HTML built from merged classification output (`outputs/classified_startups_v2.csv`).

### Key Files

- **Classification results dashboard**: `data visualization/01_Presentation_Materials/v2_dashboard.html` (rebuild with `python "data visualization/02_Analysis_Code/build_v2_dashboard.py"`)
- **Batch Processing Scripts**: 
  - `GPT-5-mini batch API processing/scripts/MTA_multi_batch_gpt5_mini.py`
  - `GPT-5-nano batch API processing/scripts/MTA_multi_batch_gpt5_nano.py`
- **CSV Results**: 
  - `GPT-5-mini batch API processing/output/concatenated_batches_gpt5_mini.csv`
  - `GPT-5-nano batch API processing/output/classified_startups_gpt5_nano.csv`
