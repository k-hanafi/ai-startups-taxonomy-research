#!/usr/bin/env python3
"""
Classification Analysis Script - GPT-5-mini vs GPT-5-nano
Week 7 Dataset (~270,000 startups)

Analyzes classification differences between two models and generates insights.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Paths
MINI_CSV = "../../GPT-5-mini/processing/classified_startups_multi_batch.csv"
NANO_CSV = "../../GPT-5-nano/processing/classified_startups_multi_batch.csv"
OUTPUT_DIR = "../03_Data_Files"

print("="*70)
print("GPT-5 MODEL COMPARISON ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/5] Loading datasets...")
mini_df = pd.read_csv(MINI_CSV)
nano_df = pd.read_csv(NANO_CSV)

print(f"  Mini dataset: {len(mini_df):,} startups")
print(f"  Nano dataset: {len(nano_df):,} startups")

# Ensure correct data types
mini_df['AI_native'] = pd.to_numeric(mini_df['AI_native'], errors='coerce')
nano_df['AI_native'] = pd.to_numeric(nano_df['AI_native'], errors='coerce')
mini_df['Confidence_1to5'] = pd.to_numeric(mini_df['Confidence_1to5'], errors='coerce')
nano_df['Confidence_1to5'] = pd.to_numeric(nano_df['Confidence_1to5'], errors='coerce')

# Find common startups
mini_ids = set(mini_df['CompanyID'])
nano_ids = set(nano_df['CompanyID'])
common_ids = mini_ids.intersection(nano_ids)

print(f"\n  Common startups: {len(common_ids):,}")
print(f"  Mini-only: {len(mini_ids - nano_ids):,}")
print(f"  Nano-only: {len(nano_ids - mini_ids):,}")

# Filter to common IDs and align
mini_common = mini_df[mini_df['CompanyID'].isin(common_ids)].copy()
nano_common = nano_df[nano_df['CompanyID'].isin(common_ids)].copy()

# Sort by CompanyID to ensure alignment
mini_common = mini_common.sort_values('CompanyID').reset_index(drop=True)
nano_common = nano_common.sort_values('CompanyID').reset_index(drop=True)

# Save aligned datasets
mini_common.to_csv(f"{OUTPUT_DIR}/mini_aligned.csv", index=False)
nano_common.to_csv(f"{OUTPUT_DIR}/nano_aligned.csv", index=False)
print(f"   Saved aligned datasets to {OUTPUT_DIR}/")

# ============================================================================
# 2. CALCULATE KEY METRICS
# ============================================================================

print("\n[2/5] Calculating key metrics...")

total_startups = len(mini_common)

# Classification metrics
mini_ai_count = mini_common['AI_native'].sum()
nano_ai_count = nano_common['AI_native'].sum()
mini_ai_rate = (mini_ai_count / total_startups) * 100
nano_ai_rate = (nano_ai_count / total_startups) * 100

# Agreement metrics
agreements = (mini_common['AI_native'] == nano_common['AI_native']).sum()
disagreements = total_startups - agreements
agreement_rate = (agreements / total_startups) * 100

# Confidence metrics
mini_conf_mean = mini_common['Confidence_1to5'].mean()
nano_conf_mean = nano_common['Confidence_1to5'].mean()
mini_conf_std = mini_common['Confidence_1to5'].std()
nano_conf_std = nano_common['Confidence_1to5'].std()

# Correlation
correlation = mini_common['Confidence_1to5'].corr(nano_common['Confidence_1to5'])

# Disagreement breakdown
disagreement_mask = mini_common['AI_native'] != nano_common['AI_native']
mini_ai_nano_not = ((mini_common['AI_native'] == 1) & (nano_common['AI_native'] == 0)).sum()
nano_ai_mini_not = ((nano_common['AI_native'] == 1) & (mini_common['AI_native'] == 0)).sum()

print(f"\n  CLASSIFICATION RATES:")
print(f"    Mini AI-native: {mini_ai_count:,} ({mini_ai_rate:.2f}%)")
print(f"    Nano AI-native: {nano_ai_count:,} ({nano_ai_rate:.2f}%)")
print(f"    Difference: {abs(nano_ai_rate - mini_ai_rate):.2f}% ({'+' if nano_ai_rate > mini_ai_rate else '-'})")

print(f"\n  AGREEMENT:")
print(f"    Agreement rate: {agreement_rate:.2f}% ({agreements:,} startups)")
print(f"    Disagreement rate: {100-agreement_rate:.2f}% ({disagreements:,} startups)")

print(f"\n  CONFIDENCE:")
print(f"    Mini mean: {mini_conf_mean:.2f} (σ={mini_conf_std:.2f})")
print(f"    Nano mean: {nano_conf_mean:.2f} (σ={nano_conf_std:.2f})")
print(f"    Correlation: {correlation:.3f}")

print(f"\n  DISAGREEMENTS:")
print(f"    Mini→AI, Nano→Not: {mini_ai_nano_not:,} ({mini_ai_nano_not/disagreements*100:.1f}%)")
print(f"    Nano→AI, Mini→Not: {nano_ai_mini_not:,} ({nano_ai_mini_not/disagreements*100:.1f}%)")

# ============================================================================
# 3. CREATE DISAGREEMENT DATASET
# ============================================================================

print("\n[3/5] Creating disagreement dataset...")

disagreement_df = pd.DataFrame({
    'CompanyID': mini_common[disagreement_mask]['CompanyID'],
    'CompanyName': mini_common[disagreement_mask]['CompanyName'],
    'Mini_AI_native': mini_common[disagreement_mask]['AI_native'],
    'Nano_AI_native': nano_common[disagreement_mask]['AI_native'],
    'Mini_Confidence': mini_common[disagreement_mask]['Confidence_1to5'],
    'Nano_Confidence': nano_common[disagreement_mask]['Confidence_1to5'],
    'Mini_Reasons': mini_common[disagreement_mask]['Reasons_3_points'],
    'Nano_Reasons': nano_common[disagreement_mask]['Reasons_3_points'],
    'Disagreement_Type': [
        'Mini→AI, Nano→Not' if m == 1 else 'Nano→AI, Mini→Not'
        for m in mini_common[disagreement_mask]['AI_native']
    ]
}).reset_index(drop=True)

disagreement_df.to_csv(f"{OUTPUT_DIR}/disagreement_dataset.csv", index=False)
print(f"   Saved {len(disagreement_df):,} disagreements to {OUTPUT_DIR}/disagreement_dataset.csv")

# ============================================================================
# 4. AGREEMENT BY CONFIDENCE LEVEL
# ============================================================================

print("\n[4/5] Analyzing agreement by confidence level...")

# For each confidence level, calculate agreement rate
agreement_by_conf = []
for conf_level in range(1, 6):
    # Mini at this confidence level
    mini_mask = mini_common['Confidence_1to5'] == conf_level
    if mini_mask.sum() > 0:
        mini_agree_rate = (mini_common[mini_mask]['AI_native'] == nano_common[mini_mask]['AI_native']).mean() * 100
        agreement_by_conf.append({
            'Model': 'Mini',
            'Confidence_Level': conf_level,
            'Agreement_Rate': mini_agree_rate,
            'Count': mini_mask.sum()
        })
    
    # Nano at this confidence level
    nano_mask = nano_common['Confidence_1to5'] == conf_level
    if nano_mask.sum() > 0:
        nano_agree_rate = (mini_common[nano_mask]['AI_native'] == nano_common[nano_mask]['AI_native']).mean() * 100
        agreement_by_conf.append({
            'Model': 'Nano',
            'Confidence_Level': conf_level,
            'Agreement_Rate': nano_agree_rate,
            'Count': nano_mask.sum()
        })

agreement_by_conf_df = pd.DataFrame(agreement_by_conf)
agreement_by_conf_df.to_csv(f"{OUTPUT_DIR}/agreement_by_confidence.csv", index=False)
print(f"   Saved confidence analysis to {OUTPUT_DIR}/agreement_by_confidence.csv")

# ============================================================================
# 5. SAVE SUMMARY METRICS
# ============================================================================

print("\n[5/5] Saving summary metrics...")

metrics = {
    'total_startups': total_startups,
    'mini_ai_count': int(mini_ai_count),
    'nano_ai_count': int(nano_ai_count),
    'mini_ai_rate': round(mini_ai_rate, 2),
    'nano_ai_rate': round(nano_ai_rate, 2),
    'agreements': int(agreements),
    'disagreements': int(disagreements),
    'agreement_rate': round(agreement_rate, 2),
    'mini_conf_mean': round(mini_conf_mean, 2),
    'nano_conf_mean': round(nano_conf_mean, 2),
    'mini_conf_std': round(mini_conf_std, 2),
    'nano_conf_std': round(nano_conf_std, 2),
    'correlation': round(correlation, 3),
    'mini_ai_nano_not': int(mini_ai_nano_not),
    'nano_ai_mini_not': int(nano_ai_mini_not)
}

# Save as JSON for easy dashboard loading
import json
with open(f"{OUTPUT_DIR}/summary_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"   Saved summary metrics to {OUTPUT_DIR}/summary_metrics.json")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nNext steps:")
print(f"  1. Run generate_visualizations.py to create charts")
print(f"  2. Run build_final_dashboard.py for full dashboard")
print(f"  3. View results in 01_Presentation_Materials/charts/mckinsey_dashboard_final.html")

