"""
Merge all batch output files into final consolidated CSV
"""

import pandas as pd
import os
from datetime import datetime

print("="*70)
print("MERGING ALL BATCH OUTPUTS - GPT-5-NANO")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Find all batch output files
batch_files = sorted([f for f in os.listdir(".") if f.startswith("batch_") and f.endswith("_output.csv")])

if not batch_files:
    print("[ERROR] ERROR: No batch output files found!")
    exit(1)

print(f"\n Found {len(batch_files)} batch output files")

# Load and concatenate all batch results
all_results = []
total_rows = 0

for batch_file in batch_files:
    try:
        df = pd.read_csv(batch_file)
        all_results.append(df)
        total_rows += len(df)
        print(f"[OK] {batch_file}: {len(df):,} rows")
    except Exception as e:
        print(f"[ERROR] Failed to load {batch_file}: {e}")

# Concatenate all results
if all_results:
    print(f"\n[IN PROGRESS] Merging {len(all_results)} files...")
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Ensure correct column order
    column_order = ['CompanyID', 'CompanyName', 'AI_native', 'Confidence_1to5', 
                   'Reasons_3_points', 'Sources_used', 'Verification_critique']
    final_df = final_df[column_order]
    
    # Save final output
    output_file = "classified_startups_multi_batch.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("[OK] SUCCESS - MERGE COMPLETE!")
    print("="*70)
    print(f" Output file: {output_file}")
    print(f" Total rows: {len(final_df):,}")
    print(f" Columns: {', '.join(final_df.columns)}")
    
    # Print statistics
    ai_native_count = (final_df['AI_native'].astype(str) == '1').sum()
    not_ai_native_count = (final_df['AI_native'].astype(str) == '0').sum()
    
    print(f"\n CLASSIFICATION STATISTICS:")
    print(f"   AI-Native:     {ai_native_count:,} ({ai_native_count/len(final_df)*100:.1f}%)")
    print(f"   Not AI-Native: {not_ai_native_count:,} ({not_ai_native_count/len(final_df)*100:.1f}%)")
    
    # Check for duplicates
    duplicates = final_df['CompanyID'].duplicated().sum()
    if duplicates > 0:
        print(f"\n[WARNING]  Warning: {duplicates:,} duplicate CompanyIDs found")
    else:
        print(f"\n[OK] No duplicate CompanyIDs found")
    
    print("="*70)
    
else:
    print("\n[ERROR] ERROR: No results to merge!")

