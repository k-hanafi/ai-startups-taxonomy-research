#!/usr/bin/env python3
"""
Quick script to check if a batch is making progress.
Run this to see if the batch is truly stuck or just slow.
"""

from openai import OpenAI
import os
import time
from datetime import datetime

# Read API key
with open("../api_key.txt", "r") as f:
    lines = f.readlines()
    api_key = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            api_key = line
            break

if not api_key:
    print("[ERROR] API key not found")
    exit(1)

client = OpenAI(api_key=api_key)

# Find the latest batch
batch_files = sorted([f for f in os.listdir('.') if f.startswith('batch_') and f.endswith('_id.txt')])
if not batch_files:
    print("No batch ID files found")
    exit(1)

latest_batch_file = batch_files[-1]
batch_num = latest_batch_file.replace('batch_', '').replace('_id.txt', '')

with open(latest_batch_file, 'r') as f:
    batch_id = f.read().strip()

print(f"=== MONITORING BATCH {batch_num} PROGRESS ===\n")
print("Checking progress every 2 minutes...")
print("Press Ctrl+C to stop\n")

previous_completed = None
stuck_count = 0

while True:
    try:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        if hasattr(batch, 'request_counts') and batch.request_counts:
            total = batch.request_counts.total
            completed = batch.request_counts.completed
            failed = batch.request_counts.failed
            remaining = total - completed - failed
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] Status: {status.upper()} | Progress: {completed:,}/{total:,} ({completed/total*100:.1f}%) | Remaining: {remaining:,}")
            
            # Check if progress is stuck
            if previous_completed is not None:
                if completed == previous_completed:
                    stuck_count += 1
                    print(f"         [WARNING]  No progress for {stuck_count} checks")
                    if stuck_count >= 5:  # 10 minutes with no progress
                        print(f"\n[WARNING]  WARNING: Batch appears stuck (no progress for 10+ minutes)")
                        print(f"   Consider canceling if this continues")
                else:
                    stuck_count = 0
                    progress = completed - previous_completed
                    print(f"         [OK] Progress: +{progress:,} requests")
            
            previous_completed = completed
            
            if status == "completed":
                print(f"\n[OK] Batch completed!")
                break
            elif status == "failed":
                print(f"\n[ERROR] Batch failed!")
                break
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status.upper()}")
        
        time.sleep(120)  # Check every 2 minutes
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(120)

