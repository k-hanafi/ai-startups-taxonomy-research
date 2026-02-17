#!/usr/bin/env python3
"""
Comprehensive batch status checker for GPT-5-mini
Shows status of all batches, their progress, and what needs to be done next.
"""

from openai import OpenAI
import os
from datetime import datetime, timedelta
import glob

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

print("="*70)
print("COMPREHENSIVE BATCH STATUS CHECK - GPT-5-MINI")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Find all batch ID files
batch_id_files = sorted(glob.glob("batch_*_id.txt"), key=lambda x: int(x.replace('batch_', '').replace('_id.txt', '')))
# Find all JSONL request files
jsonl_files = sorted(glob.glob("batch_*_requests.jsonl"), key=lambda x: int(x.replace('batch_', '').replace('_requests.jsonl', '')))
# Find all result files
result_files = sorted(glob.glob("batch_*_results.jsonl"), key=lambda x: int(x.replace('batch_', '').replace('_results.jsonl', '')))
# Find all output CSV files
output_files = sorted(glob.glob("batch_*_output.csv"), key=lambda x: int(x.replace('batch_', '').replace('_output.csv', '')))

print(f"Found {len(batch_id_files)} batch ID files")
print(f"Found {len(jsonl_files)} JSONL request files")
print(f"Found {len(result_files)} result files")
print(f"Found {len(output_files)} output CSV files\n")

# Check each batch
batches_info = []

for batch_id_file in batch_id_files:
    batch_num = batch_id_file.replace('batch_', '').replace('_id.txt', '')
    
    with open(batch_id_file, 'r') as f:
        batch_id = f.read().strip()
    
    try:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        info = {
            'batch_num': batch_num,
            'batch_id': batch_id,
            'status': status,
            'has_jsonl': f"batch_{batch_num}_requests.jsonl" in jsonl_files,
            'has_results': f"batch_{batch_num}_results.jsonl" in result_files,
            'has_output': f"batch_{batch_num}_output.csv" in output_files,
        }
        
        if hasattr(batch, 'request_counts') and batch.request_counts:
            info['total'] = batch.request_counts.total
            info['completed'] = batch.request_counts.completed
            info['failed'] = batch.request_counts.failed
            info['remaining'] = info['total'] - info['completed'] - info['failed']
        
        if hasattr(batch, 'created_at'):
            info['created_at'] = datetime.fromtimestamp(batch.created_at)
            info['elapsed'] = datetime.now() - info['created_at']
        
        if hasattr(batch, 'completed_at') and batch.completed_at:
            info['completed_at'] = datetime.fromtimestamp(batch.completed_at)
        
        batches_info.append(info)
        
    except Exception as e:
        print(f"[WARNING]  Batch {batch_num}: Error retrieving - {e}")
        batches_info.append({
            'batch_num': batch_num,
            'status': 'error',
            'error': str(e)
        })

# Display summary
print("\n" + "="*70)
print("BATCH STATUS SUMMARY")
print("="*70)

completed_count = 0
in_progress_count = 0
failed_count = 0
cancelled_count = 0
finalizing_count = 0

for info in batches_info:
    batch_num = info['batch_num']
    status = info.get('status', 'unknown').upper()
    
    print(f"\n Batch {batch_num}: {status}")
    
    if status == 'COMPLETED':
        completed_count += 1
        print(f"   [OK] Completed successfully")
        if 'completed' in info:
            print(f"   Progress: {info['completed']:,}/{info['total']:,} requests")
        if 'has_results' in info and info['has_results']:
            print(f"   [OK] Results downloaded")
        if 'has_output' in info and info['has_output']:
            print(f"   [OK] Output CSV created")
        if 'completed_at' in info:
            print(f"   Completed: {info['completed_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    elif status == 'IN_PROGRESS':
        in_progress_count += 1
        print(f"   [PENDING] Currently processing...")
        if 'completed' in info:
            progress_pct = (info['completed'] / info['total'] * 100) if info['total'] > 0 else 0
            print(f"   Progress: {info['completed']:,}/{info['total']:,} ({progress_pct:.1f}%)")
            print(f"   Remaining: {info['remaining']:,} requests")
            print(f"   Failed: {info['failed']:,} requests")
        
        if 'elapsed' in info:
            print(f"   Elapsed time: {info['elapsed']}")
        
        if 'completed' in info and info['completed'] > 0 and 'elapsed' in info:
            elapsed_seconds = info['elapsed'].total_seconds()
            if elapsed_seconds > 0:
                rate = info['completed'] / elapsed_seconds
                if rate > 0 and 'remaining' in info:
                    eta_seconds = info['remaining'] / rate
                    eta = timedelta(seconds=int(eta_seconds))
                    print(f"   Processing rate: {rate:.2f} req/sec")
                    print(f"   Estimated time remaining: {eta}")
    
    elif status == 'FINALIZING':
        finalizing_count += 1
        print(f"   [IN PROGRESS] Finalizing (almost done!)")
        if 'completed' in info:
            progress_pct = (info['completed'] / info['total'] * 100) if info['total'] > 0 else 0
            print(f"   Progress: {info['completed']:,}/{info['total']:,} ({progress_pct:.1f}%)")
        if 'elapsed' in info:
            print(f"   Elapsed time: {info['elapsed']}")
    
    elif status == 'FAILED':
        failed_count += 1
        print(f"   [ERROR] Failed")
        if 'failed' in info:
            print(f"   Failed requests: {info['failed']:,}")
    
    elif status == 'CANCELLED':
        cancelled_count += 1
        print(f"   [WARNING]  Cancelled")
        if 'completed' in info:
            print(f"   Progress before cancellation: {info['completed']:,}/{info['total']:,}")
    
    else:
        print(f"   Status: {status}")

# Check for batches that have JSONL files but no batch ID (not uploaded yet)
jsonl_batch_nums = set([int(f.replace('batch_', '').replace('_requests.jsonl', '')) for f in jsonl_files])
uploaded_batch_nums = set([int(info['batch_num']) for info in batches_info if 'status' in info])
not_uploaded = jsonl_batch_nums - uploaded_batch_nums

if not_uploaded:
    print(f"\n" + "="*70)
    print("BATCHES READY TO UPLOAD (JSONL exists but not uploaded)")
    print("="*70)
    for batch_num in sorted(not_uploaded):
        jsonl_file = f"batch_{batch_num}_requests.jsonl"
        file_size = os.path.getsize(jsonl_file) / (1024 * 1024)
        print(f"    Batch {batch_num}: {jsonl_file} ({file_size:.1f} MB) - Ready to upload")

# Summary
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"[OK] Completed: {completed_count}")
print(f"[IN PROGRESS] Finalizing: {finalizing_count}")
print(f"[PENDING] In Progress: {in_progress_count}")
print(f"[ERROR] Failed: {failed_count}")
print(f"[WARNING]  Cancelled: {cancelled_count}")
if not_uploaded:
    print(f" Ready to Upload: {len(not_uploaded)}")

print(f"\n" + "="*70)
print("NEXT STEPS")
print("="*70)

if in_progress_count > 0 or finalizing_count > 0:
    print("1. Wait for in-progress/finalizing batches to complete")
if not_uploaded:
    print(f"2. Upload {len(not_uploaded)} batch(es) that are ready")
    print("   Run: python3 MTA_multi_batch_gpt5_mini.py")
if completed_count > 0:
    # Check if any completed batches don't have results
    for info in batches_info:
        if info.get('status') == 'completed':
            if not info.get('has_results', False):
                print(f"3. Download results for batch {info['batch_num']}")
if failed_count > 0 or cancelled_count > 0:
    print("4. Review failed/cancelled batches and decide if retry is needed")

print()

