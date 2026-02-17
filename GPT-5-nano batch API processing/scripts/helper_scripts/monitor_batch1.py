"""
Monitor batch 1 status in real-time
"""

from openai import OpenAI
from datetime import datetime
import time

# Read API key
with open("../api_key.txt", "r") as f:
    lines = f.readlines()
    api_key = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            api_key = line
            break

client = OpenAI(api_key=api_key)

# Read batch 1 ID
with open("batch_1_id.txt", "r") as f:
    batch_id = f.read().strip()

print("="*70)
print("MONITORING BATCH 1")
print("="*70)
print(f"Batch ID: {batch_id}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nPress Ctrl+C to stop monitoring (batch will continue on OpenAI's servers)")
print("="*70)

check_count = 0
try:
    while True:
        check_count += 1
        
        # Check status
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        # Clear screen effect with separator
        print(f"\n{''*70}")
        print(f"CHECK #{check_count} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{''*70}")
        
        # Status indicator
        if status == "completed":
            print(f"Status: [COMPLETED]")
        elif status == "failed":
            print(f"Status: [FAILED]")
        elif status == "in_progress":
            print(f"Status: [IN PROGRESS]")
        elif status == "validating":
            print(f"Status: [VALIDATING]")
        elif status == "finalizing":
            print(f"Status: [FINALIZING]")
        else:
            print(f"Status: {status.upper()}")
        
        # Progress
        if hasattr(batch, 'request_counts') and batch.request_counts:
            total = batch.request_counts.total
            completed = batch.request_counts.completed
            failed = batch.request_counts.failed
            
            if total > 0:
                percent = (completed / total) * 100
                print(f"\nProgress:")
                print(f"  Total:     {total:,}")
                print(f"  Completed: {completed:,} ({percent:.1f}%)")
                print(f"  Failed:    {failed:,}")
                
                # Progress bar
                bar_width = 50
                filled = int(bar_width * percent / 100)
                bar = "" * filled + "" * (bar_width - filled)
                print(f"  [{bar}] {percent:.1f}%")
        
        # Check if done
        if status in ["completed", "failed", "expired", "cancelled"]:
            print("\n" + "="*70)
            if status == "completed":
                print("[SUCCESS] BATCH 1 COMPLETED!")
                print("\nNext step: Run the following to download results:")
                print("  python check_and_download_batch1.py")
            else:
                print(f"[WARNING] BATCH ENDED WITH STATUS: {status.upper()}")
            print("="*70)
            break
        
        # Wait before next check
        print(f"\nNext check in 60 seconds... (Ctrl+C to stop)")
        time.sleep(60)
        
except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("[WARNING] Monitoring stopped by user")
    print("="*70)
    print(f"Batch ID: {batch_id}")
    print(f"Last status: {status}")
    print("\nThe batch continues processing on OpenAI's servers.")
    print("Run this script again anytime to check progress.")
    print("="*70)

