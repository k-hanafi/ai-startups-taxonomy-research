"""
Check status of all batches for GPT-5-nano
"""

from openai import OpenAI
from datetime import datetime

# Read API key
with open("../api_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

print("="*70)
print("GPT-5-NANO BATCH STATUS CHECK")
print("="*70)
print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    batches = client.batches.list(limit=20)
    
    print(f"Found {len(batches.data)} batches:")
    print()
    
    for batch in batches.data:
        print(f"Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Created: {batch.created_at}")
        
        if hasattr(batch, 'request_counts') and batch.request_counts:
            print(f"  Total: {batch.request_counts.total}")
            print(f"  Completed: {batch.request_counts.completed}")
            print(f"  Failed: {batch.request_counts.failed}")
        
        print()
    
except Exception as e:
    print(f"Error: {e}")
