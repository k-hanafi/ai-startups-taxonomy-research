"""
Check status of all batches
"""

from openai import OpenAI
import os

# Read API key
with open("../api_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# Check all batch ID files
for i in range(1, 6):
    batch_id_file = f"batch_{i}_id.txt"
    
    if not os.path.exists(batch_id_file):
        print(f"Batch {i}: Not uploaded yet\n")
        continue
    
    with open(batch_id_file, "r") as f:
        batch_id = f.read().strip()
    
    try:
        batch = client.batches.retrieve(batch_id)
        
        print(f"Batch {i}:")
        print(f"  ID: {batch_id}")
        print(f"  Status: {batch.status.upper()}")
        
        if hasattr(batch, 'request_counts') and batch.request_counts:
            print(f"  Requests: {batch.request_counts.total:,}")
            print(f"  Completed: {batch.request_counts.completed:,}")
            print(f"  Failed: {batch.request_counts.failed:,}")
        
        if hasattr(batch, 'errors') and batch.errors:
            print(f"  ERRORS: {batch.errors}")
        
        print()
        
    except Exception as e:
        print(f"Batch {i}: Error - {e}\n")

