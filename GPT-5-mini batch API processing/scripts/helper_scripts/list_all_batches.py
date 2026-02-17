"""
List all batches in your OpenAI account
"""

from openai import OpenAI

# Read API key
with open("../api_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

print("Fetching all batches in your account...\n")

try:
    # List all batches
    batches = client.batches.list(limit=20)
    
    if not batches.data:
        print("No batches found.")
    else:
        print(f"Found {len(batches.data)} batch(es):\n")
        
        for i, batch in enumerate(batches.data, 1):
            print(f"{i}. Batch ID: {batch.id}")
            print(f"   Status: {batch.status.upper()}")
            
            if hasattr(batch, 'request_counts') and batch.request_counts:
                print(f"   Requests: {batch.request_counts.total:,}")
                print(f"   Completed: {batch.request_counts.completed:,}")
                print(f"   Failed: {batch.request_counts.failed:,}")
            
            if hasattr(batch, 'created_at'):
                from datetime import datetime
                created = datetime.fromtimestamp(batch.created_at)
                print(f"   Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if batch.status in ['in_progress', 'validating', 'cancelling']:
                print(f"   [WARNING]  This batch is occupying your token queue!")
            
            print()
    
except Exception as e:
    print(f"Error: {e}")

