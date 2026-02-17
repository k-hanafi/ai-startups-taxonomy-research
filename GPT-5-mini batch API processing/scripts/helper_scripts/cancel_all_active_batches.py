"""
Cancel ALL active batches immediately
"""

from openai import OpenAI

# Read API key
with open("../api_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

print("Fetching all batches...")

try:
    batches = client.batches.list(limit=20)
    
    cancelled_count = 0
    
    for batch in batches.data:
        if batch.status in ['validating', 'in_progress', 'finalizing']:
            print(f"\nCancelling batch: {batch.id}")
            print(f"  Status: {batch.status}")
            
            try:
                client.batches.cancel(batch.id)
                print(f"  [OK] Cancelled!")
                cancelled_count += 1
            except Exception as e:
                print(f"  [ERROR] Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Total batches cancelled: {cancelled_count}")
    print(f"{'='*50}")
    
except Exception as e:
    print(f"Error: {e}")

