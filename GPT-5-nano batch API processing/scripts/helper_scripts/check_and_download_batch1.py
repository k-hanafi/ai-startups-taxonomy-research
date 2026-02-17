"""
Quick script to check batch 1 status and download results if available
"""

from openai import OpenAI
import json
import csv
import io
import pandas as pd

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

print(f"Checking batch 1: {batch_id}")
print("="*70)

# Check status
batch = client.batches.retrieve(batch_id)
print(f"Status: {batch.status}")
print(f"Request counts:")
print(f"  - Total: {batch.request_counts.total}")
print(f"  - Completed: {batch.request_counts.completed}")
print(f"  - Failed: {batch.request_counts.failed}")

if batch.status == "completed":
    print("\n[OK] Batch is completed! Downloading results...")
    
    # Get output file ID
    output_file_id = batch.output_file_id
    
    if not output_file_id:
        print("[ERROR] No output file ID found!")
    else:
        # Download the results
        result = client.files.content(output_file_id)
        result_content = result.content
        
        # Save raw JSONL
        with open("batch_1_results.jsonl", "wb") as f:
            f.write(result_content)
        
        print(f"[OK] Downloaded batch_1_results.jsonl")
        
        # Parse JSONL and convert to CSV
        def parse_classification_result(content):
            """Parse AI response into structured fields"""
            result = {
                'CompanyID': '',
                'CompanyName': '',
                'AI_native': '',
                'Confidence_1to5': '',
                'Reasons_3_points': '',
                'Sources_used': '',
                'Verification_critique': ''
            }
            
            try:
                csv_reader = csv.reader(io.StringIO(content.strip()))
                rows = list(csv_reader)
                
                if rows and len(rows) > 0:
                    data_row = rows[0]
                    
                    if len(data_row) >= 7:
                        result['CompanyID'] = data_row[0].strip()
                        result['CompanyName'] = data_row[1].strip()
                        result['AI_native'] = data_row[2].strip()
                        result['Confidence_1to5'] = data_row[3].strip()
                        result['Reasons_3_points'] = data_row[4].strip()
                        result['Sources_used'] = data_row[5].strip()
                        result['Verification_critique'] = data_row[6].strip()
                        
            except Exception as e:
                print(f"Warning: Failed to parse result: {e}")
            
            return result
        
        parsed_results = []
        
        with open("batch_1_results.jsonl", "r") as f:
            for line in f:
                try:
                    response_obj = json.loads(line.strip())
                    ai_response = response_obj["response"]["body"]["choices"][0]["message"]["content"]
                    parsed = parse_classification_result(ai_response)
                    
                    if parsed['CompanyID']:
                        parsed_results.append(parsed)
                        
                except Exception as e:
                    print(f"Warning: Failed to parse line: {e}")
        
        # Save to CSV
        if parsed_results:
            results_df = pd.DataFrame(parsed_results)
            column_order = ['CompanyID', 'CompanyName', 'AI_native', 'Confidence_1to5', 
                          'Reasons_3_points', 'Sources_used', 'Verification_critique']
            results_df = results_df[column_order]
            results_df.to_csv("batch_1_output.csv", index=False)
            print(f"[OK] Saved {len(parsed_results)} results to batch_1_output.csv")
        else:
            print("[ERROR] No valid results parsed")
            
elif batch.status == "failed":
    print("\n[ERROR] Batch failed!")
    if hasattr(batch, 'errors') and batch.errors:
        print(f"Errors: {batch.errors}")
elif batch.status == "expired":
    print("\n[ERROR] Batch expired!")
elif batch.status == "cancelled":
    print("\n[ERROR] Batch was cancelled!")
else:
    print(f"\n[PENDING] Batch is still processing (status: {batch.status})")
    print("Please wait and run this script again later.")

