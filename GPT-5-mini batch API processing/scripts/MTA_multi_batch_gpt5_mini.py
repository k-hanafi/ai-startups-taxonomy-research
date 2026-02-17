"""
Multi-Batch Startup Classification Script - FULL DATASET

Splits the dataset into multiple batches, uploads them sequentially to OpenAI,
monitors progress, and merges results into a single CSV file.

This script handles the full lifecycle of batch processing:
1. Splits the large input CSV into smaller chunks (batches) to respect API and file size limits.
2. Uploads each batch to the OpenAI Batch API sequentially.
3. Monitors the status of each batch until completion.
4. Downloads the results and merges them into a final output file.
"""

import os
import json
import time
import csv
import io
import re
from datetime import datetime
import pandas as pd
from openai import OpenAI

# Read API key from external file to ensure security
try:
    with open("../api_key.txt", "r") as f:
        # Read non-empty lines that don't start with #
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        if not lines:
            raise ValueError("No API key found in api_key.txt")
        api_key = lines[0]
except FileNotFoundError:
    raise FileNotFoundError("api_key.txt not found")

client = OpenAI(api_key=api_key)

# Configuration
INPUT_CSV = "../../company_us_short_long_desc_.csv"
OUTPUT_CSV = "classified_startups_gpt5_mini.csv"
SYSTEM_PROMPT_FILE = "../system_prompt.txt"
MODEL_NAME = "gpt-5-mini"

# OpenAI Batch API limits
MAX_REQUESTS_PER_BATCH = 50000
MAX_FILE_SIZE_MB = 100

# Token limits
MAX_ENQUEUED_TOKENS = 40000000
ESTIMATED_TOKENS_PER_REQUEST = 3600

print(f"Starting classification with {MODEL_NAME}")
print(f"Input: {INPUT_CSV}")
print(f"Output: {OUTPUT_CSV}")

def extract_year_from_date(date_str):
    """Extracts a 4-digit year from various date string formats."""
    if pd.isna(date_str) or date_str == '' or date_str == 'N/A':
        return 'N/A'
    
    date_str = str(date_str).strip()
    # Look for 4 digits starting with 19 or 20
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    return year_match.group(0) if year_match else 'N/A'

def format_user_message(row):
    """
    Formats a single startup's data into the text prompt for the API.
    Constructs a structured string with ID, name, descriptions, keywords, and year.
    """
    # Safely get values handling NaNs
    def get_val(key, default='N/A'):
        val = row.get(key, default)
        return 'N/A' if pd.isna(val) else str(val).strip()

    company_id = get_val('org_uuid')
    company_name = get_val('name')
    
    short_desc = row.get('short_description', '')
    short_desc = str(short_desc).strip() if not pd.isna(short_desc) else 'N/A'
    
    long_desc = row.get('Long description', '')
    long_desc = str(long_desc).strip() if not pd.isna(long_desc) else 'N/A'
    
    # Combine category lists into keywords
    cat_list = str(row.get('category_list', '')).strip() if not pd.isna(row.get('category_list')) else ''
    cat_groups = str(row.get('category_groups_list', '')).strip() if not pd.isna(row.get('category_groups_list')) else ''
    
    if cat_list and cat_groups:
        keywords = f"{cat_list}, {cat_groups}"
    else:
        keywords = cat_list or cat_groups or 'N/A'
    
    year_founded = extract_year_from_date(row.get('founded_date'))
    
    return f"""INPUT:
CompanyID: {company_id}
CompanyName: {company_name}
Short Description: {short_desc}
Long Description: {long_desc}
Keywords: {keywords}
YearFounded: {year_founded}"""

def calculate_batch_sizes():
    """
    Calculates optimal batch size based on API limits.
    Considers token limits, file size limits, and max requests per batch.
    """
    print("Calculating batch sizes...")
    df = pd.read_csv(INPUT_CSV)
    total_startups = len(df)
    print(f"Total startups: {total_startups:,}")
    
    # Calculate limit based on tokens (e.g. 40M tokens queue)
    max_requests_tokens = MAX_ENQUEUED_TOKENS // ESTIMATED_TOKENS_PER_REQUEST
    
    # Calculate limit based on file size (approx 14.2KB per request)
    estimated_mb_per_request = 14.2 / 1024
    max_requests_size = int(MAX_FILE_SIZE_MB / estimated_mb_per_request)
    
    # Use the most restrictive limit to ensure stability
    batch_size = min(max_requests_tokens, max_requests_size, MAX_REQUESTS_PER_BATCH)
    print(f"Batch size: {batch_size:,}")
    
    num_batches = (total_startups + batch_size - 1) // batch_size
    return num_batches, total_startups // num_batches

def create_batch_file(batch_num, total_batches, batch_size):
    """Creates a JSONL file for a specific batch of startups."""
    jsonl_filename = f"batch_files/batch_requests/batch_{batch_num}_requests.jsonl"
    
    if os.path.exists(jsonl_filename):
        print(f"Batch {batch_num} file already exists")
        return jsonl_filename
        
    print(f"Creating batch {batch_num} file...")
    
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        system_prompt = f.read().strip()
        
    df = pd.read_csv(INPUT_CSV)
    start_idx = (batch_num - 1) * batch_size
    end_idx = len(df) if batch_num == total_batches else start_idx + batch_size
    
    batch_df = df.iloc[start_idx:end_idx]
    
    with open(jsonl_filename, "w") as f:
        for idx, row in batch_df.iterrows():
            user_msg = format_user_message(row)
            
            # Create unique custom_id for request tracking
            org_uuid = row.get('org_uuid')
            custom_id = f"startup-{str(org_uuid).strip()}" if not pd.isna(org_uuid) else f"startup-{idx}"
            
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ]
                }
            }
            f.write(json.dumps(request) + "\n")
            
    return jsonl_filename

def upload_batch(batch_num):
    """Uploads the batch file to OpenAI and creates the batch job."""
    jsonl_filename = f"batch_files/batch_requests/batch_{batch_num}_requests.jsonl"
    batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
    
    try:
        # Upload file first
        with open(jsonl_filename, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
            
        # Create batch job
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Save ID for monitoring
        with open(batch_id_file, "w") as f:
            f.write(batch.id)
            
        print(f"Batch {batch_num} uploaded (ID: {batch.id})")
        return batch.id
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

def monitor_batch(batch_num):
    """Polls the status of a batch until it completes or fails."""
    batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
    try:
        with open(batch_id_file, "r") as f:
            batch_id = f.read().strip()
    except:
        return "failed"
        
    print(f"Monitoring batch {batch_num}...")
    
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            print(f"Status: {status} ({datetime.now().strftime('%H:%M:%S')})")
            
            if status == "completed":
                return "completed"
            elif status in ["failed", "expired", "cancelled"]:
                return status
                
            # Wait 60 seconds before checking again to avoid rate limits
            time.sleep(60)
        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(60)

def download_results(batch_num):
    """Downloads the result file from a completed batch and parses it."""
    batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
    results_csv = f"batch_files/batch_outputs/batch_{batch_num}_output.csv"
    
    try:
        with open(batch_id_file, "r") as f:
            batch_id = f.read().strip()
            
        batch = client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            print("No output file found")
            return
            
        content = client.files.content(batch.output_file_id).content
        
        # Process results
        results = []
        for line in content.splitlines():
            try:
                resp = json.loads(line)
                ai_content = resp["response"]["body"]["choices"][0]["message"]["content"]
                
                # Parse CSV response (expecting specific columns from system prompt)
                reader = csv.reader(io.StringIO(ai_content.strip()))
                rows = list(reader)
                if rows and len(rows[0]) >= 7:
                    row = rows[0]
                    results.append({
                        'CompanyID': row[0].strip(),
                        'CompanyName': row[1].strip(),
                        'AI_native': row[2].strip(),
                        'Confidence_1to5': row[3].strip(),
                        'Reasons_3_points': row[4].strip(),
                        'Sources_used': row[5].strip(),
                        'Verification_critique': row[6].strip()
                    })
            except:
                continue
                
        if results:
            pd.DataFrame(results).to_csv(results_csv, index=False)
            print(f"Saved {len(results)} results to {results_csv}")
            
    except Exception as e:
        print(f"Download failed: {e}")

def merge_results():
    """Combines all individual batch output files into one final CSV."""
    print("Merging results...")
    files = sorted([f for f in os.listdir("batch_files/batch_outputs") if f.endswith("_output.csv")])
    
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(os.path.join("batch_files/batch_outputs", f)))
        except:
            continue
            
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Merged {len(final_df)} rows to {OUTPUT_CSV}")

def main():
    num_batches, batch_size = calculate_batch_sizes()
    
    # Process sequentially to avoid hitting total account token limits
    for i in range(1, num_batches + 1):
        print(f"\nProcessing Batch {i}/{num_batches}")
        
        create_batch_file(i, num_batches, batch_size)
        if upload_batch(i):
            if monitor_batch(i) == "completed":
                download_results(i)
                
                # Cleanup large request file to save disk space
                try:
                    os.remove(f"batch_files/batch_requests/batch_{i}_requests.jsonl")
                except:
                    pass
                    
    merge_results()

if __name__ == "__main__":
    main()
