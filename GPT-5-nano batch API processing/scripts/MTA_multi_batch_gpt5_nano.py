"""
Multi-Batch Startup Classification Script - FULL DATASET
=========================================================
Author: [Your Name]
Date: October 2025
Purpose: Classify startups using OpenAI Batch API with sequential processing

This script:
1. Splits the dataset into multiple batches (accounting for 200MB file size limit)
2. Uploads batches sequentially to OpenAI
3. Monitors batches until completion
4. Downloads and merges results into a single CSV file

Output Format: 7 columns only
- CompanyID, CompanyName, AI_native, Confidence_1to5, 
  Reasons_3_points, Sources_used, Verification_critique
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

from openai import OpenAI
import pandas as pd
import json
import time
import os
import csv
import io
from datetime import datetime
import re
import re

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

# Read API key from external file (one level up in GPT-5-nano folder)
# Skip comment lines and get the first non-comment line
with open("../api_key.txt", "r") as f:
    lines = f.readlines()
    api_key = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            api_key = line
            break
    if not api_key:
        raise ValueError("No API key found in api_key.txt (only comments or empty lines)")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Configuration variables
INPUT_CSV = "../../company_us_short_long_desc_.csv"     # Input file with startup data (two levels up to Week 7)
OUTPUT_CSV = "classified_startups_gpt5_nano.csv"       # Final output file
SYSTEM_PROMPT_FILE = "../system_prompt.txt"            # System prompt (one level up)

# OpenAI Batch API limits
MAX_REQUESTS_PER_BATCH = 50000    # OpenAI's max requests per batch
MAX_FILE_SIZE_MB = 100            # Conservative limit: 100MB per batch to avoid upload timeouts
                                  # (OpenAI's max is 200MB, but large files can timeout)

# GPT-5-nano specific limits
MAX_ENQUEUED_TOKENS = 40000000    # GPT-5-nano enqueued token limit: 40M tokens
ESTIMATED_TOKENS_PER_REQUEST = 2000  # Conservative estimate: ~2K tokens per request

# GPT-5-nano rate limits
MAX_REQUESTS_PER_MINUTE = 5000    # RPM limit: 5000 requests per minute
MAX_TOKENS_PER_MINUTE = 4000000   # TPM limit: 4M tokens per minute

# Model configuration
MODEL_NAME = "gpt-5-nano"

print("="*70)
print("MULTI-BATCH STARTUP CLASSIFICATION - FULL DATASET")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input File: {INPUT_CSV}")
print(f"Output File: {OUTPUT_CSV}")
print(f"Model: {MODEL_NAME}")
print("="*70)

# ============================================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================================

def extract_year_from_date(date_str):
    """
    Extract year from date string in format like '01nov2016' or '2016-11-01'.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        str: Year as string, or 'N/A' if cannot parse
    """
    if pd.isna(date_str) or date_str == '' or date_str == 'N/A':
        return 'N/A'
    
    date_str = str(date_str).strip()
    
    # Try to extract 4-digit year from various formats
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return year_match.group(0)
    
    return 'N/A'

def format_user_message(row):
    """
    Formats a startup's data into the input message format expected by the AI.
    Maps the new dataset columns to the expected format.
    
    Args:
        row: A pandas Series containing startup data
        
    Returns:
        str: Formatted multi-line string with startup information
    """
    # Map columns from new dataset format
    # Handle NaN values properly (str(NaN) = 'nan', so check with pd.isna first)
    org_uuid_val = row.get('org_uuid', 'N/A')
    company_id = 'N/A' if pd.isna(org_uuid_val) else str(org_uuid_val).strip()
    
    name_val = row.get('name', 'N/A')
    company_name = 'N/A' if pd.isna(name_val) else str(name_val).strip()
    
    # Get short_description and Long description separately
    short_desc_val = row.get('short_description', '')
    short_desc = 'N/A' if (pd.isna(short_desc_val) or str(short_desc_val).strip() == '') else str(short_desc_val).strip()
    
    long_desc_val = row.get('Long description', '')
    long_desc = 'N/A' if (pd.isna(long_desc_val) or str(long_desc_val).strip() == '') else str(long_desc_val).strip()
    
    # Combine category_list and category_groups_list as keywords
    cat_list_val = row.get('category_list', '')
    category_list = '' if pd.isna(cat_list_val) else str(cat_list_val).strip()
    
    cat_groups_val = row.get('category_groups_list', '')
    category_groups = '' if pd.isna(cat_groups_val) else str(cat_groups_val).strip()
    
    if category_list and category_groups:
        keywords = f"{category_list}, {category_groups}".strip()
    elif category_list:
        keywords = category_list
    elif category_groups:
        keywords = category_groups
    else:
        keywords = 'N/A'
    
    # Extract year from founded_date
    founded_date = row.get('founded_date', 'N/A')
    year_founded = extract_year_from_date(founded_date)
    
    user_message = f"""INPUT:
CompanyID: {company_id}
CompanyName: {company_name}
Short Description: {short_desc}
Long Description: {long_desc}
Keywords: {keywords}
YearFounded: {year_founded}"""
    return user_message

# ============================================================================
# SECTION 4: MAIN FUNCTIONS
# ============================================================================

def calculate_batch_sizes():
    """
    Calculate how many batches we need and how many startups per batch.
    For GPT-5-nano, we must respect the 40M token enqueued limit.
    
    Returns:
        tuple: (num_batches, startups_per_batch)
    """
    print("\n[STEP 1] Calculating batch sizes for GPT-5-nano...")
    
    # Load the CSV to get total count
    df = pd.read_csv(INPUT_CSV)
    total_startups = len(df)
    print(f"[OK] Total startups: {total_startups:,}")
    
    # GPT-5-nano specific calculation based on multiple limits
    print(f"[INFO] GPT-5-nano limits:")
    print(f"       - Enqueued tokens: {MAX_ENQUEUED_TOKENS:,}")
    print(f"       - RPM: {MAX_REQUESTS_PER_MINUTE:,}")
    print(f"       - TPM: {MAX_TOKENS_PER_MINUTE:,}")
    print(f"[INFO] Estimated tokens per request: {ESTIMATED_TOKENS_PER_REQUEST:,}")
    
    # Calculate max requests per batch based on token limit
    max_requests_for_tokens = MAX_ENQUEUED_TOKENS // ESTIMATED_TOKENS_PER_REQUEST
    print(f"[INFO] Max requests per batch (token limit): {max_requests_for_tokens:,}")
    
    # Also check file size limit (CRITICAL - this is often the bottleneck)
    # Actual measured: ~14.2 KB per request (system prompt + user message + JSON overhead)
    estimated_kb_per_request = 14.2  # Updated based on actual batch file measurements
    estimated_mb_per_request = estimated_kb_per_request / 1024
    max_requests_for_size = int(MAX_FILE_SIZE_MB / estimated_mb_per_request)
    print(f"[INFO] Max requests per batch (file size): {max_requests_for_size:,}")
    print(f"[WARNING] File size limit is often the most restrictive constraint!")
    
    # Use the MOST RESTRICTIVE limit (file size is usually the bottleneck)
    max_requests_per_batch = min(max_requests_for_tokens, max_requests_for_size, MAX_REQUESTS_PER_BATCH)
    print(f"[INFO] Using limit: {max_requests_per_batch:,} requests per batch")
    print(f"[NOTE] Rate limits (RPM/TPM) don't affect batch size, only processing speed")
    
    # Calculate number of batches needed
    num_batches = (total_startups + max_requests_per_batch - 1) // max_requests_per_batch
    startups_per_batch = total_startups // num_batches
    
    # Estimate actual batch size
    estimated_batch_size_mb = (startups_per_batch * estimated_mb_per_request)
    estimated_tokens_per_batch = startups_per_batch * ESTIMATED_TOKENS_PER_REQUEST
    
    print(f"[OK] Number of batches needed: {num_batches}")
    print(f"[OK] Startups per batch: ~{startups_per_batch:,}")
    print(f"[OK] Estimated size per batch: ~{estimated_batch_size_mb:.1f} MB")
    print(f"[OK] Estimated tokens per batch: ~{estimated_tokens_per_batch:,}")
    
    return num_batches, startups_per_batch

def create_single_batch_file(batch_num, num_batches, startups_per_batch):
    """
    Create a single JSONL batch file for the specified batch number.
    
    Args:
        batch_num: The batch number to create (1, 2, 3, etc.)
        num_batches: Total number of batches
        startups_per_batch: Number of startups per batch
    """
    print(f"\n[STEP] Creating batch {batch_num} file...")
    
    # Load system prompt
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        system_prompt = f.read().strip()
    
    # Load full dataset
    df = pd.read_csv(INPUT_CSV)
    total_startups = len(df)
    
    # Calculate start and end indices for this batch
    start_idx = (batch_num - 1) * startups_per_batch
    if batch_num == num_batches:
        # Last batch gets all remaining rows
        end_idx = total_startups
    else:
        end_idx = start_idx + startups_per_batch
    
    batch_df = df.iloc[start_idx:end_idx]
    
    # Create JSONL file for this batch
    jsonl_filename = f"batch_files/batch_requests/batch_{batch_num}_requests.jsonl"
    
    # Skip if file already exists (resuming from a previous run)
    if os.path.exists(jsonl_filename):
        file_size_bytes = os.path.getsize(jsonl_filename)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"[INFO] Batch {batch_num} JSONL file already exists, skipping creation")
        print(f"     - File: {jsonl_filename}")
        print(f"     - File size: {file_size_mb:.1f} MB")
        return jsonl_filename
    
    with open(jsonl_filename, "w") as f:
        for idx, row in batch_df.iterrows():
            # Format the user message
            user_message = format_user_message(row)
            
            # Get org_uuid for custom_id (handle NaN properly)
            org_uuid_val = row.get('org_uuid', None)
            if pd.isna(org_uuid_val):
                org_uuid = f'startup-{idx}'
            else:
                org_uuid = str(org_uuid_val).strip()
            
            # Create the API request in OpenAI Batch API format
            request = {
                "custom_id": f"startup-{org_uuid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                }
            }
            
            # Write as JSON line
            f.write(json.dumps(request) + "\n")
    
    # Get file size
    file_size_bytes = os.path.getsize(jsonl_filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"[OK] Created {jsonl_filename}")
    print(f"     - Startups: {len(batch_df):,} (rows {start_idx+1} to {end_idx})")
    print(f"     - File size: {file_size_mb:.1f} MB")
    
    return jsonl_filename

def upload_batch(batch_num):
    """
    Upload a single batch file to OpenAI and save the batch ID.
    
    Args:
        batch_num: The batch number (1, 2, 3, etc.)
        
    Returns:
        str: The batch ID returned by OpenAI, or None if failed
    """
    jsonl_filename = f"batch_files/batch_requests/batch_{batch_num}_requests.jsonl"
    batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
    
    print(f"[INFO] Uploading {jsonl_filename}...")
    
    # Check file size before uploading
    file_size_bytes = os.path.getsize(jsonl_filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"[OK] Batch file size: {file_size_mb:.1f} MB")
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        print(f"[ERROR] File size {file_size_mb:.1f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB")
        return None
    
    try:
        # Step 1: Upload the file
        with open(jsonl_filename, "rb") as f:
            batch_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        print(f"[OK] File uploaded successfully")
        print(f"     - File ID: {batch_file.id}")
        
        # Step 2: Create the batch
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        print(f"[OK] Batch created successfully")
        print(f"     - Batch ID: {batch.id}")
        
        # Step 3: Save the batch ID
        with open(batch_id_file, "w") as f:
            f.write(batch.id)
        
        print(f"[OK] Batch ID saved to {batch_id_file}")
        
        return batch.id
        
    except Exception as e:
        print(f"[ERROR] Failed to upload batch: {e}")
        return None

def check_all_batches_status(num_batches):
    """
    Monitor the status of ALL batches concurrently.
    Checks every 60 seconds until all batches are completed or failed.
    
    Args:
        num_batches: Total number of batches to monitor
        
    Returns:
        dict: Mapping of batch_num to final status
    """
    print("[INFO] Starting concurrent batch monitoring...")
    print("[INFO] This will check all batches every 60 seconds")
    print("[INFO] You can safely close your laptop - batches run on OpenAI's servers\n")
    
    # Load all batch IDs
    batch_ids = {}
    for batch_num in range(1, num_batches + 1):
        batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
        try:
            with open(batch_id_file, "r") as f:
                batch_ids[batch_num] = f.read().strip()
        except FileNotFoundError:
            print(f"[ERROR] Batch ID file not found for batch {batch_num}")
            batch_ids[batch_num] = None
    
    # Track status of each batch
    batch_statuses = {num: "validating" for num in range(1, num_batches + 1)}
    
    # Monitor until all batches are done
    check_count = 0
    while True:
        check_count += 1
        print(f"\n{'='*70}")
        print(f"STATUS CHECK #{check_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        all_done = True
        
        for batch_num in range(1, num_batches + 1):
            batch_id = batch_ids[batch_num]
            
            if not batch_id:
                print(f"Batch {batch_num}: [ERROR] No batch ID")
                batch_statuses[batch_num] = "failed"
                continue
            
            try:
                # Check batch status
                batch = client.batches.retrieve(batch_id)
                status = batch.status
                batch_statuses[batch_num] = status
                
                # Display status with progress
                if status == "completed":
                    print(f"Batch {batch_num}: [COMPLETED]")
                    print(f"          - Total: {batch.request_counts.total} | "
                          f"Completed: {batch.request_counts.completed} | "
                          f"Failed: {batch.request_counts.failed}")
                elif status == "failed":
                    print(f"Batch {batch_num}: [FAILED]")
                elif status == "in_progress":
                    print(f"Batch {batch_num}: [IN PROGRESS]")
                    print(f"          - Total: {batch.request_counts.total} | "
                          f"Completed: {batch.request_counts.completed} | "
                          f"Failed: {batch.request_counts.failed}")
                    all_done = False
                else:
                    print(f"Batch {batch_num}: [{status.upper()}]")
                    all_done = False
                    
            except Exception as e:
                print(f"Batch {batch_num}: [ERROR] {e}")
                all_done = False
        
        # Check if all batches are done
        if all_done:
            print(f"\n{'='*70}")
            print("ALL BATCHES COMPLETED!")
            print(f"{'='*70}")
            break
        
        # Wait before next check
        print(f"\n[INFO] Waiting 60 seconds before next check...")
        time.sleep(60)
    
    return batch_statuses

def parse_classification_result(content):
    """
    Parse the AI's response text into structured fields.
    The AI returns CSV format: CompanyID,CompanyName,AI_Native,Confidence,Reasons,Sources,Critique
    
    Args:
        content: The AI's response text (CSV format)
        
    Returns:
        dict: Parsed fields with exact column names needed
    """
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
        # Use csv.reader to properly handle commas within quoted fields
        csv_reader = csv.reader(io.StringIO(content.strip()))
        rows = list(csv_reader)
        
        # The AI should return exactly one row of CSV data
        if rows and len(rows) > 0:
            data_row = rows[0]
            
            # Extract fields in order
            if len(data_row) >= 7:
                result['CompanyID'] = data_row[0].strip()
                result['CompanyName'] = data_row[1].strip()
                result['AI_native'] = data_row[2].strip()
                result['Confidence_1to5'] = data_row[3].strip()
                result['Reasons_3_points'] = data_row[4].strip()
                result['Sources_used'] = data_row[5].strip()
                result['Verification_critique'] = data_row[6].strip()
                
    except Exception as e:
        print(f"[WARNING] Failed to parse result: {e}")
        print(f"[WARNING] Content: {content[:200]}...")
    
    return result

def download_batch_results(batch_num):
    """
    Download results from a completed batch and save to CSV.
    
    Args:
        batch_num: The batch number to download
    """
    batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
    results_jsonl = f"batch_files/batch_results/batch_{batch_num}_results.jsonl"
    results_csv = f"batch_files/batch_outputs/batch_{batch_num}_output.csv"
    
    print(f"[INFO] Downloading results for batch {batch_num}...")
    
    # Load batch ID
    try:
        with open(batch_id_file, "r") as f:
            batch_id = f.read().strip()
    except FileNotFoundError:
        print(f"[ERROR] Batch ID file not found: {batch_id_file}")
        return
    
    try:
        # Retrieve batch information
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            print(f"[WARNING] Batch status is '{batch.status}', not 'completed'")
            return
        
        # Get output file ID
        output_file_id = batch.output_file_id
        
        if not output_file_id:
            print(f"[ERROR] No output file ID found for batch {batch_num}")
            return
        
        # Download the results
        result = client.files.content(output_file_id)
        result_content = result.content
        
        # Save raw JSONL
        with open(results_jsonl, "wb") as f:
            f.write(result_content)
        
        print(f"[OK] Downloaded raw results to {results_jsonl}")
        
        # Parse JSONL and convert to CSV
        parsed_results = []
        
        with open(results_jsonl, "r") as f:
            for line in f:
                try:
                    response_obj = json.loads(line.strip())
                    
                    # Extract the AI's response
                    ai_response = response_obj["response"]["body"]["choices"][0]["message"]["content"]
                    
                    # Parse the classification result
                    parsed = parse_classification_result(ai_response)
                    
                    if parsed['CompanyID']:  # Only add if we got valid data
                        parsed_results.append(parsed)
                        
                except Exception as e:
                    print(f"[WARNING] Failed to parse line: {e}")
        
        # Save to CSV with exact 7 columns (no merging with original data)
        if parsed_results:
            results_df = pd.DataFrame(parsed_results)
            
            # Ensure column order
            column_order = ['CompanyID', 'CompanyName', 'AI_native', 'Confidence_1to5', 'Reasons_3_points', 'Sources_used', 'Verification_critique']
            results_df = results_df[column_order]
            
            results_df.to_csv(results_csv, index=False)
            print(f"[OK] Saved {len(parsed_results)} results to {results_csv}")
        else:
            print(f"[WARNING] No valid results parsed for batch {batch_num}")
            
    except Exception as e:
        print(f"[ERROR] Failed to download batch {batch_num}: {e}")

def merge_all_results():
    """
    Merge all individual batch CSV files into a single final output file.
    Output contains only the 7 AI classification columns (no original data).
    """
    print("[INFO] Merging all batch results...")
    
    # Find all batch output files
    batch_files = sorted([os.path.join("batch_files/batch_outputs", f) for f in os.listdir("batch_files/batch_outputs") if f.startswith("batch_") and f.endswith("_output.csv")])
    
    if not batch_files:
        print("[ERROR] No batch output files found!")
        return
    
    print(f"[OK] Found {len(batch_files)} batch output files")
    
    # Load and concatenate all batch results
    all_results = []
    
    for batch_file in batch_files:
        try:
            df = pd.read_csv(batch_file)
            all_results.append(df)
            print(f"[OK] Loaded {batch_file}: {len(df)} rows")
        except Exception as e:
            print(f"[ERROR] Failed to load {batch_file}: {e}")
    
    # Concatenate all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Ensure correct column order
        column_order = ['CompanyID', 'CompanyName', 'AI_native', 'Confidence_1to5', 'Reasons_3_points', 'Sources_used', 'Verification_critique']
        final_df = final_df[column_order]
        
        # Save final output
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\n[SUCCESS] Merged results saved to {OUTPUT_CSV}")
        print(f"[INFO] Total rows: {len(final_df):,}")
        print(f"[INFO] Columns: {', '.join(final_df.columns)}")
        
        # Print statistics
        ai_native_count = (final_df['AI_native'].astype(str) == '1').sum()
        not_ai_native_count = (final_df['AI_native'].astype(str) == '0').sum()
        
        print(f"\n[STATISTICS]")
        print(f"AI-Native: {ai_native_count:,} ({ai_native_count/len(final_df)*100:.1f}%)")
        print(f"Not AI-Native: {not_ai_native_count:,} ({not_ai_native_count/len(final_df)*100:.1f}%)")
    else:
        print("[ERROR] No results to merge!")

# ============================================================================
# SECTION 5: MAIN WORKFLOW
# ============================================================================

def check_single_batch_status(batch_num):
    """
    Monitor a single batch until completion.
    
    Args:
        batch_num: The batch number to monitor
        
    Returns:
        str: Final status ('completed', 'failed', etc.)
    """
    batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
    
    try:
        with open(batch_id_file, "r") as f:
            batch_id = f.read().strip()
    except FileNotFoundError:
        print(f"[ERROR] Batch ID file not found: {batch_id_file}")
        return "failed"
    
    print(f"[INFO] Monitoring batch {batch_num} (ID: {batch_id})")
    print(f"[INFO] Checking every 60 seconds...")
    
    check_count = 0
    while True:
        check_count += 1
        
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            
            print(f"\n[CHECK #{check_count}] {datetime.now().strftime('%H:%M:%S')} - Status: {status.upper()}")
            
            if hasattr(batch, 'request_counts') and batch.request_counts:
                print(f"             Progress: {batch.request_counts.completed}/{batch.request_counts.total} completed, {batch.request_counts.failed} failed")
            
            if status == "completed":
                print(f"[SUCCESS] Batch {batch_num} completed!")
                return "completed"
            elif status == "failed" or status == "expired" or status == "cancelled":
                print(f"[ERROR] Batch {batch_num} {status}!")
                return status
            
            # Still processing, wait
            time.sleep(60)
            
        except Exception as e:
            print(f"[ERROR] Failed to check batch status: {e}")
            time.sleep(60)

def main():
    """
    Main workflow that orchestrates the entire process.
    Processes batches SEQUENTIALLY to avoid token queue limits.
    Creates and uploads ONE batch at a time, waits for completion, then moves to next.
    """
    print("\nStarting multi-batch processing workflow...")
    print("[MODE] Sequential processing (one batch at a time)")
    print("[REASON] Avoids OpenAI's 40M enqueued token limit for GPT-5-nano")
    
    # Step 1: Calculate batch sizes
    num_batches, startups_per_batch = calculate_batch_sizes()
    
    # Step 2: Process each batch SEQUENTIALLY (create → upload → monitor → download)
    print(f"\n{'='*70}")
    print("[SEQUENTIAL PROCESSING] Create → Upload → Monitor → Download")
    print(f"{'='*70}")
    print(f"Total batches: {num_batches}")
    
    # Process all batches from the beginning
    START_BATCH = 1
    print(f"[INFO] Processing all batches starting from Batch {START_BATCH}")
    remaining_batches = num_batches
    print(f"Total batches to process: {remaining_batches}")
    print(f"Estimated time: {remaining_batches} × 12-24 hours = {remaining_batches*12}-{remaining_batches*24} hours total")
    print(f"{'='*70}")
    
    for batch_num in range(START_BATCH, num_batches + 1):
        print(f"\n{'='*70}")
        print(f"PROCESSING BATCH {batch_num}/{num_batches}")
        print(f"{'='*70}")
        
        # Create this batch file
        print(f"\n[PHASE 1] Creating Batch {batch_num} file...")
        create_single_batch_file(batch_num, num_batches, startups_per_batch)
        
        # Upload
        print(f"\n[PHASE 2] Uploading Batch {batch_num}...")
        batch_id = upload_batch(batch_num)
        if not batch_id:
            print(f"[ERROR] Failed to upload batch {batch_num}. Exiting.")
            return
        
        print(f"[SUCCESS] Batch {batch_num} uploaded!")
        
        # Monitor until completion
        print(f"\n[PHASE 3] Monitoring Batch {batch_num}...")
        print(f"[INFO] This will take 12-24 hours. You can close your laptop.")
        status = check_single_batch_status(batch_num)
        
        if status != "completed":
            print(f"[ERROR] Batch {batch_num} did not complete successfully (status: {status})")
            print(f"[INFO] Continuing to next batch...")
            continue
        
        # Download
        print(f"\n[PHASE 4] Downloading Batch {batch_num} results...")
        download_batch_results(batch_num)
        
        print(f"\n[SUCCESS] Batch {batch_num} complete!")
        print(f"[PROGRESS] {batch_num}/{num_batches} batches completed")
        
        # Clean up the JSONL file to save space
        jsonl_file = f"batch_files/batch_requests/batch_{batch_num}_requests.jsonl"
        try:
            os.remove(jsonl_file)
            print(f"[INFO] Cleaned up {jsonl_file} to save space")
        except:
            pass
    
    # Step 3: Merge all results
    print(f"\n{'='*70}")
    print("[PHASE 5] MERGING ALL RESULTS")
    print(f"{'='*70}")
    merge_all_results()
    
    print("\n" + "="*70)
    print("MULTI-BATCH PROCESSING COMPLETE!")
    print("="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()

