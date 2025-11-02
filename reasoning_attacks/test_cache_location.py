#!/usr/bin/env python3
"""
Test script to verify HuggingFace cache location
"""

import os
from huggingface_hub import snapshot_download

# Set up scratch directory for model downloads
SCRATCH_DIR = "/scratch/gpfs/KOROLOVA"
HF_CACHE_DIR = f"{SCRATCH_DIR}/huggingface"

# Set multiple environment variables to ensure HF uses scratch directory
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR

# Set offline mode for HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Create the directory if it doesn't exist
os.makedirs(HF_CACHE_DIR, exist_ok=True)

print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"Target cache directory: {HF_CACHE_DIR}")

# Test with a small model first
print("\nTesting with a small model...")
try:
    snapshot_download(
        repo_id="microsoft/DialoGPT-small",  # Small test model
        cache_dir=HF_CACHE_DIR,
        local_dir_use_symlinks=False
    )
    print(f"✓ Successfully downloaded test model to {HF_CACHE_DIR}")
    
    # Check if files actually exist in the scratch directory
    import glob
    files = glob.glob(f"{HF_CACHE_DIR}/**/*", recursive=True)
    print(f"Files in cache directory: {len(files)}")
    for f in files[:5]:  # Show first 5 files
        print(f"  {f}")
        
except Exception as e:
    print(f"✗ Error downloading test model: {e}")
