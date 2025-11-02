#!/usr/bin/env python3
"""
Diagnostic script to identify why the job is failing silently
"""

import os
import sys
import traceback
import torch
import transformers

# Set up scratch directory
SCRATCH_DIR = "/scratch/gpfs/KOROLOVA"
HF_CACHE_DIR = f"{SCRATCH_DIR}/huggingface"

# Set multiple environment variables to ensure HF uses scratch directory
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# Set offline mode for HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("=== Diagnostic Script ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")

print(f"\nEnvironment variables:")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")

# Test 1: Check if cache directory exists and has models
print(f"\n=== Test 1: Cache Directory ===")
if os.path.exists(HF_CACHE_DIR):
    import glob
    files = glob.glob(f"{HF_CACHE_DIR}/**/*", recursive=True)
    print(f"✓ Cache directory exists with {len(files)} files")
    
    # Check for specific model files
    model_files = glob.glob(f"{HF_CACHE_DIR}/**/config.json", recursive=True)
    print(f"✓ Found {len(model_files)} model config files")
    
    for f in model_files[:3]:  # Show first 3
        print(f"  {f}")
else:
    print(f"✗ Cache directory {HF_CACHE_DIR} does not exist!")

# Test 2: Try importing the attack module
print(f"\n=== Test 2: Import Attack Module ===")
try:
    sys.path.append(".")
    from attack import GCGConfig, GCGResult, run
    print("✓ Attack module imported successfully")
except Exception as e:
    print(f"✗ Failed to import attack module: {e}")
    traceback.print_exc()

# Test 3: Try loading tokenizer
print(f"\n=== Test 3: Load Tokenizer ===")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        local_files_only=True,
        cache_dir=HF_CACHE_DIR
    )
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    traceback.print_exc()

# Test 4: Try loading model (this might fail due to memory)
print(f"\n=== Test 4: Load Model ===")
try:
    from transformers import AutoModelForCausalLM
    
    # Check available memory first
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i} memory: {props.total_memory / 1e9:.1f}GB total")
    
    print("Attempting to load model...")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        local_files_only=True,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ Model loaded successfully")
    
    # Check memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i} memory usage: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    traceback.print_exc()

# Test 5: Check input data
print(f"\n=== Test 5: Check Input Data ===")
input_csv = "dataset/orthogonalized_outputs_cot150_2048.csv"
if os.path.exists(input_csv):
    print(f"✓ Input CSV exists: {input_csv}")
    try:
        import pandas as pd
        df = pd.read_csv(input_csv)
        print(f"✓ CSV loaded successfully: {len(df)} rows")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
else:
    print(f"✗ Input CSV not found: {input_csv}")

print(f"\n=== Diagnostic Complete ===")
