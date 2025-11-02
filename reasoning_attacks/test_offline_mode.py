#!/usr/bin/env python3
"""
Offline mode script for running experiments when internet is not available
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Create the directory if it doesn't exist
os.makedirs(HF_CACHE_DIR, exist_ok=True)

print(f"Running in OFFLINE mode")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")

def test_model_loading():
    """Test loading models in offline mode"""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    print(f"\nTesting model loading for {model_name}")
    print("This will only work if the model is already cached locally...")
    
    try:
        # Try to load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,  # Only use local files
            cache_dir=HF_CACHE_DIR
        )
        print("✓ Tokenizer loaded successfully")
        
        # Try to load model (this might fail due to memory constraints)
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,  # Only use local files
            cache_dir=HF_CACHE_DIR,
            torch_dtype="auto",
            device_map="auto"
        )
        print("✓ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you have downloaded the models first using download.py")
        return False

if __name__ == "__main__":
    print("=== HuggingFace Offline Mode Test ===")
    
    # Check if cache directory exists and has content
    if os.path.exists(HF_CACHE_DIR):
        import glob
        files = glob.glob(f"{HF_CACHE_DIR}/**/*", recursive=True)
        print(f"Found {len(files)} files in cache directory")
        
        if len(files) > 0:
            print("Sample files:")
            for f in files[:5]:
                print(f"  {f}")
        else:
            print("Cache directory is empty!")
            print("Run 'python download.py' first to download models")
            sys.exit(1)
    else:
        print(f"Cache directory {HF_CACHE_DIR} does not exist!")
        print("Run 'python download.py' first to download models")
        sys.exit(1)
    
    # Test model loading
    success = test_model_loading()
    
    if success:
        print("\n✓ Offline mode is working correctly!")
        print("You can now run your experiments in offline mode.")
    else:
        print("\n✗ Offline mode test failed.")
        print("Make sure models are properly downloaded and cached.")
