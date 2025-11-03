import os
import sys
from huggingface_hub import snapshot_download

# Set up scratch directory for model downloads
# Can be overridden by environment variable LOCAL_SCRATCH_DIR or command line argument
if len(sys.argv) > 1:
    SCRATCH_DIR = sys.argv[1]
elif "LOCAL_SCRATCH_DIR" in os.environ:
    SCRATCH_DIR = os.environ["LOCAL_SCRATCH_DIR"]
else:
    SCRATCH_DIR = "/scratch/gpfs/KOROLOVA"  # Default fallback

HF_CACHE_DIR = f"{SCRATCH_DIR}/huggingface"

# Set multiple environment variables to ensure HF uses scratch directory
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR

# Set offline mode for HuggingFace (can be overridden)
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Create the directory if it doesn't exist
os.makedirs(HF_CACHE_DIR, exist_ok=True)

def download_model(repo_id):
    print(f"Downloading {repo_id} to {HF_CACHE_DIR}")
    print(f"Cache directory: {HF_CACHE_DIR}")
    print(f"Scratch directory: {SCRATCH_DIR}")
    # Explicitly specify cache_dir to ensure it goes to scratch
    # Allow downloads when not in strict offline mode
    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    snapshot_download(
        repo_id=repo_id, 
        cache_dir=HF_CACHE_DIR,
        local_dir_use_symlinks=False,
        local_files_only=local_files_only
    )
    print(f"Successfully downloaded {repo_id}")

# Main experiment model
download_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# Nanogcg model
download_model("mistralai/Mistral-7B-Instruct-v0.2")

# Reprompting attack models
# Attacker model (used for generating adversarial prompts)
download_model("mistralai/Mixtral-8x7B-Instruct-v0.1")

print("All models downloaded. You can now submit your SLURM job.")