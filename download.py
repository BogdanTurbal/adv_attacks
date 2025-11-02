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

def download_model(repo_id):
    print(f"Downloading {repo_id} to {HF_CACHE_DIR}")
    print(f"Cache directory: {HF_CACHE_DIR}")
    # Explicitly specify cache_dir to ensure it goes to scratch
    snapshot_download(
        repo_id=repo_id, 
        cache_dir=HF_CACHE_DIR,
        local_dir_use_symlinks=False,
        local_files_only=False  # Allow downloads when not offline
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