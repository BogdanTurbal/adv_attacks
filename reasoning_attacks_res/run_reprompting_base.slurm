#!/bin/bash

# ====================
# SLURM Arguments
# ====================
#SBATCH --job-name=reprompting_base
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/bt4811/adv_attacks/slurm_output_reprompting_base_.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bogdan.turbal.y@gmail.com

echo "Job ID"
# Set your Princeton NetID
NETID="bt4811"

# Define directories
export HOME_DIR="/scratch/bt4811/adv_attacks/reasoning_attacks_res"
export SCRATCH_DIR="/scratch/bt4811"
export PROJECT_DIR="/scratch/bt4811/adv_attacks"
export RESULTS_DIR="$SCRATCH_DIR/$NETID"

# Create necessary directories
# mkdir -p $SCRATCH_DIR
# mkdir -p $RESULTS_DIR

# Ensure experiment results directory exists (for SLURM output logs)
mkdir -p /scratch/bt4811/adv_attacks/reasoning_attacks_res/reprompting_base

# Change to project directory (usefatt)
cd $PROJECT_DIR



# ====================
# Environment Setup
# ====================
module purge

# --- OFFLINE & PATH CONFIGURATION ---
# W&B Offline Configuration
export WANDB_MODE="offline"
export WANDB_DIR="$HOME_DIR/wandb_runs"
mkdir -p $WANDB_DIR

# HuggingFace Cache Configuration
export HF_HOME="/scratch/bt4811/huggingface"

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

mkdir -p $HF_HOME

module load anaconda3

# Activate the Conda environment
source /scratch/bt4811/adv_attacks/attenv/bin/activate

# WANDB_RUN_DIR for this specific run
export WANDB_RUN_DIR="$RESULTS_DIR/wandb_runs"
mkdir -p $WANDB_RUN_DIR

# HUGGINGFACE_HUB_CACHE (same as HF_HOME, already set above)
export HUGGINGFACE_HUB_CACHE="/scratch/bt4811/huggingface"

# Note: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are already set above before module loads

# ====================
# EXECUTION
# ====================
echo "Starting reprompting_base experiment execution at $(date)"
echo "Job ID: $JOB_ID"
echo "Example range: $EXAMPLE_RANGE"
echo "Results directory: $RESULTS_DIR"

# Execute the main experiment steps (from usefatt directory, which contains run_reprompting_unified.sh)
bash ./run_reprompting_unified.sh \
    --example-range "$EXAMPLE_RANGE" \
    --job-id "$JOB_ID" \
    --results-subdir "$JOB_ID" \
    --results-dir "reprompting_base" \
    --num-iters 15 \
    --num-branches 8 \
    --memory 32 \
    --K 2 \
    --batch-size 16 \
    --max-examples -1 \
    --input-csv "reasoning_attacks/dataset/orthogonalized_outputs_cot150_2048.csv" \
    --model-name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --attacker-model-name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    --wandb-project "reprompting_attacks" \
    --wandb-entity "bogdan-turbal-y" \
    --verbose \
    --wandb-offline \
    --attacker-api \
    --attacker-api-model mistralai/mixtral-8x7b-instruct \
    --attacker-api-key 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 \
    --attacker-api-base-url https://openrouter.ai/api/v1 \
     \
     \
     \
    --num-gpus 2

# Copy results back to home directory
echo "Copying results back to home directory..."
cp -r $RESULTS_DIR/* $PROJECT_DIR/results/ 2>/dev/null || echo "No results to copy"
cp -r $WANDB_RUN_DIR $PROJECT_DIR/ 2>/dev/null || echo "No W&B logs to copy"

echo "Job $JOB_ID finished at $(date)"
