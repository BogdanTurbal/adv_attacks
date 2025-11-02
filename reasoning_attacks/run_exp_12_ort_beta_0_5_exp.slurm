#!/bin/bash

# ====================
# SLURM Arguments
# ====================
#SBATCH --job-name=exp_12_ort_beta_0_5_exp
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_output_exp_12_ort_beta_0_5_exp_%j.out
#SBATCH --constraint=gpu80
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bogdan.turbal.y@gmail.com

# Set your Princeton NetID
NETID="bt4811"

# Define directories
export HOME_DIR="/scratch/gpfs/KOROLOVA/bt4811/reasoning_attacks_res"
export SCRATCH_DIR="/scratch/gpfs/KOROLOVA"
export PROJECT_DIR="/scratch/gpfs/KOROLOVA/bt4811/reasoning_attacks_res"
export RESULTS_DIR="$SCRATCH_DIR/$NETID"

# Create necessary directories
mkdir -p $SCRATCH_DIR
mkdir -p $RESULTS_DIR

# Change to project directory
cd $PROJECT_DIR/reasoning_attacks

# ====================
# Environment Setup
# ====================
module purge
module load anaconda3/2025.6
module load cuda/12.4

# Activate the Conda environment
source ~/.bashrc
conda activate reasoning_env

# --- OFFLINE & PATH CONFIGURATION ---
# W&B Offline Configuration
export WANDB_MODE="offline"
export WANDB_RUN_DIR="$RESULTS_DIR/wandb_runs"
mkdir -p $WANDB_RUN_DIR

# Set HuggingFace cache to scratch directory
export HF_HOME="/scratch/gpfs/KOROLOVA/huggingface"
export HUGGINGFACE_HUB_CACHE="/scratch/gpfs/KOROLOVA/huggingface"

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

mkdir -p $HF_HOME

# ====================
# EXECUTION
# ====================
echo "Starting exp_12_ort_beta_0_5_exp experiment execution at $(date)"
echo "Job ID: $JOB_ID"
echo "Example range: $EXAMPLE_RANGE"
echo "Results directory: $RESULTS_DIR"

# Execute the main experiment steps
bash ./run_experiment_unified.sh \
    --example-range "$EXAMPLE_RANGE" \
    --job-id "$JOB_ID" \
    --results-subdir "$JOB_ID" \
    --results-dir "exp_12_results_ort_beta_0_5" \
    --beta 0.5 \
    --num-steps 150 \
    --num-target-tokens 20 \
    --num-refusal-tokens 45 \
    --num-gpus 4 \
    --runs-per-gpu 1 \
    --max-examples -1 \
    --input-csv "dataset/orthogonalized_outputs_cot150_2048.csv" \
    --model-name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --wandb-project "orthogonalized_attacks" \
    --wandb-entity "bogdan-turbal-y" \
     \
    --verbose \
    --wandb-offline

# Copy results back to home directory
echo "Copying results back to home directory..."
cp -r $RESULTS_DIR/* $PROJECT_DIR/results/ 2>/dev/null || echo "No results to copy"
cp -r $WANDB_RUN_DIR $PROJECT_DIR/ 2>/dev/null || echo "No W&B logs to copy"

echo "Job $JOB_ID finished at $(date)"
