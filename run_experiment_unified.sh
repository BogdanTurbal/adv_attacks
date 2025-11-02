#!/bin/bash

# Unified experiment script that works with the configuration system
# This script replaces the individual run_experiment_*.sh scripts

# Parse command line arguments
EXAMPLE_RANGE=""
JOB_ID=""
RESULTS_SUBDIR=""
RESULTS_DIR_PARAM=""
BETA=""
NUM_STEPS=""
NUM_TARGET_TOKENS=""
NUM_REFUSAL_TOKENS=""
NUM_GPUS=""
RUNS_PER_GPU=""
MAX_EXAMPLES=""
INPUT_CSV=""
MODEL_NAME=""
WANDB_PROJECT=""
WANDB_ENTITY=""
TARGET_OVERRIDE=false
VERBOSE=false
WANDB_OFFLINE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --example-range)
            EXAMPLE_RANGE="$2"
            shift 2
            ;;
        --job-id)
            JOB_ID="$2"
            shift 2
            ;;
        --results-subdir)
            RESULTS_SUBDIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR_PARAM="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --num-target-tokens)
            NUM_TARGET_TOKENS="$2"
            shift 2
            ;;
        --num-refusal-tokens)
            NUM_REFUSAL_TOKENS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --runs-per-gpu)
            RUNS_PER_GPU="$2"
            shift 2
            ;;
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --input-csv)
            INPUT_CSV="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --target-override)
            TARGET_OVERRIDE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --wandb-offline)
            WANDB_OFFLINE=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$BETA" ] || [ -z "$NUM_STEPS" ] || [ -z "$NUM_TARGET_TOKENS" ] || [ -z "$NUM_REFUSAL_TOKENS" ] || [ -z "$NUM_GPUS" ]; then
    echo "Error: Missing required parameters (beta, num-steps, num-target-tokens, num-refusal-tokens, num-gpus)"
    exit 1
fi

# Set default values for optional parameters
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
INPUT_CSV=${INPUT_CSV:-"reasoning_attacks/dataset/orthogonalized_outputs_cot150_2048.csv"}
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
WANDB_PROJECT=${WANDB_PROJECT:-"attor"}
WANDB_ENTITY=${WANDB_ENTITY:-"bogdan-turbal-y"}

# Add reasoning_attacks to PYTHONPATH so we can import attack module
export PYTHONPATH="${PYTHONPATH}:$(pwd)/reasoning_attacks"

# Set results directory (inside reasoning_attacks_res)
if [ -n "$RESULTS_DIR_PARAM" ]; then
    BASE_RESULTS_DIR="/scratch/gpfs/KOROLOVA/bt4811/usefatt/reasoning_attacks_res/$RESULTS_DIR_PARAM"
else
    BASE_RESULTS_DIR="/scratch/gpfs/KOROLOVA/bt4811/usefatt/reasoning_attacks_res/results_default"
fi

if [ -n "$RESULTS_SUBDIR" ]; then
    RESULTS_DIR="$BASE_RESULTS_DIR/$RESULTS_SUBDIR"
else
    RESULTS_DIR="$BASE_RESULTS_DIR"
fi

mkdir -p $RESULTS_DIR

# Build the example range argument
EXAMPLE_RANGE_ARG=""
if [ -n "$EXAMPLE_RANGE" ]; then
    EXAMPLE_RANGE_ARG="--example-range $EXAMPLE_RANGE"
fi

# Build boolean flags
TARGET_OVERRIDE_ARG=""
if [ "$TARGET_OVERRIDE" = true ]; then
    TARGET_OVERRIDE_ARG="--target-override"
fi

VERBOSE_ARG=""
if [ "$VERBOSE" = true ]; then
    VERBOSE_ARG="--verbose"
fi

WANDB_OFFLINE_ARG=""
if [ "$WANDB_OFFLINE" = true ]; then
    WANDB_OFFLINE_ARG="--wandb-offline"
fi

# WANDB Initialization (WANDB_RUN_DIR should be set by SLURM script, but set default if not)
WANDB_RUN_DIR=${WANDB_RUN_DIR:-"/scratch/gpfs/KOROLOVA/bt4811/wandb_runs"}
mkdir -p $WANDB_RUN_DIR
python -c "import wandb; wandb.init(project='$WANDB_PROJECT', entity='$WANDB_ENTITY', dir='$WANDB_RUN_DIR')" 2>/dev/null || echo "WANDB initialization skipped (offline mode or error)"

# Run the Attack (The main experiment)
echo "Step 8/11: Running the attack experiment..."
echo "Job ID: $JOB_ID"
echo "Example range: $EXAMPLE_RANGE"
echo "Results directory: $RESULTS_DIR"
echo "Beta: $BETA"
echo "Steps: $NUM_STEPS"
echo "Model: $MODEL_NAME"
echo "Input CSV: $INPUT_CSV"

python -m reasoning_attacks.attack.run_experiments \
    --beta $BETA \
    --num-steps $NUM_STEPS \
    --num-target-tokens $NUM_TARGET_TOKENS \
    --num-refusal-tokens $NUM_REFUSAL_TOKENS \
    --num-gpus $NUM_GPUS \
    --runs-per-gpu $RUNS_PER_GPU \
    --max-examples $MAX_EXAMPLES \
    --input-csv "$INPUT_CSV" \
    $EXAMPLE_RANGE_ARG \
    $VERBOSE_ARG \
    --results-dir "$RESULTS_DIR" \
    $TARGET_OVERRIDE_ARG \
    $WANDB_OFFLINE_ARG

# Check results
echo "Step 10/11: Final result check"
find $RESULTS_DIR -type f -name "*.csv"
find reasoning_attacks/attack/results -type f 2>/dev/null || echo "No results in reasoning_attacks/attack/results"
