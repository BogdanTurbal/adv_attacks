#!/bin/bash

# Unified reprompting attack experiment script
# This script runs reprompting-based adversarial attacks

# Parse command line arguments
EXAMPLE_RANGE=""
JOB_ID=""
RESULTS_SUBDIR=""
RESULTS_DIR_PARAM=""
NUM_ITERS=""
NUM_BRANCHES=""
MEMORY=""
K=""
BATCH_SIZE=""
MAX_EXAMPLES=""
INPUT_CSV=""
MODEL_NAME=""
ATTACKER_MODEL_NAME=""
WANDB_PROJECT=""
WANDB_ENTITY=""
VERBOSE=false
WANDB_OFFLINE=false
ATTACKER_API=false
ATTACKER_API_MODEL=""
ATTACKER_API_KEY=""
ATTACKER_API_BASE_URL="https://openrouter.ai/api/v1"

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
        --num-iters)
            NUM_ITERS="$2"
            shift 2
            ;;
        --num-branches)
            NUM_BRANCHES="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --K)
            K="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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
        --attacker-model-name)
            ATTACKER_MODEL_NAME="$2"
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
        --verbose)
            VERBOSE=true
            shift
            ;;
        --wandb-offline)
            WANDB_OFFLINE=true
            shift
            ;;
        --attacker-quantize)
            ATTACKER_QUANTIZE=true
            shift
            ;;
        --attacker-quantize-bits)
            ATTACKER_QUANTIZE_BITS="$2"
            shift 2
            ;;
        --attacker-use-flash-attention)
            ATTACKER_USE_FLASH_ATTENTION=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --attacker-api)
            ATTACKER_API=true
            shift
            ;;
        --attacker-api-model)
            ATTACKER_API_MODEL="$2"
            shift 2
            ;;
        --attacker-api-key)
            ATTACKER_API_KEY="$2"
            shift 2
            ;;
        --attacker-api-base-url)
            ATTACKER_API_BASE_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$NUM_ITERS" ] || [ -z "$NUM_BRANCHES" ] || [ -z "$MEMORY" ] || [ -z "$K" ] || [ -z "$BATCH_SIZE" ]; then
    echo "Error: Missing required parameters (num-iters, num-branches, memory, K, batch-size)"
    exit 1
fi

# Set default values for optional parameters
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
INPUT_CSV=${INPUT_CSV:-"reasoning_attacks/dataset/orthogonalized_outputs_cot150_2048.csv"}
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
ATTACKER_MODEL_NAME=${ATTACKER_MODEL_NAME:-"mistralai/Mixtral-8x7B-Instruct-v0.1"}
WANDB_PROJECT=${WANDB_PROJECT:-"reprompting_attacks"}
WANDB_ENTITY=${WANDB_ENTITY:-"bogdan-turbal-y"}
NUM_GPUS=${NUM_GPUS:-4}  # Default to 4 GPUs

# Add Adversarial-Reasoning to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Adversarial-Reasoning"

# Set results directory (inside reasoning_attacks_res)
if [ -n "$RESULTS_DIR_PARAM" ]; then
    BASE_RESULTS_DIR="/u/bt4811/reasoning_attacks_res/$RESULTS_DIR_PARAM"
else
    BASE_RESULTS_DIR="/u/bt4811/reasoning_attacks_res/results_default"
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
VERBOSE_ARG=""
if [ "$VERBOSE" = true ]; then
    VERBOSE_ARG="--verbose"
fi

WANDB_OFFLINE_ARG=""
if [ "$WANDB_OFFLINE" = true ]; then
    WANDB_OFFLINE_ARG="--wandb-offline"
fi

ATTACKER_QUANTIZE_ARG=""
if [ "$ATTACKER_QUANTIZE" = true ]; then
    ATTACKER_QUANTIZE_ARG="--attacker-quantize"
fi

ATTACKER_QUANTIZE_BITS_ARG=""
if [ -n "$ATTACKER_QUANTIZE_BITS" ]; then
    ATTACKER_QUANTIZE_BITS_ARG="--attacker-quantize-bits $ATTACKER_QUANTIZE_BITS"
fi

ATTACKER_USE_FLASH_ATTENTION_ARG=""
if [ "$ATTACKER_USE_FLASH_ATTENTION" = true ]; then
    ATTACKER_USE_FLASH_ATTENTION_ARG="--attacker-use-flash-attention"
fi

ATTACKER_API_ARG=""
if [ "$ATTACKER_API" = true ]; then
    ATTACKER_API_ARG="--attacker-api"
fi

ATTACKER_API_MODEL_ARG=""
if [ -n "$ATTACKER_API_MODEL" ]; then
    ATTACKER_API_MODEL_ARG="--attacker-api-model $ATTACKER_API_MODEL"
fi

ATTACKER_API_KEY_ARG=""
if [ -n "$ATTACKER_API_KEY" ]; then
    ATTACKER_API_KEY_ARG="--attacker-api-key $ATTACKER_API_KEY"
fi

ATTACKER_API_BASE_URL_ARG=""
if [ -n "$ATTACKER_API_BASE_URL" ]; then
    ATTACKER_API_BASE_URL_ARG="--attacker-api-base-url $ATTACKER_API_BASE_URL"
fi

# WANDB Initialization (WANDB_RUN_DIR should be set by SLURM script, but set default if not)
WANDB_RUN_DIR=${WANDB_RUN_DIR:-"/u/bt4811/wandb_runs"}
mkdir -p $WANDB_RUN_DIR
python -c "import wandb; wandb.init(project='$WANDB_PROJECT', entity='$WANDB_ENTITY', dir='$WANDB_RUN_DIR')" 2>/dev/null || echo "WANDB initialization skipped (offline mode or error)"

# Run the Reprompting Attack (The main experiment)
echo "Running reprompting attack experiment..."
echo "Job ID: $JOB_ID"
echo "Example range: $EXAMPLE_RANGE"
echo "Results directory: $RESULTS_DIR"
echo "Num iters: $NUM_ITERS"
echo "Num branches: $NUM_BRANCHES"
echo "Memory: $MEMORY"
echo "K: $K"
echo "Batch size: $BATCH_SIZE"
echo "Model: $MODEL_NAME"
echo "Attacker model: $ATTACKER_MODEL_NAME"
echo "Input CSV: $INPUT_CSV"

python Adversarial-Reasoning/run_reprompting_attack.py \
    $EXAMPLE_RANGE_ARG \
    --job-id "$JOB_ID" \
    --results-dir "$RESULTS_DIR" \
    --num-iters $NUM_ITERS \
    --num-branches $NUM_BRANCHES \
    --memory $MEMORY \
    --K $K \
    --batch-size $BATCH_SIZE \
    --max-examples $MAX_EXAMPLES \
    --input-csv "$INPUT_CSV" \
    --model-name "$MODEL_NAME" \
    --attacker-model-name "$ATTACKER_MODEL_NAME" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    $VERBOSE_ARG \
    $WANDB_OFFLINE_ARG \
    $ATTACKER_API_ARG \
    $ATTACKER_API_MODEL_ARG \
    $ATTACKER_API_KEY_ARG \
    $ATTACKER_API_BASE_URL_ARG \
    $ATTACKER_QUANTIZE_ARG \
    $ATTACKER_QUANTIZE_BITS_ARG \
    $ATTACKER_USE_FLASH_ATTENTION_ARG \
    --num-gpus $NUM_GPUS

# Check results
echo "Final result check"
find $RESULTS_DIR -type f -name "*.csv" 2>/dev/null || echo "No CSV results found"

