#!/bin/bash

# Script to copy results from scratch directory back to home directory
# This can be run manually if needed or after job completion

NETID="bt4811"
HOME_DIR="/home/$NETID"
SCRATCH_DIR="/scratch/gpfs/KOROLOVA"
PROJECT_DIR="$HOME_DIR/reasoning_llm_att"
RESULTS_DIR="$SCRATCH_DIR/$NETID"

echo "Copying results from scratch to home directory..."

# Create results directory in home if it doesn't exist
mkdir -p "$PROJECT_DIR/results"

# Copy results
if [ -d "$RESULTS_DIR/results" ]; then
    echo "Copying experiment results..."
    cp -r "$RESULTS_DIR/results"/* "$PROJECT_DIR/results/" 2>/dev/null || echo "No results to copy"
else
    echo "No results directory found in scratch"
fi

# Copy W&B logs
if [ -d "$RESULTS_DIR/wandb_runs" ]; then
    echo "Copying W&B logs..."
    cp -r "$RESULTS_DIR/wandb_runs" "$PROJECT_DIR/" 2>/dev/null || echo "No W&B logs to copy"
else
    echo "No W&B logs found in scratch"
fi

echo "Copy completed!"
