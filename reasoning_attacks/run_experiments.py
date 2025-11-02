#!/usr/bin/env python3
"""
Unified Experiment Manager
Reads configuration from YAML and spawns SLURM jobs for different experiment setups.
"""

import argparse
import os
import subprocess
import sys
import yaml
import pandas as pd
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from math import ceil


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)


def validate_subexperiment(subexp_name: str, subexp_config: Dict[str, Any], global_config: Dict[str, Any]) -> bool:
    """Validate that a subexperiment configuration is complete."""
    required_fields = ['results_dir', 'model', 'dataset', 'experiment', 'parallel']
    
    for field in required_fields:
        if field not in subexp_config:
            print(f"Error: Subexperiment '{subexp_name}' missing required field: {field}")
            return False
    
    # Validate experiment parameters
    exp_params = subexp_config['experiment']
    required_exp_fields = ['beta', 'num_steps', 'num_target_tokens', 'num_refusal_tokens', 'num_gpus']
    
    for field in required_exp_fields:
        if field not in exp_params:
            print(f"Error: Subexperiment '{subexp_name}' experiment config missing: {field}")
            return False
    
    # Validate parallel parameters
    parallel_params = subexp_config['parallel']
    required_parallel_fields = ['num_jobs', 'range_type', 'range_spec']
    
    for field in required_parallel_fields:
        if field not in parallel_params:
            print(f"Error: Subexperiment '{subexp_name}' parallel config missing: {field}")
            return False
    
    return True


def generate_slurm_script(subexp_name: str, subexp_config: Dict[str, Any], global_config: Dict[str, Any]) -> str:
    """Generate SLURM script content for a subexperiment."""
    
    # Get configurations
    global_settings = global_config['global']
    slurm_defaults = global_config['slurm_defaults']
    slurm_overrides = subexp_config.get('slurm', {})
    experiment_params = subexp_config['experiment']
    parallel_params = subexp_config['parallel']
    
    # Merge SLURM settings (overrides take precedence)
    slurm_settings = {**slurm_defaults, **slurm_overrides}
    
    # Get model and dataset info
    model_name = global_config['models'][subexp_config['model']]['name']
    dataset_csv = global_config['datasets'][subexp_config['dataset']]['input_csv']
    
    # Generate SLURM script
    # Build constraint line conditionally
    constraint_line = ""
    if 'constraint' in slurm_settings and slurm_settings['constraint']:
        constraint_line = f"#SBATCH --constraint={slurm_settings['constraint']}\n"
    
    script_content = f"""#!/bin/bash

# ====================
# SLURM Arguments
# ====================
#SBATCH --job-name={subexp_name}
#SBATCH --nodes={slurm_settings['nodes']}
#SBATCH --ntasks={slurm_settings['ntasks']}
#SBATCH --cpus-per-task={slurm_settings['cpus_per_task']}
#SBATCH --mem={slurm_settings['mem']}
#SBATCH --time={slurm_settings['time']}
#SBATCH --gres={slurm_settings['gres']}
#SBATCH --output=slurm_output_{subexp_name}_%j.out
{constraint_line}#SBATCH --mail-type={slurm_settings['mail_type']}
#SBATCH --mail-user={global_settings['email']}

# Set your Princeton NetID
NETID="{global_settings['netid']}"

# Define directories
export HOME_DIR="{global_settings['home_dir']}"
export SCRATCH_DIR="{global_settings['scratch_dir']}"
export PROJECT_DIR="{global_settings['project_dir']}"
export RESULTS_DIR="$SCRATCH_DIR/$NETID"

# Create necessary directories
mkdir -p $SCRATCH_DIR
mkdir -p $RESULTS_DIR

# Change to project directory
cd $PROJECT_DIR/reasoning_attacks

# ====================
# Environment Setup
# ====================
module purge"""
    
    # Add module loads
    for module in global_settings['environment']['modules']:
        script_content += f"\nmodule load {module}"
    
    script_content += f"""

# Activate the Conda environment
source ~/.bashrc
conda activate {global_settings['environment']['conda_env']}

# --- OFFLINE & PATH CONFIGURATION ---
# W&B Offline Configuration
export WANDB_MODE="{'offline' if global_settings['environment']['wandb']['offline_mode'] else 'online'}"
export WANDB_RUN_DIR="$RESULTS_DIR/wandb_runs"
mkdir -p $WANDB_RUN_DIR

# Set HuggingFace cache to scratch directory
export HF_HOME="{global_settings['environment']['huggingface']['cache_dir']}"
export HUGGINGFACE_HUB_CACHE="{global_settings['environment']['huggingface']['cache_dir']}"

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE="{'1' if global_settings['environment']['huggingface']['offline_mode'] else '0'}"
export TRANSFORMERS_OFFLINE="{'1' if global_settings['environment']['huggingface']['offline_mode'] else '0'}"

mkdir -p $HF_HOME

# ====================
# EXECUTION
# ====================
echo "Starting {subexp_name} experiment execution at $(date)"
echo "Job ID: $JOB_ID"
echo "Example range: $EXAMPLE_RANGE"
echo "Results directory: $RESULTS_DIR"

# Execute the main experiment steps
bash ./run_experiment_unified.sh \\
    --example-range "$EXAMPLE_RANGE" \\
    --job-id "$JOB_ID" \\
    --results-subdir "$JOB_ID" \\
    --results-dir "{subexp_config['results_dir']}" \\
    --beta {experiment_params['beta']} \\
    --num-steps {experiment_params['num_steps']} \\
    --num-target-tokens {experiment_params['num_target_tokens']} \\
    --num-refusal-tokens {experiment_params['num_refusal_tokens']} \\
    --num-gpus {experiment_params['num_gpus']} \\
    --runs-per-gpu {experiment_params['runs_per_gpu']} \\
    --max-examples {experiment_params['max_examples']} \\
    --input-csv "{dataset_csv}" \\
    --model-name "{model_name}" \\
    --wandb-project "{subexp_config.get('wandb_project', 'attor')}" \\
    --wandb-entity "{global_settings['environment']['wandb']['entity']}" \\
    {"--target-override" if experiment_params.get('target_override', False) else ""} \\
    {"--verbose" if experiment_params.get('verbose', False) else ""} \\
    {"--wandb-offline" if global_settings['environment']['wandb']['offline_mode'] else ""}

# Copy results back to home directory
echo "Copying results back to home directory..."
cp -r $RESULTS_DIR/* $PROJECT_DIR/results/ 2>/dev/null || echo "No results to copy"
cp -r $WANDB_RUN_DIR $PROJECT_DIR/ 2>/dev/null || echo "No W&B logs to copy"

echo "Job $JOB_ID finished at $(date)"
"""
    
    return script_content


def compute_uniform_distribution(num_jobs: int, num_gpus: int, total_examples_in_range: int) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    """
    Compute uniform examples per GPU such that num_jobs * num_gpus * examples_per_gpu >= total_examples_in_range.
    Returns (examples_per_gpu, assignments) where assignments is list of (job_id, gpu_id, start_idx, end_idx).
    """
    total_capacity = num_jobs * num_gpus
    examples_per_gpu = ceil(total_examples_in_range / total_capacity)
    
    assignments = []
    current_idx = 0
    
    for job_id in range(1, num_jobs + 1):
        for gpu_id in range(num_gpus):
            start_idx = current_idx
            end_idx = min(current_idx + examples_per_gpu, total_examples_in_range)
            if start_idx < total_examples_in_range:
                assignments.append((job_id, gpu_id, start_idx, end_idx))
            current_idx = end_idx
            if current_idx >= total_examples_in_range:
                break
        if current_idx >= total_examples_in_range:
            break
    
    return examples_per_gpu, assignments


def get_total_examples_in_range(dataset_csv: str, range_type: str, range_spec: str) -> Tuple[int, int, int]:
    """Load dataset and compute total examples in the specified range.
    Returns (total_examples, range_start_idx, range_end_idx) where indices are in original dataset."""
    try:
        # Resolve relative paths relative to the script's directory
        if not os.path.isabs(dataset_csv):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_csv = os.path.join(script_dir, dataset_csv)
        df = pd.read_csv(dataset_csv)
        total_examples = len(df)
        
        if range_type == "percentage":
            start_pct, end_pct = map(float, range_spec.split(':'))
            start_idx = int(start_pct * total_examples)
            end_idx = int(end_pct * total_examples)
        elif range_type == "integer":
            start_idx, end_idx = map(int, range_spec.split(':'))
        else:
            start_idx = 0
            end_idx = total_examples
        
        examples_in_range = end_idx - start_idx
        return examples_in_range, start_idx, end_idx
    except Exception as e:
        print(f"Warning: Could not load dataset to compute total examples: {e}")
        print("Falling back to percentage-based distribution")
        return -1, -1, -1  # Signal to use old method


def submit_parallel_jobs(subexp_name: str, subexp_config: Dict[str, Any], global_config: Dict[str, Any], dry_run: bool = False) -> List[str]:
    """Submit parallel jobs for a subexperiment."""
    
    parallel_params = subexp_config['parallel']
    num_jobs = parallel_params['num_jobs']
    range_type = parallel_params['range_type']
    range_spec = parallel_params['range_spec']
    results_dir = subexp_config['results_dir']
    num_gpus = subexp_config['experiment']['num_gpus']
    
    # Get dataset path
    dataset_csv = global_config['datasets'][subexp_config['dataset']]['input_csv']
    
    print(f"Submitting {num_jobs} parallel jobs for subexperiment '{subexp_name}'")
    print(f"Range: {range_type} {range_spec}")
    print(f"Results directory: {results_dir}")
    print(f"GPUs per job: {num_gpus}")
    
    # Try to compute uniform distribution
    total_examples_in_range, range_start_offset, range_end_offset = get_total_examples_in_range(dataset_csv, range_type, range_spec)
    
    if total_examples_in_range > 0:
        print(f"Total examples in range: {total_examples_in_range}")
        print(f"Original dataset range: {range_start_offset} to {range_end_offset}")
        examples_per_gpu, assignments = compute_uniform_distribution(num_jobs, num_gpus, total_examples_in_range)
        print(f"Examples per GPU: {examples_per_gpu}")
        print(f"Total capacity: {num_jobs * num_gpus * examples_per_gpu} examples")
        
        # Convert relative indices to absolute dataset indices
        assignments_with_offset = []
        for job_id, gpu_id, rel_start, rel_end in assignments:
            abs_start = range_start_offset + rel_start
            abs_end = range_start_offset + rel_end
            assignments_with_offset.append((job_id, gpu_id, abs_start, abs_end))
        
        # Create results directory (will save CSV after job submission with SLURM IDs)
        base_results_dir = f"/scratch/gpfs/KOROLOVA/bt4811/b2/{results_dir}"
        os.makedirs(base_results_dir, exist_ok=True)
        assignment_csv_path = os.path.join(base_results_dir, f"{subexp_name}_assignment.csv")
        
        # Group assignments by job_id to create job ranges (using absolute indices)
        job_ranges = {}
        for job_id, gpu_id, abs_start, abs_end in assignments_with_offset:
            if job_id not in job_ranges:
                job_ranges[job_id] = {'min': abs_start, 'max': abs_end}
            else:
                job_ranges[job_id]['min'] = min(job_ranges[job_id]['min'], abs_start)
                job_ranges[job_id]['max'] = max(job_ranges[job_id]['max'], abs_end)
        
        # Convert to percentage or integer ranges based on original range_type
        if range_type == "percentage":
            # Load dataset to get total count for percentage conversion
            try:
                # Resolve relative paths relative to the script's directory
                dataset_csv_resolved = dataset_csv
                if not os.path.isabs(dataset_csv_resolved):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    dataset_csv_resolved = os.path.join(script_dir, dataset_csv_resolved)
                df = pd.read_csv(dataset_csv_resolved)
                total_examples = len(df)
                # Convert to percentage ranges
                ranges_to_submit = {}
                for job_id, rng in job_ranges.items():
                    start_pct = rng['min'] / total_examples
                    end_pct = rng['max'] / total_examples
                    ranges_to_submit[job_id] = f"{start_pct:.6f}:{end_pct:.6f}"
            except:
                # Fallback to integer ranges if we can't load dataset
                ranges_to_submit = {job_id: f"{rng['min']}:{rng['max']}" for job_id, rng in job_ranges.items()}
        else:
            ranges_to_submit = {job_id: f"{rng['min']}:{rng['max']}" for job_id, rng in job_ranges.items()}
    
    else:
        # Fallback to old percentage-based method
        print("Using percentage-based distribution (old method)")
        ranges_to_submit = {}
        if range_type == "percentage":
            start_pct, end_pct = map(float, range_spec.split(':'))
            total_range = end_pct - start_pct
            range_per_job = total_range / num_jobs
            
            for i in range(num_jobs):
                job_id = i + 1
                job_start = start_pct + i * range_per_job
                job_end = start_pct + (i + 1) * range_per_job
                ranges_to_submit[job_id] = f"{job_start:.6f}:{job_end:.6f}"
        elif range_type == "integer":
            start_idx, end_idx = map(int, range_spec.split(':'))
            total_range = end_idx - start_idx
            range_per_job = total_range // num_jobs
            
            for i in range(num_jobs):
                job_id = i + 1
                job_start = start_idx + i * range_per_job
                job_end = start_idx + (i + 1) * range_per_job
                if i == num_jobs - 1:
                    job_end = end_idx
                ranges_to_submit[job_id] = f"{job_start}:{job_end}"
    
    # Generate SLURM script
    slurm_script_content = generate_slurm_script(subexp_name, subexp_config, global_config)
    
    # Write SLURM script to file
    slurm_script_path = f"run_{subexp_name}.slurm"
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script_content)
    
    print(f"Generated SLURM script: {slurm_script_path}")
    
    # Submit jobs and capture SLURM job IDs
    slurm_job_ids = {}  # Maps job_id -> SLURM job ID
    submitted_job_ids = []
    
    for job_id, example_range in ranges_to_submit.items():
        job_id_str = str(job_id)
        print(f"Job {job_id_str}: range {example_range}")
        
        # Build command
        cmd = [
            'sbatch',
            f'--export=EXAMPLE_RANGE={example_range},JOB_ID={job_id_str},RESULTS_DIR={results_dir}',
            f'--job-name={subexp_name}_job_{job_id_str}',
            slurm_script_path
        ]
        
        if not dry_run:
            # Submit SLURM job
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    output = result.stdout.strip()
                    submitted_job_ids.append(output)
                    # Extract SLURM job ID from "Submitted batch job 1805283"
                    if "Submitted batch job" in output:
                        slurm_id = output.split()[-1]
                        slurm_job_ids[job_id] = slurm_id
                    else:
                        slurm_job_ids[job_id] = output
                    print(f"✓ Submitted job {job_id_str}: {output}")
                else:
                    print(f"✗ Failed to submit job {job_id_str}: {result.stderr}")
                    slurm_job_ids[job_id] = "FAILED"
            except Exception as e:
                print(f"✗ Error submitting job {job_id_str}: {e}")
                slurm_job_ids[job_id] = "ERROR"
        else:
            print(f"  [DRY RUN] Would submit: {' '.join(cmd)}")
            slurm_job_ids[job_id] = "dry_run"
            submitted_job_ids.append(f"dry_run_job_{job_id_str}")
    
    # Now write/update the assignment CSV with SLURM job IDs
    if total_examples_in_range > 0:
        # Create results directory if not already created
        base_results_dir = f"/scratch/gpfs/KOROLOVA/bt4811/b2/{results_dir}"
        os.makedirs(base_results_dir, exist_ok=True)
        assignment_csv_path = os.path.join(base_results_dir, f"{subexp_name}_assignment.csv")
        
        with open(assignment_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['job_id', 'gpu_number', 'start_example_id', 'end_example_id', 'num_examples', 'slurm_job_id', 'results_dir', 'config_reference'])
            config_ref = subexp_name  # Using subexperiment name as config reference
            for job_id, gpu_id, abs_start, abs_end in assignments_with_offset:
                num_examples = abs_end - abs_start
                slurm_id = slurm_job_ids.get(job_id, "UNKNOWN")
                # Results are saved to: base_results_dir/job_id (e.g., /scratch/gpfs/KOROLOVA/bt4811/b2/exp_20_results_ort_beta_0_5/1)
                results_subdir = os.path.join(base_results_dir, str(job_id))
                # end_example_id is inclusive in the CSV but exclusive in our code
                writer.writerow([job_id, gpu_id, abs_start, abs_end - 1, num_examples, slurm_id, results_subdir, config_ref])
        
        print(f"✓ Assignment CSV saved to: {assignment_csv_path} (with SLURM job IDs and results directories)")
    
    return submitted_job_ids


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Unified Experiment Manager")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--subexperiment", "-s", help="Subexperiment name to run")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without submitting jobs")
    parser.add_argument("--list", action="store_true", help="List available subexperiments")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # List subexperiments if requested
    if args.list:
        print("Available subexperiments:")
        for name, subexp in config['subexperiments'].items():
            description = subexp.get('description', 'No description')
            print(f"  {name}: {description}")
        return
    
    # Validate subexperiment exists (only if not listing)
    if not args.list and not args.subexperiment:
        print("Error: Must specify either --subexperiment or --list")
        sys.exit(1)
    
    if args.subexperiment and args.subexperiment not in config['subexperiments']:
        print(f"Error: Subexperiment '{args.subexperiment}' not found in configuration")
        print("Use --list to see available subexperiments")
        sys.exit(1)
    
    subexp_config = config['subexperiments'][args.subexperiment]
    
    # Validate configuration
    if not validate_subexperiment(args.subexperiment, subexp_config, config):
        sys.exit(1)
    
    # Submit jobs
    print(f"\n{'='*60}")
    print(f"Running subexperiment: {args.subexperiment}")
    print(f"Description: {subexp_config.get('description', 'No description')}")
    print(f"{'='*60}")
    
    job_ids = submit_parallel_jobs(args.subexperiment, subexp_config, config, args.dry_run)
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would have submitted {len(job_ids)} jobs")
    else:
        print(f"\n✓ Successfully submitted {len(job_ids)} jobs")
        print("Monitor jobs with: squeue -u $USER")
        print("Check job status with: scontrol show job [job_id]")


if __name__ == "__main__":
    main()
