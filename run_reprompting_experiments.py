#!/usr/bin/env python3
"""
Reprompting Attack Experiment Manager
Similar to run_experiments.py but for reprompting attacks
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
    required_fields = ['results_dir', 'model', 'attacker_model', 'dataset', 'experiment', 'parallel']
    
    for field in required_fields:
        if field not in subexp_config:
            print(f"Error: Subexperiment '{subexp_name}' missing required field: {field}")
            return False
    
    # Validate experiment parameters
    exp_params = subexp_config['experiment']
    required_exp_fields = ['num_iters', 'num_branches', 'memory', 'K', 'batch_size']
    
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
    """Generate SLURM script content for a reprompting subexperiment."""
    
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
    attacker_model_name = global_config['models'][subexp_config['attacker_model']]['name']
    dataset_csv = global_config['datasets'][subexp_config['dataset']]['input_csv']
    
    # Get OpenRouter API settings from global config
    openrouter_config = global_config['global'].get('environment', {}).get('openrouter', {})
    openrouter_api_key = openrouter_config.get('api_key', '')
    openrouter_model = openrouter_config.get('model', 'mistralai/mixtral-8x7b-instruct')
    openrouter_base_url = openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
    
    # Get attacker_api setting from experiment config
    attacker_api = experiment_params.get('attacker_api', False)
    
    # Get results directory for this experiment (where SLURM logs will be saved)
    results_dir = subexp_config['results_dir']
    experiment_results_dir = f"/scratch/gpfs/KOROLOVA/bt4811/usefatt/reasoning_attacks_res/{results_dir}"
    
    # Generate SLURM script
    # Build constraint line conditionally
    constraint_line = ""
    if 'constraint' in slurm_settings and slurm_settings['constraint'] is not None and slurm_settings['constraint'] != "":
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
#SBATCH --output={experiment_results_dir}/slurm_output_{subexp_name}_%j.out
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

# Ensure experiment results directory exists (for SLURM output logs)
mkdir -p {experiment_results_dir}

# Change to project directory (usefatt)
cd $PROJECT_DIR

# ====================
# Environment Setup
# ====================
module purge

# --- OFFLINE & PATH CONFIGURATION ---
# W&B Offline Configuration
export WANDB_MODE="{'offline' if global_settings['environment']['wandb']['offline_mode'] else 'online'}"
export WANDB_DIR="$HOME_DIR/wandb_runs"
mkdir -p $WANDB_DIR

# HuggingFace Cache Configuration
export HF_HOME="{global_settings['environment']['huggingface']['cache_dir']}"

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE="{'1' if global_settings['environment']['huggingface']['offline_mode'] else '0'}"
export TRANSFORMERS_OFFLINE="{'1' if global_settings['environment']['huggingface']['offline_mode'] else '0'}"

mkdir -p $HF_HOME
"""
    
    # Add module loads
    for module in global_settings['environment']['modules']:
        script_content += f"\nmodule load {module}"
    
    script_content += f"""

# Activate the Conda environment
source ~/.bashrc
conda activate {global_settings['environment']['conda_env']}

# WANDB_RUN_DIR for this specific run
export WANDB_RUN_DIR="$RESULTS_DIR/wandb_runs"
mkdir -p $WANDB_RUN_DIR

# HUGGINGFACE_HUB_CACHE (same as HF_HOME, already set above)
export HUGGINGFACE_HUB_CACHE="{global_settings['environment']['huggingface']['cache_dir']}"

# Note: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are already set above before module loads

# ====================
# EXECUTION
# ====================
echo "Starting {subexp_name} experiment execution at $(date)"
echo "Job ID: $JOB_ID"
echo "Example range: $EXAMPLE_RANGE"
echo "Results directory: $RESULTS_DIR"

# Execute the main experiment steps (from usefatt directory, which contains run_reprompting_unified.sh)
bash ./run_reprompting_unified.sh \\
    --example-range "$EXAMPLE_RANGE" \\
    --job-id "$JOB_ID" \\
    --results-subdir "$JOB_ID" \\
    --results-dir "{subexp_config['results_dir']}" \\
    --num-iters {experiment_params['num_iters']} \\
    --num-branches {experiment_params['num_branches']} \\
    --memory {experiment_params['memory']} \\
    --K {experiment_params['K']} \\
    --batch-size {experiment_params['batch_size']} \\
    --max-examples {experiment_params.get('max_examples', -1)} \\
    --input-csv "{dataset_csv}" \\
    --model-name "{model_name}" \\
    --attacker-model-name "{attacker_model_name}" \\
    --wandb-project "{subexp_config.get('wandb_project', 'reprompting_attacks')}" \\
    --wandb-entity "{global_settings['environment']['wandb']['entity']}" \\
    {"--verbose" if experiment_params.get('verbose', False) else ""} \\
    {"--wandb-offline" if global_settings['environment']['wandb']['offline_mode'] else ""} \\
    {"--attacker-api" if attacker_api else ""} \\
    {f"--attacker-api-model {openrouter_model}" if attacker_api else ""} \\
    {f"--attacker-api-key {openrouter_api_key}" if attacker_api else ""} \\
    {f"--attacker-api-base-url {openrouter_base_url}" if attacker_api else ""} \\
    {"--attacker-quantize" if experiment_params.get('attacker_quantize', False) and not attacker_api else ""} \\
    {f"--attacker-quantize-bits {experiment_params.get('attacker_quantize_bits', 8)}" if experiment_params.get('attacker_quantize', False) and not attacker_api else ""} \\
    {"--attacker-use-flash-attention" if experiment_params.get('attacker_use_flash_attention', False) and not attacker_api else ""} \\
    --num-gpus {experiment_params.get('num_gpus', 4)}

# Copy results back to home directory
echo "Copying results back to home directory..."
cp -r $RESULTS_DIR/* $PROJECT_DIR/results/ 2>/dev/null || echo "No results to copy"
cp -r $WANDB_RUN_DIR $PROJECT_DIR/ 2>/dev/null || echo "No W&B logs to copy"

echo "Job $JOB_ID finished at $(date)"
"""
    
    return script_content


def get_total_examples_in_range(dataset_csv: str, range_type: str, range_spec: str) -> Tuple[int, int, int]:
    """Load dataset and compute total examples in the specified range."""
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
        return -1, -1, -1


def generate_python_command(subexp_name: str, subexp_config: Dict[str, Any], global_config: Dict[str, Any], example_range: str, job_id: str) -> str:
    """Generate Python command for running reprompting attack directly (without SLURM)."""
    
    global_settings = global_config['global']
    experiment_params = subexp_config['experiment']
    
    # Get model and dataset info
    model_name = global_config['models'][subexp_config['model']]['name']
    attacker_model_name = global_config['models'][subexp_config['attacker_model']]['name']
    dataset_csv = global_config['datasets'][subexp_config['dataset']]['input_csv']
    
    # Get OpenRouter API settings
    openrouter_config = global_settings.get('environment', {}).get('openrouter', {})
    openrouter_api_key = openrouter_config.get('api_key', '')
    openrouter_model = openrouter_config.get('model', 'mistralai/mixtral-8x7b-instruct')
    openrouter_base_url = openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
    
    # Get attacker_api setting
    attacker_api = experiment_params.get('attacker_api', False)
    
    # Build Python command
    results_dir = subexp_config['results_dir']
    project_dir = global_settings['project_dir']
    
    # For Colab, use relative paths; otherwise use absolute
    if '/content' in project_dir:
        script_path = "Adversarial-Reasoning/run_reprompting_attack.py"
    else:
        script_path = f"{project_dir}/Adversarial-Reasoning/run_reprompting_attack.py"
    
    cmd_parts = [
        "python",
        script_path,
        f"--example-range {example_range}",
        f"--job-id {job_id}",
        f"--results-dir {results_dir}/{job_id}",
        f"--num-iters {experiment_params['num_iters']}",
        f"--num-branches {experiment_params['num_branches']}",
        f"--memory {experiment_params['memory']}",
        f"--K {experiment_params['K']}",
        f"--batch-size {experiment_params['batch_size']}",
        f"--max-examples {experiment_params.get('max_examples', -1)}",
        f"--input-csv {dataset_csv}",
        f"--model-name {model_name}",
        f"--attacker-model-name {attacker_model_name}",
        f"--wandb-project {subexp_config.get('wandb_project', 'reprompting_attacks')}",
        f"--wandb-entity {global_settings['environment']['wandb']['entity']}",
    ]
    
    if experiment_params.get('verbose', False):
        cmd_parts.append("--verbose")
    
    if global_settings['environment']['wandb']['offline_mode']:
        cmd_parts.append("--wandb-offline")
    
    if attacker_api:
        cmd_parts.append("--attacker-api")
        cmd_parts.append(f"--attacker-api-model {openrouter_model}")
        cmd_parts.append(f"--attacker-api-key {openrouter_api_key}")
        cmd_parts.append(f"--attacker-api-base-url {openrouter_base_url}")
    else:
        if experiment_params.get('attacker_quantize', False):
            cmd_parts.append("--attacker-quantize")
            cmd_parts.append(f"--attacker-quantize-bits {experiment_params.get('attacker_quantize_bits', 8)}")
        if experiment_params.get('attacker_use_flash_attention', False):
            cmd_parts.append("--attacker-use-flash-attention")
    
    cmd_parts.append(f"--num-gpus {experiment_params.get('num_gpus', 1)}")
    
    return " \\\n    ".join(cmd_parts)


def submit_parallel_jobs(subexp_name: str, subexp_config: Dict[str, Any], global_config: Dict[str, Any], dry_run: bool = False, no_slurm: bool = False) -> List[str]:
    """Submit parallel jobs for a reprompting subexperiment."""
    
    parallel_params = subexp_config['parallel']
    num_jobs = parallel_params['num_jobs']
    range_type = parallel_params['range_type']
    range_spec = parallel_params['range_spec']
    results_dir = subexp_config['results_dir']
    
    # Get dataset path
    dataset_csv = global_config['datasets'][subexp_config['dataset']]['input_csv']
    
    if no_slurm:
        print(f"Generating Python commands for {num_jobs} parallel jobs for subexperiment '{subexp_name}'")
    else:
        print(f"Submitting {num_jobs} parallel jobs for subexperiment '{subexp_name}'")
    print(f"Range: {range_type} {range_spec}")
    print(f"Results directory: {results_dir}")
    
    # Try to compute uniform distribution
    total_examples_in_range, range_start_offset, range_end_offset = get_total_examples_in_range(dataset_csv, range_type, range_spec)
    
    # Generate SLURM script (still useful for reference even in no_slurm mode)
    slurm_script_content = generate_slurm_script(subexp_name, subexp_config, global_config)
    
    # Write SLURM script to file (save in reasoning_attacks_res directory)
    global_settings = global_config['global']
    reasoning_attacks_res_dir = global_settings.get('home_dir', "/scratch/gpfs/KOROLOVA/bt4811/usefatt/reasoning_attacks_res")
    os.makedirs(reasoning_attacks_res_dir, exist_ok=True)
    slurm_script_path = os.path.join(reasoning_attacks_res_dir, f"run_{subexp_name}.slurm")
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script_content)
    
    if not no_slurm:
        print(f"Generated SLURM script: {slurm_script_path}")
    
    # Compute ranges for each job
    if total_examples_in_range > 0:
        examples_per_job = ceil(total_examples_in_range / num_jobs)
        ranges_to_submit = {}
        
        if range_type == "percentage":
            # Load dataset to get total count for percentage conversion
            try:
                if not os.path.isabs(dataset_csv):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    dataset_csv_resolved = os.path.join(script_dir, dataset_csv)
                else:
                    dataset_csv_resolved = dataset_csv
                df = pd.read_csv(dataset_csv_resolved)
                total_examples = len(df)
                
                for i in range(num_jobs):
                    job_id = i + 1
                    job_start_idx = range_start_offset + i * examples_per_job
                    job_end_idx = min(range_start_offset + (i + 1) * examples_per_job, range_end_offset)
                    start_pct = job_start_idx / total_examples
                    end_pct = job_end_idx / total_examples
                    ranges_to_submit[job_id] = f"{start_pct:.6f}:{end_pct:.6f}"
            except:
                # Fallback
                start_pct, end_pct = map(float, range_spec.split(':'))
                total_range = end_pct - start_pct
                range_per_job = total_range / num_jobs
                for i in range(num_jobs):
                    job_id = i + 1
                    job_start = start_pct + i * range_per_job
                    job_end = start_pct + (i + 1) * range_per_job
                    ranges_to_submit[job_id] = f"{job_start:.6f}:{job_end:.6f}"
        else:
            # Integer range
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
    else:
        # Fallback
        if range_type == "percentage":
            start_pct, end_pct = map(float, range_spec.split(':'))
            total_range = end_pct - start_pct
            range_per_job = total_range / num_jobs
            ranges_to_submit = {}
            for i in range(num_jobs):
                job_id = i + 1
                job_start = start_pct + i * range_per_job
                job_end = start_pct + (i + 1) * range_per_job
                ranges_to_submit[job_id] = f"{job_start:.6f}:{job_end:.6f}"
        else:
            start_idx, end_idx = map(int, range_spec.split(':'))
            total_range = end_idx - start_idx
            range_per_job = total_range // num_jobs
            ranges_to_submit = {}
            for i in range(num_jobs):
                job_id = i + 1
                job_start = start_idx + i * range_per_job
                job_end = start_idx + (i + 1) * range_per_job
                if i == num_jobs - 1:
                    job_end = end_idx
                ranges_to_submit[job_id] = f"{job_start}:{job_end}"
    
    # Submit jobs or generate Python commands
    slurm_job_ids = {}
    submitted_job_ids = []
    
    if no_slurm:
        # Generate Python commands instead of submitting SLURM jobs
        print(f"\n{'='*80}")
        print(f"PYTHON COMMANDS FOR {subexp_name} (Copy these to run in Colab or locally)")
        print(f"{'='*80}\n")
        
        commands_file = os.path.join(reasoning_attacks_res_dir, f"{subexp_name}_commands.txt")
        
        with open(commands_file, 'w') as f:
            f.write(f"# Python commands for subexperiment: {subexp_name}\n")
            f.write(f"# Generated for {num_jobs} parallel jobs\n")
            f.write(f"# Results directory: {results_dir}\n\n")
            
            for job_id, example_range in ranges_to_submit.items():
                job_id_str = str(job_id)
                print(f"\n{'─'*80}")
                print(f"Job {job_id_str}: range {example_range}")
                print(f"{'─'*80}")
                
                python_cmd = generate_python_command(subexp_name, subexp_config, global_config, example_range, job_id_str)
                
                print(f"\n{python_cmd}\n")
                
                f.write(f"# Job {job_id_str}: range {example_range}\n")
                f.write(f"{python_cmd}\n\n")
                submitted_job_ids.append(f"job_{job_id_str}")
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {len(ranges_to_submit)} Python commands")
        print(f"Commands saved to: {commands_file}")
        print(f"{'='*80}\n")
        
        return submitted_job_ids
    
    # Original SLURM submission code
    for job_id, example_range in ranges_to_submit.items():
        job_id_str = str(job_id)
        print(f"Job {job_id_str}: range {example_range}")
        
        cmd = [
            'sbatch',
            f'--export=EXAMPLE_RANGE={example_range},JOB_ID={job_id_str},RESULTS_DIR={results_dir}',
            f'--job-name={subexp_name}_job_{job_id_str}',
            slurm_script_path
        ]
        
        if not dry_run:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    output = result.stdout.strip()
                    submitted_job_ids.append(output)
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
    
    return submitted_job_ids


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Reprompting Attack Experiment Manager")
    parser.add_argument("--config", "-c", default="config_reprompting.yaml", help="Configuration file path")
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
    
    # Validate subexperiment exists
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
    
    # Check for no_slurm flag
    no_slurm = config['global'].get('no_slurm', False)
    
    # Submit jobs
    print(f"\n{'='*60}")
    print(f"Running subexperiment: {args.subexperiment}")
    print(f"Description: {subexp_config.get('description', 'No description')}")
    if no_slurm:
        print(f"Mode: NO SLURM (Python commands will be generated)")
    print(f"{'='*60}")
    
    job_ids = submit_parallel_jobs(args.subexperiment, subexp_config, config, args.dry_run, no_slurm)
    
    if no_slurm:
        print(f"\n✓ Generated {len(job_ids)} Python commands")
        print("Copy the commands above or from the commands file to run in Colab or locally")
    elif args.dry_run:
        print(f"\n[DRY RUN] Would have submitted {len(job_ids)} jobs")
    else:
        print(f"\n✓ Successfully submitted {len(job_ids)} jobs")
        print("Monitor jobs with: squeue -u $USER")
        print("Check job status with: scontrol show job [job_id]")


if __name__ == "__main__":
    main()
