# Unified Experiment Management System

This system provides a configuration-driven approach to managing multiple experiment types with different parameters, parallel job execution, and result consolidation.

## 🎯 **Key Features**

- **Configuration-driven**: All experiment parameters defined in YAML files
- **Multiple experiment types**: Support for GCG, orthogonalized attacks, and custom setups
- **Parallel execution**: Automatic job splitting and parallel submission
- **Flexible ranges**: Support for percentage and integer-based example selection
- **Result consolidation**: Automatic merging of results from parallel jobs
- **Easy management**: Single command to run any experiment type

## 📁 **File Structure**

```
reasoning-manipulation/
├── config.yaml                    # Main configuration file
├── config_gcg_examples.yaml       # GCG experiment examples
├── config_ort_examples.yaml       # Orthogonalized attack examples
├── run_experiments.py             # Main experiment manager
├── run_experiment_unified.sh      # Unified experiment script
├── consolidate_results_unified.py # Unified consolidation script
└── run_*.slurm                    # Generated SLURM scripts
```

## 🚀 **Quick Start**

### 1. List Available Experiments
```bash
python run_experiments.py --list
```

### 2. Run a Specific Experiment
```bash
# Run orthogonalized attack with beta=0.5
python run_experiments.py --subexperiment ort_v3_1f

# Run GCG short experiment
python run_experiments.py --config config_gcg_examples.yaml --subexperiment gcg_short
```

### 3. Dry Run (Test Without Submitting)
```bash
python run_experiments.py --subexperiment ort_v3_1f --dry-run
```

### 4. Consolidate Results
```bash
# Consolidate all experiments
python consolidate_results_unified.py

# Consolidate specific experiment
python consolidate_results_unified.py --subexperiment ort_v3_1f
```

## 📋 **Configuration Structure**

### Global Settings
```yaml
global:
  netid: "bt4811"
  scratch_dir: "/scratch/gpfs/KOROLOVA"
  email: "your@email.com"
  environment:
    conda_env: "reasoning_env"
    modules: ["anaconda3/2025.6", "cuda/12.4"]
```

### Subexperiments
```yaml
subexperiments:
  my_experiment:
    description: "My custom experiment"
    results_dir: "results_my_experiment"
    model: "deepseek_r1"
    dataset: "cot150"
    
    experiment:
      beta: 0.5
      num_steps: 150
      num_target_tokens: 20
      num_refusal_tokens: 45
      num_gpus: 8
      target_override: true
    
    parallel:
      num_jobs: 4
      range_type: "percentage"
      range_spec: "0.0:1.0"
```

## 🔧 **Usage Examples**

### Example 1: Run Orthogonalized Attack
```bash
# Submit 4 parallel jobs for orthogonalized attack
python run_experiments.py --subexperiment ort_v3_1f

# Monitor jobs
squeue -u $USER

# After completion, consolidate results
python consolidate_results_unified.py --subexperiment ort_v3_1f
```

### Example 2: Run GCG Experiments
```bash
# Use GCG-specific config
python run_experiments.py --config config_gcg_examples.yaml --subexperiment gcg_short

# Run multiple GCG experiments
python run_experiments.py --config config_gcg_examples.yaml --subexperiment gcg_long
```

### Example 3: Custom Experiment
```yaml
# Add to config.yaml
subexperiments:
  my_custom:
    description: "Custom experiment with specific parameters"
    results_dir: "results_custom"
    model: "deepseek_r1"
    dataset: "cot150"
    
    experiment:
      beta: 0.3
      num_steps: 200
      num_target_tokens: 25
      num_refusal_tokens: 50
      num_gpus: 4
      target_override: true
    
    parallel:
      num_jobs: 2
      range_type: "integer"
      range_spec: "100:500"
```

```bash
# Run custom experiment
python run_experiments.py --subexperiment my_custom
```

## 📊 **Result Organization**

Results are organized in the following structure:
```
/scratch/gpfs/KOROLOVA/bt4811/b2/
├── results_ort_v3_1f/
│   ├── 1/                    # Job 1 results
│   ├── 2/                    # Job 2 results
│   ├── 3/                    # Job 3 results
│   ├── 4/                    # Job 4 results
│   └── final/                # Consolidated results
│       └── consolidated_results.csv
├── results_gcg_short/
│   ├── 1/, 2/
│   └── final/
└── results_custom/
    ├── 1/, 2/
    └── final/
```

## 🛠 **Advanced Usage**

### Custom Configuration Files
```bash
# Create your own config file
cp config.yaml my_experiments.yaml

# Edit my_experiments.yaml with your experiments

# Run experiments from your config
python run_experiments.py --config my_experiments.yaml --subexperiment my_experiment
```

### Batch Processing
```bash
# Run multiple experiments sequentially
for exp in ort_v3_1f gcg_short gcg_long; do
    python run_experiments.py --subexperiment $exp
    # Wait for completion, then consolidate
    python consolidate_results_unified.py --subexperiment $exp
done
```

### Monitoring and Debugging
```bash
# Check job status
squeue -u $USER

# View job output
tail -f slurm_output_ort_v3_1f_*.out

# Check specific job details
scontrol show job [job_id]
```

## 🔍 **Troubleshooting**

### Common Issues

1. **Job fails to submit**
   - Check SLURM configuration in config file
   - Verify GPU availability: `sinfo -p gpu`

2. **Missing results**
   - Check example range doesn't exceed dataset size
   - Verify input CSV file exists

3. **Consolidation errors**
   - Ensure all jobs completed successfully
   - Check file permissions in results directory

### Debug Mode
```bash
# Dry run to see what would be submitted
python run_experiments.py --subexperiment ort_v3_1f --dry-run

# Check configuration
python run_experiments.py --list
```

## 📈 **Performance Tips**

1. **GPU Allocation**: Adjust `num_gpus` based on available resources
2. **Job Splitting**: Use more jobs for better parallelization
3. **Time Limits**: Set appropriate SLURM time limits based on experiment complexity
4. **Memory**: Adjust memory requirements based on model size

## 🎉 **Benefits**

- **Unified Interface**: Single command for all experiment types
- **Easy Configuration**: YAML-based configuration is human-readable
- **Parallel Execution**: Automatic job splitting and submission
- **Result Management**: Automatic consolidation and organization
- **Reproducibility**: All parameters tracked in configuration
- **Scalability**: Easy to add new experiment types and parameters
