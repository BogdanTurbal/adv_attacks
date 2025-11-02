#!/usr/bin/env python3
"""
Unified Results Consolidation Script
Consolidates results from multiple parallel job subfolders using configuration.
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import glob
from pathlib import Path


def load_config(config_path: str) -> dict:
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


def consolidate_subexperiment_results(subexp_name: str, subexp_config: dict, global_config: dict) -> bool:
    """Consolidate results for a specific subexperiment."""
    
    global_settings = global_config['global']
    results_dir_name = subexp_config['results_dir']
    
    # Construct base results directory path
    base_results_dir = f"{global_settings['scratch_dir']}/{global_settings['netid']}/b2/{results_dir_name}"
    final_results_dir = f"{base_results_dir}/final"
    
    print(f"Consolidating results for subexperiment: {subexp_name}")
    print(f"Base results directory: {base_results_dir}")
    print(f"Final results directory: {final_results_dir}")
    
    # Create final results directory
    os.makedirs(final_results_dir, exist_ok=True)
    
    # Find all CSV files in subdirectories
    csv_pattern = f"{base_results_dir}/**/*.csv"
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    # Exclude files in final directory
    csv_files = [f for f in csv_files if "/final/" not in f]
    
    if not csv_files:
        print(f"No CSV files found in {base_results_dir}")
        return False
    
    print(f"Found {len(csv_files)} CSV files to consolidate:")
    for file in csv_files:
        print(f"  {file}")
    
    # Read and concatenate all CSV files
    all_dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add source file information
            df['source_file'] = os.path.basename(csv_file)
            df['source_subdir'] = os.path.basename(os.path.dirname(csv_file))
            df['subexperiment'] = subexp_name
            all_dataframes.append(df)
            print(f"Loaded {len(df)} rows from {csv_file}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_dataframes:
        print("No valid CSV files found")
        return False
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by prompt_idx if it exists
    if 'prompt_idx' in combined_df.columns:
        combined_df = combined_df.sort_values('prompt_idx')
    
    # Save consolidated results
    output_file = os.path.join(final_results_dir, "consolidated_results.csv")
    combined_df.to_csv(output_file, index=False)
    
    print(f"Consolidated {len(combined_df)} total rows")
    print(f"Saved to: {output_file}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total experiments: {len(combined_df)}")
    if 'success' in combined_df.columns:
        successful = combined_df['success'].sum()
        failed = len(combined_df) - successful
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
    
    if 'source_subdir' in combined_df.columns:
        print("\nResults by subdirectory:")
        subdir_counts = combined_df['source_subdir'].value_counts()
        for subdir, count in subdir_counts.items():
            print(f"  {subdir}: {count} experiments")
    
    # Copy other files (logs, etc.) to final directory
    print("\nCopying other result files...")
    other_files = glob.glob(f"{base_results_dir}/**/*.log", recursive=True)
    other_files.extend(glob.glob(f"{base_results_dir}/**/*.txt", recursive=True))
    other_files.extend(glob.glob(f"{base_results_dir}/**/*.out", recursive=True))
    
    for file in other_files:
        if "/final/" not in file:
            subdir = os.path.basename(os.path.dirname(file))
            filename = os.path.basename(file)
            dest_file = os.path.join(final_results_dir, f"{subdir}_{filename}")
            try:
                import shutil
                shutil.copy2(file, dest_file)
                print(f"Copied {file} -> {dest_file}")
            except Exception as e:
                print(f"Error copying {file}: {e}")
    
    print(f"\nConsolidation complete for {subexp_name}!")
    print(f"Final results saved to: {final_results_dir}")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Unified Results Consolidation")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--subexperiment", "-s", help="Specific subexperiment to consolidate (if not provided, consolidates all)")
    parser.add_argument("--list", action="store_true", help="List available subexperiments")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # List subexperiments if requested
    if args.list:
        print("Available subexperiments:")
        for name, subexp in config['subexperiments'].items():
            description = subexp.get('description', 'No description')
            results_dir = subexp.get('results_dir', 'No results dir')
            print(f"  {name}: {description} (results: {results_dir})")
        return
    
    # Determine which subexperiments to consolidate
    if args.subexperiment:
        if args.subexperiment not in config['subexperiments']:
            print(f"Error: Subexperiment '{args.subexperiment}' not found in configuration")
            print("Use --list to see available subexperiments")
            sys.exit(1)
        subexperiments_to_process = {args.subexperiment: config['subexperiments'][args.subexperiment]}
    else:
        subexperiments_to_process = config['subexperiments']
    
    # Consolidate results for each subexperiment
    print(f"{'='*60}")
    print(f"Consolidating results for {len(subexperiments_to_process)} subexperiment(s)")
    print(f"{'='*60}")
    
    success_count = 0
    for subexp_name, subexp_config in subexperiments_to_process.items():
        print(f"\n{'-'*40}")
        try:
            if consolidate_subexperiment_results(subexp_name, subexp_config, config):
                success_count += 1
        except Exception as e:
            print(f"Error consolidating {subexp_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Consolidation complete!")
    print(f"Successfully consolidated: {success_count}/{len(subexperiments_to_process)} subexperiments")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
