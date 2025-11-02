#!/usr/bin/env python3
"""
Concatenate CSV files from numbered subfolders.

Usage:
    python concatenate_csvs.py <input_folder1> [input_folder2 ...] <output_folder/filename.csv> [--sort]

Arguments:
    input_folder(s): One or more paths to folders containing numbered subfolders (1, 2, ..., n)
    output_folder/filename.csv: Output path including folder and filename
    --sort: Optional flag to sort the final CSV by prompt_idx column
"""

import argparse
import os
import pandas as pd
from pathlib import Path
import glob


def get_numbered_subfolders(base_folder):
    """Get all numbered subfolders, sorted numerically."""
    subfolders = []
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            try:
                folder_num = int(item)
                subfolders.append((folder_num, item_path))
            except ValueError:
                # Skip non-numeric folders
                continue
    # Sort by numeric value
    subfolders.sort(key=lambda x: x[0])
    return [path for _, path in subfolders]


def find_csv_files(folder):
    """Find all CSV files in a folder."""
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    return csv_files


def concatenate_csvs(input_folders, output_path, sort_by_prompt_idx=False):
    """
    Concatenate all CSV files from numbered subfolders in one or more input folders.
    
    Args:
        input_folders: List of paths to folders containing numbered subfolders
        output_path: Full output path including folder and filename
        sort_by_prompt_idx: If True, sort final CSV by prompt_idx column
    """
    # Collect all CSV files from all input folders
    all_dataframes = []
    total_files = 0
    
    for input_folder in input_folders:
        input_folder = Path(input_folder)
        
        if not input_folder.exists():
            print(f"Warning: Input folder does not exist: {input_folder}, skipping...")
            continue
        
        print(f"\nProcessing folder: {input_folder}")
        
        # Get all numbered subfolders
        subfolders = get_numbered_subfolders(input_folder)
        
        if not subfolders:
            print(f"  Warning: No numbered subfolders found in {input_folder}, skipping...")
            continue
        
        print(f"  Found {len(subfolders)} numbered subfolders")
        
        # Collect CSV files from this input folder
        for subfolder in subfolders:
            csv_files = find_csv_files(subfolder)
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    all_dataframes.append(df)
                    total_files += 1
                    print(f"    Loaded: {csv_file} ({len(df)} rows)")
                except Exception as e:
                    print(f"    Warning: Could not read {csv_file}: {e}")
    
    if not all_dataframes:
        raise ValueError("No CSV files found or all CSV files were empty/corrupted")
    
    print(f"\n{'='*60}")
    print(f"Total CSV files loaded from {len(input_folders)} folder(s): {total_files}")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined dataframe shape: {combined_df.shape}")
    
    # Sort by prompt_idx if requested
    if sort_by_prompt_idx:
        if 'prompt_idx' in combined_df.columns:
            combined_df = combined_df.sort_values('prompt_idx').reset_index(drop=True)
            print("Sorted by prompt_idx")
        else:
            print("Warning: 'prompt_idx' column not found. Cannot sort.")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved concatenated CSV to: {output_path}")
    print(f"Final shape: {combined_df.shape}")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate CSV files from numbered subfolders in one or more folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python concatenate_csvs.py b2/exp_20_results_ort_beta_0_5 output/combined.csv
    python concatenate_csvs.py folder1 folder2 folder3 output/combined.csv --sort
    python concatenate_csvs.py b2/exp_20_results_ort_beta_0_5 b2/exp_21_results_ort_beta_0_5 output/all_combined.csv --sort
        """
    )
    parser.add_argument(
        "input_folders",
        type=str,
        nargs="+",
        help="One or more paths to folders containing numbered subfolders (1, 2, ..., n)"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path including folder and filename (e.g., output/combined.csv)"
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort the final CSV by prompt_idx column"
    )
    
    args = parser.parse_args()
    
    concatenate_csvs(args.input_folders, args.output_path, args.sort)


if __name__ == "__main__":
    main()

