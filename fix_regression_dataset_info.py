#!/usr/bin/env python3
"""
Script to fix regression dataset info JSON files with correct Excel file paths.
This script will scan the data/dataset directory and update all regression dataset info files
to use the correct paths to Excel files instead of generic "data/raw" paths.
"""

import json
import os
import glob
from pathlib import Path

def find_excel_files(base_path):
    """Find all Excel files in the dataset directory structure."""
    excel_files = {}
    
    # Search in dataset_raw, dataset_preprocess, and dataset_target
    for subdir in ['dataset_raw', 'dataset_preprocess', 'dataset_target']:
        subdir_path = os.path.join(base_path, subdir)
        if os.path.exists(subdir_path):
            # Find all Excel files recursively
            pattern = os.path.join(subdir_path, '**', '*.xlsx')
            files = glob.glob(pattern, recursive=True)
            excel_files[subdir] = files
    
    return excel_files

def get_config_type_from_filename(filename):
    """Extract configuration type from filename."""
    if 'ALL' in filename:
        return 'ALL'
    elif 'C6_FITC' in filename:
        return 'C6_FITC'
    elif 'C6_HPTS' in filename:
        return 'C6_HPTS'
    elif 'FITC_HPTS' in filename:
        return 'FITC_HPTS'
    elif 'Fish' in filename:
        return 'Fish'
    return None

def filter_files_by_config(excel_files, config_type):
    """Filter Excel files based on configuration type."""
    filtered = {'input': [], 'targets': []}
    
    if config_type == 'ALL':
        # ALL includes all types
        for subdir, files in excel_files.items():
            if subdir == 'dataset_raw':
                filtered['input'].extend(files)
            elif subdir == 'dataset_target':
                filtered['targets'].extend(files)
    elif config_type in ['C6_FITC', 'C6_HPTS']:
        # Filter for C6 related files
        for subdir, files in excel_files.items():
            c6_files = [f for f in files if 'C6' in f or 'c6' in f]
            if config_type == 'C6_FITC':
                c6_files = [f for f in c6_files if 'FITC' in f or 'fitc' in f]
            elif config_type == 'C6_HPTS':
                c6_files = [f for f in c6_files if 'HPTS' in f or 'hpts' in f]
            
            if subdir == 'dataset_raw':
                filtered['input'].extend(c6_files)
            elif subdir == 'dataset_target':
                filtered['targets'].extend(c6_files)
    elif config_type == 'FITC_HPTS':
        # Filter for FITC+HPTS files
        for subdir, files in excel_files.items():
            fitc_hpts_files = [f for f in files if ('FITC' in f and 'HPTS' in f) or ('fitc' in f and 'hpts' in f)]
            if subdir == 'dataset_raw':
                filtered['input'].extend(fitc_hpts_files)
            elif subdir == 'dataset_target':
                filtered['targets'].extend(fitc_hpts_files)
    elif config_type == 'Fish':
        # Filter for Fish related files
        for subdir, files in excel_files.items():
            fish_files = [f for f in files if 'Fish' in f or 'fish' in f]
            if subdir == 'dataset_raw':
                filtered['input'].extend(fish_files)
            elif subdir == 'dataset_target':
                filtered['targets'].extend(fish_files)
    
    return filtered

def create_dataset_entries(input_files, target_files):
    """Create dataset entries matching input files with target files."""
    entries = []
    
    for input_file in input_files:
        # Try to find matching target files
        input_basename = os.path.basename(input_file)
        input_name = os.path.splitext(input_basename)[0]
        
        # Find target files with similar names
        matching_targets = []
        for target_file in target_files:
            target_basename = os.path.basename(target_file)
            target_name = os.path.splitext(target_basename)[0]
            
            # Simple matching based on filename similarity
            if input_name in target_name or target_name in input_name:
                matching_targets.append(target_file)
        
        # If no specific matches found, use first two target files as default
        if not matching_targets and target_files:
            matching_targets = target_files[:2]
        
        if matching_targets:
            entries.append({
                "input": input_file,
                "targets": matching_targets[:2]  # Limit to 2 targets as per original format
            })
    
    return entries

def fix_regression_dataset_info_file(filepath, excel_files):
    """Fix a single regression dataset info file."""
    print(f"Fixing {filepath}...")
    
    # Determine config type from filename
    config_type = get_config_type_from_filename(os.path.basename(filepath))
    if not config_type:
        print(f"Could not determine config type for {filepath}")
        return False
    
    # Filter files based on config type
    filtered_files = filter_files_by_config(excel_files, config_type)
    
    if not filtered_files['input']:
        print(f"No input files found for config type {config_type}")
        return False
    
    # Create dataset entries
    train_entries = create_dataset_entries(filtered_files['input'], filtered_files['targets'])
    
    # Create the structure with cross-validation folds
    dataset_structure = []
    
    # Create 5 folds as per original structure
    for fold in range(5):
        fold_data = {
            "fold": fold,
            "inner_fold": 0,
            "train": train_entries.copy(),
            "validation": train_entries[:1] if train_entries else [],  # Use first entry for validation
            "test": train_entries[:2] if len(train_entries) >= 2 else train_entries.copy()  # Use first two for test
        }
        dataset_structure.append(fold_data)
    
    # Write the fixed file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_structure, f, indent=4, ensure_ascii=False)
        print(f"Successfully fixed {filepath}")
        return True
    except Exception as e:
        print(f"Error writing {filepath}: {e}")
        return False

def main():
    """Main function to fix all regression dataset info files."""
    # Define paths
    project_root = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote"
    dataset_path = os.path.join(project_root, "data", "dataset")
    config_path = os.path.join(project_root, "configs", "regression")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all Excel files
    print("Scanning for Excel files...")
    excel_files = find_excel_files(dataset_path)
    
    print("Found Excel files:")
    for subdir, files in excel_files.items():
        print(f"  {subdir}: {len(files)} files")
        for file in files[:3]:  # Show first 3 files as example
            print(f"    - {file}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")
    
    # Find all regression dataset info files
    regression_info_files = glob.glob(os.path.join(config_path, "regression_dataset_info_*.json"))
    
    print(f"\nFound {len(regression_info_files)} regression dataset info files to fix:")
    for file in regression_info_files:
        print(f"  - {os.path.basename(file)}")
    
    # Fix each file
    success_count = 0
    for filepath in regression_info_files:
        if fix_regression_dataset_info_file(filepath, excel_files):
            success_count += 1
    
    print(f"\nFixed {success_count}/{len(regression_info_files)} files successfully.")
    return success_count == len(regression_info_files)

if __name__ == "__main__":
    main()