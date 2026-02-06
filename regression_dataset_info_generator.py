#!/usr/bin/env python3
"""
Regression Dataset Info Generator

This script automatically generates regression_dataset_info_xxx.json files
based on existing regression_config_xxx.json files and actual dataset structure.

The generated dataset_info files contain train/validation/test splits with
input-target file pairs for regression tasks.
"""

import json
import os
import glob
from pathlib import Path
from sklearn.model_selection import KFold
import random


class RegressionDatasetInfoGenerator:
    def __init__(self, config_dir="configs/regression", paths_config="configs/paths.json"):
        self.config_dir = Path(config_dir)
        self.paths_config = Path(paths_config)
        self.paths = self._load_paths()
        
    def _load_paths(self):
        """Load paths configuration"""
        with open(self.paths_config, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_dataset_files(self, dataset_name):
        """Get input and target files for a dataset"""
        # Get preprocessed input files
        preprocess_path = Path(self.paths["data"]["regression"]["processed"]) / dataset_name
        if not preprocess_path.exists():
            return None, None
            
        # Get all xlsx files and filter out invalid ones
        all_files = list(preprocess_path.glob("*.xlsx"))
        # Filter out Sample.xlsx and other invalid files
        invalid_files = {"Sample.xlsx", "sample.xlsx", "template.xlsx", "Template.xlsx"}
        input_files = [f for f in all_files if f.name not in invalid_files]
        
        if not input_files:
            return None, None
            
        # Get target files for each component
        target_path = Path(self.paths["data"]["regression"]["target"]) / dataset_name
        if not target_path.exists():
            return None, None
            
        component_dirs = [d for d in target_path.iterdir() if d.is_dir() and d.name.startswith("Component")]
        if not component_dirs:
            return None, None
            
        # Sort components by number
        component_dirs.sort(key=lambda x: int(x.name.replace("Component", "")))
        
        # Create mapping from input files to target files
        dataset_pairs = []
        for input_file in input_files:
            # Find corresponding target files in each component
            targets = []
            for comp_dir in component_dirs:
                # Look for exact match first
                target_file = comp_dir / input_file.name
                if target_file.exists():
                    targets.append(str(target_file.absolute()))
                else:
                    # If exact match not found, look for similar files
                    # This is a fallback strategy
                    comp_files = list(comp_dir.glob("*.xlsx"))
                    if comp_files:
                        # Use a simple matching strategy based on file name similarity
                        best_match = min(comp_files, key=lambda x: abs(len(x.stem) - len(input_file.stem)))
                        targets.append(str(best_match.absolute()))
            
            if len(targets) == len(component_dirs):
                dataset_pairs.append({
                    "input": str(input_file.absolute()),
                    "targets": targets
                })
        
        return dataset_pairs, len(component_dirs)
    
    def _create_cross_validation_splits(self, dataset_pairs, n_folds=5, random_state=42):
        """Create cross-validation splits for the dataset"""
        random.seed(random_state)
        random.shuffle(dataset_pairs)
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = []
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(dataset_pairs)):
            # Split train_val into train and validation (80% train, 20% validation)
            train_val_data = [dataset_pairs[i] for i in train_val_idx]
            test_data = [dataset_pairs[i] for i in test_idx]
            
            # Further split train_val into train and validation
            val_size = len(train_val_data) // 5  # 20% for validation
            val_data = train_val_data[:val_size]
            train_data = train_val_data[val_size:]
            
            splits.append({
                "fold": fold_idx,
                "inner_fold": 0,  # For nested cross-validation if needed
                "train": train_data,
                "validation": val_data,
                "test": test_data
            })
        
        return splits
    
    def generate_dataset_info(self, config_file):
        """Generate dataset_info file based on config file"""
        config_path = self.config_dir / config_file
        if not config_path.exists():
            print(f"Warning: Config file {config_file} not found")
            return False
            
        # Extract dataset name from config filename
        dataset_name = config_file.replace("regression_config_", "").replace(".json", "")
        
        # Load config to check for special settings
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Get dataset files
        dataset_pairs, num_components = self._get_dataset_files(dataset_name)
        if not dataset_pairs:
            print(f"Warning: No valid dataset pairs found for {dataset_name}")
            return False
        
        print(f"Found {len(dataset_pairs)} data pairs for {dataset_name} with {num_components} components")
        
        # Create cross-validation splits
        splits = self._create_cross_validation_splits(dataset_pairs)
        
        # Generate output filename
        output_file = self.config_dir / f"regression_dataset_info_{dataset_name}.json"
        
        # Save dataset info
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=4, ensure_ascii=False)
        
        print(f"Generated: {output_file}")
        return True
    
    def generate_all_dataset_info(self):
        """Generate dataset_info files for all existing config files"""
        config_files = list(self.config_dir.glob("regression_config_*.json"))
        
        if not config_files:
            print("No regression config files found")
            return
        
        print(f"Found {len(config_files)} regression config files")
        
        success_count = 0
        for config_file in config_files:
            if self.generate_dataset_info(config_file.name):
                success_count += 1
        
        print(f"\nSuccessfully generated {success_count}/{len(config_files)} dataset_info files")


def main():
    """Main function"""
    generator = RegressionDatasetInfoGenerator()
    generator.generate_all_dataset_info()


if __name__ == "__main__":
    main()