#!/usr/bin/env python3
"""
Regression Configuration File Generator

This script automatically generates regression_config_xxx.json files based on:
1. Dataset types defined in paths.json
2. Actual Component folder structure in dataset_target
3. Folder name mappings for unified naming conventions

Author: Assistant
Date: 2024
"""

import json
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
sys.path.append(str(project_root))

from src.utils.path_manager import PathManager


class RegressionConfigGenerator:
    """Generator for regression configuration files"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the generator
        
        Args:
            project_root: Project root directory. If None, auto-detect from current file location.
        """
        if project_root is None:
            # Auto-detect project root (3 levels up from this file)
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parents[3]
        else:
            self.project_root = Path(project_root)
            
        # Initialize PathManager with default config path (let it auto-detect)
        self.path_manager = PathManager()
        
        # Load paths configuration
        self.dataset_paths = {
            "dataset_raw": self.path_manager.config["data"]["dataset"]["raw"],
            "dataset_processed": self.path_manager.config["data"]["dataset"]["processed"],
            "dataset_target": self.path_manager.config["data"]["dataset"]["target"]
        }
        self.dataset_types = list(self.path_manager.config.get("dataset_types", {}).keys())
        self.folder_name_mapping = self.path_manager.config.get("folder_name_mapping", {})
        
        # Default configuration template
        self.default_config_template = {
            "data_split": {
                "train": 0.7,
                "val": 0.15,
                "test": 0.15
            },
            "is_cross_validation": True
        }
        
        # Special configurations for specific dataset types
        self.special_configs = {
            "Fish": {
                "is_cross_validation": False,
                "is_mixup": False
            }
        }
    
    def get_folder_name(self, dataset_type: str) -> str:
        """Get the actual folder name for a dataset type
        
        Args:
            dataset_type: Standardized dataset type name (e.g., "C6_FITC")
            
        Returns:
            Actual folder name (e.g., "C6_FITC" after renaming)
        """
        return self.folder_name_mapping.get(dataset_type, dataset_type)
    
    def count_components(self, dataset_type: str) -> int:
        """Count the number of Component folders for a dataset type
        
        Args:
            dataset_type: Dataset type name
            
        Returns:
            Number of Component folders found
        """
        folder_name = self.get_folder_name(dataset_type)
        target_path = self.project_root / self.dataset_paths["dataset_target"] / folder_name
        
        if not target_path.exists():
            print(f"Warning: Target path does not exist: {target_path}")
            return 0
        
        # Count Component folders
        component_count = 0
        for item in target_path.iterdir():
            if item.is_dir() and item.name.startswith("Component"):
                component_count += 1
        
        return component_count
    
    def generate_config_for_type(self, dataset_type: str) -> Dict[str, Any]:
        """Generate configuration for a specific dataset type
        
        Args:
            dataset_type: Dataset type name
            
        Returns:
            Configuration dictionary
        """
        folder_name = self.get_folder_name(dataset_type)
        component_count = self.count_components(dataset_type)
        
        if component_count == 0:
            raise ValueError(f"No Component folders found for dataset type: {dataset_type}")
        
        # Generate base paths
        config = {
            "dataset_raw": str(self.project_root / self.dataset_paths["dataset_raw"] / folder_name),
            "dataset_processed": str(self.project_root / self.dataset_paths["dataset_processed"] / folder_name),
            "dateset_result": str(self.project_root / "results" / "regression" / f"regression_{dataset_type}")
        }
        
        # Generate target paths based on component count
        for i in range(1, component_count + 1):
            target_key = f"dataset_target{i}"
            target_path = str(self.project_root / self.dataset_paths["dataset_target"] / folder_name / f"Component{i}")
            config[target_key] = target_path
        
        # Add default configuration
        config.update(self.default_config_template)
        
        # Apply special configurations if any
        if dataset_type in self.special_configs:
            config.update(self.special_configs[dataset_type])
        
        return config
    
    def write_config_file(self, dataset_type: str, config: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Write configuration to JSON file
        
        Args:
            dataset_type: Dataset type name
            config: Configuration dictionary
            output_dir: Output directory. If None, use default configs/regression
            
        Returns:
            Path to the written file
        """
        if output_dir is None:
            output_dir = str(self.project_root / "configs/regression")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        filename = f"regression_config_{dataset_type}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Write configuration
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        return filepath
    
    def generate_all_configs(self, output_dir: Optional[str] = None) -> List[str]:
        """Generate configuration files for all dataset types
        
        Args:
            output_dir: Output directory. If None, use default configs/regression
            
        Returns:
            List of generated file paths
        """
        generated_files = []
        
        for dataset_type in self.dataset_types:
            try:
                print(f"Generating config for {dataset_type}...")
                
                # Generate configuration
                config = self.generate_config_for_type(dataset_type)
                
                # Write to file
                filepath = self.write_config_file(dataset_type, config, output_dir)
                generated_files.append(filepath)
                
                # Print summary
                component_count = self.count_components(dataset_type)
                print(f"  ✓ Generated {filepath}")
                print(f"  ✓ Components: {component_count}")
                print(f"  ✓ Special config: {'Yes' if dataset_type in self.special_configs else 'No'}")
                
            except Exception as e:
                print(f"  ✗ Error generating config for {dataset_type}: {e}")
        
        return generated_files
    
    def update_existing_configs(self, output_dir: Optional[str] = None) -> List[str]:
        """Update existing configuration files while preserving manual modifications
        
        Args:
            output_dir: Output directory. If None, use default configs/regression
            
        Returns:
            List of updated file paths
        """
        if output_dir is None:
            output_dir = str(self.project_root / "configs/regression")
        
        updated_files = []
        
        for dataset_type in self.dataset_types:
            try:
                filename = f"regression_config_{dataset_type}.json"
                filepath = os.path.join(output_dir, filename)
                
                # Generate new configuration
                new_config = self.generate_config_for_type(dataset_type)
                
                # If file exists, preserve certain manual settings
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_config = json.load(f)
                    
                    # Preserve manual modifications to data_split if they exist
                    if "data_split" in existing_config:
                        new_config["data_split"] = existing_config["data_split"]
                
                # Write updated configuration
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, indent=4, ensure_ascii=False)
                
                updated_files.append(filepath)
                print(f"Updated {filepath}")
                
            except Exception as e:
                print(f"Error updating config for {dataset_type}: {e}")
        
        return updated_files


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate regression configuration files automatically"
    )
    parser.add_argument(
        "--type", 
        type=str, 
        help="Generate config for specific dataset type (e.g., C6_FITC, Fish)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for config files"
    )
    parser.add_argument(
        "--update", 
        action="store_true", 
        help="Update existing configs while preserving manual modifications"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = RegressionConfigGenerator()
    
    print("🔧 Regression Configuration Generator")
    print("=" * 50)
    
    try:
        if args.type:
            # Generate config for specific type
            print(f"Generating config for dataset type: {args.type}")
            
            if args.type not in generator.dataset_types:
                print(f"Error: Unknown dataset type '{args.type}'")
                print(f"Available types: {', '.join(generator.dataset_types)}")
                return
            
            config = generator.generate_config_for_type(args.type)
            filepath = generator.write_config_file(args.type, config, args.output_dir)
            
            component_count = generator.count_components(args.type)
            print(f"✓ Generated: {filepath}")
            print(f"✓ Components: {component_count}")
            print(f"✓ Special config: {'Yes' if args.type in generator.special_configs else 'No'}")
            
        elif args.update:
            # Update existing configs
            print("Updating existing configuration files...")
            updated_files = generator.update_existing_configs(args.output_dir)
            print(f"✓ Updated {len(updated_files)} configuration files")
            
        else:
            # Generate all configs
            print("Generating configuration files for all dataset types...")
            generated_files = generator.generate_all_configs(args.output_dir)
            print(f"✓ Generated {len(generated_files)} configuration files")
        
        print("\n🎉 Configuration generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())