#!/usr/bin/env python3
"""
Regression Configuration Automation Pipeline

This script provides a complete automation pipeline for generating regression
configuration files and dataset info files. It can be used as a command-line
tool to automate the entire configuration process.

Usage:
    python regression_automation_pipeline.py [options]

Options:
    --config-only       Generate only config files
    --dataset-info-only Generate only dataset info files
    --clean             Clean existing files before generation
    --help              Show this help message
"""

import sys
import argparse
import subprocess
from pathlib import Path
import json


class RegressionAutomationPipeline:
    def __init__(self):
        self.project_root = Path(".")
        self.config_generator = "regression_config_generator.py"
        self.dataset_info_generator = "regression_dataset_info_generator.py"
        self.config_dir = Path("configs/regression")
        
    def check_dependencies(self):
        """Check if required generator scripts exist"""
        missing = []
        
        if not Path(self.config_generator).exists():
            missing.append(self.config_generator)
        
        if not Path(self.dataset_info_generator).exists():
            missing.append(self.dataset_info_generator)
            
        if missing:
            print(f"Error: Missing required files: {', '.join(missing)}")
            return False
        
        return True
    
    def clean_existing_files(self):
        """Clean existing generated files"""
        print("🧹 Cleaning existing generated files...")
        
        # Remove existing config files
        config_files = list(self.config_dir.glob("regression_config_*.json"))
        for file in config_files:
            file.unlink()
            print(f"   Removed: {file}")
        
        # Remove existing dataset info files
        dataset_info_files = list(self.config_dir.glob("regression_dataset_info_*.json"))
        for file in dataset_info_files:
            file.unlink()
            print(f"   Removed: {file}")
        
        print(f"   Cleaned {len(config_files)} config files and {len(dataset_info_files)} dataset info files")
    
    def generate_configs(self):
        """Generate regression config files"""
        print("⚙️  Generating regression config files...")
        
        try:
            result = subprocess.run(
                [sys.executable, self.config_generator],
                capture_output=True,
                text=True,
                check=True
            )
            
            print("   Config generation output:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   Error generating configs: {e}")
            print(f"   Error output: {e.stderr}")
            return False
    
    def generate_dataset_info(self):
        """Generate regression dataset info files"""
        print("📊 Generating regression dataset info files...")
        
        try:
            result = subprocess.run(
                [sys.executable, self.dataset_info_generator],
                capture_output=True,
                text=True,
                check=True
            )
            
            print("   Dataset info generation output:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   Error generating dataset info: {e}")
            print(f"   Error output: {e.stderr}")
            return False
    
    def verify_generated_files(self):
        """Verify that files were generated correctly"""
        print("✅ Verifying generated files...")
        
        config_files = list(self.config_dir.glob("regression_config_*.json"))
        dataset_info_files = list(self.config_dir.glob("regression_dataset_info_*.json"))
        
        print(f"   Found {len(config_files)} config files:")
        for file in sorted(config_files):
            print(f"     - {file.name}")
        
        print(f"   Found {len(dataset_info_files)} dataset info files:")
        for file in sorted(dataset_info_files):
            print(f"     - {file.name}")
        
        # Check for matching pairs
        config_datasets = set()
        for file in config_files:
            dataset_name = file.name.replace("regression_config_", "").replace(".json", "")
            config_datasets.add(dataset_name)
        
        dataset_info_datasets = set()
        for file in dataset_info_files:
            dataset_name = file.name.replace("regression_dataset_info_", "").replace(".json", "")
            dataset_info_datasets.add(dataset_name)
        
        # Check for mismatches
        missing_dataset_info = config_datasets - dataset_info_datasets
        missing_config = dataset_info_datasets - config_datasets
        
        if missing_dataset_info:
            print(f"   ⚠️  Warning: Config files without dataset info: {missing_dataset_info}")
        
        if missing_config:
            print(f"   ⚠️  Warning: Dataset info files without config: {missing_config}")
        
        matching_pairs = len(config_datasets & dataset_info_datasets)
        print(f"   ✅ {matching_pairs} matching config-dataset_info pairs found")
        
        return len(config_files) > 0 or len(dataset_info_files) > 0
    
    def run_full_pipeline(self, clean=False, config_only=False, dataset_info_only=False):
        """Run the complete automation pipeline"""
        print("🚀 Starting Regression Configuration Automation Pipeline")
        print("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Clean existing files if requested
        if clean:
            self.clean_existing_files()
            print()
        
        success = True
        
        # Generate configs
        if not dataset_info_only:
            if not self.generate_configs():
                success = False
            print()
        
        # Generate dataset info
        if not config_only:
            if not self.generate_dataset_info():
                success = False
            print()
        
        # Verify results
        if not self.verify_generated_files():
            success = False
        
        print()
        if success:
            print("🎉 Pipeline completed successfully!")
        else:
            print("❌ Pipeline completed with errors!")
        
        return success


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Regression Configuration Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python regression_automation_pipeline.py                    # Run full pipeline
  python regression_automation_pipeline.py --clean           # Clean and regenerate all
  python regression_automation_pipeline.py --config-only     # Generate only configs
  python regression_automation_pipeline.py --dataset-info-only # Generate only dataset info
        """
    )
    
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Generate only regression config files"
    )
    
    parser.add_argument(
        "--dataset-info-only",
        action="store_true",
        help="Generate only regression dataset info files"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing files before generation"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.config_only and args.dataset_info_only:
        print("Error: Cannot specify both --config-only and --dataset-info-only")
        return 1
    
    # Run pipeline
    pipeline = RegressionAutomationPipeline()
    success = pipeline.run_full_pipeline(
        clean=args.clean,
        config_only=args.config_only,
        dataset_info_only=args.dataset_info_only
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())