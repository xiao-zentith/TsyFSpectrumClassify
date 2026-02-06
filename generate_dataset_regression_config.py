#!/usr/bin/env python3
"""
Dataset Regression Configuration Generator

This script generates regression configuration files for the dataset_regression structure.
It creates the necessary directory structure and configuration files based on the paths.json configuration.

Author: Assistant
Date: 2024
"""

import json
import os
from pathlib import Path
import sys

def create_directory_structure(base_path):
    """Create the basic directory structure for dataset_regression"""
    directories = [
        "raw",
        "preprocess", 
        "target"
    ]
    
    for dir_name in directories:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return directories

def generate_regression_config(dataset_name, project_root, config_template=None):
    """Generate a regression configuration file for a dataset"""
    
    if config_template is None:
        config_template = {
            "dataset_name": dataset_name,
            "dataset_raw": f"{project_root}/data/dataset_regression/raw",
            "dataset_processed": f"{project_root}/data/dataset_regression/preprocess", 
            "is_cross_validation": True,
            "is_mixup": True,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "dataset_target1": f"{project_root}/data/dataset_regression/target/Component1",
            "dataset_target2": f"{project_root}/data/dataset_regression/target/Component2"
        }
    
    # Update dataset name
    config_template["dataset_name"] = dataset_name
    
    return config_template

def save_config_file(config, config_path, dataset_name):
    """Save configuration to JSON file"""
    config_file = config_path / f"regression_config_{dataset_name}.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"生成配置文件: {config_file}")
    return config_file

def main():
    """Main function to generate dataset regression configurations"""
    
    # Get project root
    project_root = Path(__file__).parent.absolute()
    
    # Create dataset_regression directory structure
    dataset_regression_path = project_root / "data" / "dataset_regression"
    print(f"创建数据集目录结构: {dataset_regression_path}")
    
    # Create directories
    create_directory_structure(dataset_regression_path)
    
    # Create target component directories (example structure)
    target_path = dataset_regression_path / "target"
    for i in range(1, 5):  # Create Component1 to Component4 directories
        component_dir = target_path / f"Component{i}"
        component_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建组件目录: {component_dir}")
    
    # Configuration path
    config_path = project_root / "configs" / "regression"
    config_path.mkdir(parents=True, exist_ok=True)
    
    # Generate basic regression configuration
    dataset_name = "dataset_regression"
    config = generate_regression_config(dataset_name, str(project_root))
    
    # Add additional target components (up to 4 components as example)
    for i in range(3, 5):  # Add Component3 and Component4
        config[f"dataset_target{i}"] = f"{project_root}/data/dataset_regression/target/Component{i}"
    
    # Save configuration file
    config_file = save_config_file(config, config_path, dataset_name)
    
    # Create a README file for the dataset_regression directory
    readme_content = """# Dataset Regression

这个目录包含用于回归分析的数据集结构：

## 目录结构

- `raw/`: 原始数据文件
- `preprocess/`: 预处理后的数据文件  
- `target/`: 目标数据文件
  - `Component1/`: 组件1的目标数据
  - `Component2/`: 组件2的目标数据
  - `Component3/`: 组件3的目标数据
  - `Component4/`: 组件4的目标数据

## 配置文件

相应的回归配置文件已生成在 `configs/regression/regression_config_dataset_regression.json`

## 使用说明

1. 将原始数据文件放入 `raw/` 目录
2. 将预处理后的数据文件放入 `preprocess/` 目录
3. 将目标数据文件按组件分别放入 `target/ComponentX/` 目录
4. 根据实际数据调整配置文件中的参数
"""
    
    readme_file = dataset_regression_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"创建说明文件: {readme_file}")
    
    print("\n=== 生成完成 ===")
    print(f"数据集目录: {dataset_regression_path}")
    print(f"配置文件: {config_file}")
    print(f"说明文件: {readme_file}")
    
    return config_file

if __name__ == "__main__":
    main()