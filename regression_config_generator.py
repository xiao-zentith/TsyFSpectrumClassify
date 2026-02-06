#!/usr/bin/env python3
"""
自动生成回归任务配置文件的脚本
基于文件系统结构自动发现数据集并生成对应的配置文件
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.path_manager import PathManager


class RegressionConfigGenerator:
    def __init__(self):
        """初始化配置生成器"""
        self.path_manager = PathManager()
        self.project_root = Path(self.path_manager.project_root)
        
        # 获取数据集路径
        self.dataset_raw_path = self.project_root / self.path_manager.config["data"]["dataset"]["raw"]
        self.dataset_target_path = self.project_root / self.path_manager.config["data"]["dataset"]["target"]
        self.dataset_processed_path = self.project_root / self.path_manager.config["data"]["dataset"]["processed"]
        
        # 获取特殊配置
        self.special_configs = self.path_manager.config.get("regression", {}).get("special_configs", {})
        
        print(f"数据集原始路径: {self.dataset_raw_path}")
        print(f"数据集目标路径: {self.dataset_target_path}")
        print(f"数据集预处理路径: {self.dataset_processed_path}")

    def discover_datasets(self):
        """自动发现所有数据集"""
        datasets = []
        
        if not self.dataset_raw_path.exists():
            print(f"警告: 原始数据集路径不存在: {self.dataset_raw_path}")
            return datasets
            
        # 扫描dataset_raw文件夹中的所有子文件夹
        for item in self.dataset_raw_path.iterdir():
            if item.is_dir():
                dataset_name = item.name
                datasets.append(dataset_name)
                print(f"发现数据集: {dataset_name}")
        
        return sorted(datasets)

    def count_components(self, dataset_name):
        """计算指定数据集的Component数量"""
        target_path = self.dataset_target_path / dataset_name
        
        if not target_path.exists():
            print(f"警告: 目标路径不存在: {target_path}")
            return 0
            
        # 计算Component文件夹数量
        component_count = 0
        for item in target_path.iterdir():
            if item.is_dir() and item.name.startswith("Component"):
                component_count += 1
                
        print(f"数据集 {dataset_name} 检测到 {component_count} 个Component")
        return component_count

    def generate_config(self, dataset_name):
        """为指定数据集生成配置文件"""
        print(f"\n正在为数据集 '{dataset_name}' 生成配置文件...")
        
        # 检测Component数量
        component_count = self.count_components(dataset_name)
        
        if component_count == 0:
            print(f"错误: 数据集 {dataset_name} 没有找到Component文件夹")
            return False
            
        # 基础配置
        config = {
            "dataset_name": dataset_name,
            "dataset_raw": str(self.dataset_raw_path / dataset_name),
            "dataset_processed": str(self.dataset_processed_path / dataset_name),
            "is_cross_validation": True,
            "is_mixup": True,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001
        }
        
        # 动态生成多个dataset_target路径
        for i in range(1, component_count + 1):
            target_key = f"dataset_target{i}"
            target_path = str(self.dataset_target_path / dataset_name / f"Component{i}")
            config[target_key] = target_path
            
        # 应用特殊配置
        if dataset_name in self.special_configs:
            special_config = self.special_configs[dataset_name]
            config.update(special_config)
            print(f"应用特殊配置: {special_config}")
        else:
            print("未应用特殊配置")
            
        # 保存配置文件
        config_filename = f"regression_config_{dataset_name}.json"
        config_path = self.project_root / "configs" / "regression" / config_filename
        
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"配置文件已生成: {config_path}")
        return True

    def generate_all_configs(self):
        """为所有发现的数据集生成配置文件"""
        print("开始自动生成回归配置文件...\n")
        
        datasets = self.discover_datasets()
        
        if not datasets:
            print("未发现任何数据集")
            return
            
        success_count = 0
        for dataset_name in datasets:
            if self.generate_config(dataset_name):
                success_count += 1
                
        print(f"\n配置文件生成完成! 成功生成 {success_count}/{len(datasets)} 个配置文件")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 为指定数据集生成配置
        dataset_name = sys.argv[1]
        generator = RegressionConfigGenerator()
        generator.generate_config(dataset_name)
    else:
        # 为所有数据集生成配置
        generator = RegressionConfigGenerator()
        generator.generate_all_configs()


if __name__ == "__main__":
    main()