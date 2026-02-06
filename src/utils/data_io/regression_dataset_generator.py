#!/usr/bin/env python3
"""
回归数据集信息生成器
自动生成回归数据集信息文件，支持不同的数据集类型组合
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.path_manager import PathManager


class SimpleKFold:
    """简单的K折交叉验证实现，避免sklearn依赖"""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X):
        """生成K折分割的索引"""
        n_samples = len(X)
        indices = list(range(n_samples))
        
        if self.shuffle:
            random.seed(self.random_state)
            random.shuffle(indices)
        
        fold_sizes = [n_samples // self.n_splits] * self.n_splits
        for i in range(n_samples % self.n_splits):
            fold_sizes[i] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = indices[:start] + indices[stop:]
            yield train_indices, test_indices
            current = stop


class RegressionDatasetGenerator:
    """回归数据集信息生成器"""
    
    def __init__(self, config_path=None):
        """初始化生成器"""
        self.path_manager = PathManager(config_path)
        self.project_root = self.path_manager.project_root
        
    def scan_excel_files(self, dataset_type="ALL"):
        """扫描指定类型的Excel文件"""
        dataset_root = self.path_manager.get_path("data", "dataset", "root")
        
        # 获取数据集类型对应的文件夹
        type_folders = self.path_manager.config.get("dataset_types", {}).get(dataset_type, [])
        
        files_info = {
            "raw": [],
            "processed": [],
            "target": []
        }
        
        # 扫描原始数据文件
        raw_path = self.path_manager.get_path("data", "dataset_regression", "raw")
        processed_path = self.path_manager.get_path("data", "dataset_regression", "processed")
        target_path = self.path_manager.get_path("data", "dataset_regression", "target")
        
        if dataset_type == "ALL":
            # 扫描所有文件夹
            for subdir in ["C6", "FITC", "Fish", "HPTS"]:
                files_info["raw"].extend(self._scan_folder(raw_path / subdir))
                files_info["processed"].extend(self._scan_folder(processed_path / subdir))
                files_info["target"].extend(self._scan_folder(target_path / subdir))
        else:
            # 扫描特定类型的文件夹
            for folder_type in type_folders:
                files_info["raw"].extend(self._scan_folder(raw_path / folder_type))
                files_info["processed"].extend(self._scan_folder(processed_path / folder_type))
                files_info["target"].extend(self._scan_folder(target_path / folder_type))
        
        return files_info
    
    def _scan_folder(self, folder_path):
        """扫描文件夹中的Excel文件"""
        if not folder_path.exists():
            return []
        
        excel_files = []
        for file_path in folder_path.rglob("*.xlsx"):
            excel_files.append(str(file_path))
        
        return excel_files
    
    def match_files(self, raw_files, target_files):
        """匹配原始文件和目标文件"""
        matched_pairs = []
        
        for raw_file in raw_files:
            raw_basename = Path(raw_file).stem
            
            # 查找匹配的目标文件
            matching_targets = []
            for target_file in target_files:
                target_basename = Path(target_file).stem
                
                # 简单的文件名匹配逻辑
                if self._files_match(raw_basename, target_basename):
                    matching_targets.append(target_file)
            
            if matching_targets:
                # 限制为最多2个目标文件（Component1和Component2）
                matched_pairs.append({
                    "input": raw_file,
                    "targets": matching_targets[:2]
                })
        
        return matched_pairs
    
    def _files_match(self, raw_name, target_name):
        """判断文件是否匹配"""
        # 移除常见的前缀和后缀
        raw_clean = raw_name.replace("_extracted", "").replace("mixed_", "").replace("noise_", "")
        target_clean = target_name.replace("_extracted", "").replace("mixed_", "").replace("noise_", "")
        
        # 检查是否包含相同的核心名称
        return raw_clean in target_clean or target_clean in raw_clean
    
    def generate_cross_validation_splits(self, matched_pairs, n_folds=5, n_inner_folds=4):
        """生成交叉验证数据分割"""
        if not matched_pairs:
            return []
        
        outer_cv = SimpleKFold(n_splits=n_folds, shuffle=True, random_state=42)
        dataset_splits = []
        
        for fold_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(matched_pairs)):
            train_val_pairs = [matched_pairs[i] for i in train_val_indices]
            test_pairs = [matched_pairs[i] for i in test_indices]
            
            inner_cv = SimpleKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
            
            for inner_fold_idx, (train_indices, val_indices) in enumerate(inner_cv.split(train_val_pairs)):
                train_pairs = [train_val_pairs[i] for i in train_indices]
                val_pairs = [train_val_pairs[i] for i in val_indices]
                
                fold_data = {
                    "fold": fold_idx,
                    "inner_fold": inner_fold_idx,
                    "train": train_pairs,
                    "validation": val_pairs,
                    "test": test_pairs
                }
                
                dataset_splits.append(fold_data)
        
        return dataset_splits
    
    def generate_dataset_info_from_config(self, config_file_path, output_file=None):
        """从regression配置文件生成dataset_info"""
        print(f"正在从配置文件生成dataset_info: {config_file_path}")
        
        # 读取配置文件
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        dataset_name = config.get("dataset_name", "Unknown")
        dataset_raw = config.get("dataset_raw", "")
        dataset_processed = config.get("dataset_processed", "")
        is_cross_validation = config.get("is_cross_validation", True)
        
        # 获取所有target路径
        target_paths = []
        for key, value in config.items():
            if key.startswith("dataset_target") and value:
                target_paths.append(value)
        
        if not target_paths:
            print(f"警告: 配置文件中没有找到target路径")
            return None
        
        print(f"数据集: {dataset_name}")
        print(f"原始数据路径: {dataset_raw}")
        print(f"预处理数据路径: {dataset_processed}")
        print(f"目标路径数量: {len(target_paths)}")
        print(f"交叉验证: {is_cross_validation}")
        
        # 扫描实际的数据文件
        matched_pairs = self._scan_and_match_files_from_paths(dataset_raw, target_paths)
        
        if not matched_pairs:
            print(f"警告: 没有找到匹配的文件对")
            return None
        
        print(f"找到 {len(matched_pairs)} 个匹配的文件对")
        
        # 生成交叉验证分割
        if is_cross_validation:
            dataset_info = self.generate_cross_validation_splits(matched_pairs)
        else:
            # 如果不使用交叉验证，生成简单的训练集
            dataset_info = [{
                "fold": 0,
                "inner_fold": 0,
                "train": [{"input": pair["input"], "targets": pair["targets"]} for pair in matched_pairs],
                "test": []
            }]
        
        # 保存文件
        if output_file is None:
            config_filename = Path(config_file_path).stem
            output_file = self.project_root / "configs" / "preprocessing" / f"dataset_info_{config_filename.replace('regression_config_', '')}.json"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset_info已保存到: {output_file}")
        return dataset_info
    
    def _scan_and_match_files_from_paths(self, raw_path, target_paths):
        """从指定路径扫描并匹配文件"""
        matched_pairs = []
        
        # 扫描原始数据文件
        raw_path = Path(raw_path)
        if not raw_path.exists():
            print(f"警告: 原始数据路径不存在: {raw_path}")
            return []
        
        raw_files = list(raw_path.rglob("*.xlsx"))
        if not raw_files:
            print(f"警告: 在原始数据路径中没有找到Excel文件: {raw_path}")
            return []
        
        print(f"在原始数据路径中找到 {len(raw_files)} 个Excel文件")
        
        # 为每个原始文件寻找对应的目标文件
        for raw_file in raw_files:
            raw_name = raw_file.stem
            target_files = []
            
            # 在每个target路径中寻找对应的文件
            for target_path in target_paths:
                target_path = Path(target_path)
                if target_path.exists():
                    # 寻找匹配的目标文件
                    matching_targets = list(target_path.rglob(f"*{raw_name}*.xlsx"))
                    if not matching_targets:
                        # 如果没有找到精确匹配，尝试寻找任何Excel文件
                        matching_targets = list(target_path.rglob("*.xlsx"))
                    
                    if matching_targets:
                        target_files.append(str(matching_targets[0]))  # 取第一个匹配的文件
                    else:
                        print(f"警告: 在 {target_path} 中没有找到与 {raw_name} 匹配的目标文件")
            
            if len(target_files) == len(target_paths):
                matched_pairs.append({
                    "input": str(raw_file),
                    "targets": target_files
                })
            else:
                print(f"警告: 文件 {raw_name} 没有找到完整的目标文件集合")
        
        return matched_pairs

    def generate_dataset_info(self, dataset_type="ALL", output_file=None):
        """生成数据集信息文件"""
        print(f"正在生成 {dataset_type} 类型的数据集信息...")
        
        # 扫描文件
        files_info = self.scan_excel_files(dataset_type)
        
        print(f"找到文件:")
        print(f"  原始文件: {len(files_info['raw'])}")
        print(f"  目标文件: {len(files_info['target'])}")
        
        if not files_info['raw'] or not files_info['target']:
            print(f"警告: {dataset_type} 类型的文件不足，跳过生成")
            return False
        
        # 匹配文件
        matched_pairs = self.match_files(files_info['raw'], files_info['target'])
        
        if not matched_pairs:
            print(f"警告: 没有找到匹配的文件对，跳过生成")
            return False
        
        print(f"匹配到 {len(matched_pairs)} 个文件对")
        
        # 生成交叉验证分割
        dataset_splits = self.generate_cross_validation_splits(matched_pairs)
        
        # 确定输出文件路径
        if output_file is None:
            config_path = self.path_manager.get_path("configs", "regression")
            output_file = config_path / f"regression_dataset_info_{dataset_type}.json"
        
        # 写入文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_splits, f, indent=4, ensure_ascii=False)
            
            print(f"成功生成: {output_file}")
            return True
            
        except Exception as e:
            print(f"生成文件时出错: {e}")
            return False
    
    def generate_all_dataset_types(self):
        """生成所有数据集类型的信息文件"""
        dataset_types = self.path_manager.config.get("dataset_types", {}).keys()
        
        success_count = 0
        total_count = len(dataset_types)
        
        for dataset_type in dataset_types:
            if self.generate_dataset_info(dataset_type):
                success_count += 1
        
        print(f"\n生成完成: {success_count}/{total_count} 个数据集类型")
        return success_count == total_count
    
    def update_regression_configs(self):
        """更新回归配置文件中的路径"""
        config_path = self.path_manager.get_path("configs", "regression")
        
        # 获取新的路径
        dataset_raw = str(self.path_manager.get_path("data", "dataset", "raw"))
        dataset_processed = str(self.path_manager.get_path("data", "dataset", "processed"))
        dataset_target = str(self.path_manager.get_path("data", "dataset", "target"))
        
        # 更新所有回归配置文件
        config_files = list(config_path.glob("regression_config_*.json"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新路径
                config["dataset_raw"] = dataset_raw
                config["dataset_processed"] = dataset_processed
                config["dataset_target1"] = dataset_target
                config["dataset_target2"] = dataset_target
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                
                print(f"已更新配置文件: {config_file.name}")
                
            except Exception as e:
                print(f"更新配置文件 {config_file.name} 时出错: {e}")


    def generate_all_dataset_info_from_configs(self):
        """为所有regression配置文件生成dataset_info"""
        configs_dir = self.project_root / "configs" / "regression"
        
        if not configs_dir.exists():
            print(f"错误: 配置目录不存在: {configs_dir}")
            return
        
        # 查找所有regression配置文件
        config_files = list(configs_dir.glob("regression_config_*.json"))
        
        if not config_files:
            print(f"警告: 在 {configs_dir} 中没有找到regression配置文件")
            return
        
        print(f"找到 {len(config_files)} 个regression配置文件")
        
        success_count = 0
        for config_file in config_files:
            try:
                print(f"\n{'='*50}")
                result = self.generate_dataset_info_from_config(config_file)
                if result:
                    success_count += 1
                    print(f"✓ 成功生成 {config_file.name} 的dataset_info")
                else:
                    print(f"✗ 生成 {config_file.name} 的dataset_info失败")
            except Exception as e:
                print(f"✗ 处理 {config_file.name} 时出错: {e}")
        
        print(f"\n{'='*50}")
        print(f"总结: 成功生成 {success_count}/{len(config_files)} 个dataset_info文件")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="回归数据集信息生成器")
    parser.add_argument("--mode", choices=["scan", "config", "all"], default="config",
                       help="运行模式: scan(扫描数据集), config(从配置文件生成), all(生成所有)")
    parser.add_argument("--config", type=str, help="指定配置文件路径")
    parser.add_argument("--dataset-type", type=str, default="ALL", help="数据集类型")
    parser.add_argument("--output", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    generator = RegressionDatasetGenerator()
    
    if args.mode == "scan":
        # 扫描数据集模式
        generator.generate_dataset_info(args.dataset_type, args.output)
    elif args.mode == "config":
        if args.config:
            # 从指定配置文件生成
            generator.generate_dataset_info_from_config(args.config, args.output)
        else:
            # 从所有配置文件生成
            generator.generate_all_dataset_info_from_configs()
    elif args.mode == "all":
        # 生成所有类型
        generator.generate_all_dataset_types()
        generator.generate_all_dataset_info_from_configs()
    
    print("完成!")


if __name__ == "__main__":
    main()