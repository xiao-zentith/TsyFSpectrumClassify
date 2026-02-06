#!/usr/bin/env python3
"""
最终配置文件整理脚本
Final Config Files Cleanup Script
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

class FinalConfigCleanup:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.configs_dir = self.project_root / "configs"
        self.log_file = self.project_root / "final_config_cleanup.log"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def find_remaining_configs(self):
        """查找剩余的配置文件"""
        remaining_configs = []
        
        # 检查classfication/Utils/dataset_info.json
        classification_config = self.project_root / "origin" / "classfication" / "Utils" / "dataset_info.json"
        if classification_config.exists():
            remaining_configs.append(("classification_dataset_info.json", classification_config))
        
        # 检查data目录中的配置文件
        data_configs = [
            ("data/dataset/dataset_preprocess/C6 + FITC/dataset_info.json", "dataset_info_c6_fitc_preprocess.json"),
            ("data/dataset/dataset_preprocess/FITC + hpts/dataset_info.json", "dataset_info_fitc_hpts_preprocess.json"),
            ("data/dataset/dataset_preprocess/C6 + hpts/dataset_info.json", "dataset_info_c6_hpts_preprocess.json"),
            ("data/dataset_classify/dataset_info.json", "dataset_classify_info.json")
        ]
        
        for relative_path, target_name in data_configs:
            config_path = self.project_root / relative_path
            if config_path.exists():
                remaining_configs.append((target_name, config_path))
        
        return remaining_configs
    
    def preview_cleanup(self):
        """预览清理操作"""
        self.log("🔍 查找剩余的配置文件...")
        remaining_configs = self.find_remaining_configs()
        
        print("\n" + "="*60)
        print("📋 发现的剩余配置文件:")
        print("="*60)
        
        if not remaining_configs:
            print("  ✅ 没有发现剩余的配置文件")
        else:
            for target_name, source_path in remaining_configs:
                print(f"  📄 {source_path.relative_to(self.project_root)} → configs/{target_name}")
        
        print("="*60)
        return remaining_configs
    
    def move_remaining_configs(self, remaining_configs):
        """移动剩余的配置文件"""
        if not remaining_configs:
            self.log("ℹ️ 没有需要移动的配置文件")
            return 0
        
        self.log("📁 开始移动剩余的配置文件...")
        self.configs_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        for target_name, source_path in remaining_configs:
            target_path = self.configs_dir / target_name
            
            try:
                shutil.copy2(source_path, target_path)
                self.log(f"✅ 复制配置文件: {target_name}")
                moved_count += 1
            except Exception as e:
                self.log(f"❌ 复制失败 {target_name}: {e}")
        
        return moved_count
    
    def verify_final_structure(self):
        """验证最终的配置目录结构"""
        self.log("🔍 验证最终配置目录结构...")
        
        print("\n" + "="*60)
        print("📁 configs/ 目录最终内容:")
        print("="*60)
        
        if self.configs_dir.exists():
            config_files = sorted(self.configs_dir.glob("*.json"))
            for config_file in config_files:
                file_size = config_file.stat().st_size
                size_str = f"({file_size:,} bytes)" if file_size > 1024 else f"({file_size} bytes)"
                print(f"  📄 {config_file.name} {size_str}")
            
            print(f"\n📊 总计: {len(config_files)} 个配置文件")
        else:
            print("  ❌ configs/ 目录不存在")
        
        print("="*60)
    
    def create_config_index(self):
        """创建配置文件索引"""
        index_file = self.configs_dir / "CONFIG_INDEX.md"
        
        config_descriptions = {
            "config.json": "主配置文件 - 包含主要数据集路径和训练参数",
            "dataset_classify_config.json": "分类数据集配置 - 分类任务的数据集配置",
            "dataset_info.json": "数据集信息 - 主要数据集的详细信息",
            "classification_dataset_info.json": "分类数据集信息 - 分类任务的数据集详细信息",
            "dataset_classify_info.json": "分类数据集基本信息",
            "dataset_info_c6_fitc_preprocess.json": "C6+FITC预处理数据集信息",
            "dataset_info_fitc_hpts_preprocess.json": "FITC+HPTS预处理数据集信息", 
            "dataset_info_c6_hpts_preprocess.json": "C6+HPTS预处理数据集信息"
        }
        
        # 添加regression配置文件描述
        regression_configs = [
            ("regression_config_ALL.json", "回归配置 - 全部数据"),
            ("regression_config_C6_FITC.json", "回归配置 - C6+FITC数据"),
            ("regression_config_C6_HPTS.json", "回归配置 - C6+HPTS数据"),
            ("regression_config_FITC_HPTS.json", "回归配置 - FITC+HPTS数据"),
            ("regression_config_Fish.json", "回归配置 - Fish数据"),
            ("regression_dataset_info_ALL.json", "回归数据集信息 - 全部数据"),
            ("regression_dataset_info_C6_FITC.json", "回归数据集信息 - C6+FITC数据"),
            ("regression_dataset_info_C6_HPTS.json", "回归数据集信息 - C6+HPTS数据"),
            ("regression_dataset_info_FITC_HPTS.json", "回归数据集信息 - FITC+HPTS数据"),
            ("regression_dataset_info_Fish.json", "回归数据集信息 - Fish数据")
        ]
        
        for config_name, description in regression_configs:
            config_descriptions[config_name] = description
        
        content = "# 配置文件索引\n\n"
        content += "本目录包含项目的所有配置文件。\n\n"
        content += "## 配置文件说明\n\n"
        
        if self.configs_dir.exists():
            config_files = sorted(self.configs_dir.glob("*.json"))
            for config_file in config_files:
                description = config_descriptions.get(config_file.name, "配置文件")
                content += f"- **{config_file.name}**: {description}\n"
        
        content += "\n## 使用说明\n\n"
        content += "1. 主配置文件 `config.json` 包含了项目的基本配置\n"
        content += "2. 回归相关配置文件以 `regression_` 开头\n"
        content += "3. 分类相关配置文件以 `classification_` 或 `dataset_classify_` 开头\n"
        content += "4. 数据集信息文件以 `dataset_info` 开头\n\n"
        content += f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.log(f"✅ 创建配置文件索引: {index_file.name}")
    
    def run(self):
        """执行完整的清理流程"""
        self.log("🚀 开始最终配置文件清理")
        
        # 预览操作
        remaining_configs = self.preview_cleanup()
        
        if remaining_configs:
            # 用户确认
            print(f"\n📋 发现 {len(remaining_configs)} 个剩余配置文件")
            confirm = input("是否继续移动这些配置文件? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                self.log("❌ 用户取消操作")
                return
            
            # 移动配置文件
            moved_count = self.move_remaining_configs(remaining_configs)
            self.log(f"✅ 移动完成: {moved_count} 个配置文件")
        
        # 验证最终结构
        self.verify_final_structure()
        
        # 创建配置文件索引
        self.create_config_index()
        
        self.log("✅ 最终配置文件清理完成")
        print(f"\n✅ 最终配置文件清理完成！")
        print(f"📝 详细日志: {self.log_file}")
        print(f"📋 配置索引: {self.configs_dir}/CONFIG_INDEX.md")

if __name__ == "__main__":
    cleanup = FinalConfigCleanup()
    cleanup.run()