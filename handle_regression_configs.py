#!/usr/bin/env python3
"""
处理遗漏的regression配置文件
Handle Missing Regression Config Files
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

class RegressionConfigHandler:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.origin_config_dir = self.project_root / "origin" / "regression" / "config"
        self.target_config_dir = self.project_root / "configs"
        self.log_file = self.project_root / "regression_config_handler.log"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def identify_config_files(self):
        """识别需要处理的配置文件"""
        config_files = []
        dataset_info_files = []
        
        if not self.origin_config_dir.exists():
            self.log(f"❌ 源配置目录不存在: {self.origin_config_dir}")
            return config_files, dataset_info_files
        
        for file_path in self.origin_config_dir.glob("*.json"):
            if file_path.name.startswith("config_"):
                config_files.append(file_path)
            elif file_path.name.startswith("dataset_info_"):
                dataset_info_files.append(file_path)
        
        self.log(f"📋 发现 {len(config_files)} 个配置文件")
        self.log(f"📋 发现 {len(dataset_info_files)} 个数据集信息文件")
        
        return config_files, dataset_info_files
    
    def preview_operation(self):
        """预览操作"""
        self.log("🔍 预览操作...")
        config_files, dataset_info_files = self.identify_config_files()
        
        print("\n" + "="*60)
        print("📋 将要处理的配置文件:")
        print("="*60)
        
        for file_path in config_files:
            target_name = f"regression_{file_path.name}"
            print(f"  📄 {file_path.name} → configs/{target_name}")
        
        for file_path in dataset_info_files:
            target_name = f"regression_{file_path.name}"
            print(f"  📄 {file_path.name} → configs/{target_name}")
        
        print("="*60)
        return config_files, dataset_info_files
    
    def update_config_paths(self, config_file, target_file):
        """更新配置文件中的路径"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 更新路径
            path_mappings = {
                'dataset_raw': lambda path: path.replace('/dataset/', '/data/dataset/'),
                'dataset_processed': lambda path: path.replace('/dataset/', '/data/dataset/'),
                'dataset_target1': lambda path: path.replace('/dataset/', '/data/dataset/'),
                'dataset_target2': lambda path: path.replace('/dataset/', '/data/dataset/'),
                'dataset_target3': lambda path: path.replace('/dataset/', '/data/dataset/'),
                'dataset_target4': lambda path: path.replace('/dataset/', '/data/dataset/'),
            }
            
            updated = False
            for key, update_func in path_mappings.items():
                if key in config and isinstance(config[key], str):
                    old_path = config[key]
                    new_path = update_func(old_path)
                    if old_path != new_path:
                        config[key] = new_path
                        updated = True
            
            # 保存更新后的配置
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            if updated:
                self.log(f"✅ 更新配置文件路径: {target_file.name}")
            else:
                self.log(f"📄 复制配置文件: {target_file.name}")
                
        except Exception as e:
            self.log(f"❌ 处理配置文件失败 {config_file.name}: {e}")
            # 如果更新失败，直接复制原文件
            shutil.copy2(config_file, target_file)
    
    def move_config_files(self, config_files, dataset_info_files):
        """移动配置文件"""
        self.log("📁 开始移动配置文件...")
        
        # 确保目标目录存在
        self.target_config_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        
        # 处理配置文件
        for config_file in config_files:
            target_name = f"regression_{config_file.name}"
            target_file = self.target_config_dir / target_name
            
            self.update_config_paths(config_file, target_file)
            moved_count += 1
        
        # 处理数据集信息文件
        for info_file in dataset_info_files:
            target_name = f"regression_{info_file.name}"
            target_file = self.target_config_dir / target_name
            
            shutil.copy2(info_file, target_file)
            self.log(f"✅ 复制数据集信息文件: {target_name}")
            moved_count += 1
        
        return moved_count
    
    def verify_structure(self):
        """验证最终结构"""
        self.log("🔍 验证最终结构...")
        
        print("\n" + "="*60)
        print("📁 configs/ 目录最终内容:")
        print("="*60)
        
        if self.target_config_dir.exists():
            config_files = sorted(self.target_config_dir.glob("*.json"))
            for config_file in config_files:
                print(f"  📄 {config_file.name}")
        else:
            print("  ❌ configs/ 目录不存在")
        
        print("="*60)
    
    def run(self):
        """执行完整的处理流程"""
        self.log("🚀 开始处理遗漏的regression配置文件")
        
        # 预览操作
        config_files, dataset_info_files = self.preview_operation()
        
        if not config_files and not dataset_info_files:
            self.log("ℹ️ 没有发现需要处理的配置文件")
            return
        
        # 用户确认
        total_files = len(config_files) + len(dataset_info_files)
        print(f"\n📋 总共需要处理 {total_files} 个文件")
        
        confirm = input("是否继续执行? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            self.log("❌ 用户取消操作")
            return
        
        # 执行移动
        moved_count = self.move_config_files(config_files, dataset_info_files)
        
        # 验证结构
        self.verify_structure()
        
        self.log(f"✅ 处理完成: 成功处理 {moved_count} 个配置文件")
        print(f"\n✅ 处理完成！")
        print(f"📝 详细日志: {self.log_file}")

if __name__ == "__main__":
    handler = RegressionConfigHandler()
    handler.run()