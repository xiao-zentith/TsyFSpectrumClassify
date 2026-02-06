#!/usr/bin/env python3
"""
配置文件重新整理脚本
根据配置文件的原始位置和用途，将它们组织到合理的子文件夹结构中
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

class ConfigReorganizer:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.configs_dir = self.base_dir / "configs"
        self.log_file = self.base_dir / "config_reorganize.log"
        
        # 定义新的文件夹结构
        self.folder_structure = {
            "main": {
                "description": "主要配置文件",
                "files": ["config.json"]
            },
            "classification": {
                "description": "分类任务相关配置",
                "files": [
                    "classification_dataset_info.json",
                    "dataset_classify_config.json", 
                    "dataset_classify_info.json"
                ]
            },
            "regression": {
                "description": "回归任务相关配置",
                "files": [
                    "regression_config_ALL.json",
                    "regression_config_C6_FITC.json",
                    "regression_config_C6_HPTS.json",
                    "regression_config_FITC_HPTS.json",
                    "regression_config_Fish.json",
                    "regression_dataset_info_ALL.json",
                    "regression_dataset_info_C6_FITC.json",
                    "regression_dataset_info_C6_HPTS.json",
                    "regression_dataset_info_FITC_HPTS.json",
                    "regression_dataset_info_Fish.json"
                ]
            },
            "preprocessing": {
                "description": "数据预处理相关配置",
                "files": [
                    "dataset_info.json",
                    "dataset_info_c6_fitc_preprocess.json",
                    "dataset_info_c6_hpts_preprocess.json",
                    "dataset_info_fitc_hpts_preprocess.json"
                ]
            }
        }
    
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    
    def preview_reorganization(self):
        """预览重新整理计划"""
        self.log("📋 配置文件重新整理预览")
        self.log("=" * 60)
        
        total_files = 0
        for folder_name, folder_info in self.folder_structure.items():
            self.log(f"\n📁 {folder_name}/ - {folder_info['description']}")
            for file_name in folder_info['files']:
                source_path = self.configs_dir / file_name
                if source_path.exists():
                    self.log(f"  ✅ {file_name}")
                    total_files += 1
                else:
                    self.log(f"  ❌ {file_name} (文件不存在)")
        
        self.log(f"\n📊 总计: {total_files} 个配置文件将被重新整理")
        self.log("=" * 60)
        
        return total_files > 0
    
    def create_folder_structure(self):
        """创建文件夹结构"""
        self.log("🏗️ 创建文件夹结构...")
        
        for folder_name in self.folder_structure.keys():
            folder_path = self.configs_dir / folder_name
            folder_path.mkdir(exist_ok=True)
            self.log(f"  📁 创建文件夹: {folder_name}/")
    
    def move_config_files(self):
        """移动配置文件到对应文件夹"""
        self.log("📦 移动配置文件...")
        
        moved_count = 0
        for folder_name, folder_info in self.folder_structure.items():
            folder_path = self.configs_dir / folder_name
            
            for file_name in folder_info['files']:
                source_path = self.configs_dir / file_name
                target_path = folder_path / file_name
                
                if source_path.exists() and source_path != target_path:
                    try:
                        shutil.move(str(source_path), str(target_path))
                        self.log(f"  ✅ 移动: {file_name} -> {folder_name}/")
                        moved_count += 1
                    except Exception as e:
                        self.log(f"  ❌ 移动失败: {file_name} - {e}")
                elif target_path.exists():
                    self.log(f"  ℹ️ 已存在: {folder_name}/{file_name}")
        
        self.log(f"📊 成功移动 {moved_count} 个配置文件")
        return moved_count
    
    def create_folder_readmes(self):
        """为每个文件夹创建README文件"""
        self.log("📝 创建文件夹说明文档...")
        
        for folder_name, folder_info in self.folder_structure.items():
            folder_path = self.configs_dir / folder_name
            readme_path = folder_path / "README.md"
            
            # 统计实际存在的文件
            existing_files = []
            for file_name in folder_info['files']:
                file_path = folder_path / file_name
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    existing_files.append((file_name, file_size))
            
            readme_content = f"""# {folder_name.title()} 配置文件

{folder_info['description']}

## 文件列表

"""
            
            for file_name, file_size in existing_files:
                readme_content += f"- **{file_name}** ({file_size:,} bytes)\n"
            
            readme_content += f"""
## 说明

本文件夹包含 {len(existing_files)} 个配置文件，用于{folder_info['description']}。

最后更新: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            self.log(f"  📄 创建说明文档: {folder_name}/README.md")
    
    def update_main_index(self):
        """更新主配置索引文件"""
        self.log("📋 更新主配置索引...")
        
        index_path = self.configs_dir / "CONFIG_INDEX.md"
        
        index_content = """# 配置文件索引

本目录包含项目的所有配置文件，按功能分类组织。

## 文件夹结构

"""
        
        for folder_name, folder_info in self.folder_structure.items():
            folder_path = self.configs_dir / folder_name
            file_count = len([f for f in folder_info['files'] 
                            if (folder_path / f).exists()])
            
            index_content += f"### 📁 {folder_name}/\n"
            index_content += f"{folder_info['description']}\n"
            index_content += f"包含 {file_count} 个配置文件\n\n"
        
        index_content += """## 使用说明

1. **main/** - 包含项目的主要配置文件
2. **classification/** - 分类任务相关的所有配置
3. **regression/** - 回归任务相关的所有配置  
4. **preprocessing/** - 数据预处理相关的配置

每个文件夹都包含详细的 README.md 说明文档。

"""
        
        index_content += f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)
        
        self.log("  📄 更新配置索引: CONFIG_INDEX.md")
    
    def verify_structure(self):
        """验证最终结构"""
        self.log("🔍 验证最终配置文件结构...")
        self.log("=" * 60)
        
        total_files = 0
        for folder_name in self.folder_structure.keys():
            folder_path = self.configs_dir / folder_name
            if folder_path.exists():
                files = list(folder_path.glob("*.json"))
                self.log(f"📁 {folder_name}/ ({len(files)} 个配置文件)")
                
                for file_path in sorted(files):
                    file_size = file_path.stat().st_size
                    self.log(f"  📄 {file_path.name} ({file_size:,} bytes)")
                    total_files += 1
        
        self.log("=" * 60)
        self.log(f"📊 总计: {total_files} 个配置文件")
        
        return total_files
    
    def run(self):
        """执行完整的重新整理流程"""
        self.log("🚀 开始配置文件重新整理")
        
        # 预览
        if not self.preview_reorganization():
            self.log("❌ 没有找到需要整理的配置文件")
            return False
        
        # 用户确认
        print("\n是否继续执行重新整理? (y/N): ", end="")
        if input().lower() != 'y':
            self.log("❌ 用户取消操作")
            return False
        
        try:
            # 执行整理
            self.create_folder_structure()
            moved_count = self.move_config_files()
            self.create_folder_readmes()
            self.update_main_index()
            
            # 验证结果
            final_count = self.verify_structure()
            
            self.log(f"✅ 配置文件重新整理完成！")
            self.log(f"📝 详细日志: {self.log_file}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ 重新整理过程中出现错误: {e}")
            return False

def main():
    """主函数"""
    print("🔧 配置文件重新整理工具")
    print("=" * 50)
    
    reorganizer = ConfigReorganizer()
    success = reorganizer.run()
    
    if success:
        print("\n✅ 配置文件重新整理完成！")
        print(f"📝 详细日志: {reorganizer.log_file}")
        print(f"📋 配置索引: {reorganizer.configs_dir}/CONFIG_INDEX.md")
    else:
        print("\n❌ 配置文件重新整理失败")

if __name__ == "__main__":
    main()