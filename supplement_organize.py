#!/usr/bin/env python3
"""
补充整理脚本 - 处理遗漏的配置文件和数据集整理
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

class SupplementOrganizer:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.configs_dir = self.project_root / "configs"
        self.data_dir = self.project_root / "data"
        self.log_file = self.project_root / "supplement_organize.log"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def get_missing_items(self):
        """获取遗漏的配置文件和需要整理的数据集"""
        return {
            # 遗漏的配置文件
            'config_files': [
                'config.json',  # 根目录的主配置文件
                'dataset_classify/config.json',  # 分类数据集配置
                'dataset_info.json'  # 数据集信息文件
            ],
            # 需要整理的数据集目录
            'dataset_dirs': [
                'dataset',
                'dataset_classify'
            ],
            # 其他可能遗漏的文件
            'other_files': [
                'dataset.zip'  # 数据集压缩包
            ]
        }
    
    def preview_supplement(self):
        """预览补充整理操作"""
        self.log("预览补充整理操作...")
        
        missing_items = self.get_missing_items()
        
        print("\n📁 将要移动到 configs/ 的配置文件:")
        print("=" * 60)
        existing_configs = []
        missing_configs = []
        
        for config_file in missing_items['config_files']:
            config_path = self.project_root / config_file
            if config_path.exists():
                existing_configs.append(config_file)
                target_name = config_path.name
                if config_file == 'dataset_classify/config.json':
                    target_name = 'dataset_classify_config.json'
                print(f"✅ {config_file} → configs/{target_name}")
            else:
                missing_configs.append(config_file)
                print(f"❌ {config_file} (不存在)")
        
        print("\n💾 数据集目录整理方案:")
        print("=" * 60)
        existing_datasets = []
        missing_datasets = []
        
        for dataset_dir in missing_items['dataset_dirs']:
            dataset_path = self.project_root / dataset_dir
            if dataset_path.exists():
                existing_datasets.append(dataset_dir)
                print(f"✅ {dataset_dir}/ → data/{dataset_dir}/ (移动)")
            else:
                missing_datasets.append(dataset_dir)
                print(f"❌ {dataset_dir}/ (不存在)")
        
        print("\n📦 其他文件:")
        print("=" * 60)
        existing_others = []
        missing_others = []
        
        for other_file in missing_items['other_files']:
            other_path = self.project_root / other_file
            if other_path.exists():
                existing_others.append(other_file)
                print(f"✅ {other_file} → data/{other_file}")
            else:
                missing_others.append(other_file)
                print(f"❌ {other_file} (不存在)")
        
        print(f"\n📊 统计:")
        print(f"  - 可移动配置文件: {len(existing_configs)}")
        print(f"  - 可移动数据集目录: {len(existing_datasets)}")
        print(f"  - 可移动其他文件: {len(existing_others)}")
        print(f"  - 缺失项目: {len(missing_configs) + len(missing_datasets) + len(missing_others)}")
        
        return existing_configs, existing_datasets, existing_others
    
    def create_directories(self):
        """创建必要的目录结构"""
        self.log("创建目录结构...")
        
        # 确保configs目录存在
        self.configs_dir.mkdir(exist_ok=True)
        self.log(f"✅ 确保目录存在: {self.configs_dir}")
        
        # 创建data目录
        self.data_dir.mkdir(exist_ok=True)
        self.log(f"✅ 创建目录: {self.data_dir}")
        
        return True
    
    def move_config_files(self, existing_configs):
        """移动配置文件到configs目录"""
        self.log("移动配置文件...")
        
        success_count = 0
        for config_file in existing_configs:
            try:
                src_path = self.project_root / config_file
                
                # 确定目标文件名
                if config_file == 'dataset_classify/config.json':
                    target_name = 'dataset_classify_config.json'
                else:
                    target_name = src_path.name
                
                dest_path = self.configs_dir / target_name
                
                # 如果目标已存在，先删除
                if dest_path.exists():
                    dest_path.unlink()
                
                # 复制文件（保留原文件，因为可能被其他脚本使用）
                shutil.copy2(str(src_path), str(dest_path))
                self.log(f"✅ 复制配置文件: {config_file} → configs/{target_name}")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 复制配置文件失败 {config_file}: {str(e)}")
        
        return success_count
    
    def organize_datasets(self, existing_datasets, existing_others):
        """整理数据集目录"""
        self.log("整理数据集目录...")
        
        success_count = 0
        
        # 移动数据集目录
        for dataset_dir in existing_datasets:
            try:
                src_path = self.project_root / dataset_dir
                dest_path = self.data_dir / dataset_dir
                
                # 如果目标已存在，先删除
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                
                shutil.move(str(src_path), str(dest_path))
                self.log(f"✅ 移动数据集: {dataset_dir}/ → data/{dataset_dir}/")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 移动数据集失败 {dataset_dir}: {str(e)}")
        
        # 移动其他文件
        for other_file in existing_others:
            try:
                src_path = self.project_root / other_file
                dest_path = self.data_dir / other_file
                
                # 如果目标已存在，先删除
                if dest_path.exists():
                    dest_path.unlink()
                
                shutil.move(str(src_path), str(dest_path))
                self.log(f"✅ 移动文件: {other_file} → data/{other_file}")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 移动文件失败 {other_file}: {str(e)}")
        
        return success_count
    
    def create_data_symlinks(self):
        """在根目录创建数据集的符号链接以保持兼容性"""
        self.log("创建数据集符号链接...")
        
        dataset_dirs = ['dataset', 'dataset_classify']
        success_count = 0
        
        for dataset_dir in dataset_dirs:
            try:
                link_path = self.project_root / dataset_dir
                target_path = self.data_dir / dataset_dir
                
                # 如果符号链接已存在，跳过
                if link_path.exists() and link_path.is_symlink():
                    self.log(f"⚠️ 符号链接已存在: {dataset_dir}")
                    continue
                
                # 如果目标存在且不是符号链接，跳过
                if link_path.exists() and not link_path.is_symlink():
                    self.log(f"⚠️ 目录已存在（非符号链接）: {dataset_dir}")
                    continue
                
                # 创建符号链接
                if target_path.exists():
                    link_path.symlink_to(target_path, target_is_directory=True)
                    self.log(f"✅ 创建符号链接: {dataset_dir} → data/{dataset_dir}")
                    success_count += 1
                else:
                    self.log(f"❌ 目标目录不存在，无法创建符号链接: data/{dataset_dir}")
            except Exception as e:
                self.log(f"❌ 创建符号链接失败 {dataset_dir}: {str(e)}")
        
        return success_count
    
    def update_config_paths(self):
        """更新配置文件中的路径"""
        self.log("更新配置文件路径...")
        
        # 更新主配置文件
        main_config_path = self.configs_dir / "config.json"
        if main_config_path.exists():
            try:
                import json
                with open(main_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新路径
                project_root_str = str(self.project_root)
                for key, value in config.items():
                    if isinstance(value, str) and project_root_str in value:
                        # 将绝对路径更新为相对于项目根目录的路径
                        new_value = value.replace(project_root_str, ".")
                        new_value = new_value.replace("/dataset/", "/data/dataset/")
                        config[key] = new_value
                
                # 保存更新后的配置
                with open(main_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.log("✅ 更新主配置文件路径")
            except Exception as e:
                self.log(f"❌ 更新主配置文件失败: {str(e)}")
        
        # 更新分类配置文件
        classify_config_path = self.configs_dir / "dataset_classify_config.json"
        if classify_config_path.exists():
            try:
                import json
                with open(classify_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新路径为相对路径
                if 'dataset_raw' in config:
                    config['dataset_raw'] = "./data/dataset_classify/dataset_raw"
                if 'dataset_processed' in config:
                    config['dataset_processed'] = "./data/dataset_classify/dataset_preprocess"
                
                # 保存更新后的配置
                with open(classify_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.log("✅ 更新分类配置文件路径")
            except Exception as e:
                self.log(f"❌ 更新分类配置文件失败: {str(e)}")
    
    def create_data_readme(self):
        """在data目录创建说明文件"""
        readme_content = f"""# Data Directory - 数据目录

这个目录包含了项目的所有数据集和相关文件。

## 目录结构

### 数据集目录
- `dataset/` - 主要数据集目录
  - `dataset_raw/` - 原始数据
  - `dataset_resized/` - 调整大小后的数据
  - `dataset_preprocess/` - 预处理后的数据
  - `dataset_target/` - 目标数据
- `dataset_classify/` - 分类数据集目录
  - `dataset_raw/` - 原始分类数据
  - `dataset_preprocess/` - 预处理后的分类数据
  - `dataset_noise/` - 噪声数据
  - `dataset_preprocess2/` - 二次预处理数据

### 数据文件
- `dataset.zip` - 数据集压缩包

## 符号链接

为了保持与现有代码的兼容性，在项目根目录创建了以下符号链接：
- `dataset` → `data/dataset`
- `dataset_classify` → `data/dataset_classify`

## 配置文件

相关的配置文件已移动到 `configs/` 目录：
- `configs/config.json` - 主配置文件
- `configs/dataset_classify_config.json` - 分类数据集配置
- `configs/dataset_info.json` - 数据集信息

## 整理时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 注意事项
- 数据集目录已从根目录移动到此处以保持项目结构清晰
- 通过符号链接保持了向后兼容性
- 配置文件中的路径已相应更新
"""
        
        readme_path = self.data_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        self.log(f"✅ 创建数据目录说明文件: {readme_path}")
    
    def execute_supplement(self):
        """执行补充整理"""
        self.log("开始补充整理...")
        
        # 1. 预览操作
        existing_configs, existing_datasets, existing_others = self.preview_supplement()
        
        total_items = len(existing_configs) + len(existing_datasets) + len(existing_others)
        if total_items == 0:
            self.log("没有需要补充整理的项目")
            return True
        
        # 2. 用户确认
        print(f"\n📦 即将补充整理 {total_items} 个项目")
        print("这个操作将:")
        print("- 将配置文件复制到 configs/ 目录")
        print("- 将数据集目录移动到 data/ 目录")
        print("- 创建符号链接保持兼容性")
        print("- 更新配置文件中的路径")
        response = input("确认执行补充整理？(y/N): ")
        
        if response.lower() != 'y':
            self.log("用户取消补充整理操作")
            return False
        
        # 3. 创建目录结构
        if not self.create_directories():
            self.log("❌ 创建目录结构失败")
            return False
        
        # 4. 移动配置文件
        config_success = self.move_config_files(existing_configs)
        
        # 5. 整理数据集
        dataset_success = self.organize_datasets(existing_datasets, existing_others)
        
        # 6. 创建符号链接
        symlink_success = self.create_data_symlinks()
        
        # 7. 更新配置文件路径
        self.update_config_paths()
        
        # 8. 创建说明文件
        self.create_data_readme()
        
        # 9. 最终验证
        self.final_verification()
        
        total_success = config_success + dataset_success + symlink_success
        self.log(f"补充整理完成: 成功处理 {total_success} 个项目")
        
        return total_success > 0
    
    def final_verification(self):
        """最终验证项目结构"""
        self.log("执行最终验证...")
        
        print("\n📁 补充整理后的项目结构:")
        print("=" * 60)
        
        # 显示主要目录
        main_dirs = ['src', 'notebooks', 'scripts', 'tests', 'configs', 'docs', 'data', 'origin']
        for dir_name in main_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/")
            else:
                print(f"❌ {dir_name}/ (缺失)")
        
        # 显示configs目录内容
        print(f"\n⚙️ configs/ 目录内容:")
        if self.configs_dir.exists():
            for item in self.configs_dir.iterdir():
                print(f"  📄 {item.name}")
        
        # 显示data目录内容
        print(f"\n💾 data/ 目录内容:")
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                else:
                    print(f"  📄 {item.name}")
        
        # 显示符号链接
        print(f"\n🔗 符号链接:")
        for link_name in ['dataset', 'dataset_classify']:
            link_path = self.project_root / link_name
            if link_path.exists() and link_path.is_symlink():
                target = link_path.readlink()
                print(f"  ✅ {link_name} → {target}")
            elif link_path.exists():
                print(f"  📁 {link_name} (目录，非符号链接)")
            else:
                print(f"  ❌ {link_name} (不存在)")
        
        self.log("最终验证完成")

def main():
    """主函数"""
    print("📦 补充整理工具")
    print("=" * 60)
    print("功能:")
    print("- 将遗漏的配置文件整理到 configs/ 目录")
    print("- 将数据集目录整理到 data/ 目录")
    print("- 创建符号链接保持兼容性")
    print("- 更新配置文件中的路径")
    print("- 创建详细的说明文档")
    print("=" * 60)
    
    organizer = SupplementOrganizer()
    
    # 执行补充整理
    if organizer.execute_supplement():
        print("\n✅ 补充整理成功完成！")
        print("⚙️ 配置文件已整理到 configs/ 目录")
        print("💾 数据集已整理到 data/ 目录")
        print("🔗 创建了符号链接保持兼容性")
        print(f"📝 详细日志: {organizer.log_file}")
        print(f"📖 查看说明: {organizer.data_dir}/README.md")
        return 0
    else:
        print("\n❌ 补充整理失败或被取消")
        return 1

if __name__ == "__main__":
    sys.exit(main())