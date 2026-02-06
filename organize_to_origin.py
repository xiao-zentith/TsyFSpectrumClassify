#!/usr/bin/env python3
"""
整理原始文件到origin文件夹
将已重构的原始文件移动到origin/目录中保存，而不是删除
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

class ProjectOrganizer:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.origin_dir = self.project_root / "origin"
        self.log_file = self.project_root / "organize.log"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def get_items_to_organize(self):
        """获取需要整理的原始文件和目录列表"""
        return {
            # 原始目录（已复制到src/下）
            'directories': [
                'Utils',
                'regression', 
                'classfication',  # 注意这里是原始的拼写错误
                'augmentation',
                'preprocess',
                'UI_version',
                'model_demo'
            ],
            # 原始文件（已复制到新位置）
            'files': [
                'demo.py',
                'add_noise.py'
            ],
            # 重构相关的临时文件
            'restructure_files': [
                'complete_restructure.py',
                'fixed_restructure.py', 
                'run_restructure.py',
                'test_restructure.py',
                'update_imports.py',
                'restructure_project.py',
                'safe_restructure.py',
                'RESTRUCTURE_GUIDE.md',
                'QUICK_START.md',
                'refactor_plan.md',
                'import_update_report.txt',
                'test_report.txt',
                'restructure.log'
            ]
        }
    
    def preview_organization(self):
        """预览将要整理的文件和目录"""
        self.log("预览整理操作...")
        
        items_to_organize = self.get_items_to_organize()
        
        print("\n📁 将要移动到 origin/ 的原始目录:")
        print("=" * 60)
        existing_dirs = []
        missing_dirs = []
        
        for dir_name in items_to_organize['directories']:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                existing_dirs.append(dir_name)
                print(f"✅ {dir_name}/ → origin/{dir_name}/")
            else:
                missing_dirs.append(dir_name)
                print(f"❌ {dir_name}/ (不存在)")
        
        print("\n📄 将要移动到 origin/ 的原始文件:")
        print("=" * 60)
        existing_files = []
        missing_files = []
        
        for file_name in items_to_organize['files']:
            file_path = self.project_root / file_name
            if file_path.exists():
                existing_files.append(file_name)
                print(f"✅ {file_name} → origin/{file_name}")
            else:
                missing_files.append(file_name)
                print(f"❌ {file_name} (不存在)")
        
        print("\n🛠️ 将要移动到 origin/restructure_scripts/ 的重构文件:")
        print("=" * 60)
        restructure_existing = []
        restructure_missing = []
        
        for file_name in items_to_organize['restructure_files']:
            file_path = self.project_root / file_name
            if file_path.exists():
                restructure_existing.append(file_name)
                print(f"✅ {file_name} → origin/restructure_scripts/{file_name}")
            else:
                restructure_missing.append(file_name)
                print(f"❌ {file_name} (不存在)")
        
        print(f"\n📊 统计:")
        print(f"  - 可移动目录: {len(existing_dirs)}")
        print(f"  - 可移动文件: {len(existing_files)}")
        print(f"  - 可移动重构文件: {len(restructure_existing)}")
        print(f"  - 缺失项目: {len(missing_dirs) + len(missing_files) + len(restructure_missing)}")
        
        return existing_dirs, existing_files, restructure_existing
    
    def verify_new_structure(self):
        """验证新的目录结构是否完整"""
        self.log("验证新目录结构...")
        
        required_dirs = [
            'src',
            'src/utils',
            'src/classification', 
            'src/regression',
            'src/augmentation',
            'src/preprocessing',
            'src/ui',
            'notebooks',
            'scripts',
            'tests'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.log(f"❌ 缺少新目录结构: {missing_dirs}")
            return False
        else:
            self.log("✅ 新目录结构完整")
            return True
    
    def create_origin_structure(self):
        """创建origin目录结构"""
        self.log("创建origin目录结构...")
        
        # 创建主要的origin目录
        self.origin_dir.mkdir(exist_ok=True)
        
        # 创建重构脚本子目录
        restructure_scripts_dir = self.origin_dir / "restructure_scripts"
        restructure_scripts_dir.mkdir(exist_ok=True)
        
        self.log(f"✅ 创建目录: {self.origin_dir}")
        self.log(f"✅ 创建目录: {restructure_scripts_dir}")
        
        return True
    
    def organize_project(self):
        """执行项目整理"""
        self.log("开始项目整理...")
        
        # 1. 验证新结构
        if not self.verify_new_structure():
            self.log("❌ 新目录结构不完整，停止整理")
            return False
        
        # 2. 创建origin目录结构
        if not self.create_origin_structure():
            self.log("❌ 创建origin目录失败")
            return False
        
        # 3. 预览整理
        existing_dirs, existing_files, restructure_files = self.preview_organization()
        
        total_items = len(existing_dirs) + len(existing_files) + len(restructure_files)
        if total_items == 0:
            self.log("没有需要整理的项目")
            return True
        
        # 4. 用户确认
        print(f"\n📦 即将整理 {total_items} 个项目到 origin/ 文件夹")
        print("这个操作会移动文件，但不会删除它们。")
        print("原始文件将被保存在 origin/ 目录中以备将来参考。")
        response = input("确认执行整理？(y/N): ")
        
        if response.lower() != 'y':
            self.log("用户取消整理操作")
            return False
        
        # 5. 执行移动
        success_count = 0
        
        # 移动目录
        for dir_name in existing_dirs:
            try:
                src_path = self.project_root / dir_name
                dest_path = self.origin_dir / dir_name
                
                # 如果目标已存在，先删除
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                
                shutil.move(str(src_path), str(dest_path))
                self.log(f"✅ 移动目录: {dir_name}/ → origin/{dir_name}/")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 移动目录失败 {dir_name}: {str(e)}")
        
        # 移动原始文件
        for file_name in existing_files:
            try:
                src_path = self.project_root / file_name
                dest_path = self.origin_dir / file_name
                
                # 如果目标已存在，先删除
                if dest_path.exists():
                    dest_path.unlink()
                
                shutil.move(str(src_path), str(dest_path))
                self.log(f"✅ 移动文件: {file_name} → origin/{file_name}")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 移动文件失败 {file_name}: {str(e)}")
        
        # 移动重构脚本文件
        restructure_scripts_dir = self.origin_dir / "restructure_scripts"
        for file_name in restructure_files:
            try:
                src_path = self.project_root / file_name
                dest_path = restructure_scripts_dir / file_name
                
                # 如果目标已存在，先删除
                if dest_path.exists():
                    dest_path.unlink()
                
                shutil.move(str(src_path), str(dest_path))
                self.log(f"✅ 移动重构文件: {file_name} → origin/restructure_scripts/{file_name}")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 移动重构文件失败 {file_name}: {str(e)}")
        
        self.log(f"整理完成: 成功移动 {success_count}/{total_items} 个项目")
        
        # 6. 创建说明文件
        self.create_origin_readme()
        
        # 7. 最终验证
        self.final_verification()
        
        return success_count == total_items
    
    def create_origin_readme(self):
        """在origin目录中创建说明文件"""
        readme_content = f"""# Origin Files - 原始文件备份

这个目录包含了项目重构前的原始文件和目录结构。

## 目录说明

### 原始代码目录
- `Utils/` - 原始工具函数目录（已重构到 `src/utils/`）
- `regression/` - 原始回归模块（已重构到 `src/regression/`）
- `classfication/` - 原始分类模块（已重构到 `src/classification/`）
- `augmentation/` - 原始数据增强模块（已重构到 `src/augmentation/`）
- `preprocess/` - 原始预处理模块（已重构到 `src/preprocessing/`）
- `UI_version/` - 原始UI模块（已重构到 `src/ui/`）
- `model_demo/` - 原始模型演示（已重构到 `notebooks/demos/`）

### 原始文件
- `demo.py` - 原始演示文件（已重构到 `notebooks/exploration/demo.py`）
- `add_noise.py` - 原始噪声添加文件（已重构到 `src/preprocessing/add_noise.py`）

### 重构脚本
`restructure_scripts/` 目录包含了所有用于项目重构的脚本文件：
- 各种重构脚本（`*_restructure.py`）
- 重构指南和文档
- 重构过程中生成的报告文件

## 重构时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 注意事项
- 这些文件仅作为备份保存，新的项目结构位于根目录
- 如需恢复某个文件，可以从这里复制到相应的新位置
- 数据集目录（dataset/, dataset_classify/ 等）未被移动，仍在原位置
"""
        
        readme_path = self.origin_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        self.log(f"✅ 创建说明文件: {readme_path}")
    
    def final_verification(self):
        """最终验证项目结构"""
        self.log("执行最终验证...")
        
        print("\n📁 整理后的项目结构:")
        print("=" * 60)
        
        # 显示主要目录
        main_dirs = ['src', 'notebooks', 'scripts', 'tests', 'configs', 'docs', 'origin']
        for dir_name in main_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/")
            else:
                print(f"❌ {dir_name}/ (缺失)")
        
        # 显示数据目录（应该保持不变）
        data_dirs = ['dataset', 'dataset_classify', 'dataset_result']
        print("\n💾 数据目录 (保持不变):")
        for dir_name in data_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/")
            else:
                print(f"❌ {dir_name}/ (不存在)")
        
        # 显示origin目录内容
        print(f"\n📦 origin/ 目录内容:")
        if self.origin_dir.exists():
            for item in self.origin_dir.iterdir():
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                else:
                    print(f"  📄 {item.name}")
        
        self.log("最终验证完成")

def main():
    """主函数"""
    print("📦 项目整理工具")
    print("=" * 60)
    print("功能:")
    print("- 将原始文件移动到 origin/ 文件夹保存")
    print("- 将重构脚本移动到 origin/restructure_scripts/")
    print("- 保留数据集目录不变")
    print("- 创建详细的说明文档")
    print("- 不删除任何文件，只是重新组织")
    print("=" * 60)
    
    organizer = ProjectOrganizer()
    
    # 执行整理
    if organizer.organize_project():
        print("\n✅ 项目整理成功完成！")
        print("🎉 原始文件已安全保存到 origin/ 目录")
        print("📁 项目现在有了清晰的结构，同时保留了所有原始文件")
        print(f"📝 详细日志: {organizer.log_file}")
        print(f"📖 查看说明: {organizer.origin_dir}/README.md")
        return 0
    else:
        print("\n❌ 项目整理失败或被取消")
        return 1

if __name__ == "__main__":
    sys.exit(main())