#!/usr/bin/env python3
"""
清理和移动脚本 - 完成真正的项目重构
删除已复制到新位置的原始文件和目录
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

class ProjectCleanup:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.log_file = self.project_root / "cleanup.log"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def get_files_to_remove(self):
        """获取需要删除的原始文件和目录列表"""
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
            'cleanup_files': [
                'complete_restructure.py',
                'fixed_restructure.py', 
                'run_restructure.py',
                'test_restructure.py',
                'update_imports.py',
                'restructure_project.py',
                'RESTRUCTURE_GUIDE.md',
                'QUICK_START.md',
                'refactor_plan.md',
                'import_update_report.txt',
                'test_report.txt',
                'restructure.log'
            ]
        }
    
    def preview_cleanup(self):
        """预览将要删除的文件和目录"""
        self.log("预览清理操作...")
        
        items_to_remove = self.get_files_to_remove()
        
        print("\n🗑️ 将要删除的原始目录:")
        print("=" * 50)
        existing_dirs = []
        missing_dirs = []
        
        for dir_name in items_to_remove['directories']:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                existing_dirs.append(dir_name)
                print(f"✅ {dir_name}/ (已复制到 src/)")
            else:
                missing_dirs.append(dir_name)
                print(f"❌ {dir_name}/ (不存在)")
        
        print("\n📄 将要删除的原始文件:")
        print("=" * 50)
        existing_files = []
        missing_files = []
        
        for file_name in items_to_remove['files']:
            file_path = self.project_root / file_name
            if file_path.exists():
                existing_files.append(file_name)
                print(f"✅ {file_name} (已复制到新位置)")
            else:
                missing_files.append(file_name)
                print(f"❌ {file_name} (不存在)")
        
        print("\n🧹 将要删除的重构临时文件:")
        print("=" * 50)
        cleanup_existing = []
        cleanup_missing = []
        
        for file_name in items_to_remove['cleanup_files']:
            file_path = self.project_root / file_name
            if file_path.exists():
                cleanup_existing.append(file_name)
                print(f"✅ {file_name}")
            else:
                cleanup_missing.append(file_name)
                print(f"❌ {file_name} (不存在)")
        
        print(f"\n📊 统计:")
        print(f"  - 可删除目录: {len(existing_dirs)}")
        print(f"  - 可删除文件: {len(existing_files)}")
        print(f"  - 可删除临时文件: {len(cleanup_existing)}")
        print(f"  - 缺失项目: {len(missing_dirs) + len(missing_files) + len(cleanup_missing)}")
        
        return existing_dirs, existing_files, cleanup_existing
    
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
    
    def cleanup_project(self):
        """执行项目清理"""
        self.log("开始项目清理...")
        
        # 1. 验证新结构
        if not self.verify_new_structure():
            self.log("❌ 新目录结构不完整，停止清理")
            return False
        
        # 2. 预览清理
        existing_dirs, existing_files, cleanup_files = self.preview_cleanup()
        
        total_items = len(existing_dirs) + len(existing_files) + len(cleanup_files)
        if total_items == 0:
            self.log("没有需要清理的项目")
            return True
        
        # 3. 用户确认
        print(f"\n⚠️ 即将删除 {total_items} 个项目")
        print("这个操作不可逆！请确保新的目录结构中已包含所有必要文件。")
        response = input("确认执行清理？(输入 'YES' 确认): ")
        
        if response != 'YES':
            self.log("用户取消清理操作")
            return False
        
        # 4. 执行删除
        success_count = 0
        
        # 删除目录
        for dir_name in existing_dirs:
            try:
                dir_path = self.project_root / dir_name
                shutil.rmtree(dir_path)
                self.log(f"✅ 删除目录: {dir_name}/")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 删除目录失败 {dir_name}: {str(e)}")
        
        # 删除文件
        for file_name in existing_files + cleanup_files:
            try:
                file_path = self.project_root / file_name
                file_path.unlink()
                self.log(f"✅ 删除文件: {file_name}")
                success_count += 1
            except Exception as e:
                self.log(f"❌ 删除文件失败 {file_name}: {str(e)}")
        
        self.log(f"清理完成: 成功删除 {success_count}/{total_items} 个项目")
        
        # 5. 最终验证
        self.final_verification()
        
        return success_count == total_items
    
    def final_verification(self):
        """最终验证项目结构"""
        self.log("执行最终验证...")
        
        print("\n📁 清理后的项目结构:")
        print("=" * 50)
        
        # 显示主要目录
        main_dirs = ['src', 'notebooks', 'scripts', 'tests', 'configs', 'docs']
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
        
        self.log("最终验证完成")

def main():
    """主函数"""
    print("🧹 项目清理工具")
    print("=" * 50)
    print("功能:")
    print("- 删除已复制的原始文件和目录")
    print("- 完成真正的移动重构")
    print("- 保留数据集目录")
    print("- 清理临时重构文件")
    print("=" * 50)
    
    cleanup = ProjectCleanup()
    
    # 执行清理
    if cleanup.cleanup_project():
        print("\n✅ 项目清理成功完成！")
        print("🎉 现在你有了一个干净、重构后的项目结构")
        print(f"📝 详细日志: {cleanup.log_file}")
        return 0
    else:
        print("\n❌ 项目清理失败或被取消")
        return 1

if __name__ == "__main__":
    sys.exit(main())