#!/usr/bin/env python3
"""
安全版本的项目重构脚本
避免递归备份和其他潜在问题
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
import json

class SafeProjectRestructurer:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        # 在项目内部创建备份目录
        self.backup_dir = self.project_root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = self.project_root / "restructure.log"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def create_backup(self):
        """创建项目备份（仅代码文件，避免递归）"""
        self.log("开始创建项目备份...")
        
        # 要排除的目录名称（完全匹配）
        exclude_dirs = {
            'dataset', 'dataset_classify', '.venv', '__pycache__', 
            'logs', 'results', 'dataset_result', '.git', '.idea'
        }
        exclude_extensions = {'.pyc', '.pyo', '.log'}
        
        try:
            # 确保备份目录不存在，避免递归
            if self.backup_dir.exists():
                self.log(f"备份目录已存在，删除旧备份: {self.backup_dir}")
                shutil.rmtree(self.backup_dir)
            
            self.backup_dir.mkdir(exist_ok=True)
            
            def should_exclude(path_relative):
                """检查是否应该排除某个路径"""
                path_parts = path_relative.parts
                
                # 检查是否包含排除的目录
                for part in path_parts:
                    if part in exclude_dirs:
                        return True
                    # 排除所有以backup_开头的目录
                    if part.startswith('backup_'):
                        return True
                
                # 检查文件扩展名
                if path_relative.suffix in exclude_extensions:
                    return True
                    
                return False
            
            # 复制文件
            copied_count = 0
            skipped_count = 0
            
            # 只遍历直接子项，避免深度递归问题
            def copy_directory(src_dir, dest_dir, level=0):
                nonlocal copied_count, skipped_count
                
                if level > 10:  # 防止过深递归
                    self.log(f"警告：目录层级过深，跳过: {src_dir}")
                    return
                
                for item in src_dir.iterdir():
                    try:
                        rel_path = item.relative_to(self.project_root)
                        
                        if should_exclude(rel_path):
                            skipped_count += 1
                            continue
                        
                        dest_path = dest_dir / rel_path.name
                        
                        if item.is_file():
                            # 复制文件
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_path)
                            copied_count += 1
                        elif item.is_dir():
                            # 递归复制目录
                            dest_path.mkdir(exist_ok=True)
                            copy_directory(item, dest_path, level + 1)
                            
                    except Exception as e:
                        self.log(f"复制项目时出错 {item}: {str(e)}")
                        skipped_count += 1
            
            # 开始复制
            copy_directory(self.project_root, self.backup_dir)
            
            self.log(f"备份创建成功: {self.backup_dir}")
            self.log(f"共复制 {copied_count} 个文件，跳过 {skipped_count} 个项目")
            return True
            
        except Exception as e:
            self.log(f"备份创建异常: {str(e)}")
            return False
    
    def preview_changes(self):
        """预览将要进行的更改"""
        self.log("预览重构更改...")
        
        file_mapping = self.get_file_mapping()
        
        print("\n📋 将要进行的文件移动:")
        print("=" * 60)
        
        existing_files = []
        missing_files = []
        
        for src, dest in file_mapping.items():
            src_path = self.project_root / src
            if src_path.exists():
                existing_files.append((src, dest))
                print(f"✅ {src} -> {dest}")
            else:
                missing_files.append(src)
                print(f"❌ {src} (文件不存在)")
        
        print(f"\n📊 统计:")
        print(f"  - 可移动文件: {len(existing_files)}")
        print(f"  - 缺失文件: {len(missing_files)}")
        
        if missing_files:
            print(f"\n⚠️ 以下文件不存在:")
            for file in missing_files:
                print(f"  - {file}")
        
        return len(existing_files), len(missing_files)
    
    def get_new_structure(self):
        """定义新的目录结构"""
        return {
            'src': {
                'utils': {
                    'data_io': {},
                    'visualization': {},
                    'metrics': {},
                    'file_operations': {}
                },
                'classification': {
                    'models': {
                        'demo': {}
                    },
                    'utils': {}
                },
                'regression': {
                    'models': {},
                    'training': {},
                    'utils': {}
                },
                'augmentation': {},
                'preprocessing': {},
                'ui': {}
            },
            'notebooks': {
                'exploration': {},
                'experiments': {},
                'demos': {}
            },
            'tests': {
                'test_classification': {},
                'test_regression': {},
                'test_utils': {}
            },
            'scripts': {
                'training': {},
                'evaluation': {},
                'data_processing': {}
            },
            'configs': {},
            'docs': {}
        }
    
    def get_file_mapping(self):
        """定义文件映射关系"""
        return {
            # 回归模块
            'regression/model/': 'src/regression/models/',
            'regression/training/': 'src/regression/training/',
            'regression/utils/': 'src/regression/utils/',
            'regression/run_training.py': 'scripts/training/run_regression_training.py',
            'regression/batch_run.py': 'scripts/training/batch_regression_run.py',
            
            # 分类模块
            'classfication/model/': 'src/classification/models/',
            'classfication/classify_model_demo/': 'src/classification/models/demo/',
            'classfication/Utils/': 'src/classification/utils/',
            
            # 数据增强
            'augmentation/': 'src/augmentation/',
            
            # 预处理
            'preprocess/': 'src/preprocessing/',
            
            # UI
            'UI_version/': 'src/ui/',
            
            # 工具类 - 数据IO
            'Utils/extract_460.py': 'src/utils/data_io/extract_460.py',
            'Utils/extract_data.py': 'src/utils/data_io/extract_data.py',
            'Utils/generate_json.py': 'src/utils/data_io/generate_json.py',
            'Utils/load_data.py': 'src/utils/data_io/load_data.py',
            'Utils/mat_tool.py': 'src/utils/data_io/mat_tool.py',
            'Utils/merge_json.py': 'src/utils/data_io/merge_json.py',
            'Utils/merge_txt.py': 'src/utils/data_io/merge_txt.py',
            'Utils/read_mat.py': 'src/utils/data_io/read_mat.py',
            'Utils/read_matrix.py': 'src/utils/data_io/read_matrix.py',
            'Utils/read_npz.py': 'src/utils/data_io/read_npz.py',
            'Utils/restore_matrix.py': 'src/utils/data_io/restore_matrix.py',
            'Utils/spectrum_2_tsyF.py': 'src/utils/data_io/spectrum_2_tsyf.py',
            
            # 工具类 - 可视化
            'Utils/draw_2D_spectrum.py': 'src/utils/visualization/draw_2d_spectrum.py',
            'Utils/draw_2D_spectrum_xlsx.py': 'src/utils/visualization/draw_2d_spectrum_xlsx.py',
            'Utils/draw_contour.py': 'src/utils/visualization/draw_contour.py',
            'Utils/draw_label.py': 'src/utils/visualization/draw_label.py',
            'Utils/draw_radar.py': 'src/utils/visualization/draw_radar.py',
            'Utils/plot_result.py': 'src/utils/visualization/plot_result.py',
            
            # 工具类 - 指标计算
            'Utils/compute_pearson.py': 'src/utils/metrics/compute_pearson.py',
            'Utils/compute_relative_error.py': 'src/utils/metrics/compute_relative_error.py',
            'Utils/compute_similarity.py': 'src/utils/metrics/compute_similarity.py',
            'Utils/cosine_similarity.py': 'src/utils/metrics/cosine_similarity.py',
            
            # 工具类 - 文件操作
            'Utils/batch_resize.py': 'src/utils/file_operations/batch_resize.py',
            'Utils/modify_xlsx.py': 'src/utils/file_operations/modify_xlsx.py',
            'Utils/remove_txt_name.py': 'src/utils/file_operations/remove_txt_name.py',
            'Utils/resize.py': 'src/utils/file_operations/resize.py',
            'Utils/txt_2_xlsx.py': 'src/utils/file_operations/txt_2_xlsx.py',
            
            # 脚本
            'Utils/batch_test.py': 'scripts/evaluation/batch_test.py',
            
            # 笔记本
            'model_demo/': 'notebooks/demos/',
            'demo.py': 'notebooks/exploration/demo.py',
            
            # 预处理
            'add_noise.py': 'src/preprocessing/add_noise.py',
        }
    
    def restructure_project(self):
        """执行项目重构"""
        self.log("开始安全项目重构...")
        
        # 1. 预览更改
        existing_count, missing_count = self.preview_changes()
        
        if missing_count > 0:
            response = input(f"\n发现 {missing_count} 个文件缺失，是否继续？(y/N): ")
            if response.lower() != 'y':
                self.log("用户取消重构")
                return False
        
        # 2. 创建备份
        if not self.create_backup():
            self.log("备份失败，停止重构")
            return False
        
        # 3. 创建新目录结构
        self.log("创建新目录结构...")
        new_structure = self.get_new_structure()
        self.create_directory_structure(new_structure)
        
        # 4. 移动和重命名文件
        self.log("移动和重命名文件...")
        file_mapping = self.get_file_mapping()
        success_count = 0
        
        for src, dest in file_mapping.items():
            if self.copy_file_with_rename(src, dest):
                success_count += 1
        
        self.log(f"文件移动完成: {success_count}/{len(file_mapping)}")
        
        # 5. 创建__init__.py文件
        self.log("创建__init__.py文件...")
        self.create_init_files(self.project_root / 'src')
        
        # 6. 创建配置文件
        self.create_requirements_txt()
        self.create_gitignore()
        self.create_readme()
        
        self.log("安全项目重构完成！")
        return True
    
    def create_directory_structure(self, structure, base_path=None):
        """创建目录结构"""
        if base_path is None:
            base_path = self.project_root
        
        for name, subdirs in structure.items():
            dir_path = base_path / name
            dir_path.mkdir(exist_ok=True)
            self.log(f"创建目录: {dir_path}")
            
            if subdirs:
                self.create_directory_structure(subdirs, dir_path)
    
    def copy_file_with_rename(self, src_path, dest_path):
        """复制并重命名文件"""
        try:
            src = self.project_root / src_path
            dest = self.project_root / dest_path
            
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(src, dest)
                else:
                    shutil.copy2(src, dest)
                self.log(f"复制: {src_path} -> {dest_path}")
                return True
            else:
                self.log(f"源文件不存在: {src_path}")
                return False
        except Exception as e:
            self.log(f"复制文件失败 {src_path}: {str(e)}")
            return False
    
    def create_init_files(self, directory):
        """创建__init__.py文件"""
        for root, dirs, files in os.walk(directory):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'dataset', 'dataset_classify'} and not d.startswith('backup_')]
            
            root_path = Path(root)
            if root_path.name in {'src', 'utils', 'classification', 'regression', 'augmentation', 'preprocessing', 'ui', 'tests', 'scripts'}:
                init_file = root_path / '__init__.py'
                if not init_file.exists():
                    init_file.write_text('"""Package initialization file."""\n')
                    self.log(f"创建 __init__.py: {init_file}")
    
    def create_requirements_txt(self):
        """创建requirements.txt文件"""
        requirements = [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "opencv-python>=4.5.0",
            "scipy>=1.7.0",
            "tqdm>=4.62.0",
            "jupyter>=1.0.0",
            "pytest>=6.2.0"
        ]
        
        req_file = self.project_root / 'requirements.txt'
        req_file.write_text('\n'.join(requirements) + '\n')
        self.log(f"创建 requirements.txt")
    
    def create_gitignore(self):
        """创建.gitignore文件"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
dataset/
dataset_classify/
dataset_result/
*.mat
*.npz
*.pkl

# Logs
logs/
*.log

# Results
results/
backup_*/

# OS
.DS_Store
Thumbs.db
"""
        gitignore_file = self.project_root / '.gitignore'
        gitignore_file.write_text(gitignore_content)
        self.log("创建 .gitignore")
    
    def create_readme(self):
        """创建README.md文件"""
        readme_content = """# TsyF Spectrum Classification Project

## 项目结构

```
├── src/                    # 源代码
│   ├── utils/             # 工具函数
│   ├── classification/    # 分类模块
│   ├── regression/        # 回归模块
│   ├── augmentation/      # 数据增强
│   ├── preprocessing/     # 数据预处理
│   └── ui/               # 用户界面
├── notebooks/             # Jupyter notebooks
├── tests/                # 测试代码
├── scripts/              # 脚本文件
├── configs/              # 配置文件
└── docs/                 # 文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

1. 数据预处理：使用 `src/preprocessing/` 中的脚本
2. 模型训练：使用 `scripts/training/` 中的脚本
3. 模型评估：使用 `scripts/evaluation/` 中的脚本

## 重构说明

本项目已完成安全重构，主要改进：
- 模块化的代码组织
- 标准化的文件命名
- 清晰的目录结构
- 完整的测试框架
- 避免递归备份问题
"""
        readme_file = self.project_root / 'README.md'
        readme_file.write_text(readme_content)
        self.log("创建 README.md")

def main():
    """主函数"""
    print("🛡️ 安全版项目重构工具")
    print("=" * 50)
    print("特性:")
    print("- 避免递归备份")
    print("- 预览更改")
    print("- 详细日志")
    print("- 安全检查")
    print("=" * 50)
    
    restructurer = SafeProjectRestructurer()
    
    # 确认执行
    response = input("\n是否开始执行安全重构？(y/N): ")
    if response.lower() != 'y':
        print("❌ 操作已取消")
        return 1
    
    # 执行重构
    if restructurer.restructure_project():
        print("\n✅ 安全重构成功完成！")
        print(f"📁 备份位置: {restructurer.backup_dir}")
        print(f"📝 日志文件: {restructurer.log_file}")
        return 0
    else:
        print("\n❌ 重构失败，请查看日志")
        return 1

if __name__ == "__main__":
    sys.exit(main())