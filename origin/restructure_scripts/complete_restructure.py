#!/usr/bin/env python3
"""
完整的项目重构脚本
包含备份、重构、验证等功能
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

class ProjectRestructurer:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_dir = self.project_root.parent / f"TsyFSpectrumClassify_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = self.project_root / "restructure.log"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def create_backup(self):
        """创建项目备份（仅代码文件）"""
        self.log("开始创建项目备份...")
        
        # 要排除的目录和文件
        exclude_patterns = [
            'dataset/',
            'dataset_classify/',
            '.venv/',
            '__pycache__/',
            '*.pyc',
            '*.pyo',
            'logs/',
            'results/',
            'dataset_result/',
            '.git/',
            '*.log'
        ]
        
        try:
            # 使用rsync创建备份
            cmd = ['rsync', '-av'] + [f'--exclude={pattern}' for pattern in exclude_patterns]
            cmd.extend([str(self.project_root) + '/', str(self.backup_dir) + '/'])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.log(f"备份创建成功: {self.backup_dir}")
                return True
            else:
                self.log(f"备份创建失败: {result.stderr}")
                return False
        except Exception as e:
            self.log(f"备份创建异常: {str(e)}")
            return False
    
    def get_new_structure(self):
        """定义新的目录结构"""
        return {
            'data': {
                'raw': {},
                'processed': {},
                'augmented': {},
                'results': {}
            },
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
            'configs': {
                'model_configs': {},
                'data_configs': {},
                'training_configs': {}
            },
            'scripts': {},
            'docs': {}
        }
    
    def get_file_mapping(self):
        """定义文件映射关系"""
        return {
            # Utils目录 - 数据IO
            'Utils/read_mat.py': 'src/utils/data_io/mat_reader.py',
            'Utils/read)mat.py': 'src/utils/data_io/mat_reader_alt.py',
            'Utils/read_matrix.py': 'src/utils/data_io/matrix_reader.py',
            'Utils/read_npz.py': 'src/utils/data_io/npz_reader.py',
            'Utils/load_data.py': 'src/utils/data_io/data_loader.py',
            'Utils/extract_data.py': 'src/utils/data_io/data_extractor.py',
            'Utils/extract_460.py': 'src/utils/data_io/extract_460.py',
            'Utils/mat_tool.py': 'src/utils/data_io/mat_tool.py',
            'Utils/generate_json.py': 'src/utils/data_io/json_generator.py',
            
            # Utils目录 - 可视化
            'Utils/draw_2D_spectrum.py': 'src/utils/visualization/spectrum_plotter.py',
            'Utils/draw_2D_spectrum_xlsx.py': 'src/utils/visualization/spectrum_xlsx_plotter.py',
            'Utils/draw_contour.py': 'src/utils/visualization/contour_plotter.py',
            'Utils/draw_radar.py': 'src/utils/visualization/radar_plotter.py',
            'Utils/draw_label.py': 'src/utils/visualization/label_drawer.py',
            'Utils/plot_result.py': 'src/utils/visualization/result_plotter.py',
            
            # Utils目录 - 评估指标
            'Utils/compute_similarity.py': 'src/utils/metrics/similarity_calculator.py',
            'Utils/compute_pearson.py': 'src/utils/metrics/pearson_calculator.py',
            'Utils/compute_relative_error.py': 'src/utils/metrics/relative_error.py',
            'Utils/cosine_similarity.py': 'src/utils/metrics/cosine_similarity.py',
            
            # Utils目录 - 文件操作
            'Utils/batch_resize.py': 'src/utils/file_operations/batch_resizer.py',
            'Utils/resize.py': 'src/utils/file_operations/resizer.py',
            'Utils/merge_json.py': 'src/utils/file_operations/json_merger.py',
            'Utils/merge_txt.py': 'src/utils/file_operations/txt_merger.py',
            'Utils/txt_2_xlsx.py': 'src/utils/file_operations/txt_to_xlsx.py',
            'Utils/modify_xlsx.py': 'src/utils/file_operations/xlsx_modifier.py',
            'Utils/remove_txt_name.py': 'src/utils/file_operations/name_processor.py',
            
            # 预处理
            'Utils/restore_matrix.py': 'src/preprocessing/matrix_restorer.py',
            'Utils/spectrum_2_tsyF.py': 'src/preprocessing/spectrum_converter.py',
            'preprocess/ZScore_norm.py': 'src/preprocessing/zscore_normalizer.py',
            'preprocess/add_noise.py': 'src/preprocessing/noise_adder.py',
            'preprocess/augment_data.py': 'src/preprocessing/data_augmenter.py',
            'add_noise.py': 'src/preprocessing/noise_adder_main.py',
            
            # 分类模型
            'classfication/model/2D_CNN1.py': 'src/classification/models/cnn_2d_v1.py',
            'classfication/model/KNN1.py': 'src/classification/models/knn_v1.py',
            'classfication/model/LSTM1.py': 'src/classification/models/lstm_v1.py',
            'classfication/model/RF1.py': 'src/classification/models/random_forest_v1.py',
            'classfication/model/Transformer1.py': 'src/classification/models/transformer_v1.py',
            'classfication/model/SimpleCNN.py': 'src/classification/models/simple_cnn.py',
            'classfication/model/SimpleLSTM.py': 'src/classification/models/simple_lstm.py',
            'classfication/model/SimpleTransformer.py': 'src/classification/models/simple_transformer.py',
            'classfication/model/GateNetWork.py': 'src/classification/models/gate_network.py',
            'classfication/model/vote_model.py': 'src/classification/models/vote_model.py',
            
            # 分类模型演示
            'classfication/classify_model_demo/2D_CNN.py': 'src/classification/models/demo/cnn_2d_v2.py',
            'classfication/classify_model_demo/KNN.py': 'src/classification/models/demo/knn_v2.py',
            'classfication/classify_model_demo/Moe.py': 'src/classification/models/demo/moe.py',
            'model_demo/1D-CNN.py': 'src/classification/models/demo/cnn_1d.py',
            'model_demo/2D_CNN.py': 'src/classification/models/demo/cnn_2d_v3.py',
            'model_demo/KNN.py': 'src/classification/models/demo/knn_v3.py',
            'model_demo/LSTM.py': 'src/classification/models/demo/lstm.py',
            'model_demo/PLS-DA.py': 'src/classification/models/demo/pls_da.py',
            'model_demo/RandomForest.py': 'src/classification/models/demo/random_forest.py',
            'model_demo/Transformer.py': 'src/classification/models/demo/transformer_v2.py',
            'model_demo/Transformer_kimi.py': 'src/classification/models/demo/transformer_kimi.py',
            'model_demo/feature_engineering_and_CNN.py': 'src/classification/models/demo/feature_engineering_cnn.py',
            
            # 分类工具
            'classfication/Utils/ImageDataset.py': 'src/classification/utils/image_dataset.py',
            'classfication/Utils/generate_category_json.py': 'src/classification/utils/category_generator.py',
            'classfication/Utils/plot.py': 'src/classification/utils/plot_utils.py',
            'classfication/Utils/read_matrix.py': 'src/classification/utils/matrix_reader.py',
            
            # 回归模型
            'regression/model/DualSimpleCNN.py': 'src/regression/models/dual_simple_cnn.py',
            'regression/model/DualUNet.py': 'src/regression/models/dual_unet.py',
            'regression/model/DualUNet_co_encoder.py': 'src/regression/models/dual_unet_shared_encoder.py',
            'regression/model/FVGG11.py': 'src/regression/models/vgg11.py',
            'regression/model/ResNet18.py': 'src/regression/models/resnet18.py',
            'regression/model/UNet.py': 'src/regression/models/unet.py',
            
            # 回归训练
            'regression/training/CustomDataset.py': 'src/regression/training/custom_dataset.py',
            'regression/training/test_model.py': 'src/regression/training/test_model.py',
            'regression/training/train_model.py': 'src/regression/training/train_model.py',
            
            # 回归脚本
            'regression/batch_run.py': 'scripts/batch_run_regression.py',
            'regression/run_training.py': 'scripts/run_regression_training.py',
            
            # 数据增强
            'augmentation/GMM.py': 'src/augmentation/gmm.py',
            'augmentation/MixUp.py': 'src/augmentation/mixup.py',
            'augmentation/VAE.py': 'src/augmentation/vae.py',
            'augmentation/draw_contour.py': 'src/augmentation/contour_drawer.py',
            
            # UI
            'UI_version/draw_contour_ui.py': 'src/ui/contour_ui.py',
            
            # 脚本
            'demo.py': 'scripts/demo.py',
            'Utils/batch_test.py': 'scripts/batch_test.py',
            'Utils/test.py': 'scripts/test_utils.py',
            'Utils/temp.py': 'scripts/temp.py',
        }
    
    def create_directory_structure(self, structure, base_path=None):
        """创建目录结构"""
        if base_path is None:
            base_path = self.project_root
        
        for name, subdirs in structure.items():
            dir_path = base_path / name
            dir_path.mkdir(exist_ok=True)
            self.log(f"创建目录: {dir_path}")
            
            if isinstance(subdirs, dict) and subdirs:
                self.create_directory_structure(subdirs, dir_path)
    
    def copy_file_with_rename(self, src_path, dest_path):
        """复制并重命名文件"""
        src_full = self.project_root / src_path
        dest_full = self.project_root / dest_path
        
        if not src_full.exists():
            self.log(f"源文件不存在: {src_full}")
            return False
        
        # 确保目标目录存在
        dest_full.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(src_full, dest_full)
            self.log(f"复制文件: {src_path} -> {dest_path}")
            return True
        except Exception as e:
            self.log(f"复制文件失败 {src_path}: {str(e)}")
            return False
    
    def create_init_files(self, directory):
        """创建__init__.py文件"""
        for root, dirs, files in os.walk(directory):
            # 跳过某些目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'logs', 'results']]
            
            root_path = Path(root)
            if root_path.name in ['src', 'utils', 'classification', 'regression', 'augmentation', 'preprocessing', 'ui', 'tests']:
                init_file = root_path / '__init__.py'
                if not init_file.exists():
                    init_file.touch()
                    self.log(f"创建__init__.py: {init_file}")
    
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
            "scipy>=1.7.0",
            "opencv-python>=4.5.0",
            "pillow>=8.3.0",
            "jupyter>=1.0.0",
            "tqdm>=4.62.0",
            "h5py>=3.3.0"
        ]
        
        req_file = self.project_root / 'requirements.txt'
        with open(req_file, 'w') as f:
            f.write('\n'.join(requirements))
        self.log(f"创建requirements.txt: {req_file}")
    
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
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Data files
data/
dataset/
dataset_classify/
*.mat
*.npz
*.h5

# Logs and results
logs/
results/
*.log

# OS
.DS_Store
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# Model files
*.pth
*.pkl
*.joblib
"""
        
        gitignore_file = self.project_root / '.gitignore'
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        self.log(f"创建.gitignore: {gitignore_file}")
    
    def create_readme(self):
        """创建README.md文件"""
        readme_content = """# TsyF Spectrum Classification Project

## 项目简介
这是一个用于光谱分类和回归分析的机器学习项目。

## 项目结构
```
├── data/                   # 数据目录
├── src/                    # 源代码
│   ├── classification/     # 分类模块
│   ├── regression/         # 回归模块
│   ├── utils/             # 工具模块
│   ├── augmentation/      # 数据增强
│   ├── preprocessing/     # 数据预处理
│   └── ui/               # 用户界面
├── notebooks/             # Jupyter notebooks
├── tests/                # 测试代码
├── configs/              # 配置文件
├── scripts/              # 脚本文件
└── docs/                 # 文档
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
详细使用方法请参考docs目录下的文档。

## 贡献
欢迎提交Issue和Pull Request。
"""
        
        readme_file = self.project_root / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        self.log(f"创建README.md: {readme_file}")
    
    def restructure_project(self):
        """执行完整的项目重构"""
        self.log("开始项目重构...")
        
        # 1. 创建备份
        if not self.create_backup():
            self.log("备份失败，终止重构")
            return False
        
        # 2. 创建新的目录结构
        self.log("创建新的目录结构...")
        structure = self.get_new_structure()
        self.create_directory_structure(structure)
        
        # 3. 移动和重命名文件
        self.log("移动和重命名文件...")
        file_mapping = self.get_file_mapping()
        success_count = 0
        total_count = len(file_mapping)
        
        for src, dest in file_mapping.items():
            if self.copy_file_with_rename(src, dest):
                success_count += 1
        
        self.log(f"文件移动完成: {success_count}/{total_count}")
        
        # 4. 创建__init__.py文件
        self.log("创建__init__.py文件...")
        self.create_init_files(self.project_root / 'src')
        self.create_init_files(self.project_root / 'tests')
        
        # 5. 创建项目文件
        self.log("创建项目配置文件...")
        self.create_requirements_txt()
        self.create_gitignore()
        self.create_readme()
        
        self.log("项目重构完成!")
        self.log(f"备份位置: {self.backup_dir}")
        self.log(f"日志文件: {self.log_file}")
        
        return True
    
    def validate_restructure(self):
        """验证重构结果"""
        self.log("验证重构结果...")
        
        required_dirs = [
            'src/classification/models',
            'src/regression/models',
            'src/utils/data_io',
            'src/utils/visualization',
            'scripts',
            'configs'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.log(f"缺少目录: {missing_dirs}")
            return False
        else:
            self.log("目录结构验证通过")
            return True

def main():
    """主函数"""
    print("TsyF Spectrum Classification 项目重构工具")
    print("=" * 50)
    
    # 确认操作
    response = input("是否开始重构项目？这将创建备份并重新组织文件结构 (y/N): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    # 执行重构
    restructurer = ProjectRestructurer()
    
    try:
        if restructurer.restructure_project():
            if restructurer.validate_restructure():
                print("\n✅ 项目重构成功完成!")
                print(f"📁 备份位置: {restructurer.backup_dir}")
                print(f"📝 日志文件: {restructurer.log_file}")
                print("\n下一步:")
                print("1. 检查重构后的文件结构")
                print("2. 更新import语句")
                print("3. 运行测试验证功能")
            else:
                print("\n⚠️ 重构完成但验证失败，请检查日志")
        else:
            print("\n❌ 重构失败，请检查日志")
    except Exception as e:
        print(f"\n❌ 重构过程中发生异常: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())