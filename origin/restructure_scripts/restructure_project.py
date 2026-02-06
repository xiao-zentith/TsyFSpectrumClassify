#!/usr/bin/env python3
"""
脚本用于重组TsyFSpectrumClassify项目结构
"""
import os
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path('get_project_path()')

# 新的目录结构
NEW_STRUCTURE = {
    'data': {
        'raw': {},
        'processed': {},
        'augmented': {},
        'results': {}
    },
    'src': {
        'common_utils': {
            'io': {},
            'visualization': {}
        },
        'classification': {
            'models': {},
            'utils': {}
        },
        'regression': {
            'models': {},
            'utils': {}
        },
        'augmentation': {},
        'preprocessing': {},
        'ui': {}
    },
    'notebooks': {},
    'tests': {},
    'configs': {},
    'scripts': {}
}

# 文件映射关系（旧路径 -> 新路径）
FILE_MAPPING = {
    # Utils目录下的文件映射
    'Utils/read_mat.py': 'src/common_utils/io/read_mat.py',
    'Utils/read_matrix.py': 'src/common_utils/io/read_matrix.py',
    'Utils/read_npz.py': 'src/common_utils/io/read_npz.py',
    'Utils/load_data.py': 'src/common_utils/io/load_data.py',
    'Utils/extract_data.py': 'src/common_utils/io/extract_data.py',
    'Utils/extract_460.py': 'src/common_utils/io/extract_460.py',
    'Utils/mat_tool.py': 'src/common_utils/io/mat_tool.py',
    'Utils/txt_2_xlsx.py': 'src/common_utils/io/txt_2_xlsx.py',
    'Utils/modify_xlsx.py': 'src/common_utils/io/modify_xlsx.py',
    'Utils/generate_json.py': 'src/common_utils/io/generate_json.py',
    'Utils/merge_json.py': 'src/common_utils/io/merge_json.py',
    'Utils/merge_txt.py': 'src/common_utils/io/merge_txt.py',
    
    # 可视化工具
    'Utils/draw_2D_spectrum.py': 'src/common_utils/visualization/draw_2d_spectrum.py',
    'Utils/draw_2D_spectrum_xlsx.py': 'src/common_utils/visualization/draw_2d_spectrum_xlsx.py',
    'Utils/draw_contour.py': 'src/common_utils/visualization/draw_contour.py',
    'Utils/draw_radar.py': 'src/common_utils/visualization/draw_radar.py',
    'Utils/draw_label.py': 'src/common_utils/visualization/draw_label.py',
    'Utils/plot_result.py': 'src/common_utils/visualization/plot_result.py',
    
    # 相似度计算工具
    'Utils/compute_similarity.py': 'src/common_utils/compute_similarity.py',
    'Utils/compute_pearson.py': 'src/common_utils/compute_pearson.py',
    'Utils/compute_relative_error.py': 'src/common_utils/compute_relative_error.py',
    'Utils/cosine_similarity.py': 'src/common_utils/cosine_similarity.py',
    
    # 数据处理工具
    'Utils/resize.py': 'src/preprocessing/resize.py',
    'Utils/batch_resize.py': 'src/preprocessing/batch_resize.py',
    'Utils/restore_matrix.py': 'src/preprocessing/restore_matrix.py',
    'Utils/spectrum_2_tsyF.py': 'src/preprocessing/spectrum_2_tsyf.py',
    'Utils/remove_txt_name.py': 'src/preprocessing/remove_txt_name.py',
    
    # 分类相关
    'classfication/Utils/ImageDataset.py': 'src/classification/utils/image_dataset.py',
    'classfication/Utils/generate_category_json.py': 'src/classification/utils/generate_category_json.py',
    'classfication/Utils/plot.py': 'src/classification/utils/plot.py',
    'classfication/Utils/read_matrix.py': 'src/classification/utils/read_matrix.py',
    
    'classfication/model/2D_CNN1.py': 'src/classification/models/cnn_2d_v1.py',
    'classfication/model/KNN1.py': 'src/classification/models/knn_v1.py',
    'classfication/model/LSTM1.py': 'src/classification/models/lstm_v1.py',
    'classfication/model/RF1.py': 'src/classification/models/random_forest_v1.py',
    'classfication/model/SimpleCNN.py': 'src/classification/models/simple_cnn.py',
    'classfication/model/SimpleLSTM.py': 'src/classification/models/simple_lstm.py',
    'classfication/model/SimpleTransformer.py': 'src/classification/models/simple_transformer.py',
    'classfication/model/Transformer1.py': 'src/classification/models/transformer_v1.py',
    'classfication/model/GateNetWork.py': 'src/classification/models/gate_network.py',
    'classfication/model/vote_model.py': 'src/classification/models/vote_model.py',
    
    'classfication/classify_model_demo/2D_CNN.py': 'src/classification/models/demo/cnn_2d_demo.py',
    'classfication/classify_model_demo/KNN.py': 'src/classification/models/demo/knn_demo.py',
    'classfication/classify_model_demo/Moe.py': 'src/classification/models/demo/moe_demo.py',
    
    # 回归相关
    'regression/model/DualSimpleCNN.py': 'src/regression/models/dual_simple_cnn.py',
    'regression/model/DualUNet.py': 'src/regression/models/dual_unet.py',
    'regression/model/DualUNet_co_encoder.py': 'src/regression/models/dual_unet_shared_encoder.py',
    'regression/model/FVGG11.py': 'src/regression/models/vgg11.py',
    'regression/model/ResNet18.py': 'src/regression/models/resnet18.py',
    'regression/model/UNet.py': 'src/regression/models/unet.py',
    'regression/__init__.py': 'src/regression/__init__.py',
    
    'regression/training/CustomDataset.py': 'src/regression/utils/custom_dataset.py',
    'regression/training/__init__.py': 'src/regression/utils/__init__.py',
    'regression/training/test_model.py': 'src/regression/utils/test_model.py',
    'regression/training/train_model.py': 'src/regression/utils/train_model.py',
    
    'regression/batch_run.py': 'src/regression/batch_run.py',
    'regression/run_training.py': 'src/regression/run_training.py',
    
    # 数据增强
    'augmentation/GMM.py': 'src/augmentation/gmm.py',
    'augmentation/MixUp.py': 'src/augmentation/mixup.py',
    'augmentation/VAE.py': 'src/augmentation/vae.py',
    'augmentation/draw_contour.py': 'src/augmentation/draw_contour.py',
    
    # 预处理
    'preprocess/ZScore_norm.py': 'src/preprocessing/zscore_norm.py',
    'preprocess/add_noise.py': 'src/preprocessing/add_noise.py',
    'preprocess/augment_data.py': 'src/preprocessing/augment_data.py',
    'add_noise.py': 'src/preprocessing/add_noise_root.py',
    
    # UI相关
    'UI_version/draw_contour_ui.py': 'src/ui/draw_contour_ui.py',
    
    # 演示脚本
    'model_demo/1D-CNN.py': 'src/classification/models/demo/cnn_1d_demo.py',
    'model_demo/2D_CNN.py': 'src/classification/models/demo/cnn_2d_demo_v2.py',
    'model_demo/KNN.py': 'src/classification/models/demo/knn_demo_v2.py',
    'model_demo/LSTM.py': 'src/classification/models/demo/lstm_demo.py',
    'model_demo/PLS-DA.py': 'src/classification/models/demo/pls_da_demo.py',
    'model_demo/RandomForest.py': 'src/classification/models/demo/random_forest_demo.py',
    'model_demo/Transformer.py': 'src/classification/models/demo/transformer_demo_v2.py',
    'model_demo/Transformer_kimi.py': 'src/classification/models/demo/transformer_kimi_demo.py',
    'model_demo/feature_engineering_and_CNN.py': 'src/classification/models/demo/feature_engineering_cnn_demo.py',
    
    # 主脚本
    'demo.py': 'src/demo.py',
}

# 数据目录映射
DATA_MAPPING = {
    'dataset/dataset_raw': 'data/raw/classification',
    'dataset/dataset_preprocess': 'data/processed/classification',
    'dataset/dataset_resized': 'data/processed/classification_resized',
    'dataset/dataset_target': 'data/results/classification',
    
    'dataset_classify/dataset_raw': 'data/raw/classification_alt',
    'dataset_classify/dataset_preprocess': 'data/processed/classification_alt',
    'dataset_classify/dataset_preprocess2': 'data/processed/classification_alt2',
    'dataset_classify/dataset_noise': 'data/augmented/classification_noise',
    
    'regression/dataset_result': 'data/results/regression',
    'regression/logs': 'data/results/regression_logs',
    'regression/results': 'data/results/regression_output',
    'regression/config': 'configs/regression',
}

def create_directory_structure(root, structure, current_path=None):
    """创建目录结构"""
    if current_path is None:
        current_path = root
    
    for name, substructure in structure.items():
        path = current_path / name
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {path}")
        
        if substructure:  # 如果有子目录
            create_directory_structure(root, substructure, path)

def copy_file(src_path, dest_path):
    """复制文件并确保目标目录存在"""
    dest_dir = os.path.dirname(dest_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        print(f"复制文件: {src_path} -> {dest_path}")
    else:
        print(f"警告: 源文件不存在 {src_path}")

def copy_directory(src_path, dest_path):
    """复制整个目录"""
    if os.path.exists(src_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        for item in os.listdir(src_path):
            s = os.path.join(src_path, item)
            d = os.path.join(dest_path, item)
            if os.path.isdir(s):
                copy_directory(s, d)
            else:
                shutil.copy2(s, d)
        print(f"复制目录: {src_path} -> {dest_path}")
    else:
        print(f"警告: 源目录不存在 {src_path}")

def create_init_files(directory):
    """在目录及其子目录中创建__init__.py文件"""
    for root, dirs, files in os.walk(directory):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass  # 创建空文件
            print(f"创建__init__.py: {init_file}")

def main():
    # 创建新的目录结构
    create_directory_structure(PROJECT_ROOT, NEW_STRUCTURE)
    
    # 复制文件到新位置
    for old_path, new_path in FILE_MAPPING.items():
        src = PROJECT_ROOT / old_path
        dest = PROJECT_ROOT / new_path
        copy_file(src, dest)
    
    # 复制数据目录
    for old_path, new_path in DATA_MAPPING.items():
        src = PROJECT_ROOT / old_path
        dest = PROJECT_ROOT / new_path
        copy_directory(src, dest)
    
    # 在src目录及其子目录中创建__init__.py文件
    create_init_files(PROJECT_ROOT / 'src')
    
    print("项目结构重组完成！")
    print("注意: 原始文件和目录仍然保留。请在确认新结构正常工作后手动删除旧文件。")

if __name__ == "__main__":
    main()