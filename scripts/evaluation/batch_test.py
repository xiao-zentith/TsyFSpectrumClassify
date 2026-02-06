import os
import sys
import json
import torch
from pathlib import Path

# 解析项目根目录，确保可导入 src
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent.parent
sys.path.append(str(project_root))

from src.regression.models.dualsimplecnn import DualSimpleCNN
from src.regression.models.dualunet import DualUNet
from src.regression.models.dualunet_co_encoder import DualUNetSharedEncoder
from src.regression.models.fvgg11 import FVGG11, DualFVGG11
from src.regression.models.resnet18 import ResNet18
from src.regression.training.test_model import visualize_and_save_results
from src.utils.path_manager import PathManager


# 假设函数封装进utils中


def find_all_pth_files(root_dir):
    model_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pth"):
                model_files.append(os.path.join(subdir, file))
    return model_files


def extract_fold_number(path):
    for part in path.split(os.sep):
        if part.startswith("fold_"):
            return int(part.split("_")[1])
    return None


def get_model_from_path(model_path, branch_number=2):
    """根据模型路径推断模型类型并创建相应的模型实例"""
    path_lower = model_path.lower()
    
    if 'dualsimplecnn' in path_lower:
        return DualSimpleCNN(True, in_channels=1, out_channels=1, branch_number=branch_number)
    elif 'vgg11' in path_lower:
        return DualFVGG11(True, in_channels=1, out_channels=1, branch_number=branch_number)
    elif 'resnet18' in path_lower:
        return ResNet18(True, in_channels=1, out_channels=1, branch_number=branch_number)
    elif 'dualunetsharedencoder' in path_lower:
        return DualUNetSharedEncoder(True, in_channels=1, out_channels=1, branch_number=branch_number)
    elif 'dualunet' in path_lower:
        return DualUNet(True, in_channels=1, out_channels=1, branch_number=branch_number)
    else:
        # 默认使用 DualSimpleCNN
        print(f"Warning: Cannot determine model type from path {model_path}, using DualSimpleCNN as default")
        return DualSimpleCNN(True, in_channels=1, out_channels=1, branch_number=branch_number)


def main(model_root, dataset_info_path, output_summary_path):
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    model_files = find_all_pth_files(model_root)
    all_metrics = {}
    num_targets = 0

    for model_path in model_files:
        fold = extract_fold_number(model_path)
        if fold is None:
            print(f"Warning: Cannot extract fold number from {model_path}, skipping...")
            continue
            
        if fold >= len(dataset_info):
            print(f"Warning: Fold {fold} not found in dataset_info, skipping...")
            continue

        print(f"Evaluating fold {fold} from: {model_path}")
        
        # 自动识别模型类型并创建模型
        model = get_model_from_path(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue

        fold_output = os.path.dirname(model_path)
        
        try:
            results = visualize_and_save_results(model, dataset_info[fold]['test'], fold_output, fold)
            all_metrics[fold] = results
            num_targets = len(results['rmse'])  # 获取target数量
        except Exception as e:
            print(f"Error evaluating fold {fold}: {e}")
            continue

    # 汇总所有fold的平均值
    fold_count = len(all_metrics)
    metric_keys = list(all_metrics[next(iter(all_metrics))].keys())

    aggregated = {key: [0.0] * num_targets for key in metric_keys}
    best = {key: [float('inf') if 'rmse' in key else -float('inf') for _ in range(num_targets)] for key in metric_keys}

    for fold_result in all_metrics.values():
        for key in metric_keys:
            for i in range(num_targets):
                aggregated[key][i] += fold_result[key][i]
                if 'rmse' in key:
                    best[key][i] = min(best[key][i], fold_result[key][i])
                else:
                    best[key][i] = max(best[key][i], fold_result[key][i])

    avg_metrics = {f"{key}_pred{i+1}": aggregated[key][i] / fold_count for key in metric_keys for i in range(num_targets)}
    best_metrics = {f"{key}_pred{i+1}": best[key][i] for key in metric_keys for i in range(num_targets)}

    final_summary = {
        "average_metrics": avg_metrics,
        "best_metrics": best_metrics
    }

    with open(output_summary_path, 'w') as f:
        json.dump(final_summary, f, indent=4)

    print("Summary saved to:", output_summary_path)


if __name__ == "__main__":
    pm = PathManager()
    # 默认模型根目录：results/regression
    model_root_folder = Path(pm.get_path('results', 'regression'))

    # 默认数据集信息：configs/regression/regression_dataset_info_Fish.json（可按需替换）
    configs_root = Path(pm.get_path('configs', 'regression'))
    dataset_info_json = configs_root / "regression_dataset_info_Fish.json"

    output_json_path = model_root_folder / "summary_all_folds.json"

    main(str(model_root_folder), str(dataset_info_json), str(output_json_path))