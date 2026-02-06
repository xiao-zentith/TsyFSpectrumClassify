import random
import sys
from datetime import datetime
import json
import torch
import numpy as np
import os
from pathlib import Path

# 获取项目根目录并确保可导入 src
current_script_path = Path(__file__).resolve()
current_dir = current_script_path.parent  # .../scripts/training
project_root = current_dir.parent.parent  # .../TsyFSpectrumClassify_remote
sys.path.append(str(project_root))

from src.regression.models.dualsimplecnn import DualSimpleCNN
from src.regression.models.dualunet import DualUNet
from src.regression.models.dualunet_co_encoder import DualUNetSharedEncoder
from src.regression.models.fvgg11 import FVGG11, DualFVGG11
from src.regression.models.resnet18 import ResNet18
from src.regression.training.test_model import visualize_and_save_results
from src.regression.training.train_model import train_model
from src.utils.path_manager import PathManager
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for regression.')
    parser.add_argument('--config', type=str, default='Fish',
                        help='Path to the configuration file.')
    parser.add_argument('--model', type=str, default='DualSimpleCNN',
                        help='Model to use (e.g., DualSimpleCNN).')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Epochs for training.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for training.')
    parser.add_argument('--patience', type=int, default=35,
                        help='Patience for early stopping.')
    parser.add_argument('--branch_number', type=int, default=2,
                        help='The number of targets.')
    parser.add_argument('--loss_type', type=str, default='mae',
                        help='Loss type (e.g., weighted_mae).')
    parser.add_argument('--fold_number', type=str2int, default=20,
                        help='The number of folds.')
    parser.add_argument('--is_norm', type=str2bool, default=True,
                        help='Whether or not to use round-robin cross-validation.')

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2int(v) :
    return int(v)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(1)
    # device = "cuda:1"

    pm = PathManager()
    configs_root = Path(pm.get_path('configs', 'regression'))
    dataset_info_path = configs_root / f"regression_dataset_info_{args.config}.json"
    config_path = configs_root / f"regression_config_{args.config}.json"

    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
        # 统一输出到 results/regression，下层再按模型和时间戳组织
        output_root = Path(pm.get_path('results', 'regression'))

    # 创建带有时间戳的结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = output_root / f"{args.model}" / f"training_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    all_results = []
    convergence_speeds = []
    best_models_paths = []
    branch_number = args.branch_number

    total_folds = len(dataset_info)

    random.seed(42)
    # === 使用 fold_number 控制训练多少折，并随机选择 ===
    if args.fold_number > 0:
        # 确保 fold_number 不超过总折数
        selected_folds = random.sample(range(total_folds), min(args.fold_number, total_folds))
    else:
        # 如果 fold_number <= 0，则只训练一次（不分折）
        selected_folds = [0]

    print(f"Training on {len(selected_folds)} folds: {selected_folds}")

    for current_fold in selected_folds:
        fold_output_folder = os.path.join(output_folder, f'fold_{current_fold}')
        os.makedirs(fold_output_folder, exist_ok=True)

        # 构建模型
        model = None
        if args.model == 'DualSimpleCNN':
            model = DualSimpleCNN(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'VGG11':
            model = DualFVGG11(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'ResNet18':
            model = ResNet18(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'DualUNet':
            model = DualUNet(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'DualUNetSharedEncoder':
            model = DualUNetSharedEncoder(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        # 训练模型
        best_model_path, train_log = train_model(
            model,
            dataset_info[current_fold],
            fold_output_folder,
            current_fold,
            args.loss_type,
            args.epoch,
            args.batch_size,
            args.patience
        )
        best_models_paths.append(best_model_path)
        convergence_speeds.append(train_log['converged_epoch'])

        # 加载最佳模型并测试
        model.load_state_dict(torch.load(best_model_path))
        test_results = visualize_and_save_results(
            model,
            dataset_info[current_fold]['test'],
            fold_output_folder,
            current_fold,
        )
        all_results.append(test_results)

        # 保存单折日志
        fold_summary = {
            'best_model_path': best_model_path,
            'convergence_epoch': train_log['converged_epoch'],
            'training_log': train_log,
            'test_results': test_results
        }
        with open(os.path.join(fold_output_folder, f'fold_{current_fold}_summary.json'), 'w') as f:
            json.dump(fold_summary, f, indent=4)

    # === 根据第一个结果推断目标数量 ===
    num_targets = len(all_results[0]['cos_sim'])  # 假设每个 result 都包含 cos_sim 和 rmse 列表

    # === 收集所有 fold 的结果 ===
    avg_cos_sim = [[] for _ in range(num_targets)]
    avg_rmse = [[] for _ in range(num_targets)]
    avg_nrmse = [[] for _ in range(num_targets)]
    avg_wmae = [[] for _ in range(num_targets)]

    for result in all_results:
        for i in range(num_targets):
            avg_cos_sim[i].append(result['cos_sim'][i])
            avg_rmse[i].append(result['rmse'][i])
            avg_nrmse[i].append(result['nrmse'][i])
            avg_wmae[i].append(result['wmae'][i])

    # === 计算平均值和最优值 ===
    mean_cos_sim = [np.mean(scores) for scores in avg_cos_sim]
    max_cos_sim = [max(scores) for scores in avg_cos_sim]
    mean_rmse = [np.mean(rmses) for rmses in avg_rmse]
    min_rmse = [min(rmses) for rmses in avg_rmse]
    mean_nrmse = [np.mean(nrmses) for nrmses in avg_nrmse]
    min_nrmse = [min(nrmses) for nrmses in avg_nrmse]
    mean_wmae = [np.mean(wmaes) for wmaes in avg_wmae]
    min_wmae = [min(wmaes) for wmaes in avg_wmae]

    average_metrics = {
        'cos_sim': mean_cos_sim,
        'rmse': mean_rmse,
        'nrmse': mean_nrmse,
        'wmae': mean_wmae
    }

    best_metrics = {
        'cos_sim': max_cos_sim,
        'nrmse': min_nrmse,
        'rmse': min_rmse,
        'wmae': min_wmae
    }

    overall_report = {
        'input_parameters': vars(args),
        'average_metrics': average_metrics,
        'best_metrics': best_metrics,
        'convergence': {
            'average_epochs': np.mean(convergence_speeds),
            'epochs_per_fold': convergence_speeds
        },
        'folds_used': selected_folds,
        'best_models': best_models_paths,
        'training_time': timestamp
    }

    # 保存总报告
    with open(os.path.join(output_folder, 'overall_report.json'), 'w') as f:
        json.dump(overall_report, f, indent=4)





