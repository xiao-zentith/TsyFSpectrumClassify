import sys
from datetime import datetime
import json
import torch
import numpy as np
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在目录（regression目录）
current_dir = os.path.dirname(current_script_path)
# 获取项目根目录（regression的父目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from regression.model.DualSimpleCNN import DualSimpleCNN
from regression.model.DualUNet import DualUNet
from regression.model.DualUNet_co_encoder import DualUNetSharedEncoder
from regression.model.FVGG11 import FVGG11, DualFVGG11
from regression.model.ResNet18 import ResNet18
from regression.training.test_model import visualize_and_save_results
from regression.training.train_model import train_model
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
    parser.add_argument('--branch_number', type=int, default=4,
                        help='The number of targets.')
    parser.add_argument('--is_fold', type=str2bool, default=False,
                        help='Whether or not to use batch normalization.')
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


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(r"./config/dataset_info_" + args.config + ".json") as f:
        dataset_info = json.load(f)

    with open(r"./config/config_" + args.config + ".json") as config_file:
        config = json.load(config_file)
        output_folder = config.get("dataset_result", "results")

    # 创建带有时间戳的结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, f"{args.model}", f"training_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    all_results = []
    convergence_speeds = []
    best_models_paths = []
    branch_number = args.branch_number

    # 判断是否使用分折数据
    if args.is_fold:
        print("Using K-fold cross-validation.")
        for current_fold in range(len(dataset_info)):
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

    else:
        print("Training without folds (single split).")
        fold_output_folder = output_folder
        os.makedirs(fold_output_folder, exist_ok=True)

        fold_data = dataset_info[0]

        # 构建模型
        model = None
        if args.model == 'DualSimpleCNN':
            model = DualSimpleCNN(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'VGG11':
            model = DualFVGG11(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'ResNet18':
            model = ResNet18(in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'DualUNet':
            model = DualUNet(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        elif args.model == 'DualUNetSharedEncoder':
            model = DualUNetSharedEncoder(args.is_norm, in_channels=1, out_channels=1, branch_number=branch_number).to(device)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        # 训练模型
        best_model_path, train_log = train_model(
            model,
            fold_data,
            fold_output_folder,
            current_fold=0,
            num_epochs=args.epoch,
            batch_size=args.batch_size,
            patience=args.patience
        )
        best_models_paths.append(best_model_path)
        convergence_speeds.append(train_log['converged_epoch'])

        # 加载最佳模型并测试
        model.load_state_dict(torch.load(best_model_path))
        test_results = visualize_and_save_results(
            model,
            fold_data['test'],
            fold_output_folder,
            current_fold=0
        )
        all_results.append(test_results)

        # 保存日志
        fold_summary = {
            'best_model_path': best_model_path,
            'convergence_epoch': train_log['converged_epoch'],
            'training_log': train_log,
            'test_results': test_results
        }
        with open(os.path.join(fold_output_folder, f'single_split_summary.json'), 'w') as f:
            json.dump(fold_summary, f, indent=4)

    # === 根据第一个结果推断目标数量 ===
    num_targets = len(all_results[0]['cos_sim'])  # 假设每个 result 都包含 cos_sim 和 rmse 列表

    # === 收集所有 fold 的结果 ===
    avg_cos_sim = [[] for _ in range(num_targets)]
    avg_rmse = [[] for _ in range(num_targets)]

    for result in all_results:
        for i in range(num_targets):
            avg_cos_sim[i].append(result['cos_sim'][i])
            avg_rmse[i].append(result['rmse'][i])

    # === 计算平均值和最优值 ===
    mean_cos_sim = [np.mean(scores) for scores in avg_cos_sim]
    max_cos_sim = [max(scores) for scores in avg_cos_sim]
    mean_rmse = [np.mean(errors) for errors in avg_rmse]
    min_rmse = [min(errors) for errors in avg_rmse]

    # === 构建动态报告 ===
    average_metrics = {
        'cos_sim': mean_cos_sim,
        'rmse': mean_rmse
    }

    best_metrics = {
        'cos_sim': max_cos_sim,
        'rmse': min_rmse
    }

    overall_report = {
        'input_parameters': vars(args),
        'average_metrics': average_metrics,
        'best_metrics': best_metrics,
        'convergence': {
            'average_epochs': np.mean(convergence_speeds),
            'epochs_per_fold': convergence_speeds
        },
        'folds': [
            {
                'fold_number': i,
                'best_model_path': best_models_paths[i],
                'convergence_epoch': convergence_speeds[i],
                'training_log': all_results[i]['training_log'],
                'test_results': all_results[i]
            }
            for i in range(len(all_results))
        ],
        'best_models': best_models_paths,
        'training_time': timestamp
    }

    # 保存总报告
    with open(os.path.join(output_folder, 'overall_report.json'), 'w') as f:
        json.dump(overall_report, f, indent=4)





