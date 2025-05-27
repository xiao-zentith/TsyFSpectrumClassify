from datetime import datetime
import json
import numpy as np
import os
from regression.training.test_model import visualize_and_save_results
from regression.training.train_model import train_model
from regression.model.DualUNet import DualUNet

if __name__ == "__main__":
    with open('../../config/dataset_info_FITC_HPTS.json') as f:
        dataset_info = json.load(f)

    with open('../../config/config_FITC_HPTS.json') as config_file:
        config = json.load(config_file)
        output_folder = config.get("dataset_result", "results")

    # 创建带有时间戳的结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(config["dataset_result"], f"training_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    all_results = []
    convergence_speeds = []
    best_models_paths = []

    for current_fold in range(len(dataset_info)):
        fold_output_folder = os.path.join(output_folder, f'fold_{current_fold}')
        os.makedirs(fold_output_folder, exist_ok=True)

        # 训练模型
        model = DualUNet(in_channels=1, out_channels=1)
        best_model_path, train_log = train_model(
            model,
            dataset_info[current_fold],
            fold_output_folder,
            current_fold
        )
        best_models_paths.append(best_model_path)
        convergence_speeds.append(train_log['converged_epoch'])

        # 测试模型
        model = DualUNet(in_channels=1, out_channels=1)
        import torch

        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu'), weights_only=True))

        test_results = visualize_and_save_results(
            dataset_info[current_fold]['test'],
            model,
            fold_output_folder,
            current_fold
        )
        all_results.append(test_results)

        # 保存单折完整日志
        fold_summary = {
            'best_model_path': best_model_path,
            'convergence_epoch': train_log['converged_epoch'],
            'training_log': train_log,
            'test_results': test_results
        }
        with open(os.path.join(fold_output_folder, f'fold_{current_fold}_summary.json'), 'w') as f:
            json.dump(fold_summary, f, indent=4)

    # 生成总报告
    overall_report = {
        'average_metrics': {
            'cos_sim_pred1': np.mean([r['average_cos_sim_pred1'] for r in all_results]),
            'cos_sim_pred2': np.mean([r['average_cos_sim_pred2'] for r in all_results]),
            'rmse_pred1': np.mean([r['average_rmse_pred1'] for r in all_results]),
            'rmse_pred2': np.mean([r['average_rmse_pred2'] for r in all_results])
        },
        'best_metrics': {
            'cos_sim_pred1': max([r['average_cos_sim_pred1'] for r in all_results]),
            'cos_sim_pred2': max([r['average_cos_sim_pred2'] for r in all_results]),
            'rmse_pred1': min([r['average_rmse_pred1'] for r in all_results]),
            'rmse_pred2': min([r['average_rmse_pred2'] for r in all_results])
        },
        'convergence': {
            'average_epochs': np.mean(convergence_speeds),
            'epochs_per_fold': convergence_speeds
        },
        'best_models': best_models_paths,
        'training_time': timestamp
    }

    with open(os.path.join(output_folder, 'overall_report.json'), 'w') as f:
        json.dump(overall_report, f, indent=4)



