import os
import json
import torch
from src.regression.models.DualSimpleCNN import DualSimpleCNN
from regression.training.test_model import visualize_and_save_results


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


def main(model_root, dataset_info_path, output_summary_path):
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    model_files = find_all_pth_files(model_root)
    all_metrics = {}
    num_targets = 0

    for model_path in model_files:
        fold = extract_fold_number(model_path)
        # if fold is None or fold not in dataset_info:
        #     continue

        print(f"Evaluating fold {fold} from: {model_path}")
        model = DualSimpleCNN(True, in_channels=1, out_channels=1, branch_number=2)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        fold_output = os.path.dirname(model_path)
        results = visualize_and_save_results(model, dataset_info[fold]['test'], fold_output, fold)

        all_metrics[fold] = results
        num_targets = len(results['rmse'])  # 获取target数量

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
    model_root_folder = "get_data_path("raw")"  # 替换为你的根目录路径
    dataset_info_json = "get_data_path("raw")"  # 替换为你的dataset_info路径
    output_json_path = os.path.join(model_root_folder, "summary_all_folds.json")

    main(model_root_folder, dataset_info_json, output_json_path)