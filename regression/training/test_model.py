import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import json
# from regression.utils.cosine_similarity import cosine_similarity
from regression.training.CustomDataset import CustomDataset


def visualize_and_save_results(model, test_data, output_folder, current_fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model.eval()

    # 存储所有 target 的指标
    all_cos_sim = []  # 每个 target 一个 list
    all_rmse = []
    all_wmae = []
    all_pearson = []
    all_mse = []
    all_rae = []
    all_mre = []
    all_nrmse = []
    all_rrmse = []
    all_rel_wmae = []



    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, *targets = batch
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]
            num_targets = len(targets)

            # 初始化列表
            if not all_cos_sim:
                # 初始化存储列表
                all_mse = [[] for _ in range(num_targets)]
                all_rmse = [[] for _ in range(num_targets)]
                all_rae = [[] for _ in range(num_targets)]
                all_mre = [[] for _ in range(num_targets)]
                all_nrmse = [[] for _ in range(num_targets)]
                all_rrmse = [[] for _ in range(num_targets)]
                all_wmae = [[] for _ in range(num_targets)]
                all_rel_wmae = [[] for _ in range(num_targets)]
                all_cos_sim = [[] for _ in range(num_targets)]
                all_pearson = [[] for _ in range(num_targets)]

            preds = model(inputs)

            if isinstance(preds, torch.Tensor):  # 兼容单输出模型
                preds = [preds]

            input_np = inputs.squeeze().cpu().numpy()

            # 可视化设置（支持多个 target）
            # rows = (num_targets + 1) // 2 + 1  # 计算需要多少行来展示输入+预测+真实值
            # fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
            # axes = axes if isinstance(axes[0], np.ndarray) else np.array([axes])

            # # 输入图像
            # axes[0, 0].imshow(input_np, cmap='viridis')
            # axes[0, 0].set_title('Input')
            # axes[0, 0].axis('off')

            # 真实值与预测值
            for j in range(num_targets):
                pred_j = preds[j].squeeze().cpu().numpy()
                target_j = targets[j].squeeze().cpu().numpy()

                # # 绘图
                # row = (j // 2) + 1
                # col = j % 2 * 2
                #
                # axes[row, col].imshow(target_j, cmap='viridis')
                # axes[row, col].set_title(f'Target {j + 1}')
                # axes[row, col].axis('off')
                #
                # axes[row, col + 1].imshow(pred_j, cmap='viridis')
                # axes[row, col + 1].set_title(f'Predicted {j + 1}')
                # axes[row, col + 1].axis('off')

                # 保存数据
                np.savez(os.path.join(output_folder, f'test_sample_{i}_target_{j}.npz'),
                         input=input_np,
                         target=target_j,
                         pred=pred_j)

                # 计算指标
                metrics = calculate_all_metrics(pred_j, target_j)

                all_mse[j].append(metrics['mse'])
                all_rmse[j].append(metrics['rmse'])
                all_rae[j].append(metrics['rae'])
                all_mre[j].append(metrics['mre'])
                all_nrmse[j].append(metrics['nrmse'])
                all_rrmse[j].append(metrics['rrmse'])
                all_wmae[j].append(metrics['wmae'])
                all_rel_wmae[j].append(metrics['rel_wmae'])
                all_cos_sim[j].append(metrics['cos_sim'])
                all_pearson[j].append(metrics['pearson'])

            # plt.tight_layout()
            # plt.savefig(os.path.join(output_folder, f'test_sample_{i}.png'))
            # plt.close(fig)

    # 计算平均指标
    avg_mse = [np.mean(errors) for errors in all_mse]
    avg_rmse = [np.mean(errors) for errors in all_rmse]
    avg_rae = [np.mean(errors) for errors in all_rae]
    avg_mre = [np.mean(errors) for errors in all_mre]
    avg_nrmse = [np.mean(errors) for errors in all_nrmse]
    avg_rrmse = [np.mean(errors) for errors in all_rrmse]
    avg_wmae = [np.mean(errors) for errors in all_wmae]
    avg_rel_wmae = [np.mean(errors) for errors in all_rel_wmae]
    avg_cos_sim = [np.mean(scores) for scores in all_cos_sim]
    avg_pearson = [np.mean(scores) for scores in all_pearson]

    results = {
        'mse': avg_mse,
        'rmse': avg_rmse,
        'rae': avg_rae,
        'mre': avg_mre,
        'nrmse': avg_nrmse,
        'rrmse': avg_rrmse,
        'wmae': avg_wmae,
        'rel_wmae': avg_rel_wmae,
        'cos_sim': avg_cos_sim,
        'pearson': avg_pearson
    }

    # 保存测试结果
    with open(os.path.join(output_folder, f'fold_{current_fold}_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results

def continuous_weighted_mae(pred, target, base_weight=1.0, peak_factor=2.0):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    # 权重与目标强度成正比，越强的区域权重越高
    weights = base_weight + (target / (np.max(target) + 1e-8)) * (peak_factor - 1)

    wmae = np.average(np.abs(pred - target), weights=weights)
    return wmae


def calculate_all_metrics(pred, target, base_weight=1.0, peak_factor=2.0):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    # 权重计算（用于 WMAE 和 Rel-WMAE）
    weights = base_weight + (target / (np.max(target) + 1e-8)) * (peak_factor - 1)

    # 原始指标
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))

    # 相对指标
    rae = np.sum(np.abs(pred - target)) / (np.sum(np.abs(target - np.mean(target))) + 1e-8)
    mre = np.mean(np.abs(pred - target) / (np.abs(target) + 1e-8))
    rrmse = rmse / (np.mean(target) + 1e-8)

    # 新增 NRMSE: RMSE 归一化到目标值范围 [min(target), max(target)]
    target_range = np.max(target) - np.min(target)
    nrmse = rmse / (target_range + 1e-8)  # 防止除以零

    # 自定义加权 MAE
    wmae = np.average(np.abs(pred - target), weights=weights)
    rel_wmae = wmae / (np.mean(target) + 1e-8)

    # 余弦相似度
    cos_sim = cosine_similarity(pred, target).item()

    # 皮尔逊相关系数
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    pearson = np.corrcoef(pred_flat, target_flat)[0, 1].item()

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'cos_sim': float(cos_sim),
        'pearson': float(pearson),
        'mae': float(mae),
        'rae': float(rae),
        'mre': float(mre),
        'nrmse': float(nrmse),
        'rrmse': float(rrmse),
        'wmae': float(wmae),
        'rel_wmae': float(rel_wmae)
    }

def cosine_similarity(matrix_a, matrix_b):
    # 将矩阵展平成向量
    vector_a = matrix_a.flatten()
    vector_b = matrix_b.flatten()

    # 计算点积
    dot_product = np.dot(vector_a, vector_b)

    # 计算向量的模
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # 计算余弦相似度
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 如果其中一个向量为零，则相似度为0

    similarity = dot_product / (norm_a * norm_b)
    return similarity

if __name__ == "__main__":
    # 构造一个简单的测试数据（模拟光谱图像）
    from regression.model.DualSimpleCNN import DualSimpleCNN
    model = DualSimpleCNN(True, in_channels=1, out_channels=1, branch_number=4).to("cuda")
    best_model_path = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/regression/dataset_result/Fish/DualSimpleCNN/training_20250603_234941/fold_15/best_model_fold_15.pth"
    model.load_state_dict(torch.load(best_model_path))
    with open(r"/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/regression/config/dataset_info_Fish.json") as f:
        dataset_info = json.load(f)
    fold_output_folder = r"/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/regression/dataset_result/Fish/DualSimpleCNN/training_20250603_234941/fold_1"
    test_results = visualize_and_save_results(
        model,
        dataset_info[15]['test'],
        fold_output_folder,
        15,
    )