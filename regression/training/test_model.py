import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from regression.utils.cosine_similarity import cosine_similarity
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

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, *targets = batch
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]
            num_targets = len(targets)

            # 初始化列表
            if not all_cos_sim:
                all_cos_sim = [[] for _ in range(num_targets)]
                all_rmse = [[] for _ in range(num_targets)]
                all_wmae =  [[] for _ in range(num_targets)]

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
                cos_sim_j = cosine_similarity(pred_j, target_j).item()
                rmse_j = np.sqrt(np.mean((pred_j - target_j) ** 2)).item()
                wmae_j = continuous_weighted_mae(pred_j, target_j).item()

                all_cos_sim[j].append(cos_sim_j)
                all_rmse[j].append(rmse_j)
                all_wmae[j].append(wmae_j)

            # plt.tight_layout()
            # plt.savefig(os.path.join(output_folder, f'test_sample_{i}.png'))
            # plt.close(fig)

    # 计算平均指标
    avg_cos_sim = [np.mean(scores) for scores in all_cos_sim]
    avg_rmse = [np.mean(rmses) for rmses in all_rmse]
    avg_wmae = [np.mean(wmaes) for wmaes in all_wmae]

    results = {
        'cos_sim': avg_cos_sim,
        'rmse': avg_rmse,
        'wmae': avg_wmae
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
