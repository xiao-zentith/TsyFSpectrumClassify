from Utils.cosine_similarity import cosine_similarity
from regression.utils.CustomDataset import CustomDataset
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import json
def visualize_and_save_results(test_data, model, output_folder, current_fold):
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model.eval()
    all_cos_sim_pred1 = []
    all_cos_sim_pred2 = []
    all_rmse_pred1 = []
    all_rmse_pred2 = []

    with torch.no_grad():
        for i, (inputs, targets1, targets2) in enumerate(test_loader):
            inputs, targets1, targets2 = inputs, targets1, targets2

            preds1, preds2 = model(inputs)

            input_np = inputs.squeeze().cpu().numpy()
            target1_np = targets1.squeeze().cpu().numpy()
            target2_np = targets2.squeeze().cpu().numpy()
            pred1_np = preds1.squeeze().cpu().numpy()
            pred2_np = preds2.squeeze().cpu().numpy()

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            axes[0, 0].imshow(input_np, cmap='viridis')
            axes[0, 0].set_title('Input')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(target1_np, cmap='viridis')
            axes[0, 1].set_title('Target 1')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(pred1_np, cmap='viridis')
            axes[0, 2].set_title('Predicted 1')
            axes[0, 2].axis('off')

            axes[1, 1].imshow(target2_np, cmap='viridis')
            axes[1, 1].set_title('Target 2')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(pred2_np, cmap='viridis')
            axes[1, 2].set_title('Predicted 2')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'test_sample_{i}.png'))
            plt.close(fig)

            # Save predictions and targets to files
            np.savez(os.path.join(output_folder, f'test_sample_{i}_data.npz'),
                     input=input_np, target1=target1_np, target2=target2_np,
                     pred1=pred1_np, pred2=pred2_np)

            # Calculate Cosine Similarity
            cos_sim_pred1 = cosine_similarity(pred1_np, target1_np).item()
            cos_sim_pred2 = cosine_similarity(pred2_np, target2_np).item()

            # Calculate RMSE
            rmse_pred1 = np.sqrt(np.mean((pred1_np - target1_np) ** 2)).item()
            rmse_pred2 = np.sqrt(np.mean((pred2_np - target2_np) ** 2)).item()

            all_cos_sim_pred1.append(cos_sim_pred1)
            all_cos_sim_pred2.append(cos_sim_pred2)
            all_rmse_pred1.append(rmse_pred1)
            all_rmse_pred2.append(rmse_pred2)

    avg_cos_sim_pred1 = np.mean(all_cos_sim_pred1)
    avg_cos_sim_pred2 = np.mean(all_cos_sim_pred2)
    avg_rmse_pred1 = np.mean(all_rmse_pred1)
    avg_rmse_pred2 = np.mean(all_rmse_pred2)

    results = {
        'average_cos_sim_pred1': avg_cos_sim_pred1,
        'average_cos_sim_pred2': avg_cos_sim_pred2,
        'average_rmse_pred1': avg_rmse_pred1,
        'average_rmse_pred2': avg_rmse_pred2
    }

    # 在保存结果时添加fold编号
    with open(os.path.join(output_folder, f'fold_{current_fold}_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results