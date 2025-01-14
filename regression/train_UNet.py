import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from Utils.cosine_similarity import cosine_similarity
from regression.CustomDataset import CustomDataset
from regression.regression_model.DualUNet_magic import DualUNetSharedEncoder

def train_model(fold_data, num_epochs=200, batch_size=5, learning_rate=1e-3, patience=20):
    train_dataset = CustomDataset(fold_data['train'])
    val_dataset = CustomDataset(fold_data['validation'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DualUNetSharedEncoder(in_channels=1, out_channels=1)

    rmse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f'best_model_fold_{current_fold}.pth'

    for epoch in range(num_epochs):
        model.train()
        running_rmse_loss1, running_mae_loss1 = 0.0, 0.0
        running_rmse_loss2, running_mae_loss2 = 0.0, 0.0

        for inputs, outputs1, outputs2 in train_loader:
            inputs, outputs1, outputs2 = inputs, outputs1, outputs2

            optimizer.zero_grad()

            preds1, preds2 = model(inputs)

            rmse_loss1 = torch.sqrt(rmse_criterion(preds1, outputs1))
            mae_loss1 = mae_criterion(preds1, outputs1)

            rmse_loss2 = torch.sqrt(rmse_criterion(preds2, outputs2))
            mae_loss2 = mae_criterion(preds2, outputs2)

            total_loss = rmse_loss1 + mae_loss1 + rmse_loss2 + mae_loss2
            total_loss.backward()
            optimizer.step()

            running_rmse_loss1 += rmse_loss1.item() * inputs.size(0)
            running_mae_loss1 += mae_loss1.item() * inputs.size(0)
            running_rmse_loss2 += rmse_loss2.item() * inputs.size(0)
            running_mae_loss2 += mae_loss2.item() * inputs.size(0)

        epoch_rmse_loss1 = running_rmse_loss1 / len(train_dataset)
        epoch_mae_loss1 = running_mae_loss1 / len(train_dataset)
        epoch_rmse_loss2 = running_rmse_loss2 / len(train_dataset)
        epoch_mae_loss2 = running_mae_loss2 / len(train_dataset)

        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Train rmse Loss: {epoch_rmse_loss1:.4f}, {epoch_rmse_loss2:.4f}')
        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Train MAE Loss: {epoch_mae_loss1:.4f}, {epoch_mae_loss2:.4f}')

        model.eval()
        val_running_rmse_loss1, val_running_mae_loss1 = 0.0, 0.0
        val_running_rmse_loss2, val_running_mae_loss2 = 0.0, 0.0

        with torch.no_grad():
            for inputs, outputs1, outputs2 in val_loader:
                inputs, outputs1, outputs2 = inputs, outputs1, outputs2

                preds1, preds2 = model(inputs)

                rmse_loss1 = torch.sqrt(rmse_criterion(preds1, outputs1))
                mae_loss1 = mae_criterion(preds1, outputs1)

                rmse_loss2 = torch.sqrt(rmse_criterion(preds2, outputs2))
                mae_loss2 = mae_criterion(preds2, outputs2)

                val_running_rmse_loss1 += rmse_loss1.item() * inputs.size(0)
                val_running_mae_loss1 += mae_loss1.item() * inputs.size(0)
                val_running_rmse_loss2 += rmse_loss2.item() * inputs.size(0)
                val_running_mae_loss2 += mae_loss2.item() * inputs.size(0)

        val_epoch_rmse_loss1 = val_running_rmse_loss1 / len(val_dataset)
        val_epoch_mae_loss1 = val_running_mae_loss1 / len(val_dataset)
        val_epoch_rmse_loss2 = val_running_rmse_loss2 / len(val_dataset)
        val_epoch_mae_loss2 = val_running_mae_loss2 / len(val_dataset)

        val_total_loss = val_epoch_rmse_loss1 + val_epoch_mae_loss1 + val_epoch_rmse_loss2 + val_epoch_mae_loss2

        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Val rmse Loss: {val_epoch_rmse_loss1:.4f}, {val_epoch_rmse_loss2:.4f}')
        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Val MAE Loss: {val_epoch_mae_loss1:.4f}, {val_epoch_mae_loss2:.4f}')

        # Early stopping logic
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    return best_model_path


def visualize_and_save_results(test_data, model, output_folder):
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

    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    with open(r'C:\Users\xiao\PycharmProjects\TsyFSpectrumClassify\dataset\dataset_preprocess\C6 + hpts\dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
        output_folder = config.get("dataset_result", "results")

    best_models_paths = []
    for current_fold in range(len(dataset_info)):
        fold_data = dataset_info[current_fold]
        best_model_path = train_model(fold_data)
        best_models_paths.append(best_model_path)

    overall_avg_cos_sim_pred1 = 0
    overall_avg_cos_sim_pred2 = 0
    overall_avg_rmse_pred1 = 0
    overall_avg_rmse_pred2 = 0

    for current_fold in range(len(dataset_info)):
        best_model_path = best_models_paths[current_fold]
        model = DualUNetSharedEncoder(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(best_model_path, weights_only=True))  # Load the best model safely

        # Assuming the last fold's test set is used for final evaluation
        test_data = dataset_info[current_fold]['test']
        fold_output_folder = os.path.join(output_folder, f'fold_{current_fold}')
        visualize_and_save_results(test_data, model, fold_output_folder)

        with open(os.path.join(fold_output_folder, 'results.json'), 'r') as f:
            fold_results = json.load(f)
            overall_avg_cos_sim_pred1 += fold_results['average_cos_sim_pred1']
            overall_avg_cos_sim_pred2 += fold_results['average_cos_sim_pred2']
            overall_avg_rmse_pred1 += fold_results['average_rmse_pred1']
            overall_avg_rmse_pred2 += fold_results['average_rmse_pred2']

    overall_avg_cos_sim_pred1 /= len(dataset_info)
    overall_avg_cos_sim_pred2 /= len(dataset_info)
    overall_avg_rmse_pred1 /= len(dataset_info)
    overall_avg_rmse_pred2 /= len(dataset_info)

    overall_results = {
        'overall_average_cos_sim_pred1': overall_avg_cos_sim_pred1,
        'overall_average_cos_sim_pred2': overall_avg_cos_sim_pred2,
        'overall_average_rmse_pred1': overall_avg_rmse_pred1,
        'overall_average_rmse_pred2': overall_avg_rmse_pred2
    }

    with open(os.path.join(output_folder, 'overall_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)



