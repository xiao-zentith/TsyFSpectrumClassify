from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from regression.utils.CustomDataset import CustomDataset

def train_model(model, fold_data, output_folder, current_fold, num_epochs=200, batch_size=5, learning_rate=1e-3, patience=20):
    os.makedirs(output_folder, exist_ok=True)
    train_dataset = CustomDataset(fold_data['train'])
    val_dataset = CustomDataset(fold_data['validation'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    rmse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(output_folder, f'best_model_fold_{current_fold}.pth')

    # 初始化训练日志
    train_log = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse1': [],
        'train_mae1': [],
        'train_rmse2': [],
        'train_mae2': [],
        'val_rmse1': [],
        'val_mae1': [],
        'val_rmse2': [],
        'val_mae2': [],
        'converged_epoch': num_epochs  # 默认完整训练
    }

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

        # 记录训练指标
        train_log['train_rmse1'].append(epoch_rmse_loss1)
        train_log['train_mae1'].append(epoch_mae_loss1)
        train_log['train_rmse2'].append(epoch_rmse_loss2)
        train_log['train_mae2'].append(epoch_mae_loss2)
        train_total_loss = epoch_rmse_loss1 + epoch_mae_loss1 + epoch_rmse_loss2 + epoch_mae_loss2
        train_log['train_loss'].append(train_total_loss)

        # 记录验证指标
        train_log['val_rmse1'].append(val_epoch_rmse_loss1)
        train_log['val_mae1'].append(val_epoch_mae_loss1)
        train_log['val_rmse2'].append(val_epoch_rmse_loss2)
        train_log['val_mae2'].append(val_epoch_mae_loss2)
        train_log['val_loss'].append(val_total_loss)

        # Early stopping逻辑
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                train_log['converged_epoch'] = epoch + 1
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # 保存训练日志
    with open(os.path.join(output_folder, f'fold_{current_fold}_train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=4)

        # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_log['train_loss'], label='Train Loss')
    plt.plot(train_log['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {current_fold} Training/Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'fold_{current_fold}_loss_curve.png'))
    plt.close()
    return best_model_path, train_log



