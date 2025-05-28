import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from regression.training.CustomDataset import CustomDataset


def train_model(model, fold_data, output_folder, current_fold, num_epochs, batch_size, patience, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    # 自动检测 target 数量
    sample_input, *sample_targets = next(iter(train_loader))
    num_targets = len(sample_targets)

    # 初始化训练日志（动态生成）
    train_log = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [[] for _ in range(num_targets)],
        'train_mae': [[] for _ in range(num_targets)],
        'val_rmse': [[] for _ in range(num_targets)],
        'val_mae': [[] for _ in range(num_targets)],
        'converged_epoch': num_epochs
    }

    for epoch in range(num_epochs):
        model.train()
        running_rmse = [0.0 for _ in range(num_targets)]
        running_mae = [0.0 for _ in range(num_targets)]

        for batch in train_loader:
            inputs, *targets = batch
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            preds = model(inputs)

            if isinstance(preds, torch.Tensor):  # 兼容单输出模型
                preds = [preds]

            rmse_losses = []
            mae_losses = []

            for i, (pred, target) in enumerate(zip(preds, targets)):
                rmse_loss = torch.sqrt(rmse_criterion(pred, target))
                mae_loss = mae_criterion(pred, target)
                rmse_losses.append(rmse_loss)
                mae_losses.append(mae_loss)
                running_rmse[i] += rmse_loss.item() * inputs.size(0)
                running_mae[i] += mae_loss.item() * inputs.size(0)

            total_loss = sum(rmse_losses + mae_losses)
            total_loss.backward()
            optimizer.step()

        # 计算平均损失
        epoch_rmse = [rm / len(train_dataset) for rm in running_rmse]
        epoch_mae = [ma / len(train_dataset) for ma in running_mae]

        print(f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Train RMSE: {[f"{x:.4f}" for x in epoch_rmse]}')
        print(f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Train MAE: {[f"{x:.4f}" for x in epoch_mae]}')

        # 验证阶段
        model.eval()
        val_running_rmse = [0.0 for _ in range(num_targets)]
        val_running_mae = [0.0 for _ in range(num_targets)]

        with torch.no_grad():
            for batch in val_loader:
                inputs, *targets = batch
                inputs = inputs.to(device)
                targets = [t.to(device) for t in targets]

                preds = model(inputs)

                if isinstance(preds, torch.Tensor):
                    preds = [preds]

                for i, (pred, target) in enumerate(zip(preds, targets)):
                    rmse_loss = torch.sqrt(rmse_criterion(pred, target))
                    mae_loss = mae_criterion(pred, target)
                    val_running_rmse[i] += rmse_loss.item() * inputs.size(0)
                    val_running_mae[i] += mae_loss.item() * inputs.size(0)

        val_epoch_rmse = [vr / len(val_dataset) for vr in val_running_rmse]
        val_epoch_mae = [vm / len(val_dataset) for vm in val_running_mae]
        val_total_loss = sum(val_epoch_rmse + val_epoch_mae)

        print(f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Val RMSE: {[f"{x:.4f}" for x in val_epoch_rmse]}')
        print(f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Val MAE: {[f"{x:.4f}" for x in val_epoch_mae]}')

        # 更新日志
        for i in range(num_targets):
            train_log['train_rmse'][i].append(epoch_rmse[i])
            train_log['train_mae'][i].append(epoch_mae[i])
            train_log['val_rmse'][i].append(val_epoch_rmse[i])
            train_log['val_mae'][i].append(val_epoch_mae[i])

        train_log['train_loss'].append(sum(epoch_rmse + epoch_mae))
        train_log['val_loss'].append(val_total_loss)

        # Early stopping
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

    # 保存日志
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
