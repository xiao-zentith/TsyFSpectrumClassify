from datetime import datetime
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

from Utils.cosine_similarity import cosine_similarity
from regression.utils.CustomDataset import CustomDataset
from regression.regression_model.ResUNet_co_encoder import DualResNetSharedEncoder


def train_model(fold_data, output_folder, current_fold, num_epochs=200, batch_size=5, learning_rate=1e-3, patience=20):
    os.makedirs(output_folder, exist_ok=True)

    train_dataset = CustomDataset(fold_data['train'])
    val_dataset = CustomDataset(fold_data['validation'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DualResNetSharedEncoder(in_channels=1, out_channels=1)
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

        # # Early stopping logic
        # if val_total_loss < best_val_loss:
        #     best_val_loss = val_total_loss
        #     epochs_no_improve = 0
        #     torch.save(model.state_dict(), best_model_path)  # Save the best model
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f"Early stopping after {epoch + 1} epochs")
        #         break

    # return best_model_path


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


if __name__ == "__main__":
    with open(r'dataset_info_C6_FITC.json', 'r') as f:
        dataset_info = json.load(f)

    with open('config_C6_FITC.json', 'r') as config_file:
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
        best_model_path, train_log = train_model(
            dataset_info[current_fold],
            fold_output_folder,
            current_fold
        )
        best_models_paths.append(best_model_path)
        convergence_speeds.append(train_log['converged_epoch'])

        # 测试模型
        model = DualResNetSharedEncoder(in_channels=1, out_channels=1)
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



