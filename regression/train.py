import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from regression.CustomDataset import CustomDataset
from regression.regression_model.DualUNet import DualUNet


def train_model(fold_data, num_epochs=50, batch_size=4, learning_rate=1e-3):
    train_dataset = CustomDataset(fold_data['train'])
    val_dataset = CustomDataset(fold_data['validation'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DualUNet(in_channels=1, out_channels=1).cuda()

    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_mse_loss1, running_mae_loss1 = 0.0, 0.0
        running_mse_loss2, running_mae_loss2 = 0.0, 0.0

        for inputs, outputs1, outputs2 in train_loader:
            inputs, outputs1, outputs2 = inputs.cuda(), outputs1.cuda(), outputs2.cuda()

            optimizer.zero_grad()

            preds1, preds2 = model(inputs)

            mse_loss1 = mse_criterion(preds1, outputs1)
            mae_loss1 = mae_criterion(preds1, outputs1)

            mse_loss2 = mse_criterion(preds2, outputs2)
            mae_loss2 = mae_criterion(preds2, outputs2)

            total_loss = mse_loss1 + mae_loss1 + mse_loss2 + mae_loss2
            total_loss.backward()
            optimizer.step()

            running_mse_loss1 += mse_loss1.item() * inputs.size(0)
            running_mae_loss1 += mae_loss1.item() * inputs.size(0)
            running_mse_loss2 += mse_loss2.item() * inputs.size(0)
            running_mae_loss2 += mae_loss2.item() * inputs.size(0)

        epoch_mse_loss1 = running_mse_loss1 / len(train_dataset)
        epoch_mae_loss1 = running_mae_loss1 / len(train_dataset)
        epoch_mse_loss2 = running_mse_loss2 / len(train_dataset)
        epoch_mae_loss2 = running_mae_loss2 / len(train_dataset)

        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Train MSE Loss: {epoch_mse_loss1:.4f}, {epoch_mse_loss2:.4f}')
        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Train MAE Loss: {epoch_mae_loss1:.4f}, {epoch_mae_loss2:.4f}')

        model.eval()
        val_running_mse_loss1, val_running_mae_loss1 = 0.0, 0.0
        val_running_mse_loss2, val_running_mae_loss2 = 0.0, 0.0

        with torch.no_grad():
            for inputs, outputs1, outputs2 in val_loader:
                inputs, outputs1, outputs2 = inputs.cuda(), outputs1.cuda(), outputs2.cuda()

                preds1, preds2 = model(inputs)

                mse_loss1 = mse_criterion(preds1, outputs1)
                mae_loss1 = mae_criterion(preds1, outputs1)

                mse_loss2 = mse_criterion(preds2, outputs2)
                mae_loss2 = mae_criterion(preds2, outputs2)

                val_running_mse_loss1 += mse_loss1.item() * inputs.size(0)
                val_running_mae_loss1 += mae_loss1.item() * inputs.size(0)
                val_running_mse_loss2 += mse_loss2.item() * inputs.size(0)
                val_running_mae_loss2 += mae_loss2.item() * inputs.size(0)

        val_epoch_mse_loss1 = val_running_mse_loss1 / len(val_dataset)
        val_epoch_mae_loss1 = val_running_mae_loss1 / len(val_dataset)
        val_epoch_mse_loss2 = val_running_mse_loss2 / len(val_dataset)
        val_epoch_mae_loss2 = val_running_mae_loss2 / len(val_dataset)

        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Val MSE Loss: {val_epoch_mse_loss1:.4f}, {val_epoch_mse_loss2:.4f}')
        print(
            f'Fold {current_fold}, Epoch {epoch + 1}/{num_epochs} - Val MAE Loss: {val_epoch_mae_loss1:.4f}, {val_epoch_mae_loss2:.4f}')

    return model


def visualize_results(test_data, model):
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets1, targets2) in enumerate(test_loader):
            inputs, targets1, targets2 = inputs.cuda(), targets1.cuda(), targets2.cuda()

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

            axes[1, 0].imshow(input_np, cmap='viridis')
            axes[1, 0].set_title('Input')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(target2_np, cmap='viridis')
            axes[1, 1].set_title('Target 2')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(pred2_np, cmap='viridis')
            axes[1, 2].set_title('Predicted 2')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    with open(r'C:\Users\xiao\PycharmProjects\TsyFSpectrumClassify\dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    best_models = []
    for current_fold in range(len(dataset_info)):
        fold_data = dataset_info[current_fold]
        trained_model = train_model(fold_data)
        best_models.append(trained_model)

    # Assuming the last fold's test set is used for final evaluation
    test_data = dataset_info[-1]['test']
    visualize_results(test_data, best_models[-1])



