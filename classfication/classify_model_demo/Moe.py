import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import shutil
from preprocess.augment_data import AugmentData


class ImageDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class Expert(nn.Module):
    def __init__(self, image_size, num_channels, num_classes):
        super(Expert, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * (image_size // 4) * (image_size // 4), 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)  # Softmax to get gating weights
        return x


class MixtureOfExperts(nn.Module):
    def __init__(self, image_size, num_channels, num_classes, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Expert(image_size, num_channels, num_classes) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(image_size * image_size, num_experts)

    def forward(self, x):
        gates = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.experts]
        weighted_outputs = [gates[:, i].unsqueeze(1) * expert_outputs[i] for i in range(len(self.experts))]
        output = sum(weighted_outputs)
        return output


def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    x_coords = list(map(float, lines[0].strip().split()))
    y_coords = []
    data_matrix = []
    for line in lines[1:]:
        parts = list(map(float, line.strip().split()))
        y_coords.append(parts[0])
        data_matrix.append(parts[1:])

    return np.array(x_coords), np.array(y_coords), np.array(data_matrix)


def load_data(input_folder):
    features = []
    labels = []
    label_map = {}

    for label_idx, label in enumerate(os.listdir(input_folder)):
        label_map[label] = label_idx
        label_folder = os.path.join(input_folder, label)

        if not os.path.isdir(label_folder):
            continue

        for txt_file in os.listdir(label_folder):
            if not txt_file.endswith('.txt'):
                continue

            file_path = os.path.join(label_folder, txt_file)
            x_coords, y_coords, matrix = read_matrix_from_file(file_path)

            # Normalize the matrix values between 0 and 1
            matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

            features.append(matrix)
            labels.append(label_idx)

    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map


def plot_confusion_matrix(y_true, y_pred, n_classes, label_map, fold_number):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Fold {fold_number}')
    plt.show()


def train_model(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    return running_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return avg_loss, all_preds, all_labels


def zscore_normalize(X_train, X_val, X_test):
    scaler = StandardScaler()
    # Flatten each sample for normalization
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Fit on training data only
    scaler.fit(X_train_flat)
    X_train_normalized_flat = scaler.transform(X_train_flat)
    X_val_normalized_flat = scaler.transform(X_val_flat)
    X_test_normalized_flat = scaler.transform(X_test_flat)

    # Reshape back to original shape
    X_train_normalized = X_train_normalized_flat.reshape(X_train.shape)
    X_val_normalized = X_val_normalized_flat.reshape(X_val.shape)
    X_test_normalized = X_test_normalized_flat.reshape(X_test.shape)

    return X_train_normalized, X_val_normalized, X_test_normalized


def nested_k_fold_cross_validation(dataset_folder, k_outer=5, k_inner=5, random_seed=22):
    X, y, label_map = load_data(dataset_folder)
    outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=random_seed)
    inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=random_seed)

    accuracies = []
    recalls = []
    f1_scores = []

    augmenter = AugmentData()

    for i, (train_val_indices, test_indices) in enumerate(outer_cv.split(X, y)):
        print(f'Outer Fold {i + 1}/{k_outer}')
        X_train_val, X_test = X[train_val_indices], X[test_indices]
        y_train_val, y_test = y[train_val_indices], y[test_indices]

        best_val_loss = float('inf')
        best_model_state = None

        temp_folder = f'temp_{i}'
        augmenter.process_data(dataset_folder, temp_folder, random_seed)

        for j, (train_indices, val_indices) in enumerate(inner_cv.split(X_train_val, y_train_val)):
            print(f'Inner Fold {j + 1}/{k_inner} of Outer Fold {i + 1}/{k_outer}')
            X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
            y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]

            # Apply augmentation here if needed before normalizing
            # For now, we assume no additional augmentation steps are required after loading
            X_train_normalized, X_val_normalized, _ = zscore_normalize(X_train, X_val, X_test)

            train_dataset = ImageDataset(X_train_normalized, y_train)
            val_dataset = ImageDataset(X_val_normalized, y_val)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Assuming the input size is 22x22, so the flattened size should be 484
            model = MixtureOfExperts(image_size=X_train.shape[1], num_channels=1, num_classes=len(label_map),
                                     num_experts=3).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
            num_epochs = 100

            for epoch in range(num_epochs):
                train_loss = train_model(model, train_loader, criterion, optimizer, scheduler, device)
                val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        final_model = MixtureOfExperts(image_size=X_train.shape[1], num_channels=1, num_classes=len(label_map),
                                       num_experts=3).to(device)
        final_model.load_state_dict(best_model_state)

        # Perform Z-score normalization for the entire dataset_raw before splitting into test set
        X_train_val_normalized, _, X_test_normalized = zscore_normalize(X_train_val, X_val, X_test)
        test_dataset = ImageDataset(X_test_normalized, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        _, y_pred, y_true = evaluate_model(final_model, test_loader, criterion, device)
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        accuracies.append(accuracy)
        recalls.append(recall)
        f1_scores.append(f1)
        print(f"Fold {i + 1}: Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
        plot_confusion_matrix(y_true, y_pred, len(label_map), label_map, i + 1)
        shutil.rmtree(temp_folder)

    avg_accuracy = np.mean(accuracies)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


dataset_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\dataset_TsyF'
nested_k_fold_cross_validation(dataset_folder)



