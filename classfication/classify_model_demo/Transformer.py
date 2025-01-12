
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import shutil

from classfication.Utils.ImageDataset import ImageDataset
from classfication.Utils.plot import plot_confusion_matrix
from classfication.Utils.read_matrix import read_matrix_from_file, load_data


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=8, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average pooling over the sequence length
        x = self.fc(x)
        return x





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



def nested_k_fold_cross_validation(dataset_folder, k_outer=5, k_inner=5, random_seed=22):
    X, y, label_map = load_data(dataset_folder)
    outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=random_seed)
    inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=random_seed)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i, (train_val_indices, test_indices) in enumerate(outer_cv.split(X, y)):
        print(f'Outer Fold {i + 1}/{k_outer}')
        X_train_val, X_test = X[train_val_indices], X[test_indices]
        y_train_val, y_test = y[train_val_indices], y[test_indices]

        best_val_loss = float('inf')
        best_model_state = None

        temp_folder = f'temp_{i}'

        for j, (train_indices, val_indices) in enumerate(inner_cv.split(X_train_val, y_train_val)):
            print(f'Inner Fold {j + 1}/{k_inner} of Outer Fold {i + 1}/{k_outer}')
            X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
            y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]

            # Perform Z-score normalization
            X_train_normalized, X_val_normalized, _ = X_train, X_val, X_test

            train_dataset = ImageDataset(X_train_normalized, y_train)
            val_dataset = ImageDataset(X_val_normalized, y_val)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimpleTransformer(input_dim=X_train.shape[1], num_classes=len(label_map)).to(device)
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

        final_model = SimpleTransformer(input_dim=X_train.shape[1], num_classes=len(label_map)).to(device)
        final_model.load_state_dict(best_model_state)

        # Perform Z-score normalization for the entire dataset_raw before splitting into test set
        X_train_val_normalized, _, X_test_normalized = X_train_val, X_val, X_test

        test_dataset = ImageDataset(X_test_normalized, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        _, y_pred, y_true = evaluate_model(final_model, test_loader, criterion, device)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"Fold {i + 1}: Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")

        # Plot confusion matrix for each fold
        plot_confusion_matrix(y_true, y_pred, len(label_map), label_map, i + 1)
        shutil.rmtree(temp_folder)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


dataset_folder = r'C:\Users\xiao\Desktop\Draw-flatbread\data\dataset_K\dataset_TsyF'
nested_k_fold_cross_validation(dataset_folder)



