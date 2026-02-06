import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import shutil

from src.utils.data_io.matrix_reader import read_matrix_from_file
from src.classification.utils.image_dataset import ImageDataset
from src.classification.utils.plot_utils import plot_confusion_matrix
from classfication.model.SimpleTransformer import SimpleTransformer


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


def nested_k_fold_cross_validation(json_path):
    label_map = {'C6': 0, 'FITC': 1, 'hpts': 2, 'C6 + FITC': 3, 'C6 + hpts': 4, 'FITC + hpts': 5,}  # Example label map, adjust according to your dataset
    with open(json_path, 'r') as file:
        dataset_info = json.load(file)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i, fold_info in enumerate(dataset_info):
        print(f'Outer Fold {i + 1}/{len(dataset_info)}')

        X_train = [item['input'] for item in fold_info['train']]
        y_train = [label_map[item['category']] for item in fold_info['train']]
        X_val = [item['input'] for item in fold_info['validation']]
        y_val = [label_map[item['category']] for item in fold_info['validation']]
        X_test = [item['input'] for item in fold_info['test']]
        y_test = [label_map[item['category']] for item in fold_info['test']]

        best_val_loss = float('inf')
        best_model_state = None

        train_dataset = ImageDataset(X_train, y_train)
        val_dataset = ImageDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine input dimension based on the first training sample
        input_dim = read_matrix_from_file(X_train[0])[2].flatten().shape[0]
        seq_length = read_matrix_from_file(X_train[0])[2].shape[0]  # Number of rows in the matrix

        # Ensure that the number of heads divides the hidden dimension
        hidden_dim = 64
        num_heads = 4
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"

        model = SimpleTransformer(input_dim=input_dim, num_heads=num_heads, hidden_dim=hidden_dim,
                                  output_dim=len(label_map)).to(device)
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

        final_model = SimpleTransformer(input_dim=input_dim, num_heads=num_heads, hidden_dim=hidden_dim,
                                        output_dim=len(label_map)).to(device)
        final_model.load_state_dict(best_model_state)

        test_dataset = ImageDataset(X_test, y_test)
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


    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


json_path = r'../../dataset_classify/dataset_info.json'
nested_k_fold_cross_validation(json_path)



