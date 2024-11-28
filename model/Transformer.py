import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_recall_curve, \
    roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TransformerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence length dimension (batch_size, seq_len=1, d_model)
        x = self.transformer_encoder(x).squeeze(1)  # Remove sequence length dimension
        x = self.fc(x)
        return x

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

            feature_vector = matrix.flatten()
            feature_vector = np.concatenate((x_coords, y_coords, feature_vector))

            features.append(feature_vector)
            labels.append(label_idx)

    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map

def perform_feature_engineering(X):
    print(f"Original shape of X: {X.shape}")
    pca = PCA(n_components=min(3, X.shape[1]))
    X_pca = pca.fit_transform(X)
    print(f"Shape of X after PCA: {X_pca.shape}")
    return X_pca

def plot_classification_metrics(y_true, y_pred, y_scores, n_classes, label_map):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))

    plt.figure(figsize=(8, 8))

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     plt.plot(fpr[i], tpr[i], label=f'Class {list(label_map.keys())[i]} (area = {roc_auc[i]:.2f})')
    #
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    #
    # precision = dict()
    # recall = dict()
    # for i in range(n_classes):
    #     precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
    #     plt.plot(recall[i], precision[i], label=f'Class {list(label_map.keys())[i]}')
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curve')
    # plt.legend(loc="best")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

def train_model(model, dataloader, criterion, optimizer, device):
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
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            scores = torch.softmax(outputs, dim=1)
            preds = torch.argmax(scores, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    return avg_loss, all_preds, all_labels, all_scores

def main(train_folder, test_folder):
    X_train, y_train, label_map = load_data(train_folder)
    X_test, y_test, _ = load_data(test_folder)
    n_classes = len(label_map)

    X_train = perform_feature_engineering(X_train)
    X_test = perform_feature_engineering(X_test)

    y_train_bin = label_binarize(y_train, classes=list(range(n_classes)))
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    train_dataset = TransformerDataset(X_train, y_train)
    test_dataset = TransformerDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(input_dim=X_train.shape[1], num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, _, _, _ = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    _, y_pred, y_true, y_scores = evaluate_model(model, test_loader, criterion, device)

    plot_classification_metrics(y_true, y_pred, y_scores, n_classes, label_map)

train_folder = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset\dataset_train'
test_folder = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset\dataset_test'

main(train_folder, test_folder)






