import json
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

from src.utils.data_io.data_loader import load_data
from src.utils.data_io.matrix_reader import read_matrix_from_file
from src.classification.utils.image_dataset import ImageDataset
from src.classification.utils.plot_utils import plot_confusion_matrix
from classfication.model.SimpleCNN import SimpleCNN
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

def train_and_evaluate_model(model, final_model, train_loader, val_loader, test_loader, device):
    print(model._get_name())
    best_val_loss = float('inf')
    best_model_state = None

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

    final_model.load_state_dict(best_model_state)

    _, y_pred, y_true = evaluate_model(final_model, test_loader, criterion, device)
    return y_pred, y_true


def train_KNN(fold_info, label_map):
    train_files = fold_info['train'] + fold_info['validation']  # 合并train和validation
    test_files = fold_info['test']

    X_train, y_train = load_data(train_files, label_map)
    X_test, y_test= load_data(test_files, label_map)
    n_classes = len(label_map)

    # 将标签二值化
    y_train_bin = label_binarize(y_train, classes=list(range(n_classes)))
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    X_train_scaled = X_train
    X_test_scaled = X_test

    # 使用GridSearchCV自动寻找最佳K值
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 21)}
    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train_bin)

    best_k = grid_search.best_params_['n_neighbors']
    print(f"Best k for Fold {fold_info['fold']}: {best_k}")

    # 使用最佳K值进行预测
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    classifier = OneVsRestClassifier(best_knn)
    y_scores = classifier.fit(X_train_scaled, y_train_bin).predict_proba(X_test_scaled)
    y_pred = classifier.predict(X_test_scaled)

    # 转换预测结果为单个类别标签
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test_bin, axis=1)

    return y_pred_labels, y_true_labels

def train_RF(fold_info, label_map):
    train_files = fold_info['train'] + fold_info['validation']  # 合并train和validation
    test_files = fold_info['test']

    X_train, y_train = load_data(train_files, label_map)
    X_test, y_test= load_data(test_files, label_map)
    n_classes = len(label_map)

    # 将标签二值化
    y_train_bin = label_binarize(y_train, classes=list(range(n_classes)))
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    X_train_scaled = X_train
    X_test_scaled = X_test

    # 使用GridSearchCV自动寻找最佳参数
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train_bin)

    best_params = grid_search.best_params_
    print(f"Best parameters for Fold {fold_info['fold']}: {best_params}")

    # 使用最佳参数进行预测
    best_rf = RandomForestClassifier(**best_params)
    classifier = OneVsRestClassifier(best_rf)
    y_scores = classifier.fit(X_train_scaled, y_train_bin).predict_proba(X_test_scaled)
    y_pred = classifier.predict(X_test_scaled)

    # 转换预测结果为单个类别标签
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test_bin, axis=1)

    return y_pred_labels, y_true_labels


def vector_voting(vectors):
    # 获取向量的数量和长度
    num_vectors = len(vectors)
    vector_length = len(vectors[0])

    # 初始化结果向量
    result_vector = [0] * vector_length

    # 对每个索引位置进行投票
    for i in range(vector_length):
        # 统计当前索引位置的所有值的出现频率
        count = Counter([v[i] for v in vectors])

        # 选择出现次数最多的值作为结果
        most_common_value, _ = count.most_common(1)[0]
        result_vector[i] = most_common_value

    return result_vector


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

        y_pred_KNN,_ = train_KNN(fold_info, label_map)
        y_pred_RF,_ = train_RF(fold_info, label_map)


        X_train = [item['input'] for item in fold_info['train']]
        y_train = [label_map[item['category']] for item in fold_info['train']]
        X_val = [item['input'] for item in fold_info['validation']]
        y_val = [label_map[item['category']] for item in fold_info['validation']]
        X_test = [item['input'] for item in fold_info['test']]
        y_test = [label_map[item['category']] for item in fold_info['test']]

        train_dataset = ImageDataset(X_train, y_train)
        val_dataset = ImageDataset(X_val, y_val)
        test_dataset = ImageDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CNN_model = SimpleCNN(image_size=read_matrix_from_file(X_train[0])[2].shape[0], num_channels=1, num_classes=len(label_map)).to(device)
        final_CNN_model = SimpleCNN(image_size=read_matrix_from_file(X_train[0])[2].shape[0], num_channels=1,
                                num_classes=len(label_map)).to(device)

        # Determine input dimension based on the first training sample
        input_dim = read_matrix_from_file(X_train[0])[2].flatten().shape[0]

        # Ensure that the number of heads divides the hidden dimension
        hidden_dim = 64
        num_heads = 4
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"
        Transformer_model = SimpleTransformer(input_dim=input_dim, num_heads=num_heads, hidden_dim=hidden_dim,
                                  output_dim=len(label_map)).to(device)
        final_Transformer_model = SimpleTransformer(input_dim=input_dim, num_heads=num_heads, hidden_dim=hidden_dim,
                                  output_dim=len(label_map)).to(device)

        y_pred_CNN, y_true_CNN = train_and_evaluate_model(CNN_model, final_CNN_model, train_loader, val_loader, test_loader, device)
        y_pred_Transformer, y_true_Transformer = train_and_evaluate_model(Transformer_model, final_Transformer_model, train_loader, val_loader, test_loader, device)

        y_true = y_true_CNN
        # 投票集成
        y_pred = vector_voting([y_pred_KNN, y_pred_RF,y_pred_Transformer])

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



