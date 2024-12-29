import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_recall_curve, \
    roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def read_matrix_from_file(file_path):
    """
    从TXT文件中读取数据并构建矩阵
    :param file_path: 文件路径
    :return: 横坐标数组, 纵坐标数组, 数据矩阵
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取横坐标
    x_coords = list(map(float, lines[0].strip().split()))

    # 提取纵坐标和数据矩阵
    y_coords = []
    data_matrix = []
    for line in lines[1:]:
        parts = list(map(float, line.strip().split()))
        y_coords.append(parts[0])
        data_matrix.append(parts[1:])

    return np.array(x_coords), np.array(y_coords), np.array(data_matrix)


def load_data(input_folder):
    """
    加载所有数据并返回特征和标签
    :param input_folder: 输入文件夹路径
    :return: 特征数组, 标签数组, 标签映射字典
    """
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

            # 展平矩阵为特征向量
            feature_vector = matrix.flatten()

            # 将横坐标和纵坐标展平并添加到特征向量中
            feature_vector = np.concatenate((x_coords, y_coords, feature_vector))

            features.append(feature_vector)
            labels.append(label_idx)

    # 将列表转换为numpy数组
    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map


def plot_classification_metrics(y_true, y_pred, y_scores, n_classes, label_map):
    """
    绘制分类指标图，包括P-R曲线、ROC曲线及其AUC值和混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param y_scores: 分类得分
    :param n_classes: 类别数量
    :param label_map: 标签映射字典
    """
    # 计算准确率、召回率、F1分数
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))

    # P-R曲线
    plt.figure(figsize=(24, 8))

    # ROC曲线
    plt.subplot(1, 4, 1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {list(label_map.keys())[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # P-R曲线
    plt.subplot(1, 4, 2)
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
        plt.plot(recall[i], precision[i], label=f'Class {list(label_map.keys())[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="best")

    # 混淆矩阵
    plt.subplot(1, 4, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()


class Simple1DCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.2)

        # Calculate the output size of the convolutional layers
        conv_output_size = (input_shape - 2) // 2 + 1
        self.fc1 = nn.Linear(64 * conv_output_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main(train_folder, test_folder):
    """
    主函数，加载数据、训练模型并评估性能
    :param train_folder: 训练集文件夹路径
    :param test_folder: 测试集文件夹路径
    """
    # 加载训练集和测试集
    X_train, y_train, label_map = load_data(train_folder)
    X_test, y_test, _ = load_data(test_folder)
    n_classes = len(label_map)

    # 调整输入数据的形状以适应1D-CNN
    max_length = max(X_train.shape[1], X_test.shape[1])
    X_train_padded = np.pad(X_train, ((0, 0), (0, max_length - X_train.shape[1])), mode='constant')
    X_test_padded = np.pad(X_test, ((0, 0), (0, max_length - X_test.shape[1])), mode='constant')
    X_train_padded = X_train_padded.reshape((X_train_padded.shape[0], 1, X_train_padded.shape[1]))
    X_test_padded = X_test_padded.reshape((X_test_padded.shape[0], 1, X_test_padded.shape[1]))

    # 创建DataLoader
    train_dataset = CustomDataset(X_train_padded, y_train)
    test_dataset = CustomDataset(X_test_padded, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 构建1D-CNN模型
    model = Simple1DCNN(input_shape=max_length, num_classes=n_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # 评估模型
    model.eval()
    y_pred_bin = []
    y_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred_bin.extend(outputs.cpu().numpy())
            y_true_labels.extend(labels.cpu().numpy())

    y_pred_labels = np.argmax(np.array(y_pred_bin), axis=1)

    # 绘制分类指标图
    plot_classification_metrics(y_true_labels, y_pred_labels, np.array(y_pred_bin), n_classes, label_map)


# 设置输入文件夹路径
train_folder = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset_train_norm'
test_folder = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset_test_norm'

# 调用主函数
main(train_folder, test_folder)






