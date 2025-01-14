import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自定义数据集类
class SpectralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # 遍历文件夹，加载数据
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                # 读取TXT文件
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    x_coords = lines[0].strip().split()
                    y_coords = lines[1].strip().split()
                    matrix = np.array([line.strip().split() for line in lines[2:]], dtype=float)
                self.data.append(matrix)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# 数据转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    # 可以添加更多的转换
])

# 加载数据集
dataset = SpectralDataset(root_dir=r'C:\Users\xiao\Desktop\论文汇总\data\dataset_after_norm', transform=transform)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加序列维度
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc_out(x)
        return x

# 实例化模型
input_dim = dataset.data[0].shape[0] * dataset.data[0].shape[1]  # 根据数据调整
num_classes = len(os.listdir(r'C:\Users\xiao\Desktop\论文汇总\data\dataset_after_norm'))
model = TransformerModel(input_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, torch.tensor([int(label) for label in labels]))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train_model(model, train_loader, criterion, optimizer)

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels)
            y_pred.extend(predicted)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    conf_mat = confusion_matrix(y_true, y_pred)
    print(f'Accuracy: {accuracy}, Recall: {recall}, F1: {f1}, Confusion Matrix:\n{conf_mat}')

y_true = []
y_pred = []
evaluate_model(model, test_loader)

# 绘制P-R曲线和ROC曲线
y_true = np.array(y_true)
y_pred = np.array(y_pred)
precision, recall, _ = precision_recall_curve(y_true, y_pred)
roc_auc = auc(recall, precision)
plt.plot(recall, precision, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P-R Curve')
plt.legend()
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()