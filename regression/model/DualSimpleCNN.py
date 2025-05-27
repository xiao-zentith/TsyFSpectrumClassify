import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveCNN(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels):
        super(AdaptiveCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_norm = is_norm

        # 卷积层和池化层（固定结构）
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)

        # 固定全连接层
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # 假设输入尺寸为 63x63，经过两次池化后变为 13x13
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 3969)  # 特征数固定为 63 * 63

    def forward(self, x):
        original_shape = (63, 63)  # 固定输出形状为 63x21
        batch_size = x.size(0)

        if self.is_norm:
            # 编码部分（卷积+池化）
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # 第一层卷积+池化
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # 第二层卷积+池化
        else:
            # 编码部分（卷积+池化）
            x = self.pool1(F.relu(self.conv1(x)))  # 第一层卷积+池化
            x = self.pool2(F.relu(self.conv2(x)))  # 第二层卷积+池化

        # 展平维度
        x = x.view(batch_size, -1)

        # 全连接层处理
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # 还原为原始形状
        x = x.view(batch_size, self.out_channels, *original_shape)

        return x


class DualSimpleCNN(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels):
        super(DualSimpleCNN, self).__init__()
        # 创建两个独立的分支
        self.branch1 = AdaptiveCNN(is_norm, in_channels, out_channels)
        self.branch2 = AdaptiveCNN(is_norm, in_channels, out_channels)

    def forward(self, x):
        # 并行处理输入
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        return output1, output2


# 使用示例
if __name__ == "__main__":
    model = DualSimpleCNN(is_norm = False, in_channels=1, out_channels=1)

    # 输入示例：51x51 的单通道图像
    input_tensor = torch.randn(5, 1, 63, 63)

    # 前向传播，得到两个输出
    out1, out2 = model(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("分支1输出形状:", out1.shape)  # 应为 (1, 1, 63, 63)
    print("分支2输出形状:", out2.shape)  # 应为 (1, 1, 63, 63)



