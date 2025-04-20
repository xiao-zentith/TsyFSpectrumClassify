import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 卷积层和池化层（固定结构）
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # 动态全连接层（初始化时设为None，首次前向计算时创建）
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self._initialized = False  # 标记是否已初始化全连接层

    def forward(self, x):
        original_shape = x.shape[2:]  # 保存原始输入尺寸 (H, W)
        batch_size = x.size(0)

        # 编码部分（卷积+池化）
        x = self.pool1(F.relu(self.conv1(x)))  # 第一层卷积+池化
        x = self.pool2(F.relu(self.conv2(x)))  # 第二层卷积+池化

        # 动态计算展平维度
        flattened_dim = x.view(batch_size, -1).size(1)

        # 首次前向时初始化全连接层
        if not self._initialized:
            self.fc1 = nn.Linear(flattened_dim, 128)
            self.fc2 = nn.Linear(128, 256)
            # 输出层维度为 out_channels × H × W
            output_dim = self.out_channels * original_shape[0] * original_shape[1]
            self.fc3 = nn.Linear(256, output_dim)
            self._initialized = True

        # 全连接层处理
        x = x.view(batch_size, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # 还原为原始形状
        x = x.view(batch_size, self.out_channels, *original_shape)

        return x


class DualSimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualSimpleCNN, self).__init__()
        # 创建两个独立的分支
        self.branch1 = AdaptiveCNN(in_channels, out_channels)
        self.branch2 = AdaptiveCNN(in_channels, out_channels)

    def forward(self, x):
        # 并行处理输入
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        return output1, output2


# 使用示例
if __name__ == "__main__":
    # 创建并行模型（输入通道1，分支1输出1通道，分支2输出3通道）
    model = DualSimpleCNN(in_channels=1, out_channels=1)

    # 输入示例：51x51 的单通道图像
    input_tensor = torch.randn(1, 1, 51, 51)

    # 前向传播，得到两个输出
    out1, out2 = model(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("分支1输出形状:", out1.shape)  # 应为 (2, 1, 51, 51)
    print("分支2输出形状:", out2.shape)  # 应为 (2, 3, 51, 51)