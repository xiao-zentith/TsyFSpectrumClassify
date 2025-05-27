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
        self.fc1 = nn.Linear(64 * 88 * 88, 128)  # 假设输入尺寸为 63x63，经过两次池化后变为 13x13
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 3600)  # 特征数固定为 63 * 63

    def forward(self, x):
        # 记录输入的原始形状 (batch_size, channels, height, width)
        original_shape = x.shape  # 保留原始输入形状
        batch_size = original_shape[0]

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

        # 获取原始输入的高度和宽度
        original_height = original_shape[2]
        original_width = original_shape[3]

        # 还原为原始输入图像大小
        x = x.view(batch_size, self.out_channels, original_height, original_width)

        return x


class DualSimpleCNN(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels, branch_number):
        super(DualSimpleCNN, self).__init__()
        """
             :param branch_number: 分支数量，如 2、3 等
             :param is_norm: 是否启用 BatchNorm
             :param in_channels: 输入通道数
             :param out_channels: 输出通道数
             """
        self.branches = nn.ModuleList()

        for _ in range(branch_number):
            self.branches.append(AdaptiveCNN(is_norm, in_channels, out_channels))


    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return outputs  # 返回 list 形式：[output1, output2, ...]


# 使用示例
if __name__ == "__main__":
    model = DualSimpleCNN(is_norm = False, in_channels=1, out_channels=1, branch_number= 4)

    # 输入示例：51x51 的单通道图像
    input_tensor = torch.randn(5, 1, 360, 360)

    # 前向传播，得到n个输出
    outputs = model(input_tensor)

    print("输出数量:", len(outputs))  # 应为 3
    print("每个输出形状:", [out.shape for out in outputs])



