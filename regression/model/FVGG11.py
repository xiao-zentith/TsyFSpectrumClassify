from torch import nn
import torch
import torch.nn.functional as F


class FVGG11(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels):
        super(FVGG11, self).__init__()  # 基类初始化first
        self.name = 'FVGG11'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_norm = is_norm
        # ---------------------------
        # first layer
        # ---------------------------
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.bn1 = nn.BatchNorm2d(64, affine=True)

        self.relu1 = nn.ReLU(inplace=True)

        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出为

        # ---------------------------
        # second layer
        # ---------------------------
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.bn2 = nn.BatchNorm2d(128, affine=True)

        self.relu2 = nn.ReLU(inplace=True)

        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------
        # third layer
        # ---------------------------
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)

        self.bn3 = nn.BatchNorm2d(256, affine=True)

        self.relu3 = nn.ReLU(inplace=True)

        self.maxpooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------
        # fourth layer
        # ---------------------------
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.bn4 = nn.BatchNorm2d(512, affine=True)

        self.relu4 = nn.ReLU(inplace=True)

        self.maxpooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------
        # fifth layer
        # ---------------------------
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.bn5 = nn.BatchNorm2d(512, affine=True)

        self.relu5 = nn.ReLU(inplace=True)

        self.maxpooling5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------
        # full connection layer
        # ---------------------------
        self.linear1 = nn.Linear(512, 2048)  # 输入通道, 输出通道
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2048, 1024)  # 输入通道, 输出通道
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(1024, 3969)  # 输入通道, 输出通道

    # 会被自动调用, module(xxx)  -->  module.forward(xxx)
    def forward(self, x):
        out_shape = x.shape[2:]
        batch_size = x.size(0)
        # layer 1
        out = self.conv1(x)  # 输出: batch_size * 64 * (60 - 2 + 2*1)/1 * 60
        if self.is_norm:
            out = self.bn1(out)  # 输出: batch_size * 64 * (60 - 2 + 2*1)/1 * 60
        out = self.relu1(out)  # 输出: batch_size * 64 * (60 - 2 + 2*1)/1 * 60
        out = self.maxpooling1(out)  # 输出: batch_size * 64 * (60-2+1)/2 * 29

        # layer 2
        out = self.conv2(out)  # 输出: batch_size * 128 * 29 * 29
        if self.is_norm:
            out = self.bn2(out)  # 输出: batch_size * 128 * 29 * 29
        out = self.relu2(out)  # 输出: batch_size * 128 * 29 * 29
        out = self.maxpooling2(out)  # 输出: batch_size * 128 * 14 * 14

        # layer 3
        out = self.conv3(out)  # 输出: batch_size * 256 * 14 * 14
        if self.is_norm:
            out = self.bn3(out)  # 输出: batch_size * 256 * 14 * 14
        out = self.relu3(out)  # 输出: batch_size * 256 * 14 * 14
        out = self.maxpooling3(out)  # 输出: batch_size * 256 * 7 * 7

        # layer 4
        out = self.conv4(out)  # 输出: batch_size * 512 * 7 * 7
        if self.is_norm:
            out = self.bn4(out)  # 输出: batch_size * 512 * 7 * 7
        out = self.relu4(out)  # 输出: batch_size * 512 * 7 * 7
        out = self.maxpooling4(out)  # 输出: batch_size * 512 * 3 * 3

        # layer 5
        out = self.conv5(out)  # 输出: batch_size * 512 * 3 * 3
        if self.is_norm:
            out = self.bn5(out)  # 输出: batch_size * 512 * 3 * 3
        out = self.relu5(out)  # 输出: batch_size * 512 * 3 * 3
        out = self.maxpooling5(out)  # 输出: batch_size * 512 * 1 * 1

        out = out.view(-1, 512 * 1 * 1)

        out = self.linear1(out)

        out = F.relu(out)

        out = self.dropout1(out)  # dropout 增强数据, 减少过拟合的可能性

        out = self.linear2(out)

        out = F.relu(out)

        out = self.dropout2(out)

        out = self.linear3(out)

        # 还原为原始形状
        out = out.view(batch_size, self.out_channels, *out_shape)

        return out

class DualFVGG11(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels):
        super(DualFVGG11, self).__init__()
        # 创建两个独立的分支
        self.branch1 = FVGG11(is_norm, in_channels, out_channels)
        self.branch2 = FVGG11(is_norm, in_channels, out_channels)

    def forward(self, x):
        # 并行处理输入
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        return output1, output2


# 使用示例
if __name__ == "__main__":
    model = DualFVGG11(is_norm = False, in_channels=1, out_channels=1)

    # 输入示例：51x51 的单通道图像
    input_tensor = torch.randn(5, 1, 63, 63)

    # 前向传播，得到两个输出
    out1, out2 = model(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("分支1输出形状:", out1.shape)  # 应为 (1, 1, 63, 63)
    print("分支2输出形状:", out2.shape)  # 应为 (1, 1, 63, 63)