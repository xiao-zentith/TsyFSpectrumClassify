from torch import nn
import torch
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet18, self).__init__()
        # 加载未预训练的ResNet18模型，并移除最后的全连接层
        resnet18_1 = models.resnet18(pretrained=False)
        resnet18_2 = models.resnet18(pretrained=False)

        if in_channels != 3:
            # 如果输入通道数不是3，则需要修改第一层卷积层以适应不同的输入通道数
            resnet18_1.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet18_2.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 替换最后一个全连接层以适应输出通道数
        num_ftrs = resnet18_1.fc.in_features
        resnet18_1.fc = nn.Linear(num_ftrs, out_channels * 63 * 63)
        resnet18_2.fc = nn.Linear(num_ftrs, out_channels * 63 * 63)

        self.branch1 = resnet18_1
        self.branch2 = resnet18_2

        self.out_channels = out_channels

    def forward(self, x):
        # 并行处理输入
        output1 = self.branch1(x)
        output2 = self.branch2(x)

        # 还原为原始形状
        batch_size = x.size(0)
        output1 = output1.view(batch_size, self.out_channels, 63, 63)
        output2 = output2.view(batch_size, self.out_channels, 63, 63)

        return output1, output2


# 使用示例
if __name__ == "__main__":
    model = ResNet18(in_channels=1, out_channels=1)

    # 输入示例：51x51 的单通道图像
    input_tensor = torch.randn(5, 1, 63, 63)

    # 前向传播，得到两个输出
    out1, out2 = model(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("分支1输出形状:", out1.shape)  # 应为 (5, 1, 63, 63)
    print("分支2输出形状:", out2.shape)  # 应为 (5, 1, 63, 63)



