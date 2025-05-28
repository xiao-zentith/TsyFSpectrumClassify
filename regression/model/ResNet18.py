from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels, branch_number):
        """
        :param branch_number: 分支数量，如 2、3 等
        :param is_norm: 是否启用 BatchNorm
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super(ResNet18, self).__init__()
        self.branches = nn.ModuleList()
        self.out_channels = out_channels

        for _ in range(branch_number):
            branch = models.resnet18(pretrained=False)

            if is_norm:
                branch = remove_batchnorm(branch)

            # 修改第一层卷积以适配输入通道数
            if in_channels != 3:
                branch.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # 替换最后的全连接层以适配输出尺寸
            num_ftrs = branch.fc.in_features
            branch.fc = nn.Linear(num_ftrs, out_channels * 60 * 60)

            self.branches.append(branch)

    def forward(self, x):
        outputs = []
        original_shape = x.shape  # 保留原始输入形状
        # 获取原始输入的高度和宽度
        original_height = original_shape[2]
        original_width = original_shape[3]
        for branch in self.branches:
            out = branch(x)
            batch_size = x.size(0)
            out = out.view(batch_size, self.out_channels, 60, 60)
            # 插值
            out = F.interpolate(out, size=(original_height, original_width), mode='bilinear', align_corners=False)
            outputs.append(out)
        return outputs  # 返回 list of tensors

def remove_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        return nn.Identity()

    # 如果是 BasicBlock 或 Bottleneck，递归处理其子模块
    elif isinstance(m, models.resnet.BasicBlock) or isinstance(m, models.resnet.Bottleneck):
        for name, child in m.named_children():
            setattr(m, name, remove_batchnorm(child))
        return m

    # 如果是 Sequential，递归处理每个子模块
    elif isinstance(m, nn.Sequential):
        return nn.Sequential(*[remove_batchnorm(child) for child in m.children()])

    # 如果是 ResNet 主体，递归处理每个子模块
    elif isinstance(m, models.ResNet):
        for name, child in m.named_children():
            setattr(m, name, remove_batchnorm(child))
        return m

    else:
        return m

# 使用示例
if __name__ == "__main__":
    model = ResNet18(is_norm= False, in_channels=1, out_channels=1, branch_number=6)

    # 输入示例：51x51 的单通道图像
    input_tensor = torch.randn(5, 1, 360, 360)

    outputs = model(input_tensor)

    print("输出数量:", len(outputs))  # 应为 3
    print("每个输出形状:", [out.shape for out in outputs])



