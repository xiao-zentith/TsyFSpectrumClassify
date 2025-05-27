import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ResNet 风格的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 用于调整维度的下采样层（当输入输出维度不同时）

    def forward(self, x):
        identity = x  # 保存原始输入用于跳跃连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 调整维度

        out += identity  # 跳跃连接
        out = self.relu(out)
        return out


class UNET_ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNET_ResNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下采样路径（使用 ResidualBlock）
        in_ch = in_channels
        for f in features:
            self.downs.append(
                ResidualBlock(in_ch, f, stride=1,
                             downsample=self._downsample(in_ch, f))
            )
            in_ch = f

        # 瓶颈层（使用两个 ResidualBlock）
        self.bottleneck = nn.Sequential(
            ResidualBlock(features[-1], features[-1]*2,
                         downsample=self._downsample(features[-1], features[-1]*2)),
            ResidualBlock(features[-1]*2, features[-1]*2)
        )

        # 上采样路径（使用 ResidualBlock）
        for i, f in enumerate(reversed(features)):
            # 转置卷积恢复空间维度
            self.ups.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            # 残差块处理特征
            self.ups.append(
                ResidualBlock(f * 2, f,
                             downsample=self._downsample(f * 2, f))
            )

        # 最终卷积层和缩放层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.scale_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def _downsample(self, in_channels, out_channels):
        """用于调整维度的下采样层（当输入输出通道数不同时）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        skip_connections = []

        # 下采样阶段
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # 上采样阶段
        for i in range(0, len(self.ups), 2):
            # 转置卷积恢复空间维度
            x = self.ups[i](x)
            skip = skip_connections[i//2]

            # 处理维度不匹配（可能需要调整）
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                 mode='bilinear', align_corners=False)

            # 特征拼接
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](concat_skip)  # 残差块处理

        # 最终输出
        x = self.final_conv(x)
        scale_factor = self.scale_layer(torch.ones_like(x))
        scaled_out = x * scale_factor

        return scaled_out

# 测试代码
if __name__ == "__main__":
    model = UNET_ResNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 161, 161)
    output = model(x)
    print("Output shape:", output.shape)  # 应输出 (2, 1, 161, 161)