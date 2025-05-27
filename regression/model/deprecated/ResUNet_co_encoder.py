import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ResNet 风格的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 用于调整维度的下采样层（当输入输出维度不同时）

    def forward(self, x):
        identity = x  # 保存原始输入用于跳跃连接

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 调整维度

        out += identity  # 跳跃连接
        out = self.relu(out)
        return out

class UNETDecoder(nn.Module):
    def __init__(self, features, out_channels):
        super(UNETDecoder, self).__init__()
        self.ups = nn.ModuleList()
        self.features = features

        # 上采样路径（使用 ResidualBlock）
        for i, f in enumerate(reversed(features)):
            # 转置卷积恢复空间维度
            self.ups.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            # 创建 ResidualBlock（输入通道是 f*2，输出是 f）
            downsample = self._downsample(f * 2, f)
            self.ups.append(
                ResidualBlock(f * 2, f, downsample=downsample)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.scale_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def _downsample(self, in_channels, out_channels):
        """用于调整维度的下采样层（当输入输出通道数不同时）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # 转置卷积恢复空间维度
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # 处理维度不匹配
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                 mode='bilinear', align_corners=False)

            # 特征拼接
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # ResidualBlock 处理

        # 最终输出
        x = self.final_conv(x)
        scale_factor = self.scale_layer(torch.ones_like(x))
        scaled_out = x * scale_factor
        return scaled_out

class DualResNetSharedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features=[16, 32, 64]):
        super(DualResNetSharedEncoder, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下采样路径（使用 ResidualBlock）
        in_ch = in_channels
        for f in features:
            # 创建 ResidualBlock 的下采样层（调整通道数）
            downsample = self._downsample(in_ch, f)
            self.downs.append(
                ResidualBlock(in_ch, f, stride=1, downsample=downsample)
            )
            in_ch = f

        # 瓶颈层（使用 ResidualBlock）
        self.bottleneck = ResidualBlock(
            features[-1],
            features[-1] * 2,
            downsample=self._downsample(features[-1], features[-1] * 2)
        )

        # 两个解码器
        self.decoder1 = UNETDecoder(features, out_channels)
        self.decoder2 = UNETDecoder(features, out_channels)

    def _downsample(self, in_channels, out_channels):
        """用于调整维度的下采样层（当输入输出通道数不同时）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        skip_connections1 = []
        skip_connections2 = []

        for down in self.downs:
            x = down(x)
            skip_connections1.append(x)
            skip_connections2.append(x)
            x = self.pool(x)

        # 瓶颈层处理
        x = self.bottleneck(x)

        # 两个解码器独立处理
        output1 = self.decoder1(x, skip_connections1)
        output2 = self.decoder2(x, skip_connections2)

        return output1, output2

# 测试代码
if __name__ == "__main__":
    model = DualResNetSharedEncoder(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 64, 21)
    outputs = model(x)
    print("Output 1 shape:", outputs[0].shape)  # 应输出 (1, 1, 64, 21)
    print("Output 2 shape:", outputs[1].shape)  # 应输出 (1, 1, 64, 21)