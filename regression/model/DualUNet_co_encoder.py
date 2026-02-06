import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        if is_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class UNETDecoder(nn.Module):
    def __init__(self, is_norm, features, out_channels):
        super(UNETDecoder, self).__init__()
        self.ups = nn.ModuleList()

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(is_norm, feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Adding a 1x1 convolution layer to learn scaling factor
        self.scale_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        # Applying scaling layer
        scale_factor = self.scale_layer(torch.ones_like(x))
        scaled_out = x * scale_factor

        return scaled_out


class DualUNetSharedEncoder(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels, branch_number, features=[16, 32, 64]):
        """
        :param branch_number: 分支数量
        :param is_norm: 是否使用 BatchNorm
        :param in_channels: 输入通道数
        :param out_channels: 每个分支的输出通道数
        :param features: 编码器中每层的通道数列表
        """
        super(DualUNetSharedEncoder, self).__init__()

        # Shared encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(is_norm, in_channels, feature))
            in_channels = feature

        # Bottleneck layer
        self.bottleneck = DoubleConv(is_norm, features[-1], features[-1] * 2)

        # 动态创建多个解码器
        self.decoders = nn.ModuleList([
            UNETDecoder(is_norm, features, out_channels) for _ in range(branch_number)
        ])

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        outputs = []
        for decoder in self.decoders:
            output = decoder(x, skip_connections)
            outputs.append(output)

        return outputs  # 返回 list of tensors



# Example usage
if __name__ == "__main__":
    model = DualUNetSharedEncoder(is_norm = True, in_channels=1, out_channels=1, branch_number=8)
    x = torch.randn((1, 1, 63, 63))  # Batch size, channels, height, width
    outputs = model(x)

    print("输出数量:", len(outputs))  # 应为 3
    print("每个输出形状:", [out.shape for out in outputs])



