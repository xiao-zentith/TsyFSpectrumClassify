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


class UNET(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels, features):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.is_norm = is_norm

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(is_norm, in_channels, feature))
            in_channels = feature

        # Bottleneck layer
        self.bottleneck = DoubleConv(is_norm, features[-1], features[-1] * 2)

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(is_norm, feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # 添加一个1x1卷积层来学习缩放比例
        self.scale_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        # 应用缩放层
        scale_factor = self.scale_layer(torch.ones_like(x))
        scaled_out = x * scale_factor

        return scaled_out

# Example usage
if __name__ == "__main__":
    model = UNET(in_channels=1, out_channels=1)
    x = torch.randn((3, 3, 161, 161))  # Batch size, channels, height, width
    print(model(x).shape)  # Output should be (3, 1, 161, 161)



