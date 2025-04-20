import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNETDecoder(nn.Module):
    def __init__(self, features, out_channels):
        super(UNETDecoder, self).__init__()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

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
    def __init__(self, in_channels, out_channels, features=[16, 32, 64]):
        super(DualUNetSharedEncoder, self).__init__()

        # Shared encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Two separate decoders
        self.decoder1 = UNETDecoder(features, out_channels)
        self.decoder2 = UNETDecoder(features, out_channels)

    def forward(self, x):
        skip_connections1 = []
        skip_connections2 = []

        for down in self.downs:
            x = down(x)
            skip_connections1.append(x)
            skip_connections2.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        output1 = self.decoder1(x, skip_connections1)
        output2 = self.decoder2(x, skip_connections2)

        return output1, output2


# Example usage
if __name__ == "__main__":
    model = DualUNetSharedEncoder(in_channels=1, out_channels=1)
    x = torch.randn((1, 1, 64, 21))  # Batch size, channels, height, width
    outputs = model(x)
    print(outputs[0].shape)  # Output should be (1, 1, 64, 21)
    print(outputs[1].shape)  # Output should be (1, 1, 64, 21)



