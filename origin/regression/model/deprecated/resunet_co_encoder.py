import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet18 BasicBlock Definition
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Updated DoubleConv using BasicBlock
class ResNetConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetConv, self).__init__()
        self.block = BasicBlock(in_channels, out_channels, stride)

    def forward(self, x):
        return self.block(x)


# UNETDecoder using ResNetConv instead of DoubleConv
class UNETDecoder(nn.Module):
    def __init__(self, is_norm, features, out_channels):
        super(UNETDecoder, self).__init__()
        self.ups = nn.ModuleList()

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ResNetConv(feature * 2, feature))

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


class ResUNetSharedEncoder(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels, branch_number, features=[16, 32, 64]):
        """
        :param branch_number: Number of branches
        :param is_norm: Whether to use BatchNorm
        :param in_channels: Input channels
        :param out_channels: Output channels for each branch
        :param features: List of channels per layer in the encoder
        """
        super(ResUNetSharedEncoder, self).__init__()

        # Shared encoder using ResNet Blocks
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(ResNetConv(in_channels, feature))
            in_channels = feature

        # Bottleneck layer
        self.bottleneck = ResNetConv(features[-1], features[-1] * 2)

        # Dynamically create multiple decoders
        self.decoders = nn.ModuleList([
            UNETDecoder(is_norm, features, out_channels) for _ in range(branch_number)
        ])

    def forward(self, x):
        skip_connections = []

        # Encoder using ResNet blocks
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

        return outputs  # Return list of tensors


# Example usage
if __name__ == "__main__":
    model = ResUNetSharedEncoder(is_norm=True, in_channels=1, out_channels=1, branch_number=8)
    x = torch.randn((1, 1, 64, 64))  # Batch size, channels, height, width
    outputs = model(x)

    print("Number of outputs:", len(outputs))  # Should be 8
    print("Shape of each output:", [out.shape for out in outputs])