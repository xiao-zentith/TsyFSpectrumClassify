import torch
import torch.nn as nn
from regression.regression_model.UNet import UNET
from regression.regression_model.UNet_no_skip import UNETNoSkip


class DualUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[16, 32, 64]):
        super(DualUNet, self).__init__()
        self.unet1 = UNET(in_channels=in_channels, out_channels=out_channels, features=features)
        self.unet2 = UNET(in_channels=in_channels, out_channels=out_channels, features=features)

    def forward(self, x):
        output1 = self.unet1(x)
        output2 = self.unet2(x)
        return output1, output2

# Example usage
if __name__ == "__main__":
    model = DualUNet(in_channels=3, out_channels=1)
    x = torch.randn((3, 3, 161, 161))  # Batch size, channels, height, width
    outputs = model(x)
    print(outputs[0].shape)  # Output should be (3, 1, 161, 161)
    print(outputs[1].shape)  # Output should be (3, 1, 161, 161)