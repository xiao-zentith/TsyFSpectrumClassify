import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128]):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(features[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features[0], features[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(features[1])
        self.final_conv = nn.Conv2d(features[1], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.final_conv(x)
        return x

class DualSimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128]):
        super(DualSimpleCNN, self).__init__()
        self.cnn1 = SimpleCNN(in_channels=in_channels, out_channels=out_channels, features=features)
        self.cnn2 = SimpleCNN(in_channels=in_channels, out_channels=out_channels, features=features)

    def forward(self, x):
        output1 = self.cnn1(x)
        output2 = self.cnn2(x)
        return output1, output2

# Example usage
if __name__ == "__main__":
    model = DualSimpleCNN(in_channels=3, out_channels=1)
    x = torch.randn((3, 3, 161, 161))  # Batch size, channels, height, width
    outputs = model(x)
    print(outputs[0].shape)  # Output should be (3, 1, 161, 161)
    print(outputs[1].shape)  # Output should be (3, 1, 161, 161)



