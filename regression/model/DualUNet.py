import torch
import torch.nn as nn
from regression.model.UNet import UNET


class DualUNet(nn.Module):
    def __init__(self, is_norm, in_channels, out_channels, branch_number, features=[16, 32, 64]):
        super(DualUNet, self).__init__()

        self.UNets = nn.ModuleList()
        for _ in range(branch_number):
            self.UNets.append(UNET(is_norm, in_channels=in_channels, out_channels=out_channels, features=features))


    def forward(self, x):
        outputs = [UNET(x) for UNET in self.UNets]
        return outputs

# Example usage
if __name__ == "__main__":
    model = DualUNet(is_norm = False, in_channels=3, out_channels=1, branch_number=3)
    x = torch.randn((3, 3, 161, 161))  # Batch size, channels, height, width
    # 前向传播，得到n个输出
    outputs = model(x)

    print("输出数量:", len(outputs))  # 应为 3
    print("每个输出形状:", [out.shape for out in outputs])