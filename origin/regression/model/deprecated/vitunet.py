import torch
import torch.nn as nn
import torch.nn.functional as F


class NoNormTransformerLayer(nn.Module):
    """自定义无归一化的Transformer层"""

    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # 自注意力部分（无归一化）
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)

        # 前馈网络部分（无归一化）
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class ViTBlock(nn.Module):
    """适配小尺寸的ViT模块"""

    def __init__(self, in_channels, out_channels, patch_size=4, heads=2):
        super().__init__()
        self.patch_size = patch_size
        self.dim = out_channels

        # Patch嵌入层
        self.patch_embed = nn.Conv2d(
            in_channels, self.dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 动态位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 10000, self.dim))

        # 无归一化的Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=NoNormTransformerLayer(
                d_model=self.dim,
                nhead=heads,
                dim_feedforward=64
            ),
            num_layers=1
        )

        # 上采样层
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                self.dim, out_channels,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.ReLU()
        )

        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        B, C, H, W = x.shape

        # Patch嵌入
        patches = self.patch_embed(x)  # [B, D, H_p, W_p]
        B, D, H_p, W_p = patches.shape
        seq_len = H_p * W_p

        # 调整形状并添加位置编码
        patches = patches.permute(0, 2, 3, 1).reshape(B, seq_len, D)
        patches += self.pos_embed[:, :seq_len]

        # Transformer处理 (需要调整维度顺序)
        x = patches.permute(1, 0, 2)  # [seq_len, B, D]
        x = self.transformer(x)
        x = x.permute(1, 2, 0).reshape(B, D, H_p, W_p)

        # 上采样
        x = self.up(x)

        # 尺寸对齐
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x + residual


class UNETDecoder(nn.Module):
    """基于ViT的解码器"""

    def __init__(self, features, out_channels):
        super().__init__()
        self.ups = nn.ModuleList()

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ViTBlock(feature * 2, feature, patch_size=2))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.scale_layer = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def forward(self, x, skip_connections):
        skips = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]

            # 尺寸对齐
            if x.shape != skip.shape:
                x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        x = self.final_conv(x)
        return x * self.scale_layer(torch.ones_like(x))


class DualUNetSharedEncoder(nn.Module):
    """共享编码器的双UNet"""

    def __init__(self, in_channels, out_channels, features=[16, 32, 64]):
        super().__init__()
        self.downs = nn.ModuleList()

        # 构建编码器
        for feature in features:
            self.downs.append(ViTBlock(in_channels, feature, patch_size=2))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = ViTBlock(features[-1], features[-1] * 2, patch_size=2)

        # 双解码器
        self.decoder1 = UNETDecoder(features, out_channels)
        self.decoder2 = UNETDecoder(features, out_channels)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        return self.decoder1(x, skips), self.decoder2(x, skips)


if __name__ == "__main__":
    # 测试样例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualUNetSharedEncoder(
        in_channels=1,
        out_channels=1,
        features=[16, 32, 64]
    ).to(device)

    # 测试不同尺寸的输入
    test_sizes = [(64, 64), (128, 128), (96, 160)]
    for h, w in test_sizes:
        x = torch.randn(2, 1, h, w).to(device)
        out1, out2 = model(x)
        print(f"输入尺寸 ({h}, {w}) => 输出尺寸: {out1.shape}, {out2.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")