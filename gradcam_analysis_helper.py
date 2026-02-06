#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM 热力图分析助手
帮助理解和解释Grad-CAM生成的热力图
"""

import sys
sys.path.append('.')
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.utils.interpretability.gradcam import SpectralGradCAM
from src.utils.interpretability.visualization import GradCAMVisualizer
from src.regression.models.unet import UNET

def create_simple_analysis():
    """创建简单易懂的Grad-CAM分析"""
    
    print("🔍 开始创建Grad-CAM分析...")
    
    # 创建模型
    model = UNET(is_norm=False, in_channels=1, out_channels=2, features=[64, 128, 256, 512])
    model.eval()
    
    # 创建简单的测试数据
    np.random.seed(42)
    input_data = torch.zeros(1, 1, 63, 21)
    
    # 添加两个明显的"峰"
    input_data[0, 0, 20:25, 8:13] = 2.0  # 峰1
    input_data[0, 0, 40:45, 15:20] = 1.5  # 峰2
    input_data += torch.randn_like(input_data) * 0.1  # 噪声
    
    # 生成Grad-CAM
    gradcam = SpectralGradCAM(model, model_type='unet')
    cam_results = gradcam.generate_cam(input_data, target_class=None)
    
    # 创建分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Grad-CAM 热力图分析 - 简化版', fontsize=16, fontweight='bold')
    
    # 1. 原始数据
    im1 = axes[0, 0].imshow(input_data.squeeze().detach().numpy(), cmap='viridis', aspect='auto')
    axes[0, 0].set_title('原始光谱数据\n(黄色=高强度，紫色=低强度)', fontsize=12)
    axes[0, 0].set_xlabel('发射波长方向')
    axes[0, 0].set_ylabel('激发波长方向')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 标注峰位置
    axes[0, 0].add_patch(patches.Rectangle((8, 20), 5, 5, linewidth=3, edgecolor='red', facecolor='none'))
    axes[0, 0].add_patch(patches.Rectangle((15, 40), 5, 5, linewidth=3, edgecolor='red', facecolor='none'))
    axes[0, 0].text(10.5, 18, '峰1', color='red', fontsize=12, ha='center', fontweight='bold')
    axes[0, 0].text(17.5, 38, '峰2', color='red', fontsize=12, ha='center', fontweight='bold')
    
    # 2. 选择一个重要的CAM层
    layer_names = list(cam_results.keys())
    # 找激活最强的层
    best_layer = max(layer_names, key=lambda x: cam_results[x]['peak_intensity'])
    cam_data = cam_results[best_layer]['cam']
    
    # 调整CAM尺寸
    from scipy.ndimage import zoom
    if cam_data.shape != (63, 21):
        zoom_factors = (63 / cam_data.shape[0], 21 / cam_data.shape[1])
        cam_resized = zoom(cam_data, zoom_factors, order=1)
    else:
        cam_resized = cam_data
    
    im2 = axes[0, 1].imshow(cam_resized, cmap='hot', aspect='auto')
    axes[0, 1].set_title(f'Grad-CAM 热力图\n{best_layer}\n(红色=模型关注，黑色=忽略)', fontsize=12)
    axes[0, 1].set_xlabel('发射波长方向')
    axes[0, 1].set_ylabel('激发波长方向')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 叠加显示
    # 归一化数据用于叠加
    input_np = input_data.squeeze().detach().numpy()
    input_norm = (input_np - input_np.min()) / (input_np.max() - input_np.min())
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
    
    axes[1, 0].imshow(input_norm, cmap='gray', aspect='auto', alpha=0.7)
    axes[1, 0].imshow(cam_norm, cmap='hot', aspect='auto', alpha=0.5)
    axes[1, 0].set_title('叠加显示\n(灰色=原始数据，红色=模型关注)', fontsize=12)
    axes[1, 0].set_xlabel('发射波长方向')
    axes[1, 0].set_ylabel('激发波长方向')
    
    # 4. 统计信息
    axes[1, 1].axis('off')
    
    # 计算统计
    peak1_region = cam_resized[20:25, 8:13]
    peak2_region = cam_resized[40:45, 15:20]
    
    # 创建背景区域掩码
    background_mask = np.ones_like(cam_resized, dtype=bool)
    background_mask[20:25, 8:13] = False  # 排除峰1
    background_mask[40:45, 15:20] = False  # 排除峰2
    background = cam_resized[background_mask]
    
    stats_text = f"""
📊 分析结果：

🎯 模型关注度分析：
• 峰1区域平均关注度: {peak1_region.mean():.3f}
• 峰2区域平均关注度: {peak2_region.mean():.3f}
• 背景区域平均关注度: {background.mean():.3f}

🔍 解读：
• 数值越高 = 模型越关注
• 如果峰区域数值高 = 模型正确识别特征
• 如果背景数值高 = 模型可能过拟合噪声

📈 当前模型表现：
"""
    
    if peak1_region.mean() > background.mean() and peak2_region.mean() > background.mean():
        stats_text += "✅ 良好 - 模型正确关注了光谱峰"
    else:
        stats_text += "⚠️  需要改进 - 模型关注点不够准确"
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('gradcam_simple_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 简化分析图已保存为: gradcam_simple_analysis.png")
    
    return stats_text

def explain_gradcam():
    """解释Grad-CAM的基本概念"""
    explanation = """
🎓 Grad-CAM 热力图解读指南：

🔥 热力图颜色含义：
• 红色/黄色/白色 = 模型高度关注的区域
• 橙色 = 中等关注
• 黑色/深蓝色 = 模型忽略的区域

🎯 如何判断模型好坏：
1. 好的模型：热力图集中在有意义的特征上（如光谱峰）
2. 差的模型：热力图分散或集中在噪声区域

🔬 在光谱分析中的应用：
• 验证模型是否识别了正确的光谱特征
• 发现模型可能的偏见
• 指导数据预处理和特征选择

💡 实用建议：
• 多看几个不同的样本
• 对比不同模型的热力图
• 结合领域知识判断合理性
"""
    return explanation

if __name__ == "__main__":
    # 创建分析
    stats = create_simple_analysis()
    
    # 打印解释
    print("\n" + "="*60)
    print(explain_gradcam())
    print("="*60)
    
    print(f"\n📁 生成的文件：")
    print(f"• gradcam_simple_analysis.png - 简化分析图")
    print(f"• 包含原始数据、热力图、叠加显示和统计分析")