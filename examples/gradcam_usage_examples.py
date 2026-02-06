#!/usr/bin/env python3
"""
Grad-CAM Usage Examples for Spectral Data Analysis

This file contains various examples of how to use the Grad-CAM interpretability
tools with different models and data types.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.interpretability.gradcam import GradCAM, SpectralGradCAM, get_model_target_layers
from src.utils.interpretability.visualization import GradCAMVisualizer
from src.utils.interpretability.model_wrapper import SpectralModelAnalyzer
from src.regression.models.unet import UNET
from src.regression.models.dualunet import DualUNet


def example_1_basic_gradcam():
    """基本Grad-CAM使用示例"""
    print("=== 示例1: 基本Grad-CAM分析 ===")
    
    try:
        # 创建简单的UNet模型用于演示，使用正确的配置
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64])
        
        # 创建模拟的2D光谱数据 (63x63，与真实数据维度一致)
        spectral_data = np.random.randn(63, 63)
        
        # 创建SpectralGradCAM实例
        grad_cam = SpectralGradCAM(model, model_type='unet')
        
        # 生成CAM
        results = grad_cam.generate_cam(spectral_data, target_class=None)
        
        # 获取第一个层的结果作为示例
        first_layer = list(results.keys())[0]
        result = results[first_layer]
        
        print(f"CAM形状: {result['cam'].shape}")
        print(f"峰值强度: {result['peak_intensity']:.4f}")
        print(f"平均强度: {result['mean_intensity']:.4f}")
        print(f"分析的层数: {len(results)}")
        
        del grad_cam
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


def example_2_model_comparison():
    """模型比较示例"""
    print("=== 示例2: 模型比较 ===")
    
    try:
        # 创建两个不同的模型
        from src.regression.models.unet import UNET
        from src.regression.models.dualunet import DualUNet
        
        model1 = UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64])
        model2 = DualUNet(is_norm=False, in_channels=1, out_channels=1, branch_number=2, features=[16, 32, 64])
        
        models = {'UNet': model1, 'DualUNet': model2}
        
        # 创建模拟的2D光谱数据
        spectral_data = np.random.randn(63, 63)
        
        # 比较模型
        for name, model in models.items():
            try:
                grad_cam = SpectralGradCAM(model, model_type='unet')
                results = grad_cam.generate_cam(spectral_data, target_class=None)
                
                # 获取第一个层的结果
                first_layer = list(results.keys())[0]
                result = results[first_layer]
                
                print(f"{name} - CAM形状: {result['cam'].shape}, 峰值强度: {result['peak_intensity']:.4f}")
                del grad_cam
            except Exception as e:
                print(f"{name} - 错误: {e}")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


def example_3_batch_analysis():
    """批量分析示例"""
    print("=== 示例3: 批量分析 ===")
    
    try:
        # 创建模型
        from src.regression.models.unet import UNET
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64])
        
        # 创建批量数据
        batch_data = [np.random.randn(63, 63) for _ in range(3)]
        
        # 批量分析
        grad_cam = SpectralGradCAM(model, model_type='unet')
        
        results = []
        for i, data in enumerate(batch_data):
            batch_results = grad_cam.generate_cam(data, target_class=None)
            
            # 获取第一个层的结果
            first_layer = list(batch_results.keys())[0]
            result = batch_results[first_layer]
            
            results.append(result)
            print(f"样本 {i+1} - CAM形状: {result['cam'].shape}, 峰值强度: {result['peak_intensity']:.4f}")
        
        # 计算平均统计
        avg_peak = np.mean([r['peak_intensity'] for r in results])
        avg_mean = np.mean([r['mean_intensity'] for r in results])
        print(f"平均峰值强度: {avg_peak:.4f}")
        print(f"平均均值强度: {avg_mean:.4f}")
        
        del grad_cam
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


def example_4_2d_spectral_analysis():
    """2D光谱数据分析示例"""
    print("=== 示例4: 2D光谱数据分析 ===")
    
    try:
        # 创建模型
        from src.regression.models.unet import UNET
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64])
        
        # 创建2D光谱数据
        spectral_2d = np.random.randn(63, 63)
        
        # 分析2D数据
        grad_cam = SpectralGradCAM(model, model_type='unet')
        results = grad_cam.generate_cam(spectral_2d, target_class=None)
        
        # 获取第一个层的结果
        first_layer = list(results.keys())[0]
        result = results[first_layer]
        
        print(f"输入数据形状: {spectral_2d.shape}")
        print(f"CAM形状: {result['cam'].shape}")
        print(f"峰值强度: {result['peak_intensity']:.4f}")
        print(f"平均强度: {result['mean_intensity']:.4f}")
        
        # 分析重要区域
        cam = result['cam']
        threshold = np.percentile(cam, 90)  # 90%分位数作为阈值
        important_regions = np.where(cam > threshold)
        print(f"重要区域数量: {len(important_regions[0])}")
        
        del grad_cam
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


def example_5_layer_importance_analysis():
    """层重要性分析示例"""
    print("=== 示例5: 层重要性分析 ===")
    
    try:
        # 创建模型
        from src.regression.models.unet import UNET
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64])
        
        # 创建光谱数据
        spectral_data = np.random.randn(63, 63)
        
        # 分析层重要性
        grad_cam = SpectralGradCAM(model, model_type='unet')
        
        # 获取目标层
        target_layers = get_model_target_layers(model, 'unet')
        print(f"找到 {len(target_layers)} 个目标层")
        
        # 分析每个层的重要性
        layer_results = {}
        for i, layer in enumerate(target_layers[:3]):  # 只分析前3层避免过多输出
            try:
                grad_cam_layer = SpectralGradCAM(model, target_layers=[layer], model_type='unet')
                results = grad_cam_layer.generate_cam(spectral_data, target_class=None)
                
                # 获取该层的结果
                if layer in results:
                    result = results[layer]
                    layer_results[f'layer_{i}'] = result['peak_intensity']
                    print(f"层 {i+1} 峰值强度: {result['peak_intensity']:.4f}")
                else:
                    # 如果指定层不在结果中，取第一个可用层
                    first_layer = list(results.keys())[0]
                    result = results[first_layer]
                    layer_results[f'layer_{i}'] = result['peak_intensity']
                    print(f"层 {i+1} 峰值强度: {result['peak_intensity']:.4f}")
                
                del grad_cam_layer
            except Exception as e:
                print(f"层 {i+1} 分析失败: {e}")
        
        # 排序层重要性
        if layer_results:
            sorted_layers = sorted(layer_results.items(), key=lambda x: x[1], reverse=True)
            print("层重要性排序:")
            for layer_name, importance in sorted_layers:
                print(f"  {layer_name}: {importance:.4f}")
        
        del grad_cam
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


def example_6_training_integration():
    """训练流程集成示例"""
    print("=== 示例6: 训练流程集成 ===")
    
    try:
        # 创建模型
        from src.regression.models.unet import UNET
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64])
        
        # 模拟训练后的模型状态
        model.eval()
        
        # 创建测试数据
        test_data = np.random.randn(63, 63)
        
        # 在训练后进行Grad-CAM分析
        print("模拟训练完成，开始Grad-CAM分析...")
        
        grad_cam = SpectralGradCAM(model, model_type='unet')
        results = grad_cam.generate_cam(test_data, target_class=None)
        
        # 获取第一个层的结果
        first_layer = list(results.keys())[0]
        result = results[first_layer]
        
        print(f"模型预测完成")
        print(f"CAM分析结果:")
        print(f"  - CAM形状: {result['cam'].shape}")
        print(f"  - 峰值强度: {result['peak_intensity']:.4f}")
        print(f"  - 平均强度: {result['mean_intensity']:.4f}")
        print(f"  - 分析的层数: {len(results)}")
        
        # 模拟保存分析结果
        print("Grad-CAM分析结果已保存到训练日志中")
        
        del grad_cam
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Grad-CAM 使用示例")
    print("=" * 50)
    
    # 运行所有示例
    example_1_basic_gradcam()
    print()
    
    example_2_model_comparison()
    print()
    
    example_3_batch_analysis()
    print()
    
    example_4_2d_spectral_analysis()
    print()
    
    example_5_layer_importance_analysis()
    print()
    
    example_6_training_integration()
    print()
    
    print("所有示例运行完成！")