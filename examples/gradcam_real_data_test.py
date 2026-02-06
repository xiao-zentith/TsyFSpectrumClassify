#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实数据的Grad-CAM测试案例
测试Grad-CAM功能在真实光谱数据上的表现
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# 添加项目根目录到路径
sys.path.append('/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote')

from src.utils.interpretability.gradcam import SpectralGradCAM, get_model_target_layers
from src.regression.models.unet import UNET
from src.regression.models.dualunet import DualUNet


class RealDataset(Dataset):
    """真实数据集加载器"""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 读取Excel文件，跳过第一行并排除第一列
        input_df = pd.read_excel(item['input'], header=None, usecols=lambda x: x != 0, skiprows=1)
        output_df1 = pd.read_excel(item['targets'][0], header=None, usecols=lambda x: x != 0, skiprows=1)
        output_df2 = pd.read_excel(item['targets'][1], header=None, usecols=lambda x: x != 0, skiprows=1)

        # 转换为NumPy数组
        input_matrix = input_df.to_numpy()
        output_matrix1 = output_df1.to_numpy()
        output_matrix2 = output_df2.to_numpy()

        # 转换为张量并添加通道维度
        input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0)  # 添加通道维度
        output_tensor1 = torch.from_numpy(output_matrix1).float().unsqueeze(0)
        output_tensor2 = torch.from_numpy(output_matrix2).float().unsqueeze(0)

        return input_tensor, output_tensor1, output_tensor2


def load_real_data_config():
    """加载真实数据配置"""
    config_path = '/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression/regression_config_FITC_HPTS.json'
    dataset_info_path = '/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression/regression_dataset_info_FITC_HPTS.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    return config, dataset_info


def get_data_shape(data_list):
    """获取数据形状"""
    if not data_list:
        return None
    
    # 读取第一个文件获取形状
    first_item = data_list[0]
    input_df = pd.read_excel(first_item['input'], header=None, usecols=lambda x: x != 0, skiprows=1)
    return input_df.shape


def example_1_real_data_basic_analysis():
    """示例1: 使用真实数据的基础Grad-CAM分析"""
    print("=== 示例1: 真实数据基础分析 ===")
    
    try:
        # 加载配置和数据信息
        config, dataset_info = load_real_data_config()
        
        # 使用第一个fold的测试数据
        test_data = dataset_info[0]['test'][:3]  # 只取前3个样本进行测试
        
        # 获取数据形状
        data_shape = get_data_shape(test_data)
        print(f"真实数据形状: {data_shape}")
        
        # 创建模型（根据真实数据形状调整）
        # 假设数据是63x63，使用适合的特征层数
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[32, 64, 128, 256])
        model.eval()
        
        # 创建数据集和数据加载器
        dataset = RealDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 获取一个样本
        input_tensor, target1, target2 = next(iter(dataloader))
        print(f"输入张量形状: {input_tensor.shape}")
        
        # 创建Grad-CAM分析器
        grad_cam = SpectralGradCAM(model, model_type='unet')
        
        # 进行分析 - 使用正确的4D张量
        results = grad_cam.generate_cam(input_tensor, target_class=None)
        
        # 显示结果
        print(f"分析完成，共分析了 {len(results)} 个层")
        for layer_name, result in list(results.items())[:3]:  # 只显示前3个层
            print(f"层 {layer_name}:")
            print(f"  - CAM形状: {result['cam'].shape}")
            print(f"  - 峰值强度: {result['peak_intensity']:.4f}")
            print(f"  - 平均强度: {result['mean_intensity']:.4f}")
            if 'important_wavelengths' in result:
                print(f"  - 重要波长数量: {len(result['important_wavelengths'])}")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()


def example_2_real_data_model_comparison():
    """示例2: 使用真实数据的模型对比分析"""
    print("\n=== 示例2: 真实数据模型对比 ===")
    
    try:
        # 加载配置和数据信息
        config, dataset_info = load_real_data_config()
        
        # 使用第一个fold的测试数据
        test_data = dataset_info[0]['test'][:2]  # 只取前2个样本进行测试
        
        # 获取数据形状
        data_shape = get_data_shape(test_data)
        
        # 创建不同的模型
        models = {
            'UNET_Small': UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64]),
            'UNET_Medium': UNET(is_norm=False, in_channels=1, out_channels=1, features=[32, 64, 128]),
            'DualUNet': DualUNet(is_norm=False, in_channels=1, out_channels=1, branch_number=2)
        }
        
        # 创建数据集
        dataset = RealDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 获取一个样本
        input_tensor, target1, target2 = next(iter(dataloader))
        
        # 对比不同模型
        comparison_results = {}
        for model_name, model in models.items():
            try:
                model.eval()
                grad_cam = SpectralGradCAM(model, model_type='unet')
                # 进行分析
                results = grad_cam.generate_cam(input_tensor, target_class=None)
                
                # 获取第一个层的结果
                first_layer = list(results.keys())[0]
                result = results[first_layer]
                
                comparison_results[model_name] = {
                    'peak_intensity': result['peak_intensity'],
                    'mean_intensity': result['mean_intensity'],
                    'cam_shape': result['cam'].shape,
                    'layers_count': len(results)
                }
                
                print(f"{model_name} - 峰值强度: {result['peak_intensity']:.4f}, "
                      f"平均强度: {result['mean_intensity']:.4f}, "
                      f"分析层数: {len(results)}")
                
                del grad_cam
                
            except Exception as e:
                print(f"{model_name} 分析失败: {e}")
        
        # 找出最佳模型
        if comparison_results:
            best_model = max(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]['peak_intensity'])
            print(f"最高峰值强度模型: {best_model}")
        
    except Exception as e:
        print(f"模型对比失败: {e}")
        import traceback
        traceback.print_exc()


def example_3_real_data_batch_analysis():
    """示例3: 使用真实数据的批量分析"""
    print("\n=== 示例3: 真实数据批量分析 ===")
    
    try:
        # 加载配置和数据信息
        config, dataset_info = load_real_data_config()
        
        # 使用第一个fold的测试数据
        test_data = dataset_info[0]['test'][:5]  # 取前5个样本进行批量测试
        
        # 创建模型
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[32, 64, 128])
        model.eval()
        
        # 创建Grad-CAM分析器
        grad_cam = SpectralGradCAM(model, model_type='unet')
        
        # 批量分析
        batch_results = []
        for i, data_item in enumerate(test_data):
            try:
                # 加载单个样本
                input_df = pd.read_excel(data_item['input'], header=None, 
                                       usecols=lambda x: x != 0, skiprows=1)
                input_tensor = torch.from_numpy(input_df.to_numpy()).float().unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
                
                # 进行分析
                results = grad_cam.generate_cam(input_tensor, target_class=None)
                
                # 获取第一个层的结果
                first_layer = list(results.keys())[0]
                result = results[first_layer]
                
                batch_results.append({
                    'sample_id': i,
                    'peak_intensity': result['peak_intensity'],
                    'mean_intensity': result['mean_intensity'],
                    'file_name': os.path.basename(data_item['input'])
                })
                
                print(f"样本 {i+1} ({os.path.basename(data_item['input'])}) - "
                      f"峰值强度: {result['peak_intensity']:.4f}")
                
            except Exception as e:
                print(f"样本 {i+1} 分析失败: {e}")
        
        # 统计结果
        if batch_results:
            peak_intensities = [r['peak_intensity'] for r in batch_results]
            mean_intensities = [r['mean_intensity'] for r in batch_results]
            
            print(f"\n批量分析统计:")
            print(f"  - 成功分析样本数: {len(batch_results)}")
            print(f"  - 平均峰值强度: {np.mean(peak_intensities):.4f}")
            print(f"  - 峰值强度标准差: {np.std(peak_intensities):.4f}")
            print(f"  - 平均均值强度: {np.mean(mean_intensities):.4f}")
            print(f"  - 最高峰值强度样本: {max(batch_results, key=lambda x: x['peak_intensity'])['file_name']}")
        
    except Exception as e:
        print(f"批量分析失败: {e}")
        import traceback
        traceback.print_exc()


def example_4_real_data_layer_analysis():
    """示例4: 使用真实数据的层级分析"""
    print("\n=== 示例4: 真实数据层级分析 ===")
    
    try:
        # 加载配置和数据信息
        config, dataset_info = load_real_data_config()
        
        # 使用第一个fold的测试数据
        test_data = dataset_info[0]['test'][:1]  # 只取1个样本进行详细分析
        
        # 创建模型
        model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[32, 64, 128, 256])
        model.eval()
        
        # 获取目标层
        target_layers = get_model_target_layers(model, 'unet')
        print(f"找到 {len(target_layers)} 个目标层")
        
        # 加载样本
        data_item = test_data[0]
        input_df = pd.read_excel(data_item['input'], header=None, 
                               usecols=lambda x: x != 0, skiprows=1)
        input_tensor = torch.from_numpy(input_df.to_numpy()).float().unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        print(f"分析文件: {os.path.basename(data_item['input'])}")
        print(f"输入数据形状: {input_tensor.shape}")
        
        # 分析每个层
        layer_results = {}
        for i, layer in enumerate(target_layers[:5]):  # 只分析前5层
            try:
                grad_cam_layer = SpectralGradCAM(model, target_layers=[layer], model_type='unet')
                results = grad_cam_layer.generate_cam(input_tensor, target_class=None)
                
                # 获取该层的结果
                if layer in results:
                    result = results[layer]
                    layer_results[f'layer_{i}'] = result['peak_intensity']
                    print(f"层 {i+1} ({layer.__class__.__name__}) 峰值强度: {result['peak_intensity']:.4f}")
                else:
                    # 如果指定层不在结果中，取第一个可用层
                    first_layer = list(results.keys())[0]
                    result = results[first_layer]
                    layer_results[f'layer_{i}'] = result['peak_intensity']
                    print(f"层 {i+1} 峰值强度: {result['peak_intensity']:.4f}")
                
                del grad_cam_layer
                
            except Exception as e:
                print(f"层 {i+1} 分析失败: {e}")
        
        # 层重要性排序
        if layer_results:
            sorted_layers = sorted(layer_results.items(), key=lambda x: x[1], reverse=True)
            print(f"\n层重要性排序:")
            for layer_name, intensity in sorted_layers:
                print(f"  {layer_name}: {intensity:.4f}")
        
    except Exception as e:
        print(f"层级分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Grad-CAM 真实数据测试案例")
    print("=" * 50)
    
    # 检查数据文件是否存在
    config_path = '/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression/regression_config_FITC_HPTS.json'
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 运行所有示例
    example_1_real_data_basic_analysis()
    print()
    
    example_2_real_data_model_comparison()
    print()
    
    example_3_real_data_batch_analysis()
    print()
    
    example_4_real_data_layer_analysis()
    
    print("\n所有真实数据测试完成！")