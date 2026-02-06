import pandas as pd
import torch
from torch.nn import functional as F
import numpy as np

# 测试数据加载和tensor转换
test_file = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/data/dataset/dataset_preprocess/C6_FITC/C1 + F9_extracted.xlsx"

print(f"Testing file: {test_file}")

try:
    # 使用原始方法读取数据
    input_df = pd.read_excel(test_file, header=None, usecols=lambda x: x != 0, skiprows=1)
    print(f"DataFrame shape: {input_df.shape}")
    
    # 转换为numpy
    input_matrix = input_df.to_numpy()
    print(f"Numpy matrix shape: {input_matrix.shape}")
    print(f"Matrix dtype: {input_matrix.dtype}")
    
    # 检查是否有NaN值
    print(f"Has NaN values: {np.isnan(input_matrix).any()}")
    print(f"Has inf values: {np.isinf(input_matrix).any()}")
    
    # 转换为tensor
    input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0).unsqueeze(0)
    print(f"Tensor shape after unsqueeze: {input_tensor.shape}")
    
    # 尝试插值
    print("Attempting interpolation...")
    input_tensor_resized = F.interpolate(input_tensor, size=(63, 63), mode='bilinear', align_corners=False)
    print(f"Resized tensor shape: {input_tensor_resized.shape}")
    
    # 移除batch和channel维度
    final_tensor = input_tensor_resized.squeeze(0)
    print(f"Final tensor shape: {final_tensor.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()