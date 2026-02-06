import pandas as pd
import torch
from torch.nn import functional as F
import numpy as np

# 测试配置文件中引用的具体文件
test_file = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/data/dataset/dataset_preprocess/C6_FITC/mixed_C3 + F7_extracted_with_C9 + F1_extracted.xlsx"

print(f"Testing specific file: {test_file}")

try:
    # 使用原始方法读取数据
    input_df = pd.read_excel(test_file, header=None, usecols=lambda x: x != 0, skiprows=1)
    print(f"DataFrame shape: {input_df.shape}")
    
    if input_df.shape[0] == 0 or input_df.shape[1] == 0:
        print("ERROR: DataFrame is empty!")
        # 尝试其他读取方法
        print("Trying alternative reading methods...")
        
        input_df2 = pd.read_excel(test_file, header=None)
        print(f"All columns shape: {input_df2.shape}")
        print(f"First few rows:\n{input_df2.head()}")
        
        input_df3 = pd.read_excel(test_file, header=None, skiprows=1)
        print(f"Skip first row shape: {input_df3.shape}")
        
    else:
        # 转换为numpy
        input_matrix = input_df.to_numpy()
        print(f"Numpy matrix shape: {input_matrix.shape}")
        
        # 转换为tensor
        input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0).unsqueeze(0)
        print(f"Tensor shape after unsqueeze: {input_tensor.shape}")
        
        # 尝试插值
        print("Attempting interpolation...")
        input_tensor_resized = F.interpolate(input_tensor, size=(63, 63), mode='bilinear', align_corners=False)
        print(f"Resized tensor shape: {input_tensor_resized.shape}")
        print("SUCCESS: File processed correctly!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()