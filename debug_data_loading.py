import pandas as pd
import torch
from torch.nn import functional as F

# 测试数据加载
test_file = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/data/dataset/dataset_preprocess/C6_FITC/C1 + F9_extracted.xlsx"

print(f"Testing file: {test_file}")

try:
    # 尝试不同的读取方式
    print("Method 1: Original method")
    input_df = pd.read_excel(test_file, header=None, usecols=lambda x: x != 0, skiprows=1)
    print(f"Shape: {input_df.shape}")
    print(f"First few rows:\n{input_df.head()}")
    
    print("\nMethod 2: Read all columns")
    input_df2 = pd.read_excel(test_file, header=None)
    print(f"Shape: {input_df2.shape}")
    print(f"First few rows:\n{input_df2.head()}")
    
    print("\nMethod 3: Skip first row only")
    input_df3 = pd.read_excel(test_file, header=None, skiprows=1)
    print(f"Shape: {input_df3.shape}")
    print(f"First few rows:\n{input_df3.head()}")
    
except Exception as e:
    print(f"Error: {e}")