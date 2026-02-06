import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import numpy as np

# 尝试导入 tqdm 显示进度条，如果没有安装也不影响运行
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.cached_data = []
        
        # === 核心修改：初始化时一次性把所有数据读进内存 ===
        print(f"检测到数据量较小 ({len(data_list)} samples)，正在全部加载到内存以加速训练...")
        print("启动阶段会进行大量 Excel 读取，请耐心等待进度条跑完 (预计 1 分钟)...")
        
        for idx in tqdm(range(len(data_list)), desc="Loading Data"):
            self.cached_data.append(self._load_item_from_disk(idx))
            
        print("数据加载完毕！后续训练将不再读取硬盘，速度将起飞。")

    def _load_item_from_disk(self, idx):
        """这是原本的读取逻辑，只在初始化时调用一次"""
        item = self.data_list[idx]

        # 1. 读取 Input Excel
        # 保持你原本的逻辑：skiprows=1, usecols 过滤 0
        input_df = pd.read_excel(item['input'], header=None, usecols=lambda x: x != 0, skiprows=1)
        input_matrix = input_df.to_numpy()
        input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        # Resize Input (60x60)
        input_tensor_resized = F.interpolate(input_tensor, size=(60, 60), mode='bilinear', align_corners=False).squeeze(0)

        # 2. 读取 Target Excels
        target_tensors = []
        for target_path in item['targets']:
            df = pd.read_excel(target_path, header=None, usecols=lambda x: x != 0, skiprows=1)
            matrix = df.to_numpy()
            tensor = torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)
            
            # Resize Target (60x60)
            resized_tensor = F.interpolate(tensor, size=(60, 60), mode='bilinear', align_corners=False).squeeze(0)
            target_tensors.append(resized_tensor)

        return input_tensor_resized, target_tensors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # === 训练时直接从内存拿数据，0 延迟 ===
        input_tensor, target_tensors = self.cached_data[idx]
        # 保持和你原始代码一致的返回格式，这样不需要改 train_model
        return input_tensor, *target_tensors