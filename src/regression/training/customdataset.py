import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        try:
            # 读取输入数据
            input_df = pd.read_excel(item['input'], header=None, usecols=lambda x: x != 0, skiprows=1)
            
            # 检查数据是否为空
            if input_df.empty or input_df.shape[0] == 0 or input_df.shape[1] == 0:
                print(f"Warning: Empty input data in file: {item['input']}")
                print(f"DataFrame shape: {input_df.shape}")
                # 尝试其他读取方法
                input_df = pd.read_excel(item['input'], header=None, skiprows=1)
                if input_df.shape[1] > 0:
                    input_df = input_df.iloc[:, 1:]  # 移除第一列
                
            if input_df.empty or input_df.shape[0] == 0 or input_df.shape[1] == 0:
                raise ValueError(f"Cannot load valid data from {item['input']}, shape: {input_df.shape}")
            
            input_matrix = input_df.to_numpy()
            input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # 检查tensor尺寸
            if input_tensor.shape[2] == 0 or input_tensor.shape[3] == 0:
                raise ValueError(f"Input tensor has zero dimensions: {input_tensor.shape} from file {item['input']}")

            # Resize 输入到 63x63
            input_tensor_resized = F.interpolate(input_tensor, size=(63, 63), mode='bilinear', align_corners=False).squeeze(0)

            # 动态读取所有 targets
            target_tensors = []
            for target_path in item['targets']:
                try:
                    df = pd.read_excel(target_path, header=None, usecols=lambda x: x != 0, skiprows=1)
                    
                    # 检查target数据是否为空
                    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
                        print(f"Warning: Empty target data in file: {target_path}")
                        # 尝试其他读取方法
                        df = pd.read_excel(target_path, header=None, skiprows=1)
                        if df.shape[1] > 0:
                            df = df.iloc[:, 1:]  # 移除第一列
                    
                    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
                        raise ValueError(f"Cannot load valid target data from {target_path}, shape: {df.shape}")
                    
                    matrix = df.to_numpy()
                    tensor = torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    
                    # 检查tensor尺寸
                    if tensor.shape[2] == 0 or tensor.shape[3] == 0:
                        raise ValueError(f"Target tensor has zero dimensions: {tensor.shape} from file {target_path}")
                    
                    resized_tensor = F.interpolate(tensor, size=(63, 63), mode='bilinear', align_corners=False).squeeze(0)
                    target_tensors.append(resized_tensor)
                    
                except Exception as e:
                    print(f"Error loading target file {target_path}: {e}")
                    raise

            return input_tensor_resized, *target_tensors
            
        except Exception as e:
            print(f"Error in CustomDataset.__getitem__ for index {idx}")
            print(f"Input file: {item['input']}")
            print(f"Target files: {item['targets']}")
            print(f"Error: {e}")
            raise
