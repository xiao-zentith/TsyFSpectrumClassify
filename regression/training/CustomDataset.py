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

        # 读取输入数据
        input_df = pd.read_excel(item['input'], header=None, usecols=lambda x: x != 0, skiprows=1)
        input_matrix = input_df.to_numpy()
        input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Resize 输入到 63x63
        # input_tensor_resized = F.interpolate(input_tensor, size=(360, 360), mode='bilinear', align_corners=False).squeeze(0)
        input_tensor_resized = F.interpolate(input_tensor, size=(63, 63), mode='bilinear', align_corners=False).squeeze(0)

        # 动态读取所有 targets
        target_tensors = []
        for target_path in item['targets']:
            df = pd.read_excel(target_path, header=None, usecols=lambda x: x != 0, skiprows=1)
            matrix = df.to_numpy()
            tensor = torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            # resized_tensor = F.interpolate(tensor, size=(360, 360), mode='bilinear', align_corners=False).squeeze(0)
            resized_tensor = F.interpolate(tensor, size=(63, 63), mode='bilinear', align_corners=False).squeeze(0)
            target_tensors.append(resized_tensor)

        return input_tensor_resized, *target_tensors
