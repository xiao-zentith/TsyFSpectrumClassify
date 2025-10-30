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

        # Read Excel files skipping the first row and excluding the first column
        input_df = pd.read_excel(item['input'], header=None, usecols=lambda x: x != 0, skiprows=1)
        output_df1 = pd.read_excel(item['targets'][0], header=None, usecols=lambda x: x != 0, skiprows=1)
        output_df2 = pd.read_excel(item['targets'][1], header=None, usecols=lambda x: x != 0, skiprows=1)

        # Convert to NumPy array
        input_matrix = input_df.to_numpy()
        output_matrix1 = output_df1.to_numpy()
        output_matrix2 = output_df2.to_numpy()

        # Convert to tensor and add channel dimension
        input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0)
        output_tensor1 = torch.from_numpy(output_matrix1).float().unsqueeze(0)
        output_tensor2 = torch.from_numpy(output_matrix2).float().unsqueeze(0)

        return input_tensor, output_tensor1, output_tensor2

        # # Resize tensors to 63x63 using bilinear interpolation
        # input_tensor_resized = F.interpolate(input_tensor, size=(63, 63), mode='bilinear', align_corners=False)
        # output_tensor1_resized = F.interpolate(output_tensor1, size=(63, 63), mode='bilinear', align_corners=False)
        # output_tensor2_resized = F.interpolate(output_tensor2, size=(63, 63), mode='bilinear', align_corners=False)
        #
        # # Remove batch dimension
        # input_tensor_resized = input_tensor_resized.squeeze(0)
        # output_tensor1_resized = output_tensor1_resized.squeeze(0)
        # output_tensor2_resized = output_tensor2_resized.squeeze(0)
        #
        # return input_tensor_resized, output_tensor1_resized, output_tensor2_resized



