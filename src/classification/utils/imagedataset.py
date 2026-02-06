import numpy as np
from torch.utils.data import Dataset
import torch

from src.utils.data_io.matrix_reader import read_matrix_from_file


class ImageDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        x_coords, y_coords, matrix = read_matrix_from_file(file_path)
        x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y

