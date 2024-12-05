import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """Custom Dataset for loading the input data and the corresponding masks"""

    def __init__(self, data, masks):
        self.data = data
        self.masks = masks

    def __getitem__(self, index):
        return self.data[index], self.masks[index]

    def __len__(self):
        return len(self.data)