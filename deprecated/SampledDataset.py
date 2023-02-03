'''sampled dataset

samples values until without replacement each 
'''
from torch.utils.data import Dataset
import torch

class GenDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(self.x.size(0), -1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]