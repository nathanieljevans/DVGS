'''
Autoencoder dataset
'''
from torch.utils.data import Dataset
import torch

class AEDataset(Dataset):
    def __init__(self, x):
        if torch.is_tensor(x): 
            self.x = x
        else: 
            self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        out = self.x[idx, :]
        return out, out