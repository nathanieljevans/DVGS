''''''
from torch.utils.data import Dataset
import torch

class GenDataset(Dataset):
    def __init__(self, x, y, return_index=False):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(self.x.size(0), -1)
        self.return_index = return_index 

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):

        if self.return_index: 
            return self.x[idx, :], self.y[idx, :], torch.tensor([idx], dtype=torch.long)
        else:
            return self.x[idx, :], self.y[idx, :]